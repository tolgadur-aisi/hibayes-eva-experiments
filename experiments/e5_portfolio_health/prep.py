"""Data preparation for E5 portfolio health audit.

Hygiene policy (documented in FINDINGS.md):

1. DEDUP: the warehouse contains the same eval log ingested under multiple
   ``internal_id``s (same ``eval_id``/``run_id``/``created``; a log copied to
   several S3 keys is ingested once per key). We dedupe by keeping the lowest
   ``internal_id`` per (task_name, eval_id) using the eval-level extract.
2. ERRORS: sample rows with a non-empty ``error`` (harness/API failures such as
   404 model-not-found, harness AttributeError) have no score and are treated
   as missing data -> dropped and counted. They are not model failures.
3. LIMITS: rows that hit a limit (token/message/time/operator) but carry a
   valid score are KEPT. Budget exhaustion without solving is a genuine
   incorrect outcome for agentic tasks, not missing data.
4. NON-BINARY scores (partial credit) are excluded from binomial aggregation
   and counted.
5. HETEROGENEITY: task_args/solver configs vary across runs (observational
   data). Model x item crossed effects pool over configs; exact-config
   replicate groups isolate run-to-run noise for the reproducibility metric.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

from shared.bridge import DATA_DIR

SAMPLE_TASKS: dict[str, str] = {
    "cybench": "score_includes",
    "gdm_intercode_ctf": "score_includes",
    "swe_bench": "score_swe_bench_scorer",
}
LETTER = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}
MIN_MODEL_N = 50  # min binary samples for a model to enter the crossed fit
MIN_RUN_N = 20  # min samples per run for the reproducibility fit
MIN_GROUP_RUNS = 2  # min runs per exact-config replicate group


def _to_score(v: object) -> float | None:
    """Inspect letter grade or numeric string -> float score (None if unparseable)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s in LETTER:
        return LETTER[s]
    try:
        return float(s)
    except ValueError:
        return None


def _keep_internal_ids() -> tuple[set[int], pd.DataFrame]:
    """Lowest internal_id per (task_name, eval_id); returns ids + eval index."""
    ev = pd.read_parquet(DATA_DIR / "public_evals.parquet")
    keep = ev.groupby(["task_name", "eval_id"])["internal_id"].min()
    return set(keep.tolist()), ev


@dataclass
class TaskData:
    task: str
    df: pd.DataFrame  # deduped, scored, binary sample rows
    accounting: dict[str, int | float] = field(default_factory=dict)


def load_task(task: str, keep_ids: set[int]) -> TaskData:
    score_col = SAMPLE_TASKS[task]
    df = pd.read_parquet(DATA_DIR / f"{task}.samples.parquet")
    acct: dict[str, int | float] = {"raw_rows": len(df), "raw_runs": df["run_internal_id"].nunique()}

    dup_mask = ~df["run_internal_id"].isin(keep_ids)
    acct["dup_rows_removed"] = int(dup_mask.sum())
    df = df[~dup_mask].copy()
    acct["runs_after_dedup"] = df["run_internal_id"].nunique()

    df["score"] = df[score_col].map(_to_score)
    err_mask = df["error"].fillna("").ne("") | df["score"].isna()
    acct["error_or_unscored_removed"] = int(err_mask.sum())
    df = df[~err_mask]

    nonbin = ~df["score"].isin([0.0, 1.0])
    acct["nonbinary_removed"] = int(nonbin.sum())
    df = df[~nonbin].copy()

    acct["limit_rows_kept"] = int(df["limit"].fillna("").ne("").sum())
    acct["rows_final"] = len(df)
    acct["models_total"] = df["model"].nunique()
    acct["items_total"] = df["item_id"].nunique()
    # realised epochs per item within a run (for the design-effect scenario)
    per = df.groupby("run_internal_id").agg(items=("item_id", "nunique"), rows=("score", "size"))
    acct["median_epochs_per_item"] = float((per["rows"] / per["items"]).median())
    return TaskData(task=task, df=df, accounting=acct)


def crossed_cells(td: TaskData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(model, item) binomial counts for models with >= MIN_MODEL_N samples.

    Returns (cells, model_inclusion_table).
    """
    df = td.df
    per_model = df.groupby("model")["score"].agg(n="size", k="sum").reset_index()
    per_model["included"] = per_model["n"] >= MIN_MODEL_N
    kept = set(per_model.loc[per_model["included"], "model"])
    sub = df[df["model"].isin(kept)]
    cells = (
        sub.groupby(["model", "item_id"])["score"]
        .agg(n_correct="sum", n_total="size")
        .reset_index()
    )
    cells["n_correct"] = cells["n_correct"].astype(int)
    td.accounting["models_kept"] = len(kept)
    td.accounting["rows_in_kept_models"] = int(len(sub))
    td.accounting["crossed_cells"] = len(cells)
    return cells, per_model.sort_values("n", ascending=False)


def replicate_runs(td: TaskData) -> pd.DataFrame:
    """Runs (n >= MIN_RUN_N) in exact-config replicate groups (>= 2 runs).

    A group = identical (model, task_args, solver, run_epochs, item set).
    Returns one row per run with a ``group`` label, n_total, n_correct.
    """
    df = td.df
    runs = (
        df.groupby("run_internal_id")
        .agg(
            model=("model", "first"),
            task_args=("task_args", lambda s: str(s.iloc[0])),
            solver=("solver", lambda s: str(s.iloc[0])),
            run_epochs=("run_epochs", "first"),
            itemset=("item_id", lambda s: hashlib.md5(",".join(sorted(set(s))).encode()).hexdigest()[:10]),
            n_total=("score", "size"),
            n_correct=("score", "sum"),
        )
        .reset_index()
    )
    runs = runs[runs["n_total"] >= MIN_RUN_N].copy()
    key = ["model", "task_args", "solver", "run_epochs", "itemset"]
    sizes = runs.groupby(key)["run_internal_id"].transform("size")
    runs = runs[sizes >= MIN_GROUP_RUNS].copy()
    gid = runs.groupby(key).ngroup()
    runs["group"] = runs["model"].str.split("/").str[-1] + "|g" + gid.astype(str)
    runs["n_correct"] = runs["n_correct"].astype(int)
    runs["acc"] = runs["n_correct"] / runs["n_total"]
    td.accounting["repro_groups"] = runs["group"].nunique()
    td.accounting["repro_runs"] = len(runs)
    return runs[["group", "model", "run_epochs", "n_total", "n_correct", "acc"]]


def prededup_replicate_groups(task: str, min_run_n: int = 10) -> pd.DataFrame:
    """PRE-dedup replicate-group accounting (traceability for Finding 1).

    Same cleaning (error/unscored/non-binary dropped) and same group key as
    :func:`replicate_runs`, but WITHOUT the (task, eval_id) dedup and with a
    lower ``min_run_n`` threshold (10, matching the pre-dedup headline claim;
    the post-dedup reproducibility fit uses 20). Duplicate-ingest copies of a
    run land in the same group with identical accuracy, so sd_acc == 0 flags
    copies masquerading as replicates. Returns one row per group.
    """
    score_col = SAMPLE_TASKS[task]
    df = pd.read_parquet(DATA_DIR / f"{task}.samples.parquet")
    df["score"] = df[score_col].map(_to_score)
    df = df[~(df["error"].fillna("").ne("") | df["score"].isna())]
    df = df[df["score"].isin([0.0, 1.0])].copy()
    runs = (
        df.groupby("run_internal_id")
        .agg(
            model=("model", "first"),
            task_args=("task_args", lambda s: str(s.iloc[0])),
            solver=("solver", lambda s: str(s.iloc[0])),
            run_epochs=("run_epochs", "first"),
            itemset=("item_id", lambda s: hashlib.md5(",".join(sorted(set(s))).encode()).hexdigest()[:10]),
            n_total=("score", "size"),
            n_correct=("score", "sum"),
        )
        .reset_index()
    )
    runs = runs[runs["n_total"] >= min_run_n].copy()
    key = ["model", "task_args", "solver", "run_epochs", "itemset"]
    sizes = runs.groupby(key)["run_internal_id"].transform("size")
    runs = runs[sizes >= MIN_GROUP_RUNS].copy()
    runs["acc"] = runs["n_correct"] / runs["n_total"]
    grp = (
        runs.groupby(key)
        .agg(n_runs=("acc", "size"), mean_acc=("acc", "mean"), sd_acc=("acc", "std"),
             med_n=("n_total", "median"))
        .reset_index()
    )
    grp.insert(0, "task", task)
    grp["min_run_n"] = min_run_n
    grp["sd_zero"] = grp["sd_acc"] == 0
    return grp.drop(columns=["task_args", "solver", "itemset"]).assign(
        model=grp["model"].str.split("/").str[-1]
    )


def load_boolq() -> tuple[pd.DataFrame, dict[str, int | float]]:
    """boolq_preference eval-level rows: pattern scorer only, deduped by eval_id.

    Only the ``pattern`` scorer rows carry score_headline_value/stderr (the
    ``boolq_scorer`` rows have neither), so the audit uses that subset.
    Parses charity/split condition arms from task_args.
    """
    ev = pd.read_parquet(DATA_DIR / "public_evals.parquet")
    bq = ev[(ev["task_name"] == "boolq_preference") & (ev["score_headline_name"] == "pattern")].copy()
    acct: dict[str, int | float] = {"raw_rows": len(bq)}
    bq = bq.sort_values("internal_id").drop_duplicates(subset="eval_id", keep="first")
    acct["dup_rows_removed"] = acct["raw_rows"] - len(bq)
    bq = bq.dropna(subset=["score_headline_value", "score_headline_stderr"]).copy()

    def parse(s: object) -> dict:
        try:
            return json.loads(s) if isinstance(s, str) else {}
        except (TypeError, ValueError):
            return {}

    args = bq["task_args"].map(parse)
    bq["charity"] = args.map(lambda d: str(d.get("charity", "none_specified")))
    bq["split"] = args.map(lambda d: str(d.get("split", "unknown")))
    bq = bq.rename(columns={"score_headline_value": "value", "score_headline_stderr": "se"})
    # guard: stderr must be positive for the normal likelihood
    bad_se = (bq["se"] <= 0) | bq["se"].isna()
    acct["nonpositive_se_removed"] = int(bad_se.sum())
    bq = bq[~bad_se]
    acct["rows_final"] = len(bq)
    acct["models_total"] = bq["model"].nunique()
    acct["charities"] = bq["charity"].nunique()
    out = bq[["model", "charity", "split", "value", "se", "completed_samples"]].reset_index(drop=True)
    return out, acct
