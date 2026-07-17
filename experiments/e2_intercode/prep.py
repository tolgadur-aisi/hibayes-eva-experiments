"""Data hygiene + dataset construction for E2 (gdm_intercode_ctf variance decomposition).

Policies (documented in FINDINGS.md):
- Rows with a non-empty `error` are dropped (all are unscored gpt-4o 429 quota errors).
- Rows with a `limit` hit (token/message/time/operator) are KEPT with their recorded
  score: hitting the task's resource budget is a genuine failure mode, not missing data.
- Duplicate-ingested eval logs (same model + run start timestamp + identical per-sample
  scores/tokens) are collapsed to one run. Identity is asserted, not assumed.
- Config signature = task_args minus sample selectors (sample_ids, shuffle) plus solver name.
"""

import json

import pandas as pd

from shared.bridge import DATA_DIR

FAN_OUT_MODELS = [
    "anthropic/claude-opus-4-1-20250805",
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-5",
    "openai/o3",
    "openai/o4-mini",
]

VLLM_PREFIX = "vllm/"


def config_signature(task_args: str) -> str:
    """Behavioural config signature: drop sample selectors, collapse install lists."""
    d = json.loads(task_args)
    d.pop("sample_ids", None)
    d.pop("shuffle", None)
    for k in ("pip3_installs", "apt_get_installs"):
        if k in d:
            d[k] = "std"
    if "solver" in d:
        d["solver"] = json.loads(d["solver"]).get("name", "?")
    return json.dumps(d, sort_keys=True)


def load_clean() -> tuple[pd.DataFrame, dict]:
    """Load intercode samples, apply hygiene, return (df, exclusion_log)."""
    df = pd.read_parquet(DATA_DIR / "gdm_intercode_ctf.samples.parquet")
    log: dict = {"raw_rows": len(df)}

    # 1. errored / unscored rows
    err_mask = df["error"].notna() & (df["error"] != "")
    unscored_mask = df["score_includes"].isna()
    assert (err_mask == unscored_mask).all(), "error rows and unscored rows should coincide"
    log["dropped_error_rows"] = int(err_mask.sum())
    df = df[~err_mask].copy()

    # 2. binary score (only C/I remain)
    assert set(df["score_includes"].unique()) <= {"C", "I"}
    df["score"] = (df["score_includes"] == "C").astype(int)
    log["limit_rows_kept"] = int(df["limit"].notna().sum())

    # 3. config signature + solver label
    df["cfg"] = df["task_args"].map(config_signature)
    df["solver_name"] = (
        df["solver"].fillna("na").str.replace("src/agents/intercode_agent.py@", "", regex=False)
    )
    df["variant"] = df["cfg"] + "|" + df["solver_name"]

    # 4. deduplicate duplicate-ingested runs (non-fan-out only; fan-out verified clean)
    non_fan = ~df["model"].isin(FAN_OUT_MODELS)
    fp = (
        df[non_fan]
        .groupby("run_internal_id")
        .agg(
            model=("model", "first"),
            start=("created", "min"),
            rows=("epoch", "size"),
            toks=("total_tokens", "sum"),
            ncorr=("score", "sum"),
        )
        .reset_index()
    )
    fp["fkey"] = (
        fp["model"]
        + "|"
        + fp["start"].astype(str)
        + "|"
        + fp["rows"].astype(str)
        + "|"
        + fp["toks"].astype(str)
        + "|"
        + fp["ncorr"].astype(str)
    )
    keep, drop = [], []
    for _, grp in fp.groupby("fkey"):
        ids = sorted(grp["run_internal_id"])
        keep.append(ids[0])
        drop.extend(ids[1:])
        if len(ids) > 1:  # assert true row-level identity, not just fingerprint match
            piv_s = df[df["run_internal_id"].isin(ids)].pivot_table(
                index=["item_id", "epoch"], columns="run_internal_id", values="score", aggfunc="first"
            )
            piv_t = df[df["run_internal_id"].isin(ids)].pivot_table(
                index=["item_id", "epoch"], columns="run_internal_id", values="total_tokens", aggfunc="first"
            )
            assert piv_s.nunique(axis=1).eq(1).all() and piv_t.nunique(axis=1).eq(1).all(), (
                f"fingerprint-matched runs {ids} are not row-identical"
            )
    log["dup_runs_dropped"] = len(drop)
    log["unique_nonfan_runs"] = len(keep)
    df = df[df["model"].isin(FAN_OUT_MODELS) | ~df["run_internal_id"].isin(drop)].copy()

    # fan-out duplicate check (row-level): must be clean
    fan_rows = df[df["model"].isin(FAN_OUT_MODELS)]
    n_fan_dup = int(
        fan_rows.duplicated(
            subset=["model", "item_id", "created", "total_tokens", "total_time", "score"]
        ).sum()
    )
    assert n_fan_dup == 0, "unexpected duplicates in fan-out rows"

    assert not df.duplicated(subset=["run_internal_id", "item_id", "epoch"]).any()
    log["clean_rows"] = len(df)
    return df, log


def build_d1(df: pd.DataFrame) -> pd.DataFrame:
    """Primary decomposition dataset: multi-sample runs, homogeneous config.

    cfg == '{}', solver == intercode_agent; 79 items x 10 epochs per run.
    Cells: (model, run, item) with n_correct / n_total.
    """
    d = df[
        (~df["model"].isin(FAN_OUT_MODELS))
        & (~df["model"].str.startswith(VLLM_PREFIX))
        & (df["cfg"] == "{}")
        & (df["solver_name"] == "intercode_agent")
    ]
    cells = (
        d.groupby(["model", "run_internal_id", "item_id"])
        .agg(n_correct=("score", "sum"), n_total=("score", "size"))
        .reset_index()
        .rename(columns={"run_internal_id": "run", "item_id": "item"})
    )
    cells["run"] = cells["run"].astype(str)
    return cells


def build_d1x(df: pd.DataFrame) -> pd.DataFrame:
    """Extended dataset with scaffold-variant component.

    All non-fan-out, non-vllm models and ALL their config/solver variants
    (message-limit changes, prompt variants, vanilla_agent_setup for gpt-4o).
    Variant labelled within model.
    """
    d = df[(~df["model"].isin(FAN_OUT_MODELS)) & (~df["model"].str.startswith(VLLM_PREFIX))]
    cells = (
        d.groupby(["model", "variant", "run_internal_id", "item_id"])
        .agg(n_correct=("score", "sum"), n_total=("score", "size"))
        .reset_index()
        .rename(columns={"run_internal_id": "run", "item_id": "item"})
    )
    cells["variant"] = cells["model"] + "||" + cells["variant"]
    cells["run"] = cells["run"].astype(str)
    return cells


def build_d2(df: pd.DataFrame) -> pd.DataFrame:
    """Frontier fan-out dataset: day-batches as the run-analogue.

    The 5 frontier models were run as ~36k single-sample runs over 2025-09-17..21
    (14 items, hundreds of attempts each). run_internal_id is a per-attempt
    artifact there, so temporal day-batches (model x UTC date) proxy the
    'same eval relaunched on a different day' grouping.
    """
    d = df[df["model"].isin(FAN_OUT_MODELS)].copy()
    d["batch"] = d["model"] + "|" + d["created"].dt.floor("D").dt.strftime("%Y-%m-%d")
    cells = (
        d.groupby(["model", "batch", "item_id"])
        .agg(n_correct=("score", "sum"), n_total=("score", "size"))
        .reset_index()
        .rename(columns={"item_id": "item", "batch": "run"})
    )
    return cells
