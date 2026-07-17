"""Data preparation for E1: cybench sample-level cleaning.

Policy (documented in FINDINGS.md):
- Drop rows with no parseable score (bridge does this): these coincide with
  infra errors (Helm/K8s failures) where the environment never started — they
  are missing data, not model failures.
- Keep rows that hit limits (token/message/time/operator): the eval scored
  them ('I' unless the flag was found first), a genuine failure under the
  eval's resource budget.
- Normalise item_id (three naming conventions observed) into
  (challenge, variant) and merge spelling aliases across conventions.
- Deduplicate re-ingested runs: whole runs whose scored content is identical
  to another run of the same model (same challenge/variant/epoch/score/
  total_tokens/message_count multiset) — keep the lowest run_internal_id.
- Headline analyses restrict to the dominant 'hard' variant.
"""

import hashlib
import json
import re

import pandas as pd

from shared.bridge import load_samples

FOCAL_MODELS = [
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-opus-4-20250514",
    "gemma/gemma-3-27b-it",
    "mistralazure/Mistral-Large-2411",
    "openai/gpt-4o-2024-08-06",
    "openai/gpt-5",
    "openai/o1",
    "openai/o1-2024-12-17",
    "openai/o3",
    "openai/o3-mini",
]

# Models present only with the 'easy' variant on <=8 challenges (repeated
# single-epoch runs) — excluded from hard-variant fits, used as design
# illustrations only.
EASY_ONLY_MODELS = [
    "anthropic/claude-opus-4-1-20250805",
    "anthropic/claude-sonnet-4-20250514",
    "openai/o4-mini",
]

_PAREN = re.compile(r"^(.*?) \((\w+)\)$")
_SUFFIX = re.compile(r"^(.*)-(easy|hard|solution)$")

# same challenge, different spelling across item_id conventions
CHALLENGE_ALIASES = {
    "lock_talk": "locktalk",
    "loot_stash": "lootstash",
    "missing_bits": "missingbits",
    "packed_away": "packedaway",
    "emaze": "ezmaze",
}


def norm_item(item_id: str) -> tuple[str, str]:
    """Split an item_id into (challenge, variant), normalising 3 conventions."""
    m = _PAREN.match(item_id)
    if m:
        name, var = m.group(1), m.group(2)
    else:
        m2 = _SUFFIX.match(item_id)
        if m2:
            name, var = m2.group(1), m2.group(2)
        else:
            name, var = item_id, "unknown"
    name = name.lower().replace("-", "_")
    return CHALLENGE_ALIASES.get(name, name), var.removeprefix("cybench_")


def _task_args_field(s: str, key: str) -> object:
    d = json.loads(s) if s else {}
    v = d.get(key)
    if isinstance(v, list):
        v = ",".join(map(str, v))
    return v


def _dedupe_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop whole runs whose scored content duplicates an earlier run."""

    def run_hash(g: pd.DataFrame) -> str:
        content = g.sort_values(["challenge", "variant", "epoch", "total_tokens"])[
            ["challenge", "variant", "epoch", "score", "total_tokens",
             "message_count"]
        ].to_csv(index=False)
        return hashlib.md5(content.encode()).hexdigest()

    hashes = (
        df.groupby(["model", "run_internal_id"])
        .apply(run_hash, include_groups=False)
        .reset_index(name="h")
        .sort_values("run_internal_id")
    )
    dup = hashes[hashes.duplicated(["model", "h"], keep="first")]
    keep = df[~df["run_internal_id"].isin(set(dup["run_internal_id"]))]
    return keep, len(dup), len(df) - len(keep)


def load_clean(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load cybench samples, normalise, dedupe. Returns (df, audit dict)."""
    audit: dict = {}
    raw_rows = len(pd.read_parquet("data/cybench.samples.parquet"))
    df = load_samples("cybench", score_col="score_includes")
    audit["rows_raw"] = raw_rows
    audit["rows_dropped_unscored"] = raw_rows - len(df)
    audit["rows_scored"] = len(df)

    ni = df["item_id"].map(norm_item)
    df["challenge"] = ni.map(lambda t: t[0])
    df["variant"] = ni.map(lambda t: t[1])
    df["has_error"] = df["error"].fillna("").str.len() > 0
    audit["rows_with_error_but_scored"] = int(df["has_error"].sum())
    audit["rows_limit_hit_kept"] = int(df["limit"].notna().sum())

    df, n_dup_runs, n_dup_rows = _dedupe_runs(df)
    audit["duplicate_runs_dropped"] = n_dup_runs
    audit["rows_deduped"] = n_dup_rows
    audit["rows_clean"] = len(df)
    if verbose:
        print(f"[prep] audit: {audit}")
    return df.reset_index(drop=True), audit


def hard_focal(df: pd.DataFrame) -> pd.DataFrame:
    """Hard-variant rows for the focal (well-covered) models.

    Also excludes degenerate harness-failure runs: broad-coverage runs
    (>=20 attempts, >=10 challenges) with zero solves for a model whose other
    runs solve >15% — verified mechanistically (every sample terminated at
    message_count=2 with a token limit, i.e. the agent loop never ran).
    Excluded run ids and row counts are recorded in df.attrs["degenerate"].
    """
    out = df[(df["variant"] == "hard") & df["model"].isin(FOCAL_MODELS)].copy()
    out["run"] = out["run_internal_id"].astype(str)

    per_run = out.groupby(["model", "run"]).agg(
        n=("score", "size"), acc=("score", "mean"),
        chal=("challenge", "nunique"))
    model_acc = out.groupby("model")["score"].mean()
    deg = per_run[(per_run["n"] >= 20) & (per_run["chal"] >= 10)
                  & (per_run["acc"] == 0.0)
                  & (per_run.index.get_level_values("model").map(model_acc)
                     > 0.15)]
    deg_runs = deg.index.get_level_values("run").tolist()
    n_before = len(out)
    out = out[~out["run"].isin(deg_runs)]
    out = out.reset_index(drop=True)
    out.attrs["degenerate"] = {
        "runs_excluded": deg_runs, "rows_excluded": n_before - len(out)}
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 300)
    df, audit = load_clean()
    hard = hard_focal(df)
    print()
    print(
        hard.groupby("model")
        .agg(rows=("challenge", "size"), chal=("challenge", "nunique"),
             runs=("run", "nunique"), acc=("score", "mean"))
        .to_string()
    )
    print()
    cov = hard.groupby(["challenge", "model"]).size().unstack(fill_value=0)
    print("challenges by number of models covering them:")
    print((cov > 0).sum(axis=1).value_counts().sort_index())
