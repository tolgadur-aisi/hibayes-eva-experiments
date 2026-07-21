"""Step 1 of the modeling recipe: list all evals in eva and classify their outcome.

For every team_ru task, this reports how many runs exist and whether the
scores are pass/fail (binary C/I/N or 0/1), partial-credit, continuous, or
something else. Only pass/fail tasks feed the binomial model. Scope is pinned
to TEAM below -- your eva roles may see other teams' data (cast, chembio, ...)
and an unpinned scan would probe all of it, slowly.

Run from this repo on a platform dev VM with eva access:

    uv run python modeling/discovery/list_evals.py

Outputs (committed so the selection is reviewable):
    modeling/discovery/outputs/evals_inventory.csv
    modeling/discovery/outputs/pass_fail_tasks.txt
"""

import argparse
import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from eva import query, samples
from tqdm import tqdm

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# eva caches ONE psycopg connection per process, so parallel queries need
# separate processes, not threads -- and the pool must use SPAWN: fork would
# copy the parent's cached connection into every worker and they would all
# share one socket (serialized queries at best, protocol errors at worst).
# Keep the pool modest -- each spawned worker opens its own Aurora connection.
PROBE_WORKERS = 8
SPAWN = multiprocessing.get_context("spawn")

TEAM = "team_ru"

BASE_FILTERS = f"""
    e.team = '{TEAM}' AND e.status = 'success'
    AND e.model NOT LIKE 'replay/%%'
    AND e.model NOT IN ('none/none', 'mockllm/model')
"""

TASKS_SQL = f"""
SELECT e.task_name,
       COUNT(*) AS n_evals,
       COUNT(DISTINCT e.model) AS n_models,
       SUM(e.completed_samples) AS n_samples,
       MIN(e.created) AS first_run,
       MAX(e.created) AS last_run,
       STRING_AGG(DISTINCT e.score_headline_name, ', ') AS headline_scorers,
       STRING_AGG(DISTINCT e.score_headline_metric, ', ') AS headline_metrics
FROM inspect.evals e
WHERE {BASE_FILTERS}
GROUP BY e.task_name
ORDER BY n_evals DESC
"""

# A bounded probe of raw score values per task; enough to classify the outcome
# type without scanning millions of rows.
SCORE_PROBE_SQL = f"""
SELECT s.scores
FROM inspect.samples s
JOIN inspect.evals e ON s.eval_internal_id = e.internal_id
WHERE e.task_name = %(task)s AND {BASE_FILTERS}
LIMIT %(limit)s
"""

PASS_FAIL_VALUES = {"C", "I", "N", 0, 1, 0.0, 1.0, "0", "1", "0.0", "1.0", True, False}
PARTIAL_VALUES = {"P", 0.5, "0.5"}

# Ordering used to pick the most decision-relevant score column when a task
# has several (value, answer, explanation, ... all expand to score_* columns).
CLASS_PRIORITY = [
    "pass_fail",
    "pass_fail_with_partial",
    "continuous_0_1",
    "ordinal",
    "continuous",
    "other",
    "no_scores",
]


def classify_values(values: Counter) -> str:
    """Classify a task's observed score values into an outcome type."""
    if not values:
        return "no_scores"
    keys = set(values)
    if keys <= PASS_FAIL_VALUES:
        return "pass_fail"
    if keys <= (PASS_FAIL_VALUES | PARTIAL_VALUES):
        return "pass_fail_with_partial"
    numeric = []
    for k in keys:
        try:
            numeric.append(float(k))
        except (TypeError, ValueError):
            return "other"
    if all(0.0 <= v <= 1.0 for v in numeric) and len(set(numeric)) > 3:
        return "continuous_0_1"
    return "continuous" if len(set(numeric)) > 3 else "ordinal"


def column_values(df: pd.DataFrame, col: str) -> Counter:
    """Hashable scalar values of one column (dict/list cells are skipped)."""
    return Counter(
        v for v in df[col].dropna().tolist() if not isinstance(v, (dict, list))
    )


def classify_score_columns(
    df: pd.DataFrame, headline_scorers: str | None
) -> tuple[str | None, str, dict]:
    """Pick the score column that carries the task's outcome and classify it.

    eva stores the WHOLE Inspect score object per scorer, so samples() expands
    it into several score_* columns: the scorer value, but also answer /
    explanation / metadata (flag strings, free text). Pooling them all made
    every real benchmark classify as "other". Instead: prefer the headline
    scorer's own column (the one the modeling config maps), falling back to
    whichever score_* column classifies as most decision-like.

    Returns (column, outcome, top_values).
    """
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if not score_cols:
        return None, "no_scores", {}

    classified = {}
    for col in score_cols:
        vals = column_values(df, col)
        classified[col] = (classify_values(vals), vals)

    # headline scorer's exact column wins if present (e.g. score_includes);
    # among its dotted expansions (score_x.value, ...) take the best-ranked.
    for scorer in (headline_scorers or "").split(", "):
        if not scorer:
            continue
        candidates = [
            c for c in score_cols
            if c == f"score_{scorer}" or c.startswith(f"score_{scorer}.")
        ]
        if f"score_{scorer}" in candidates:
            candidates = [f"score_{scorer}"]
        if candidates:
            col = min(candidates, key=lambda c: CLASS_PRIORITY.index(classified[c][0]))
            outcome, vals = classified[col]
            return col, outcome, dict(vals.most_common(8))

    col = min(classified, key=lambda c: CLASS_PRIORITY.index(classified[c][0]))
    outcome, vals = classified[col]
    return col, outcome, dict(vals.most_common(8))


def probe_task_scores(
    task: str, headline_scorers: str | None = None, limit: int = 500
) -> tuple[str | None, str, dict]:
    """Sample raw score values for one task and classify the outcome type.

    Retries on AWS throttling: spawned workers each set up their own eva
    config/credentials, and a burst of 8 can trip SSM rate limits.
    """
    for attempt in range(3):
        try:
            df = samples(
                SCORE_PROBE_SQL,
                params={"task": task, "limit": limit},
                statement_timeout_ms=120_000,
            )
            break
        except Exception as e:
            retryable = "Throttling" in str(e) or "Rate exceeded" in str(e)
            if not retryable or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    col, outcome, top = classify_score_columns(df, headline_scorers)
    return col, outcome, {str(k): v for k, v in top.items()}


def main(min_evals: int = 5) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks_df = query(TASKS_SQL, statement_timeout_ms=240_000)
    print(f"{len(tasks_df)} {TEAM} tasks")

    # every task is listed in the inventory, but probing one-off dev runs is
    # wasted work -- only tasks with enough evals get a score probe
    headline_by_task = dict(zip(tasks_df["task_name"], tasks_df["headline_scorers"]))
    eligible = list(tasks_df[tasks_df["n_evals"] >= min_evals]["task_name"])
    print(f"probing {len(eligible)} tasks with >= {min_evals} evals")

    probed: dict[str, tuple[str | None, str, dict]] = {}
    with ProcessPoolExecutor(max_workers=PROBE_WORKERS, mp_context=SPAWN) as pool:
        futures = {
            pool.submit(probe_task_scores, task, headline_by_task.get(task)): task
            for task in eligible
        }
        progress = tqdm(
            as_completed(futures), total=len(futures), desc="probing scores", unit="task"
        )
        for future in progress:
            task = futures[future]
            try:
                probed[task] = future.result()
            except Exception as e:  # one broken task shouldn't kill discovery
                probed[task] = (None, f"probe_failed: {e}", {})
            progress.write(f"  {task}: {probed[task][1]} ({probed[task][0]})")

    columns, outcomes, value_samples = [], [], []
    for task in tasks_df["task_name"]:
        col, outcome, top = probed.get(
            task, (None, f"not_probed(<{min_evals} evals)", {})
        )
        columns.append(col or "")
        outcomes.append(outcome)
        value_samples.append(json.dumps(top))

    tasks_df["outcome_type"] = outcomes
    tasks_df["score_column"] = columns
    tasks_df["score_values_sampled"] = value_samples
    tasks_df.to_csv(OUT_DIR / "evals_inventory.csv", index=False)

    pass_fail = tasks_df[
        tasks_df["outcome_type"].isin(["pass_fail", "pass_fail_with_partial"])
    ]["task_name"].tolist()
    (OUT_DIR / "pass_fail_tasks.txt").write_text("\n".join(pass_fail) + "\n")

    print(f"\npass/fail tasks ({len(pass_fail)}): {pass_fail}")
    print(f"wrote {OUT_DIR / 'evals_inventory.csv'}")
    print(f"wrote {OUT_DIR / 'pass_fail_tasks.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-evals",
        type=int,
        default=5,
        help="Only probe score values for tasks with at least this many evals",
    )
    args = parser.parse_args()
    main(min_evals=args.min_evals)
