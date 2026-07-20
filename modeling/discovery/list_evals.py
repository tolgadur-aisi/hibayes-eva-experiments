"""Step 1 of the modeling recipe: list all evals in eva and classify their outcome.

For every task visible to your eva roles, this reports how many runs exist and
whether the scores are pass/fail (binary C/I/N or 0/1), partial-credit,
continuous, or something else. Only pass/fail tasks feed the binomial model.

Run from this repo on a platform dev VM with eva access:

    uv run python modeling/discovery/list_evals.py

Outputs (committed so the selection is reviewable):
    modeling/discovery/outputs/evals_inventory.csv
    modeling/discovery/outputs/pass_fail_tasks.txt
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from eva import query, samples

OUT_DIR = Path(__file__).resolve().parent / "outputs"

BASE_FILTERS = """
    e.status = 'success'
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

PASS_FAIL_VALUES = {"C", "I", "N", 0, 1, 0.0, 1.0, "0", "1", True, False}
PARTIAL_VALUES = {"P", 0.5, "0.5"}


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


def probe_task_scores(task: str, limit: int = 500) -> tuple[str, dict]:
    """Sample raw score values for one task and classify the outcome type."""
    df = samples(
        SCORE_PROBE_SQL,
        params={"task": task, "limit": limit},
        statement_timeout_ms=120_000,
    )
    values: Counter = Counter()
    score_cols = [c for c in df.columns if c.startswith("score_")]
    for col in score_cols:
        values.update(v for v in df[col].dropna().tolist())
    outcome = classify_values(values)
    top = dict(values.most_common(8))
    return outcome, {str(k): v for k, v in top.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks_df = query(TASKS_SQL, statement_timeout_ms=240_000)
    print(f"{len(tasks_df)} tasks visible to your eva roles")

    outcomes, value_samples = [], []
    for task in tasks_df["task_name"]:
        try:
            outcome, top = probe_task_scores(task)
        except Exception as e:  # keep going; one broken task shouldn't kill discovery
            outcome, top = f"probe_failed: {e}", {}
        outcomes.append(outcome)
        value_samples.append(json.dumps(top))
        print(f"  {task}: {outcome}")

    tasks_df["outcome_type"] = outcomes
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
    main()
