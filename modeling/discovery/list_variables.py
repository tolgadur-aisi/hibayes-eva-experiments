"""Step 2 of the modeling recipe: list variables of interest in eva and their coverage.

For each candidate covariate (scaffold ingredients, reasoning config, design
knobs) this reports its type, cardinality, and how many eval runs have it
populated -- per task and overall. Use the coverage table to pick the variable
subset the unified model conditions on; rows missing any chosen variable are
dropped by the pipeline, not imputed.

Run from this repo on a platform dev VM with eva access:

    uv run python modeling/discovery/list_variables.py            # all tasks
    uv run python modeling/discovery/list_variables.py --pass-fail-only

Outputs (committed so the selection is reviewable):
    modeling/discovery/outputs/variables_coverage.csv
    modeling/discovery/outputs/variables_values.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from eva import query

OUT_DIR = Path(__file__).resolve().parent / "outputs"

BASE_FILTERS = """
    e.status = 'success'
    AND e.model NOT LIKE 'replay/%%'
    AND e.model NOT IN ('none/none', 'mockllm/model')
"""

EVALS_SQL = f"""
SELECT e.task_name, e.model, e.solver, e.solver_args::text AS solver_args,
       e.task_args::text AS task_args,
       e.model_generate_config::text AS model_generate_config,
       e.sandbox_type, e.message_limit, e.epochs, e.epochs_reducer,
       e.task_version
FROM inspect.evals e
WHERE {BASE_FILTERS} {{task_filter}}
"""

# Plain columns: (name, type). JSON columns are expanded key-by-key below.
PLAIN_VARIABLES = [
    ("model", "categorical"),
    ("solver", "categorical"),
    ("sandbox_type", "categorical"),
    ("task_version", "categorical"),
    ("message_limit", "continuous"),
    ("epochs", "continuous"),
    ("epochs_reducer", "categorical"),
]
JSON_VARIABLES = ["task_args", "solver_args", "model_generate_config"]


def expand_json_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Expand a JSON-string column into one boolean/value column per key."""
    parsed = df[col].map(lambda v: json.loads(v) if isinstance(v, str) and v else {})
    keys = sorted({k for d in parsed if isinstance(d, dict) for k in d})
    out = pd.DataFrame(index=df.index)
    for k in keys:
        out[f"{col}.{k}"] = parsed.map(
            lambda d: d.get(k) if isinstance(d, dict) else None
        )
    return out


def summarise(series: pd.Series, name: str, vtype: str, by_task: pd.Series) -> dict:
    """Coverage summary for one variable."""
    non_null = series.notna()
    per_task = non_null.groupby(by_task).mean().round(3)
    values = series.dropna()
    # JSON values (lists/dicts) are not hashable; stringify for cardinality
    values = values.map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return {
        "variable": name,
        "type": vtype,
        "n_rows": int(len(series)),
        "n_non_null": int(non_null.sum()),
        "coverage": round(float(non_null.mean()), 3),
        "n_distinct": int(values.nunique()),
        "tasks_fully_covered": int((per_task == 1.0).sum()),
        "coverage_by_task": json.dumps(per_task.to_dict()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pass-fail-only",
        action="store_true",
        help="Restrict to tasks in outputs/pass_fail_tasks.txt (run list_evals.py first)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    task_filter = ""
    params = {}
    if args.pass_fail_only:
        tasks = (
            (OUT_DIR / "pass_fail_tasks.txt").read_text().strip().splitlines()
        )
        task_filter = "AND e.task_name = ANY(%(tasks)s)"
        params = {"tasks": tasks}
        print(f"restricting to {len(tasks)} pass/fail tasks")

    df = query(
        EVALS_SQL.format(task_filter=task_filter),
        params=params or None,
        statement_timeout_ms=240_000,
    )
    print(f"{len(df)} eval runs, {df['task_name'].nunique()} tasks")

    rows = []
    top_values: dict[str, dict] = {}
    for name, vtype in PLAIN_VARIABLES:
        rows.append(summarise(df[name], name, vtype, df["task_name"]))
        top_values[name] = (
            df[name].value_counts(dropna=True).head(10).to_dict()
        )

    for col in JSON_VARIABLES:
        expanded = expand_json_column(df, col)
        for sub in expanded.columns:
            series = expanded[sub]
            hashable = series.map(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
            )
            vtype = (
                "continuous"
                if pd.api.types.is_numeric_dtype(hashable.dropna().infer_objects())
                else "categorical"
            )
            rows.append(summarise(series, sub, vtype, df["task_name"]))
            top_values[sub] = hashable.value_counts(dropna=True).head(10).to_dict()

    coverage = pd.DataFrame(rows).sort_values("coverage", ascending=False)
    coverage.to_csv(OUT_DIR / "variables_coverage.csv", index=False)
    (OUT_DIR / "variables_values.json").write_text(
        json.dumps({k: {str(kk): vv for kk, vv in v.items()} for k, v in top_values.items()}, indent=2)
    )

    print(coverage[["variable", "type", "coverage", "n_distinct", "tasks_fully_covered"]].to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'variables_coverage.csv'}")
    print(f"wrote {OUT_DIR / 'variables_values.json'}")


if __name__ == "__main__":
    main()
