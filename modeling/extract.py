"""Extract sample-level pass/fail data from eva prod for the unified scaffold model.

Run from this repo on an AISI platform dev VM (needs VPC access to the eva
Aurora cluster, AWS credentials, and team_ru membership -- `uv run eva-check`
verifies the setup):

    uv run python modeling/extract.py

Writes one parquet per task to data/modeling/ plus an eval-level extract and a
manifest. Tasks default to the known pass/fail benchmarks; after running
modeling/discovery/list_evals.py, pass --tasks to extend:

    uv run python modeling/extract.py --tasks cybench gdm_intercode_ctf swe_bench

The extract is intentionally wide: it carries every covariate the modeling
config might select (solver, solver_args, task_args, model_generate_config,
sandbox_type, message_limit) so re-extraction is not needed when the config
changes.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from eva import query, samples

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "modeling"

# Pass/fail benchmarks on the public team_ru slice (see
# modeling/discovery/list_evals.py output for the authoritative list).
DEFAULT_TASKS = ["cybench", "gdm_intercode_ctf", "swe_bench"]

BASE_FILTERS = """
    e.team = 'team_ru' AND e.status = 'success'
    AND e.model NOT LIKE 'replay/%%'
    AND e.model NOT IN ('none/none', 'mockllm/model')
"""

SAMPLE_SQL = f"""
SELECT e.task_name, e.model, e.internal_id AS run_internal_id, e.created,
       e.epochs AS run_epochs, e.task_version,
       e.task_args::text AS task_args,
       e.solver, e.solver_args::text AS solver_args,
       e.model_generate_config::text AS model_generate_config,
       e.sandbox_type, e.message_limit,
       s.id AS item_id, s.epoch, s.scores,
       s.total_tokens, s.total_time, s.message_count, s.error, s._limit, s.retries
FROM inspect.samples s
JOIN inspect.evals e ON s.eval_internal_id = e.internal_id
WHERE e.task_name = %(task)s AND {BASE_FILTERS}
  AND e.created >= %(lo)s AND e.created < %(hi)s
"""

EVAL_SQL = f"""
SELECT e.internal_id, e.eval_id, e.run_id, e.task_name, e.task_version,
       e.task_args::text AS task_args, e.solver, e.solver_args::text AS solver_args,
       e.model, e.model_generate_config::text AS model_generate_config,
       e.sandbox_type, e.message_limit, e.created, e.epochs,
       e.dataset_samples, e.total_samples, e.completed_samples,
       e.score_headline_name, e.score_headline_metric,
       e.score_headline_value, e.score_headline_stderr
FROM inspect.evals e
WHERE e.task_name = ANY(%(tasks)s) AND {BASE_FILTERS}
"""


def quarters(start: date, end: date) -> list[tuple[str, str]]:
    """Quarterly [lo, hi) date ranges covering start..end."""
    edges = []
    y, m = start.year, ((start.month - 1) // 3) * 3 + 1
    while date(y, m, 1) <= end:
        edges.append(date(y, m, 1).isoformat())
        m += 3
        if m > 12:
            m, y = m - 12, y + 1
    edges.append(date(y, m, 1).isoformat())
    return list(zip(edges[:-1], edges[1:]))


def extract_samples(task: str) -> None:
    out = DATA_DIR / f"{task}.samples.parquet"
    if out.exists():
        print(f"[skip] {out.name} exists")
        return
    chunks = []
    for lo, hi in quarters(date(2024, 10, 1), date(2026, 8, 1)):
        df = samples(
            SAMPLE_SQL,
            params={"task": task, "lo": lo, "hi": hi},
            statement_timeout_ms=240_000,
        )
        if len(df):
            chunks.append(df)
        print(f"  {task} {lo}..{hi}: {len(df)} rows")
    if not chunks:
        print(f"[warn] {task}: no rows at all")
        return
    df = pd.concat(chunks, ignore_index=True)
    # eva.samples() expands the scores JSONB into score_* columns; keep everything.
    df.to_parquet(out, index=False)
    print(f"[done] {out.name}: {len(df)} rows, cols={list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== eval-level extract ===")
    evals_df = query(
        EVAL_SQL, params={"tasks": args.tasks}, statement_timeout_ms=240_000
    )
    evals_df.to_parquet(DATA_DIR / "evals.parquet", index=False)
    print(f"[done] evals.parquet: {len(evals_df)} rows")
    print(evals_df.groupby("task_name").size().to_string())

    print("\n=== sample-level extracts ===")
    for task in args.tasks:
        extract_samples(task)

    manifest = {
        "extracted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "tasks": args.tasks,
        "filters": "team=team_ru, status=success, no replay/mock/none models",
        "files": sorted(p.name for p in DATA_DIR.glob("*.parquet")),
    }
    (DATA_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("\nMANIFEST written.")


if __name__ == "__main__":
    main()
