"""Discover pass/fail evals in eva and extract their sample-level data.

The one eva-facing step of the pipeline. Runs discovery (which tasks are
pass/fail, which covariates have coverage -- see modeling/discovery/), then
extracts every task that discovery classified as pass/fail AND that has a
score column configured in modeling/config.yaml. Tasks discovery surfaces
beyond the config are reported, not extracted: adding a benchmark to the
model stays a reviewed config change, not a side effect of the warehouse
growing.

Run from this repo on an AISI platform dev VM (needs VPC access to the eva
Aurora cluster, AWS credentials, and team_ru membership -- `uv run eva-check`
verifies the setup):

    uv run python -m modeling.extract
    uv run python -m modeling.extract --tasks cybench swe_bench   # skip discovery

Writes one parquet per task to data/modeling/ plus an eval-level extract and a
manifest. The extract is intentionally wide: it carries every covariate the
modeling config might select (solver, solver_args, task_args,
model_generate_config, sandbox_type, message_limit) so re-extraction is not
needed when the config changes.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from eva import query, samples
from tqdm import tqdm

from modeling.discovery import list_evals, list_variables

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "modeling"
CONFIG = Path(__file__).resolve().parent / "config.yaml"
PASS_FAIL_TASKS = (
    Path(__file__).resolve().parent / "discovery" / "outputs" / "pass_fail_tasks.txt"
)

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
       e.sandbox_type, e.message_limit, e.token_limit, e.time_limit,
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
       e.sandbox_type, e.message_limit, e.token_limit, e.time_limit,
       e.created, e.epochs,
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


# eva caches ONE psycopg connection per process, so parallel pulls need
# separate processes, not threads. Each worker holds its own Aurora connection.
PULL_WORKERS = 8


def pull_quarter(task: str, lo: str, hi: str) -> pd.DataFrame:
    """One (task, quarter) chunk -- runs in a worker process."""
    return samples(
        SAMPLE_SQL,
        params={"task": task, "lo": lo, "hi": hi},
        statement_timeout_ms=240_000,
    )


def extract_samples(tasks: list[str]) -> None:
    """Pull all (task, quarter) chunks in parallel, write one parquet per task."""
    todo = []
    for task in tasks:
        if (DATA_DIR / f"{task}.samples.parquet").exists():
            print(f"[skip] {task}.samples.parquet exists")
        else:
            todo.append(task)
    if not todo:
        return

    window = quarters(date(2024, 10, 1), date(2026, 8, 1))
    chunks: dict[str, list[pd.DataFrame]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=PULL_WORKERS) as pool:
        futures = {
            pool.submit(pull_quarter, task, lo, hi): (task, lo, hi)
            for task in todo
            for lo, hi in window
        }
        progress = tqdm(
            as_completed(futures), total=len(futures), desc="sample chunks", unit="chunk"
        )
        for future in progress:
            task, lo, hi = futures[future]
            df = future.result()
            if len(df):
                chunks[task].append(df)
                progress.write(f"  {task} {lo}..{hi}: {len(df)} rows")

    for task in todo:
        if not chunks[task]:
            print(f"[warn] {task}: no rows at all")
            continue
        out = DATA_DIR / f"{task}.samples.parquet"
        df = pd.concat(chunks[task], ignore_index=True)
        # eva.samples() expands the scores JSONB into score_* columns; keep everything.
        df.to_parquet(out, index=False)
        print(f"[done] {out.name}: {len(df)} rows, cols={list(df.columns)}")


def configured_benchmarks() -> set[str]:
    """Benchmarks with a score column in the config's coerce_pass_fail_score."""
    config = yaml.safe_load(CONFIG.read_text())
    for proc in config["data_process"]["processors"]:
        if isinstance(proc, dict) and "coerce_pass_fail_score" in proc:
            return set(proc["coerce_pass_fail_score"]["score_column_by_benchmark"])
    raise ValueError(f"no coerce_pass_fail_score processor found in {CONFIG}")


def discover_tasks() -> list[str]:
    """Run discovery, then pick the tasks that are pass/fail AND configured."""
    print("=== discovery: outcome types ===")
    list_evals.main()
    print("\n=== discovery: variable coverage ===")
    list_variables.main(pass_fail_only=True)

    discovered = set(PASS_FAIL_TASKS.read_text().split())
    configured = configured_benchmarks()
    if unconfigured := sorted(discovered - configured):
        print(
            f"\n[extract] pass/fail tasks with no score column in {CONFIG.name} -- "
            f"skipping: {unconfigured}\n"
            f"[extract] to include one, add it to coerce_pass_fail_score."
            f"score_column_by_benchmark and re-run."
        )
    if stale := sorted(configured - discovered):
        print(
            f"\n[extract] configured benchmarks that discovery did NOT classify as "
            f"pass/fail (check discovery/outputs/evals_inventory.csv): {stale}"
        )
    tasks = sorted(discovered & configured)
    if not tasks:
        raise SystemExit(
            "[extract] no benchmark is both discovered pass/fail and configured; "
            "nothing to extract."
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit task list; skips discovery and the config cross-check.",
    )
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else discover_tasks()
    print(f"\n[extract] extracting {len(tasks)} benchmarks: {tasks}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== eval-level extract ===")
    evals_df = query(
        EVAL_SQL, params={"tasks": tasks}, statement_timeout_ms=240_000
    )
    evals_df.to_parquet(DATA_DIR / "evals.parquet", index=False)
    print(f"[done] evals.parquet: {len(evals_df)} rows")
    print(evals_df.groupby("task_name").size().to_string())

    print("\n=== sample-level extracts ===")
    extract_samples(tasks)

    manifest = {
        "extracted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "tasks": tasks,
        "filters": "team=team_ru, status=success, no replay/mock/none models",
        "files": sorted(p.name for p in DATA_DIR.glob("*.parquet")),
    }
    (DATA_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("\nMANIFEST written.")


if __name__ == "__main__":
    main()
