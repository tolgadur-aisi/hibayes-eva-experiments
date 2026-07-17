"""One-pass extraction of public team_ru eval data from eva prod to local parquet.

Run from the eva client project (needs its env + VPC access):
    cd ~/dev/eva/client && uv run --with pyarrow python ~/dev/hibayes-experiments/shared/extract.py

All experiments read the parquet caches in ~/dev/hibayes-experiments/data/ —
they must NOT query prod directly.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from eva import query, samples

DATA_DIR = Path("~/dev/hibayes-experiments/data").expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_TASKS = [
    "cybench",
    "gdm_intercode_ctf",
    "swe_bench",
    "boolq_preference",
    "gsm8k-agent-0",
]
# Sample-level pulls (boolq excluded: ~68M rows, eval-level only)
SAMPLE_LEVEL_TASKS = ["cybench", "gdm_intercode_ctf", "swe_bench", "gsm8k-agent-0"]

BASE_FILTERS = """
    e.team = 'team_ru' AND e.status = 'success'
    AND e.model NOT LIKE 'replay/%%'
    AND e.model NOT IN ('none/none', 'mockllm/model')
"""

SAMPLE_SQL = f"""
SELECT e.model, e.internal_id AS run_internal_id, e.created, e.epochs AS run_epochs,
       e.task_version, e.task_args::text AS task_args, e.solver,
       s.id AS item_id, s.epoch, s.scores, s.total_tokens, s.total_time,
       s.message_count, s.error, s._limit, s.retries
FROM inspect.samples s
JOIN inspect.evals e ON s.eval_internal_id = e.internal_id
WHERE e.task_name = %(task)s AND {BASE_FILTERS}
  AND e.created >= %(lo)s AND e.created < %(hi)s
"""

EVAL_SQL = f"""
SELECT e.internal_id, e.eval_id, e.run_id, e.task_name, e.task_version,
       e.task_args::text AS task_args, e.solver, e.model, e.created, e.epochs,
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
    # eva.samples() expands scores JSONB to score_* columns; keep everything.
    df.to_parquet(out, index=False)
    print(f"[done] {out.name}: {len(df)} rows, cols={list(df.columns)}")


def main() -> None:
    print("=== eval-level extract (all public tasks) ===")
    evals_df = query(EVAL_SQL, params={"tasks": PUBLIC_TASKS}, statement_timeout_ms=240_000)
    evals_df.to_parquet(DATA_DIR / "public_evals.parquet", index=False)
    print(f"[done] public_evals.parquet: {len(evals_df)} rows")
    print(evals_df.groupby("task_name").size().to_string())

    print("\n=== sample-level extracts ===")
    for task in SAMPLE_LEVEL_TASKS:
        extract_samples(task)

    print("\n=== scorer inventory (headline scorers per public task) ===")
    inv = (
        evals_df.groupby(["task_name", "score_headline_name", "score_headline_metric"], dropna=False)
        .size()
        .reset_index(name="n_runs")
    )
    inv.to_json(DATA_DIR / "scorer_inventory.json", orient="records", indent=2)
    print(inv.to_string(index=False))

    manifest = {
        "extracted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "filters": "team=team_ru, status=success, no replay/mock/none models",
        "files": sorted(p.name for p in DATA_DIR.glob("*.parquet")),
    }
    (DATA_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("\nMANIFEST written.")


if __name__ == "__main__":
    main()
