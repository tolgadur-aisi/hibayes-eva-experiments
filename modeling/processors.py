"""Custom hibayes processors for the eva unified scaffold model.

Each processor does one thing and is invoked from modeling/config.yaml, in the
same style as the hibayes docs and the syco-at-t example. Expected input is the
eva sample-level extract (see modeling/extract.py): one row per sample attempt
with task_name, model, solver, solver_args, task_args and score_* columns.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
from hibayes.analysis import AnalysisState
from hibayes.process import DataProcessor, process
from hibayes.ui import ModellingDisplay

# Inspect letter grades -> numeric. "P" (partial credit) has no binomial
# interpretation and is dropped by coerce_pass_fail_score.
LETTER_SCORES = {"C": 1.0, "I": 0.0, "N": 0.0, "P": 0.5}


def _coerce_one(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):  # JSONB true/false scorers
        return float(value)
    s = str(value).strip()
    if s in LETTER_SCORES:
        return LETTER_SCORES[s]
    if s.lower() in ("true", "false"):
        return float(s.lower() == "true")
    try:
        return float(s)
    except ValueError:
        return None


@process
def coerce_pass_fail_score(
    score_column_by_benchmark: dict[str, str] | None = None,
) -> DataProcessor:
    """Build a binary `score` column from each benchmark's scorer column.

    Args:
        score_column_by_benchmark: Mapping from task_name to the score_* column
            holding that benchmark's headline scorer values, e.g.
            {"cybench": "score_includes", "swe_bench": "score_swe_bench_scorer"}.
            Rows whose value does not coerce to exactly 0 or 1 (unparseable,
            missing, or partial credit) are dropped with a logged count.
    """
    score_column_by_benchmark = score_column_by_benchmark or {}

    def processor(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> AnalysisState:
        df = state.processed_data
        missing = set(df["task_name"].unique()) - set(score_column_by_benchmark)
        if missing:
            raise ValueError(
                f"No score column configured for benchmarks: {sorted(missing)}. "
                f"Add them to coerce_pass_fail_score.score_column_by_benchmark."
            )

        score = pd.Series(index=df.index, dtype=float)
        for benchmark, col in score_column_by_benchmark.items():
            mask = df["task_name"] == benchmark
            if not mask.any():
                continue
            if col not in df.columns:
                raise ValueError(
                    f"{benchmark}: column {col!r} not in data. Available score "
                    f"columns: {sorted(c for c in df.columns if c.startswith('score_'))}"
                )
            score.loc[mask] = df.loc[mask, col].map(_coerce_one)

        df["score"] = score
        binary = df["score"].isin([0.0, 1.0])
        # report per benchmark: a systematic scorer mismatch must be loud
        empty = []
        for benchmark, group in df.groupby("task_name"):
            n_bad = int((~binary[group.index]).sum())
            if display and n_bad:
                display.logger.info(
                    f"coerce_pass_fail_score: {benchmark}: dropping "
                    f"{n_bad}/{len(group)} rows with non-binary scores "
                    f"(missing, unparseable, or partial credit)"
                )
            if n_bad == len(group):
                empty.append(str(benchmark))
        if empty:
            raise ValueError(
                f"coerce_pass_fail_score: every row of {empty} was dropped -- "
                f"the configured score column is not binary pass/fail for "
                f"these benchmarks. Fix score_column_by_benchmark or remove them."
            )
        state.processed_data = df[binary].reset_index(drop=True)
        return state

    return processor


def canonical_args(
    raw: object,
    excluded: set[str] | None = None,
    included: set[str] | None = None,
) -> str:
    """Canonicalise a JSON args blob into a stable label fragment.

    If `included` is given, only those keys are kept (allowlist mode);
    otherwise all keys except `excluded` are kept. Shared by derive_scaffold
    and the synthetic-data generator so the expected scaffold labels in
    parameter-recovery tests can never drift from the pipeline's actual labels.
    """
    excluded = excluded or set()
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        return str(parsed)
    items = sorted(
        (k, v)
        for k, v in parsed.items()
        if (k in included if included is not None else k not in excluded)
    )
    return ",".join(f"{k}={json.dumps(v, sort_keys=True)}" for k, v in items)


@process
def derive_scaffold(
    exclude_task_args: list[str] | None = None,
    include_solver_args: bool = True,
    scaffold_keys: list[str] | None = None,
) -> DataProcessor:
    """Derive a `scaffold` label from solver + solver_args + task_args.

    "Scaffold" is not a column in eva; it is everything about how the model was
    run that is not the model or the benchmark items. We canonicalise it as
    `solver` plus the sorted key=value pairs of solver_args and task_args.

    On real data, args carry many incidental knobs (limits, seeds, tool lists)
    that can fragment the scaffold factor into singletons -- watch the logged
    cardinality, and switch to the `scaffold_keys` allowlist (informed by
    discovery/list_variables.py) once you know which keys matter.

    Args:
        exclude_task_args: task_args keys that select *data* rather than
            configure the agent (e.g. dataset variants or sample filters).
            These belong to item identity, not the scaffold.
        include_solver_args: Whether solver_args participate in the label.
        scaffold_keys: If set, allowlist mode -- only these keys of task_args
            and solver_args define the scaffold; exclude_task_args is ignored.
    """
    exclude = set(exclude_task_args or [])
    include = set(scaffold_keys) if scaffold_keys is not None else None

    def processor(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> AnalysisState:
        df = state.processed_data
        solver = df["solver"].fillna("none").astype(str)
        parts = [solver, df["task_args"].map(lambda v: canonical_args(v, exclude, include))]
        if include_solver_args:
            parts.append(df["solver_args"].map(lambda v: canonical_args(v, included=include)))
        label = parts[0]
        for part in parts[1:]:
            label = label + "|" + part
        df["scaffold"] = label.str.rstrip("|")

        if display:
            counts = df["scaffold"].value_counts()
            models_per_scaffold = df.groupby("scaffold")["model"].nunique()
            display.logger.info(
                f"derive_scaffold: {len(counts)} distinct scaffolds across "
                f"{df['task_name'].nunique()} benchmarks; "
                f"{int((models_per_scaffold == 1).sum())} scaffolds seen with "
                f"only one model (these confound scaffold with model). "
                f"If cardinality is high, set scaffold_keys to an allowlist."
            )
        return state

    return processor


@process
def add_benchmark_item() -> DataProcessor:
    """Add `benchmark` and `benchmark_item` columns.

    benchmark_item is namespaced by benchmark (cybench/foo, swe_bench/foo) so
    identical item ids in different benchmarks never collide, which is what
    lets the model nest item effects within benchmarks.
    """

    def processor(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> AnalysisState:
        df = state.processed_data
        df["benchmark"] = df["task_name"].astype(str)
        df["benchmark_item"] = df["benchmark"] + "/" + df["item_id"].astype(str)
        if display:
            display.logger.info(
                f"add_benchmark_item: {df['benchmark_item'].nunique()} items "
                f"in {df['benchmark'].nunique()} benchmarks"
            )
        return state

    return processor


@process
def snapshot_trials(path: str = "modeling/.output/trials.parquet") -> DataProcessor:
    """Save the trial-level processed data to parquet before aggregation.

    Place this immediately before groupby in the config: the snapshot then
    holds one row per individual trial with score, model, scaffold, benchmark
    and benchmark_item -- exactly what raw-data plots (modeling/plot_raw.py)
    need, and what the aggregated processed_data.parquet no longer contains.
    """

    def processor(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> AnalysisState:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        state.processed_data.to_parquet(out, index=False)
        if display:
            display.logger.info(
                f"snapshot_trials: {len(state.processed_data)} trial rows -> {out}"
            )
        return state

    return processor


@process
def extract_item_benchmark_index() -> DataProcessor:
    """Map each benchmark_item level to its benchmark level, as a jax array.

    Must run *after* extract_features so the categorical level orderings
    (state.coords) exist. Adds features["item_benchmark_index"], which the
    hierarchical model uses to give every item a prior centred on its
    benchmark's mean difficulty -- the "benchmark_item inherits from
    benchmark" structure.
    """

    def processor(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> AnalysisState:
        coords = state.coords or {}
        if "benchmark_item" not in coords or "benchmark" not in coords:
            raise ValueError(
                "extract_item_benchmark_index must run after extract_features "
                "with both 'benchmark_item' and 'benchmark' as categorical features."
            )
        item_to_benchmark = (
            state.processed_data.drop_duplicates("benchmark_item")
            .set_index("benchmark_item")["benchmark"]
        )
        benchmark_pos = {b: i for i, b in enumerate(coords["benchmark"])}
        index = [
            benchmark_pos[item_to_benchmark[item]] for item in coords["benchmark_item"]
        ]
        state.features["item_benchmark_index"] = jnp.asarray(index, dtype=jnp.int32)

        # arviz labelling for the per-benchmark item spread parameter
        state.dims["benchmark_item_sigma"] = ["benchmark"]
        if display:
            display.logger.info(
                f"extract_item_benchmark_index: mapped {len(index)} items to "
                f"{len(benchmark_pos)} benchmarks"
            )
        return state

    return processor
