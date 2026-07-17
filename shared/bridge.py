"""Shared bridge: eva parquet extracts -> hibayes AnalysisState -> fitted models.

IMPORT THIS FIRST in every experiment script: it sets the numpyro host device
count before jax initialises (required for parallel chains on CPU).

Typical experiment flow:

    from shared.bridge import (load_samples, binomial_agg, make_state,
                               run_processors, fit, diagnostics)
    from hibayes.process import groupby, extract_features, extract_observed_feature
    from hibayes.model.models import two_level_group_binomial

    df = load_samples("cybench", score_col="score_includes")
    agg = binomial_agg(df, by=["model", "item_id"])
    state = make_state(agg)
    state = run_processors(
        state,
        extract_features(continuous_features=["n_total"], categorical_features=["model"]),
        extract_observed_feature(feature_name="n_correct"),
    )
    mas = fit(state, two_level_group_binomial(), tag="v1")
    print(diagnostics(mas))
"""

import numpyro

numpyro.set_host_device_count(4)  # before any jax usage

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import arviz as az  # noqa: E402
import pandas as pd  # noqa: E402
from hibayes.analysis_state import AnalysisState, ModelAnalysisState  # noqa: E402
from hibayes.model.model_config import ModelConfig  # noqa: E402
from hibayes.model.fit import fit as _mcmc_fit  # noqa: E402
from hibayes.platform.config import PlatformConfig  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Inspect letter grades -> numeric. "P" (partial) maps to 0.5 and is EXCLUDED
# by binomial_agg (document the count when it happens).
LETTER_SCORES = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def coerce_score(series: pd.Series) -> pd.Series:
    """Coerce an Inspect score column (letters or numeric strings) to float."""

    def one(v: object) -> float | None:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        if s in LETTER_SCORES:
            return LETTER_SCORES[s]
        try:
            return float(s)
        except ValueError:
            return None

    return series.map(one)


def load_samples(task: str, score_col: str) -> pd.DataFrame:
    """Load a sample-level parquet extract and attach a float `score` column.

    Drops rows with unparseable/missing scores (count printed). Rows with
    sample-level `error` set are kept — filtering is an analysis decision.
    """
    df = pd.read_parquet(DATA_DIR / f"{task}.samples.parquet")
    if score_col not in df.columns:
        raise ValueError(f"{task}: no column {score_col!r}. Have: {sorted(c for c in df.columns if c.startswith('score_'))}")
    df["score"] = coerce_score(df[score_col])
    n_bad = int(df["score"].isna().sum())
    if n_bad:
        print(f"[bridge] {task}: dropping {n_bad}/{len(df)} rows with unparseable {score_col}")
        df = df.dropna(subset=["score"])
    return df.reset_index(drop=True)


def load_evals() -> pd.DataFrame:
    """Eval-level extract for all public tasks (one row per eval run)."""
    return pd.read_parquet(DATA_DIR / "public_evals.parquet")


def binomial_agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate binary scores to n_correct/n_total by group.

    Non-binary scores (e.g. partial credit 0.5) are excluded with a warning —
    binomial likelihoods need 0/1 outcomes.
    """
    binary = df[df["score"].isin([0.0, 1.0])]
    n_excluded = len(df) - len(binary)
    if n_excluded:
        print(f"[bridge] binomial_agg: excluded {n_excluded}/{len(df)} non-binary score rows")
    out = (
        binary.groupby(by, dropna=False)
        .agg(n_correct=("score", "sum"), n_total=("score", "count"))
        .reset_index()
    )
    out["n_correct"] = out["n_correct"].astype(int)
    return out


def make_state(df: pd.DataFrame) -> AnalysisState:
    state = AnalysisState(data=df)
    state.processed_data = df.copy()
    return state


def run_processors(state: AnalysisState, *processors) -> AnalysisState:
    for proc in processors:
        state = proc(state, None)
    return state


def fit(
    state: AnalysisState,
    model,
    tag: str | None = None,
    samples: int = 1500,
    warmup: int = 1000,
    chains: int = 4,
    seed: int = 0,
    **fit_overrides,
) -> ModelAnalysisState:
    """Fit a hibayes/numpyro model against the state's extracted features."""
    config = ModelConfig.from_dict(
        {
            "tag": tag,
            "fit": {
                "samples": samples,
                "warmup": warmup,
                "chains": chains,
                "seed": seed,
                "progress_bar": False,
                **fit_overrides,
            },
        }
    )
    mas = ModelAnalysisState(
        model=model,
        model_config=config,
        platform_config=PlatformConfig(num_devices=4, chain_method="parallel"),
        features=state.features,
        coords=state.coords,
        dims=state.dims,
    )
    _mcmc_fit(mas)
    return mas


def diagnostics(mas: ModelAnalysisState) -> dict:
    """Convergence summary: max r_hat, min ESS, divergence count."""
    summ = az.summary(mas.inference_data, round_to=4)
    diverging = int(mas.inference_data.sample_stats["diverging"].values.sum())
    return {
        "model": mas.model.__name__,
        "tag": mas.model_config.tag,
        "max_r_hat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "n_divergences": diverging,
        "n_params": len(summ),
    }


def save_outputs(mas: ModelAnalysisState, out_dir: Path, name: str) -> dict:
    """Save inference data + diagnostics; returns the diagnostics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mas.inference_data.to_netcdf(out_dir / f"{name}.idata.nc")
    diag = diagnostics(mas)
    (out_dir / f"{name}.diagnostics.json").write_text(json.dumps(diag, indent=2))
    return diag
