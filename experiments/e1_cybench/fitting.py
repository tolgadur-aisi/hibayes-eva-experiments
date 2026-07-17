"""Fit wrappers: aggregate cells -> hibayes state -> fitted model + coords."""

import arviz as az
import numpy as np
import pandas as pd
from hibayes.model.models import simplified_group_binomial_exponential
from hibayes.process import extract_features, extract_observed_feature

from experiments.e1_cybench.models import (
    challenge_run_binomial,
    model_challenge_run_binomial,
)
from shared.bridge import fit, make_state, run_processors

HDI_PROB = 0.94


def fit_challenge_only(agg: pd.DataFrame, tag: str, **fit_kw):
    """Per-model difficulty spectrum: hierarchical binomial over challenges.

    agg: columns challenge, n_correct, n_total.
    Priors tuned vs hibayes defaults (see synthetic recovery): challenge
    spread is several logits, so sigma ~ Exponential(0.5), mu ~ N(0, 2).
    """
    df = agg.rename(columns={"challenge": "group"}).copy()
    state = make_state(df)
    state = run_processors(
        state,
        extract_features(categorical_features=["group"],
                         continuous_features=["n_total"]),
        extract_observed_feature(feature_name="n_correct"),
    )
    mas = fit(
        state,
        simplified_group_binomial_exponential(
            prior_mu_overall_loc=0.0,
            prior_mu_overall_scale=2.0,
            prior_sigma_group_rate=0.5,
        ),
        tag=tag,
        **fit_kw,
    )
    return mas, state.coords["group"]


def fit_crossed(cells: pd.DataFrame, tag: str, **fit_kw):
    """Single-model crossed challenge x run fit.

    cells: columns challenge, run, n_correct, n_total.
    """
    state = make_state(cells.copy())
    state = run_processors(
        state,
        extract_features(categorical_features=["challenge", "run"],
                         continuous_features=["n_total"]),
        extract_observed_feature(feature_name="n_correct"),
    )
    mas = fit(state, challenge_run_binomial(), tag=tag, **fit_kw)
    return mas, dict(state.coords)


def fit_joint(cells: pd.DataFrame, tag: str, **fit_kw):
    """Joint model + challenge + run fit.

    cells: columns model, challenge, run, n_correct, n_total.
    """
    state = make_state(cells.copy())
    state = run_processors(
        state,
        extract_features(categorical_features=["model", "challenge", "run"],
                         continuous_features=["n_total"]),
        extract_observed_feature(feature_name="n_correct"),
    )
    state.dims["model_ability"] = ["model"]
    mas = fit(state, model_challenge_run_binomial(), tag=tag, **fit_kw)
    return mas, dict(state.coords)


def draws(mas, var: str) -> np.ndarray:
    """Posterior draws for a variable, flattened over (chain, draw)."""
    da = mas.inference_data.posterior[var]
    return da.stack(sample=("chain", "draw")).transpose("sample", ...).values


def hdi(x: np.ndarray) -> tuple[float, float]:
    lo, hi = az.hdi(np.asarray(x), hdi_prob=HDI_PROB)
    return float(lo), float(hi)


CROSSED_REPORTED = ["mu", "sigma_challenge", "sigma_run",
                    "challenge_effects", "run_effects"]
JOINT_REPORTED = ["model_ability", "sigma_challenge", "sigma_run",
                  "challenge_effects", "run_effects"]


def reported_diagnostics(mas, var_names: list[str]) -> dict:
    """Convergence restricted to the quantities actually reported.

    The centred parameterisation leaves the mean-mode of the raw z_challenge
    latents non-identified (it cancels in every deterministic), so overall
    r_hat can sit slightly above 1.01 while all reported quantities converge.
    """
    s = az.summary(mas.inference_data, var_names=var_names, round_to=4)
    return {
        "reported_max_r_hat": float(s["r_hat"].max()),
        "reported_min_ess_bulk": float(s["ess_bulk"].min()),
    }
