"""Posterior health metrics for E5.

All metrics are computed from posterior draws; every league-table number has a
94% HDI. Definitions (stated in FINDINGS.md):

- discrimination_ratio: per-draw SD across model effects (logit scale) divided
  by the median posterior SD of a single model effect. >> 1 means the
  benchmark separates models well beyond single-model uncertainty.
- p_best: per-draw max over models of the item-marginal success probability
  mean_i sigmoid(model_effect + item_effect_i) over the observed item pool.
- rho: intra-item (beta-binomial-style) correlation at the average model:
  var_i(p_i) / (pbar (1 - pbar)). Design effect for m epochs/item:
  Deff = 1 + (m - 1) rho.
- run inflation: sqrt(binom_var + (sigma_run * p(1-p))^2) / sqrt(binom_var)
  at the median replicate-group accuracy and median run size (delta method).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import xarray as xr
from scipy.special import expit

HDI_PROB = 0.94


def hdi(x: np.ndarray) -> tuple[float, float]:
    lo, hi = az.hdi(np.asarray(x), hdi_prob=HDI_PROB)
    return float(lo), float(hi)


def _draws(idata, var: str) -> np.ndarray:
    """Posterior draws flattened to (n_draws, *shape)."""
    v = idata.posterior[var].values
    return v.reshape(-1, *v.shape[2:])


def crossed_metrics(idata, model_names: list[str], m_real: float = 1.0) -> dict:
    me = _draws(idata, "model_effects")  # (D, M)
    u = _draws(idata, "item_id_effects")  # (D, I)
    n_draws, n_models = me.shape

    # discrimination
    spread = me.std(axis=1, ddof=1)
    per_model_sd = me.std(axis=0, ddof=1)
    ratio = spread / np.median(per_model_sd)

    # item-marginal ability per model (probability scale)
    p_models = np.empty((n_draws, n_models))
    for j in range(n_models):
        p_models[:, j] = expit(me[:, j][:, None] + u).mean(axis=1)
    p_best = p_models.max(axis=1)
    best_idx = np.bincount(p_models.argmax(axis=1), minlength=n_models)
    best_model = model_names[int(best_idx.argmax())]

    # overdispersion at the average model
    mbar = me.mean(axis=1)
    p_i = expit(mbar[:, None] + u)
    pbar = p_i.mean(axis=1)
    rho = p_i.var(axis=1, ddof=1) / (pbar * (1.0 - pbar))
    deff10 = 1.0 + 9.0 * rho
    deff_real = 1.0 + (m_real - 1.0) * rho

    out = {
        "disc_ratio_mean": float(ratio.mean()),
        "disc_ratio_hdi": hdi(ratio),
        "p_best_mean": float(p_best.mean()),
        "p_best_hdi": hdi(p_best),
        "p_sat_gt90": float((p_best > 0.90).mean()),
        "p_floor_lt10": float((p_best < 0.10).mean()),
        "best_model": best_model,
        "rho_mean": float(rho.mean()),
        "rho_hdi": hdi(rho),
        "deff_epochs10_mean": float(deff10.mean()),
        "deff_epochs10_hdi": hdi(deff10),
        "deff_realized_mean": float(deff_real.mean()),
        "m_real": float(m_real),
    }
    abilities = {
        "model": model_names,
        "p_mean": p_models.mean(axis=0),
        "p_hdi_low": np.array([hdi(p_models[:, j])[0] for j in range(n_models)]),
        "p_hdi_high": np.array([hdi(p_models[:, j])[1] for j in range(n_models)]),
        "logit_mean": me.mean(axis=0),
        "logit_sd": per_model_sd,
    }
    item_p_mean = p_i.mean(axis=0)  # posterior mean difficulty per item
    return {"summary": out, "abilities": abilities, "item_p_mean": item_p_mean}


def repro_metrics(idata, pbar_med: float, n_med: float) -> dict:
    s = _draws(idata, "sigma_run")  # (D,)
    binom_var = pbar_med * (1 - pbar_med) / n_med
    extra_sd = s * pbar_med * (1 - pbar_med)  # delta method, prob scale
    inflation = np.sqrt(binom_var + extra_sd**2) / np.sqrt(binom_var)
    eff_n = n_med / inflation**2
    return {
        "sigma_run_mean": float(s.mean()),
        "sigma_run_hdi": hdi(s),
        "run_sd_extra_pp_mean": float(extra_sd.mean() * 100),
        "run_sd_extra_pp_hdi": tuple(x * 100 for x in hdi(extra_sd)),
        "run_inflation_mean": float(inflation.mean()),
        "run_inflation_hdi": hdi(inflation),
        "run_eff_n_median": float(np.median(eff_n)),
        "repro_pbar_med": pbar_med,
        "repro_n_med": n_med,
    }


def boolq_metrics(idata, model_names: list[str], se_med: float) -> dict:
    theta = _draws(idata, "model_effects")  # (D, M) probability scale
    sig_char = _draws(idata, "sigma_char")  # (D, M)
    sig_common = _draws(idata, "sigma_char_common")
    s_run = _draws(idata, "sigma_run")

    spread = theta.std(axis=1, ddof=1)
    per_model_sd = theta.std(axis=0, ddof=1)
    ratio = spread / np.median(per_model_sd)
    p_best = theta.max(axis=1)
    best_idx = np.bincount(theta.argmax(axis=1), minlength=theta.shape[1])
    best_model = model_names[int(best_idx.argmax())]

    # condition-effect SD (analogue of challenge variance): total per model
    total_char = np.sqrt(sig_char**2 + sig_common[:, None] ** 2)  # (D, M)
    max_char = total_char.max(axis=1)
    med_char = np.median(total_char, axis=1)

    inflation = np.sqrt(se_med**2 + s_run**2) / se_med

    summary = {
        "disc_ratio_mean": float(ratio.mean()),
        "disc_ratio_hdi": hdi(ratio),
        "p_best_mean": float(p_best.mean()),
        "p_best_hdi": hdi(p_best),
        "p_sat_gt90": float((p_best > 0.90).mean()),
        "p_floor_lt10": float((p_best < 0.10).mean()),
        "best_model": best_model,
        "cond_sd_median_mean": float(med_char.mean()),
        "cond_sd_median_hdi": hdi(med_char),
        "cond_sd_max_mean": float(max_char.mean()),
        "cond_sd_max_hdi": hdi(max_char),
        "sigma_run_mean": float(s_run.mean()),
        "sigma_run_hdi": hdi(s_run),
        "run_inflation_mean": float(inflation.mean()),
        "run_inflation_hdi": hdi(inflation),
        "se_med": se_med,
    }
    abilities = {
        "model": model_names,
        "p_mean": theta.mean(axis=0),
        "p_hdi_low": np.array([hdi(theta[:, j])[0] for j in range(theta.shape[1])]),
        "p_hdi_high": np.array([hdi(theta[:, j])[1] for j in range(theta.shape[1])]),
        "cond_sd_mean": total_char.mean(axis=0),
    }
    return {"summary": summary, "abilities": abilities}


def headline_param_diagnostics(idata, fit: str, params: list[str]) -> list[dict]:
    """Per-parameter r_hat / ESS for the HEADLINE (reported) parameters of a fit.

    One row per named parameter (worst case over its vector elements). This is
    the artifact separating headline convergence from global max-r_hat, which
    for the crossed/boolq fits is dominated by nuisance non-centred z's.
    """
    summ = az.summary(idata, var_names=params, round_to=4)
    base = summ.index.str.replace(r"\[.*\]$", "", regex=True)
    rows = []
    for p in params:
        sub = summ[base == p]
        rows.append({
            "fit": fit, "param": p, "n_elements": len(sub),
            "max_r_hat": float(sub["r_hat"].max()),
            "min_ess_bulk": float(sub["ess_bulk"].min()),
            "min_ess_tail": float(sub["ess_tail"].min()),
        })
    return rows


def _chain_diag(arr: np.ndarray) -> tuple[float, float]:
    """(r_hat, ess_bulk) of a derived per-draw quantity, keeping chain structure."""
    ds = xr.Dataset({"x": xr.DataArray(arr, dims=("chain", "draw"))})
    return float(az.rhat(ds)["x"].values), float(az.ess(ds)["x"].values)


def derived_diagnostics_crossed(idata) -> dict:
    """Convergence of the headline DERIVED metrics (prob scale).

    Global max r_hat can be dominated by nuisance non-centred z's for items
    solved by all/no models (their logits are weakly identified in the tail);
    the reported metrics live on the probability scale where those saturate,
    so this is the gate that matters for the league table.
    """
    post = idata.posterior
    me = post["model_effects"].values  # (C, D, M)
    u = post["item_id_effects"].values  # (C, D, I)
    n_models = me.shape[2]
    spread = me.std(axis=2, ddof=1)
    mbar = me.mean(axis=2)
    p_i = expit(mbar[..., None] + u)
    pbar = p_i.mean(axis=2)
    rho = p_i.var(axis=2, ddof=1) / (pbar * (1 - pbar))
    p_m = np.stack(
        [expit(me[..., j : j + 1] + u).mean(axis=2) for j in range(n_models)], axis=-1
    )
    p_best = p_m.max(axis=-1)
    out = {}
    for name, arr in [("disc_spread", spread), ("rho", rho), ("p_best", p_best)]:
        r, e = _chain_diag(arr)
        out[f"{name}_r_hat"] = round(r, 4)
        out[f"{name}_ess_bulk"] = round(e, 1)
    return out


def derived_diagnostics_boolq(idata) -> dict:
    post = idata.posterior
    th = post["model_effects"].values
    s_run = post["sigma_run"].values
    out = {}
    for name, arr in [
        ("disc_spread", th.std(axis=2, ddof=1)),
        ("p_best", th.max(axis=2)),
        ("sigma_run", s_run),
    ]:
        r, e = _chain_diag(arr)
        out[f"{name}_r_hat"] = round(r, 4)
        out[f"{name}_ess_bulk"] = round(e, 1)
    return out


def verdict(m: dict, n_models: int) -> tuple[str, str]:
    """Rule-based verdict. Priority: saturating/floor > noisy > underpowered > healthy."""
    tags, why = [], []
    if m.get("p_sat_gt90", 0) > 0.5:
        tags.append("saturating")
        why.append(f"P(best ability > 0.90) = {m['p_sat_gt90']:.2f}")
    if m.get("p_floor_lt10", 0) > 0.5:
        tags.append("near-floor")
        why.append(f"P(best ability < 0.10) = {m['p_floor_lt10']:.2f}")
    infl = m.get("run_inflation_mean")
    if infl is not None and infl > 2.0:
        tags.append("noisy")
        why.append(f"same-config run SD is {infl:.1f}x binomial/naive SE")
    if m.get("disc_ratio_hdi", (99, 99))[0] < 1.0 or n_models < 5:
        tags.append("underpowered")
        why.append(
            f"discrimination ratio HDI low = {m['disc_ratio_hdi'][0]:.1f}"
            if m.get("disc_ratio_hdi", (99,))[0] < 1.0
            else f"only {n_models} models"
        )
    if not tags:
        tags.append("healthy")
        why.append("separates models, no saturation, run noise near binomial")
    return " + ".join(tags), "; ".join(why)
