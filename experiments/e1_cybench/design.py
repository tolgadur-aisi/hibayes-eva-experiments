"""Q2: epoch reliability, ICC, design effect, and the epochs-vs-challenges
budget curve, derived from the claude-3-7-sonnet crossed fit.

Estimator analysed: the standard benchmark score, i.e. the unweighted mean
over c challenges of per-challenge accuracy from k epochs (N = c*k attempts).

Curve A ("cybench-like" superpopulation, unlimited challenge pool):
    Var(p_hat; k) = (k * sigma_p^2 + wbar) / N,
where sigma_p^2 = Var(p_i) between challenges and wbar = E[p_i(1-p_i)] is the
mean within-challenge Bernoulli variance. Equivalent to the classic design
effect 1 + (k-1)*rho with rho = sigma_p^2 / (sigma_p^2 + wbar).

Curve B (this fixed 40-challenge benchmark, challenges sampled without
replacement):
    Var(p_hat; k, c=N/k) = (1 - c/C) * S_p^2 / c + wbar_40 / (c*k),
using the 40 fitted challenge probabilities.

Both are evaluated per posterior draw (of mu, sigma_challenge, challenge
effects), giving posterior uncertainty bands on the curve itself.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

from shared.bridge import diagnostics
from experiments.e1_cybench.fitting import draws, fit_challenge_only

BUDGET_N = 400
POOL_C = 40
K_GRID_A = np.array([1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200])
K_GRID_B = np.array([10, 16, 20, 25, 40, 50, 80, 100, 200])


def _moments_superpop(mu: np.ndarray, sigma_c: np.ndarray, rng, n_mc=2000):
    """Per-draw MC moments of the challenge-probability distribution."""
    z = rng.normal(0, 1, n_mc)
    p = expit(mu[:, None] + sigma_c[:, None] * z[None, :])  # (draw, mc)
    sigma_p2 = p.var(axis=1)
    wbar = (p * (1 - p)).mean(axis=1)
    return sigma_p2, wbar


def design_curves(mas_crossed, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    mu = draws(mas_crossed, "mu")
    sigma_c = draws(mas_crossed, "sigma_challenge")
    sigma_r = draws(mas_crossed, "sigma_run")
    a = draws(mas_crossed, "challenge_effects")  # (sample, challenge)

    # thin draws for the MC moment computation
    idx = rng.choice(len(mu), size=min(1500, len(mu)), replace=False)
    sigma_p2, wbar = _moments_superpop(mu[idx], sigma_c[idx], rng)
    rho_obs = sigma_p2 / (sigma_p2 + wbar)

    rows = []
    for k in K_GRID_A:
        var = (k * sigma_p2 + wbar) / BUDGET_N
        se = np.sqrt(var)
        rows.append({
            "curve": "A_superpopulation", "k": int(k), "c": BUDGET_N / k,
            "se_median": float(np.median(se)),
            "se_q3": float(np.quantile(se, 0.03)),
            "se_q97": float(np.quantile(se, 0.97)),
        })

    # fixed benchmark: use the fitted 40 challenge probs per draw
    p40 = expit(mu[idx][:, None] + a[idx])  # (draw, 40)
    S2 = p40.var(axis=1, ddof=1)
    wbar40 = (p40 * (1 - p40)).mean(axis=1)
    for k in K_GRID_B:
        c = BUDGET_N / k
        if c > POOL_C:
            continue
        var = (1 - c / POOL_C) * S2 / c + wbar40 / (c * k)
        se = np.sqrt(var)
        rows.append({
            "curve": "B_fixed_benchmark_40", "k": int(k), "c": c,
            "se_median": float(np.median(se)),
            "se_q3": float(np.quantile(se, 0.03)),
            "se_q97": float(np.quantile(se, 0.97)),
        })

    pi2_3 = np.pi ** 2 / 3
    icc_latent = sigma_c ** 2 / (sigma_c ** 2 + sigma_r ** 2 + pi2_3)
    icc_latent_norun = sigma_c ** 2 / (sigma_c ** 2 + pi2_3)
    summary = {
        "rho_obs_median": float(np.median(rho_obs)),
        "rho_obs_q3": float(np.quantile(rho_obs, 0.03)),
        "rho_obs_q97": float(np.quantile(rho_obs, 0.97)),
        "icc_latent_median": float(np.median(icc_latent)),
        "icc_latent_q3": float(np.quantile(icc_latent, 0.03)),
        "icc_latent_q97": float(np.quantile(icc_latent, 0.97)),
        "icc_latent_norun_median": float(np.median(icc_latent_norun)),
        "deff_k10_median": float(np.median(1 + 9 * rho_obs)),
        "deff_k100_median": float(np.median(1 + 99 * rho_obs)),
        "sigma_run_median": float(np.median(sigma_r)),
    }
    return pd.DataFrame(rows), summary


def mc_check_formula(mas_crossed, seed: int = 11) -> pd.DataFrame:
    """Frequentist Monte-Carlo check of the curve-A formula (no MCMC):
    simulate the estimator at posterior-mean parameters."""
    rng = np.random.default_rng(seed)
    mu = float(np.mean(draws(mas_crossed, "mu")))
    sigma_c = float(np.mean(draws(mas_crossed, "sigma_challenge")))
    rows = []
    for k in (10, 100):
        c = BUDGET_N // k
        n_rep = 200_000
        a = rng.normal(0, sigma_c, (n_rep, c))
        p = expit(mu + a)
        x = rng.binomial(k, p) / k
        emp_se = float(x.mean(axis=1).std())
        z = rng.normal(0, 1, 100_000)
        pz = expit(mu + sigma_c * z)
        formula_se = float(np.sqrt((k * pz.var() + (pz * (1 - pz)).mean())
                                   / BUDGET_N))
        rows.append({"k": k, "c": c, "empirical_se": emp_se,
                     "formula_se": formula_se})
    return pd.DataFrame(rows)


def mcmc_design_validation(mas_crossed, out_dir, seed: int = 13,
                           n_rep: int = 4) -> pd.DataFrame:
    """Fit the hierarchical model to synthetic data at two budget-matched
    designs and report the posterior SD of population-mean accuracy —
    the Bayesian analogue of the curve-A SE."""
    rng = np.random.default_rng(seed)
    mu = float(np.mean(draws(mas_crossed, "mu")))
    sigma_c = float(np.mean(draws(mas_crossed, "sigma_challenge")))
    rows = []
    for k, c in ((10, 40), (100, 4)):
        for rep in range(n_rep):
            a = rng.normal(0, sigma_c, c)
            p = expit(mu + a)
            agg = pd.DataFrame({
                "challenge": [f"c{i:02d}" for i in range(c)],
                "n_correct": rng.binomial(k, p),
                "n_total": k,
            })
            mas, _ = fit_challenge_only(
                agg, tag=f"design_k{k}_rep{rep}",
                samples=1000, warmup=800)
            mud = draws(mas, "overall_mean")
            sgd = draws(mas, "sigma_group")
            # population-mean accuracy per posterior draw (MC over challenges)
            z = rng.normal(0, 1, 800)
            pop_acc = expit(mud[:, None] + sgd[:, None] * z[None, :]).mean(axis=1)
            d = diagnostics(mas)
            rows.append({"k": k, "c": c, "rep": rep,
                         "post_sd_pop_acc": float(pop_acc.std()),
                         "post_mean_pop_acc": float(pop_acc.mean()),
                         "max_r_hat": d["max_r_hat"],
                         "min_ess_bulk": d["min_ess_bulk"],
                         "n_divergences": d["n_divergences"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "q2_design_mcmc_validation.csv", index=False)
    return df
