"""Synthetic-data recovery checks for the three E5 models.

Each simulator draws data with KNOWN parameters matching the real designs'
shape; run.py fits the same models used on real data and records recovery
(pass/fail with stated tolerances) in outputs/synthetic_recovery.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def sim_crossed(seed: int = 11) -> tuple[pd.DataFrame, dict]:
    """8 models (logit spread like cybench), 100 items, 5 trials/cell, 30% cells missing."""
    rng = np.random.default_rng(seed)
    true_model = np.linspace(-1.5, 1.5, 8)
    sigma_item = 1.2
    true_item = rng.normal(0, sigma_item, 100)
    rows = []
    for m in range(8):
        for i in range(100):
            if rng.random() < 0.30:
                continue
            n = 5
            p = expit(true_model[m] + true_item[i])
            rows.append({"model": f"m{m:02d}", "item_id": f"i{i:03d}",
                         "n_correct": int(rng.binomial(n, p)), "n_total": n})
    df = pd.DataFrame(rows)
    # true rho at the average model, over the realised item pool
    p_i = expit(true_model.mean() + true_item)
    rho = p_i.var(ddof=1) / (p_i.mean() * (1 - p_i.mean()))
    # The fitted model centres item effects (sum-to-zero), so its model-effect
    # estimand is true_model + realised mean of the item pool.
    truth = {
        "model_effects": true_model + true_item.mean(),
        "sigma_item": sigma_item,
        "rho": float(rho),
    }
    return df, truth


def sim_crossed_harsh(seed: int = 21) -> tuple[pd.DataFrame, dict]:
    """Harsh regime matching real cybench: 45 models; 134 items from a BIMODAL
    pool (60% hard N(-6, 1.5), 40% easy N(3, 1.2) -> ~half the items never
    solved, rho ~ 0.67 at the average model, weakly identified z_item tails —
    the exact regime that inflates global r_hat on the real fits); 70% cells
    missing; real-like cell sizes. The Gaussian item prior is deliberately
    misspecified against this pool, so recovery is checked on the league-table
    estimands, which remain IDENTIFIED in this regime: centred model-effect
    contrasts, item-marginal probability-scale abilities, and rho. Raw model
    intercepts are not an identified estimand here: never-solved items leave
    the item-pool mean logit — hence a common intercept offset — data-free
    (the depth of a never-solved item can trade off against the intercepts),
    so raw-intercept HDI coverage is not a meaningful check. The truth dict
    still carries raw effects for the correlation check, which a common
    offset barely affects.
    """
    rng = np.random.default_rng(seed)
    true_model = np.sort(np.clip(rng.normal(-2.2, 2.2, 45), -6.0, 2.0))
    hard = rng.random(134) < 0.60
    true_item = np.where(hard, rng.normal(-6.0, 1.5, 134), rng.normal(3.0, 1.2, 134))
    cell_n = np.array([5, 10, 20, 30, 60, 130])
    cell_w = np.array([0.20, 0.45, 0.10, 0.10, 0.10, 0.05])
    rows = []
    for m in range(45):
        for i in range(134):
            if rng.random() < 0.70:
                continue
            n = int(rng.choice(cell_n, p=cell_w))
            p = expit(true_model[m] + true_item[i])
            rows.append({"model": f"m{m:02d}", "item_id": f"i{i:03d}",
                         "n_correct": int(rng.binomial(n, p)), "n_total": n})
    df = pd.DataFrame(rows)
    p_i = expit(true_model.mean() + true_item)
    rho = p_i.var(ddof=1) / (p_i.mean() * (1 - p_i.mean()))
    truth = {
        "model_effects": true_model + true_item.mean(),
        "model_effects_centred": true_model - true_model.mean(),
        "p_marginal": expit(true_model[:, None] + true_item[None, :]).mean(axis=1),
        "item_sd_realized": float(true_item.std(ddof=1)),
        "rho": float(rho),
    }
    return df, truth


def sim_run_repro(sigma_run: float, seed: int = 12) -> tuple[pd.DataFrame, dict]:
    """30 replicate groups x 2-4 runs of n=150, known sigma_run (logit)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(30):
        mu = rng.uniform(-1.0, 1.0)
        for r in range(int(rng.integers(2, 5))):
            logit = mu + rng.normal(0, sigma_run)
            n = 150
            rows.append({"group": f"g{g:02d}", "n_total": n,
                         "n_correct": int(rng.binomial(n, expit(logit)))})
    return pd.DataFrame(rows), {"sigma_run": sigma_run}


def sim_boolq(seed: int = 13) -> tuple[pd.DataFrame, dict]:
    """5 models x 40 charities x 2 splits, known theta / condition SDs / sigma_run."""
    rng = np.random.default_rng(seed)
    theta = np.array([0.55, 0.65, 0.75, 0.85, 0.90])
    sigma_char_m = np.array([0.02, 0.05, 0.08, 0.12, 0.16])
    sigma_common = 0.04
    sigma_run = 0.02
    b_split = {"a_ref": 0.0, "b_alt": -0.03}
    delta = rng.normal(0, sigma_common, 40)
    gamma = rng.normal(0, 1, (5, 40)) * sigma_char_m[:, None]
    rows = []
    for m in range(5):
        for c in range(40):
            for split, b in b_split.items():
                for _ in range(int(rng.integers(1, 3))):
                    se = 0.006
                    mu = theta[m] + b + delta[c] + gamma[m, c]
                    rows.append({"model": f"m{m}", "charity": f"c{c:02d}", "split": split,
                                 "value": rng.normal(mu, np.sqrt(se**2 + sigma_run**2)),
                                 "se": se})
    # The fitted model centres delta and each gamma row (sum-to-zero), so its
    # theta estimand is theta + realised mean(delta) + realised mean(gamma_m).
    truth = {"theta": theta + delta.mean() + gamma.mean(axis=1),
             "sigma_char_m": sigma_char_m,
             "sigma_char_common": sigma_common, "sigma_run": sigma_run}
    return pd.DataFrame(rows), truth
