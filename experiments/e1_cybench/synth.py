"""Synthetic-recovery checks: simulate data with known parameters in the
exact shape of the real designs, fit the same models, verify recovery.

Targets are the *likelihood-identified* quantities: with a single realized
challenge panel, the identified location is mu + mean(a_realized) and the
identified spread is the realized SD of the effects — comparing to the
hyperparameters instead would confound recovery with panel sampling noise
(SD 2.5/sqrt(40) ~ 0.4 on the location).
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from experiments.e1_cybench.fitting import (
    draws,
    fit_challenge_only,
    fit_crossed,
    fit_joint,
    hdi,
)

RNG_SEED = 20260716


def _row(name, true, x):
    lo, hi = hdi(x)
    return {
        "param": name,
        "true": float(true),
        "post_mean": float(np.mean(x)),
        "hdi_3%": lo,
        "hdi_97%": hi,
        "in_hdi": bool(lo <= true <= hi),
    }


def _flag(name, value, ok, true=np.nan):
    return {"param": name, "true": true, "post_mean": float(value),
            "hdi_3%": np.nan, "hdi_97%": np.nan, "in_hdi": bool(ok)}


def synth_challenge_only(out_rows: list, **fit_kw):
    """40 challenges x 10 attempts (the o3/gpt-5-style single-run design)."""
    rng = np.random.default_rng(RNG_SEED)
    mu, sigma_c, n_chal, k = -1.0, 1.8, 40, 10
    a = rng.normal(0, sigma_c, n_chal)
    p = expit(mu + a)
    agg = pd.DataFrame({
        "challenge": [f"c{i:02d}" for i in range(n_chal)],
        "n_correct": rng.binomial(k, p),
        "n_total": k,
    })
    mas, coords = fit_challenge_only(agg, tag="synth_challenge_only", **fit_kw)
    om = draws(mas, "overall_mean")
    sg = draws(mas, "sigma_group")
    ge = draws(mas, "group_effects")  # (sample, group); includes overall mean
    out_rows.append(_row("challenge_only/mu_panel", mu + a.mean(), om))
    out_rows.append(_row("challenge_only/sigma_challenge_realized",
                         a.std(ddof=1), sg))
    truth = mu + a  # coords c00..c39 are alphabetical == generation order
    est = ge.mean(axis=0)
    r = float(np.corrcoef(truth, est)[0, 1])
    n_cover = sum(hdi(ge[:, i])[0] <= truth[i] <= hdi(ge[:, i])[1]
                  for i in range(n_chal))
    # with k=10 attempts/challenge, binomial noise on the logit is ~1 logit,
    # so corr(truth, posterior mean) ~0.87 is the information ceiling
    out_rows.append(_flag("challenge_only/effects_corr", r, r > 0.85))
    out_rows.append(_flag("challenge_only/effects_94hdi_coverage",
                          n_cover / n_chal, n_cover >= 33, true=0.94))
    return mas


def synth_crossed(design: pd.DataFrame, out_rows: list, n_rep: int = 3,
                  **fit_kw):
    """Crossed challenge x run recovery on the real claude-3-7 cell design.

    Replicated over n_rep independent simulations: sigma_run is only weakly
    identified in this design (many runs cover 1-5 challenges), so single
    realizations can miss; we require coverage in a majority of replicates.
    """
    mu, sigma_c, sigma_r = -1.0, 2.5, 0.4
    chal = sorted(design["challenge"].unique())
    runs = sorted(design["run"].unique())
    last = None
    for rep in range(n_rep):
        rng = np.random.default_rng(RNG_SEED + 1 + 1000 * rep)
        a = rng.normal(0, sigma_c, len(chal))
        b = rng.normal(0, sigma_r, len(runs))
        amap = dict(zip(chal, a))
        bmap = dict(zip(runs, b))
        cells = design.copy()
        p = expit(mu + cells["challenge"].map(amap).values
                  + cells["run"].map(bmap).values)
        cells["n_correct"] = rng.binomial(cells["n_total"].values, p)
        mas, _ = fit_crossed(cells, tag=f"synth_crossed_rep{rep}", **fit_kw)
        out_rows.append(_row(f"crossed/mu_panel_rep{rep}", mu + a.mean(),
                             draws(mas, "mu")))
        out_rows.append(_row(f"crossed/sigma_challenge_realized_rep{rep}",
                             a.std(ddof=1), draws(mas, "sigma_challenge")))
        out_rows.append(_row(f"crossed/sigma_run_realized_rep{rep}",
                             b.std(ddof=1), draws(mas, "sigma_run")))
        last = mas
    return last


def synth_joint(design: pd.DataFrame, abilities: dict, out_rows: list,
                **fit_kw):
    """Joint model recovery on the real 10-model cell design."""
    rng = np.random.default_rng(RNG_SEED + 2)
    sigma_c, sigma_r = 2.5, 0.4
    chal = sorted(design["challenge"].unique())
    runs = sorted(design["run"].unique())
    a = rng.normal(0, sigma_c, len(chal))
    b = rng.normal(0, sigma_r, len(runs))
    amap = dict(zip(chal, a))
    bmap = dict(zip(runs, b))
    cells = design.copy()
    p = expit(cells["model"].map(abilities).values
              + cells["challenge"].map(amap).values
              + cells["run"].map(bmap).values)
    cells["n_correct"] = rng.binomial(cells["n_total"].values, p)
    mas, coords = fit_joint(cells, tag="synth_joint", **fit_kw)
    ab = draws(mas, "model_ability")  # (sample, model)
    models = coords["model"]
    # identified target: ability + panel mean + the model's realized mean run
    # effect (runs nest in models here, so their mean is absorbed by ability;
    # exactly so for single-run models)
    run_mean = (cells.groupby("model")["run"].unique()
                .map(lambda rs: float(np.mean([bmap[r] for r in rs]))))
    target = {m: abilities[m] + a.mean() + run_mean[m] for m in models}
    for i, name in enumerate(models):
        out_rows.append(_row(f"joint/ability[{name}]", target[name], ab[:, i]))
    out_rows.append(_row("joint/sigma_challenge_realized", a.std(ddof=1),
                         draws(mas, "sigma_challenge")))
    out_rows.append(_row("joint/sigma_run_realized", b.std(ddof=1),
                         draws(mas, "sigma_run")))

    # pairwise contrasts: pairs separated by > 0.5 logits should mostly get
    # the right sign with high posterior probability (single-run models are
    # honestly wide, so require >=90%), and NO pair may be confidently wrong
    true_v = np.array([abilities[m] for m in models])
    est_mean = ab.mean(axis=0)
    rho = float(spearmanr(true_v, est_mean).statistic)
    out_rows.append(_flag("joint/rank_spearman", rho, rho >= 0.9))
    n_big, n_ok, n_conf_wrong = 0, 0, 0
    for i in range(len(models)):
        for j in range(len(models)):
            gap = true_v[i] - true_v[j]
            if gap > 0.5:
                n_big += 1
                p_gt = float((ab[:, i] > ab[:, j]).mean())
                if p_gt > 0.8:
                    n_ok += 1
                if p_gt < 0.2:
                    n_conf_wrong += 1
    out_rows.append(_flag("joint/big_gap_pairs_recovered", n_ok / n_big,
                          n_ok / n_big >= 0.9, true=1.0))
    out_rows.append(_flag("joint/big_gap_pairs_confidently_wrong",
                          n_conf_wrong, n_conf_wrong == 0, true=0.0))
    return mas
