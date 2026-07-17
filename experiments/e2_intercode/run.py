"""E2 — gdm_intercode_ctf variance decomposition at warehouse scale.

Idempotent entrypoint: rebuilds every file in experiments/e2_intercode/outputs/.

    cd ~/dev/hibayes-experiments && uv run python -m experiments.e2_intercode.run
"""

from shared.bridge import fit, make_state, run_processors, save_outputs  # noqa: I001  (must import first: sets host device count)

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from hibayes.process import extract_features, extract_observed_feature
from scipy.special import expit

from experiments.e2_intercode import claims, prep
from experiments.e2_intercode.common import GRAY, OUT, PALETTE, RESID, SHORT, flat, hdi94
from experiments.e2_intercode.vcmodels import crossed_var, crossed_var_hetrun, crossed_var_variant

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRIORS = dict(
    prior_mu_scale=1.5,
    prior_sigma_model=1.5,
    prior_sigma_item=2.0,
    prior_sigma_inter=0.75,
    prior_sigma_run=0.75,
)


def build_state(cells: pd.DataFrame, cats: list[str]):
    state = make_state(cells)
    state = run_processors(
        state,
        extract_features(categorical_features=cats),
        extract_observed_feature(feature_name="n_correct"),
    )
    state.features["n_total"] = jnp.asarray(cells["n_total"].values, dtype=jnp.int32)
    return state


def run_model_index(cells: pd.DataFrame, coords) -> jnp.ndarray:
    """Map run level -> model level (both in extract_features' category order)."""
    run_to_model = cells.drop_duplicates("run").set_index("run")["model"]
    model_pos = {m: i for i, m in enumerate(coords["model"])}
    return jnp.asarray([model_pos[run_to_model[r]] for r in coords["run"]], dtype=jnp.int32)


def fit_and_save(cells, cats, model, name, seed=0, het_run=False):
    state = build_state(cells, cats)
    if het_run:
        state.features["run_model_index"] = run_model_index(cells, state.coords)
    mas = fit(state, model, tag=name, samples=1500, warmup=1500, chains=4, seed=seed,
              target_accept=0.99)
    post = mas.inference_data.posterior
    mas.inference_data.posterior = post.drop_vars([v for v in post.data_vars if str(v).startswith("z_")])
    diag = save_outputs(mas, OUT, name)
    print(f"[{name}] {diag}")
    return mas.inference_data, state.coords, diag


# ---------------------------------------------------------------- variance shares
def variance_shares(idata, name, has_variant=False) -> pd.DataFrame:
    """Finite-population variance components (per posterior draw) + shares.

    V_k = var(effects_k, ddof=1) on the logit scale; epoch residual fixed at
    pi^2/3 (latent-threshold logistic). Superpopulation sigmas reported too.
    """
    comps = {
        "model": flat(idata, "model_effects"),
        "challenge": flat(idata, "item_effects"),
        "model x challenge": flat(idata, "inter_effects").reshape(len(flat(idata, "mu")), -1),
        "run": flat(idata, "run_effects"),
    }
    if has_variant:
        comps["scaffold variant"] = flat(idata, "variant_effects")
    V = {k: v.var(axis=1, ddof=1) for k, v in comps.items()}
    V["epoch residual"] = np.full_like(V["model"], RESID)
    total = sum(V.values())
    sigma_map = {
        "model": "sigma_model",
        "challenge": "sigma_item",
        "model x challenge": "sigma_inter",
        "run": "sigma_run",
        "scaffold variant": "sigma_variant",
    }
    rows = []
    for k, v in V.items():
        share = v / total
        row = {
            "dataset": name,
            "component": k,
            "var_mean": v.mean(),
            "var_hdi3": hdi94(v)[0],
            "var_hdi97": hdi94(v)[1],
            "share_mean": share.mean(),
            "share_hdi3": hdi94(share)[0],
            "share_hdi97": hdi94(share)[1],
        }
        if k in sigma_map:
            s = flat(idata, sigma_map[k])
            row |= {"sigma_mean": s.mean(), "sigma_hdi3": hdi94(s)[0], "sigma_hdi97": hdi94(s)[1]}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- design effects
def sigma_run_by_model(idata, n_models: int) -> np.ndarray:
    """(draws, M) run-effect scale; broadcasts the pooled scalar model."""
    sr = flat(idata, "sigma_run")
    if sr.ndim == 1:
        sr = np.repeat(sr[:, None], n_models, axis=1)
    return sr


def design_effects(idata, coords, name, k_list=(1, 10), n_nodes=41, n_draws=500) -> pd.DataFrame:
    """Effective sample size of one fresh run of the eval, per model.

    Estimand: the model's expected accuracy on the FIXED item set. Noise: the
    run-level effect u ~ N(0, sigma_run) plus Bernoulli sampling; item effects
    are fixed strata. N_eff = pbar(1-pbar) / Var(run mean score) = the number of
    iid Bernoulli(pbar) draws giving the same precision. Gauss-Hermite
    quadrature over u; posterior subsampled to n_draws.
    """
    mu = flat(idata, "mu")
    a = flat(idata, "model_effects")
    b = flat(idata, "item_effects")
    g = flat(idata, "inter_effects")
    sr_m = sigma_run_by_model(idata, a.shape[1])
    idx = np.linspace(0, len(mu) - 1, n_draws).astype(int)
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    n_items = b.shape[1]
    rows = []
    for m, mname in enumerate(coords["model"]):
        logit = mu[idx, None] + a[idx, m, None] + b[idx] + g[idx, m]  # (draws, I)
        u = sr_m[idx, m, None] * x[None, :]  # (draws, nodes)
        P = expit(logit[:, None, :] + u[:, :, None])  # (draws, nodes, I)
        pbar = P.mean(axis=2)
        sbar = (P * (1 - P)).mean(axis=2)
        m1 = (w * pbar).sum(axis=1)
        between = (w * pbar**2).sum(axis=1) - m1**2
        wbar = (w * sbar).sum(axis=1)
        row = {
            "dataset": name,
            "model": SHORT.get(str(mname), str(mname)),
            "n_items": n_items,
            "p_mean": m1.mean(),
            "sd_between_pp": (np.sqrt(between) * 100).mean(),
            "sd_between_pp_hdi3": hdi94(np.sqrt(between) * 100)[0],
            "sd_between_pp_hdi97": hdi94(np.sqrt(between) * 100)[1],
        }
        for K in k_list:
            var = between + wbar / (n_items * K)
            neff = m1 * (1 - m1) / var
            lo, hi = hdi94(neff)
            row |= {
                f"N_nominal_K{K}": n_items * K,
                f"N_eff_K{K}": neff.mean(),
                f"N_eff_K{K}_hdi3": lo,
                f"N_eff_K{K}_hdi97": hi,
            }
        ceil = m1 * (1 - m1) / np.maximum(between, 1e-12)
        row |= {"N_eff_ceiling": np.median(ceil), "N_eff_ceiling_hdi3": hdi94(ceil)[0], "N_eff_ceiling_hdi97": hdi94(ceil)[1]}
        rows.append(row)
    return pd.DataFrame(rows)


def neff_curves(idata, coords, k_max=30, n_nodes=41, n_draws=400):
    """Posterior-mean N_eff as a function of epochs K, per model."""
    mu, a, b, g = (flat(idata, v) for v in ["mu", "model_effects", "item_effects", "inter_effects"])
    sr_m = sigma_run_by_model(idata, a.shape[1])
    idx = np.linspace(0, len(mu) - 1, n_draws).astype(int)
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    n_items = b.shape[1]
    ks = np.arange(1, k_max + 1)
    out = {}
    for m, mname in enumerate(coords["model"]):
        logit = mu[idx, None] + a[idx, m, None] + b[idx] + g[idx, m]
        u = sr_m[idx, m, None] * x[None, :]
        P = expit(logit[:, None, :] + u[:, :, None])
        pbar, sbar = P.mean(axis=2), (P * (1 - P)).mean(axis=2)
        m1 = (w * pbar).sum(axis=1)
        between = (w * pbar**2).sum(axis=1) - m1**2
        wbar = (w * sbar).sum(axis=1)
        neff = m1[:, None] * (1 - m1[:, None]) / (between[:, None] + wbar[:, None] / (n_items * ks[None, :]))
        out[SHORT.get(str(mname), str(mname))] = neff.mean(axis=0)
    return ks, out, n_items


# ---------------------------------------------------------------- synthetic recovery
def simulate_like(cells: pd.DataFrame, truth: dict, seed: int):
    """Simulate counts on the exact D1 design. truth['sigma_run'] may be a
    scalar or a {model: sd} dict (heterogeneous run noise).

    Returns (sim_cells, realized) where realized holds the finite-population
    SDs of the drawn effects — the quantity the finite-population posterior
    should recover (the superpopulation sigma is only recoverable up to the
    sampling noise of a handful of realized groups)."""
    rng = np.random.default_rng(seed)
    sim = cells.copy()
    mi, models = pd.factorize(sim["model"])
    ii, items = pd.factorize(sim["item"])
    ri, runs = pd.factorize(sim["run"])
    run_model = sim.drop_duplicates("run").set_index("run")["model"]
    a = rng.normal(0, truth["sigma_model"], len(models))
    b = rng.normal(0, truth["sigma_item"], len(items))
    g = rng.normal(0, truth["sigma_inter"], (len(models), len(items)))
    sr = truth["sigma_run"]
    run_sd = np.array([sr[run_model[r]] if isinstance(sr, dict) else sr for r in runs])
    u = rng.normal(0, 1, len(runs)) * run_sd
    logit = truth["mu"] + a[mi] + b[ii] + g[mi, ii] + u[ri]
    sim["n_correct"] = rng.binomial(sim["n_total"].values, expit(logit))
    realized = {
        "sigma_model": a.std(ddof=1),
        "sigma_item": b.std(ddof=1),
        "sigma_inter": g.ravel().std(ddof=1),
        "sigma_run": u.std(ddof=1),
    }
    return sim, realized


EFFECT_OF = {
    "sigma_model": "model_effects",
    "sigma_item": "item_effects",
    "sigma_inter": "inter_effects",
    "sigma_run": "run_effects",
}


def recovery_rows(idata, truth, realized, name):
    """Two checks per component: superpopulation sigma vs true value, and
    finite-population SD of the effects vs the realized SD of the draw."""
    rows = []
    mu = flat(idata, "mu")
    lo, hi = hdi94(mu)
    rows.append({"fit": name, "param": "mu", "kind": "superpop", "target": truth["mu"],
                 "post_mean": mu.mean(), "hdi3": lo, "hdi97": hi,
                 "in_hdi": bool(lo <= truth["mu"] <= hi)})
    for p, eff in EFFECT_OF.items():
        s = flat(idata, p)
        if s.ndim > 1:  # per-model sigma_run: compare pooled scale via effects only
            s = None
        tr = truth[p]
        if s is not None and not isinstance(tr, dict):
            lo, hi = hdi94(s)
            rows.append({"fit": name, "param": p, "kind": "superpop", "target": tr,
                         "post_mean": s.mean(), "hdi3": lo, "hdi97": hi,
                         "in_hdi": bool(lo <= tr <= hi)})
        e = flat(idata, eff)
        e = e.reshape(len(e), -1)
        fp = e.std(axis=1, ddof=1)
        lo, hi = hdi94(fp)
        rows.append({"fit": name, "param": p, "kind": "finite_pop", "target": realized[p],
                     "post_mean": fp.mean(), "hdi3": lo, "hdi97": hi,
                     "in_hdi": bool(lo <= realized[p] <= hi)})
    return rows


# ---------------------------------------------------------------- empirical checks
def dispersion_check(d1: pd.DataFrame, n_null: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Model-free run overdispersion: observed between-run variance of per-item
    correct counts vs the binomial expectation, pooled over items; Monte Carlo
    p-value under the exact binomial null."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, dm in d1.groupby("model"):
        piv = dm.pivot(index="item", columns="run", values="n_correct")
        piv = piv.dropna()
        n_per = dm["n_total"].iloc[0]
        R = piv.shape[1]
        phat = piv.sum(axis=1) / (R * n_per)
        keep = (phat > 0) & (phat < 1)
        obs = piv[keep].var(axis=1, ddof=1).sum()
        exp = (n_per * phat[keep] * (1 - phat[keep])).sum()
        # MC null: same phat, R runs of Binomial(n_per, phat)
        k = rng.binomial(n_per, np.repeat(phat[keep].values[:, None], R, axis=1)[None].repeat(n_null, 0))
        null_ratio = k.var(axis=2, ddof=1).sum(axis=1) / exp
        rows.append({
            "model": SHORT.get(model, model), "runs": R, "items_used": int(keep.sum()),
            "epochs_per_item": int(n_per),
            "var_ratio_obs_over_binom": obs / exp,
            "mc_p_value": float((null_ratio >= obs / exp).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)


def plot_shares(shares: pd.DataFrame, path: Path):
    d = shares[shares.dataset == "d1"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(d))
    ax.hlines(y, d.share_hdi3, d.share_hdi97, color=PALETTE[0], lw=2)
    ax.plot(d.share_mean, y, "o", color=PALETTE[0], ms=7)
    ax.set_yticks(y, d.component)
    ax.set_xlabel("share of total logit-scale variance (posterior mean, 94% HDI)")
    ax.set_title("intercode-ctf: where does outcome variance come from?\n(primary dataset: 12 runs x 4 models x 79 challenges x 10 epochs)")
    ax.set_xlim(0, 1)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_run_effects(idata, coords, registry: pd.DataFrame, path: Path):
    u = flat(idata, "run_effects")
    runs = [str(r) for r in coords["run"]]
    reg = registry.set_index("run")
    models = sorted(reg.loc[runs, "model"].unique())
    cmap = {m: PALETTE[i] for i, m in enumerate(models)}
    order = np.argsort([reg.loc[r, "model"] + str(reg.loc[r, "start"]) for r in runs])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for y, j in enumerate(order):
        r = runs[j]
        lo, hi = hdi94(u[:, j])
        c = cmap[reg.loc[r, "model"]]
        ax.hlines(y, lo, hi, color=c, lw=2)
        ax.plot(u[:, j].mean(), y, "o", color=c, ms=6)
    ax.set_yticks(
        range(len(order)),
        [f"{SHORT.get(reg.loc[runs[j], 'model'], '?')}  {str(reg.loc[runs[j], 'start'])[:10]}" for j in order],
        fontsize=8,
    )
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("run effect on logit scale (posterior mean, 94% HDI)")
    ax.set_title("Run-to-run drift: run effects, same model + config")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_neff(ks, curves: dict, n_items: int, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (m, v) in enumerate(sorted(curves.items())):
        ax.plot(ks, v, color=PALETTE[i], lw=2, label=m)
    # direct labels, de-collided: place bottom-up with a min separation
    ymax = max(v[-1] for v in curves.values())
    min_sep = 0.04 * ymax
    order = sorted(curves.items(), key=lambda kv: kv[1][-1])
    colors = {m: PALETTE[i] for i, (m, _) in enumerate(sorted(curves.items()))}
    last_y = None
    for m, v in order:
        y = v[-1]
        if last_y is not None and y - last_y < min_sep:
            y = last_y + min_sep
        ax.annotate(m, (ks[-1], y), xytext=(4, 0), textcoords="offset points",
                    color=colors[m], fontsize=8, va="center")
        last_y = y
    ax.plot(ks, n_items * ks, color=GRAY, lw=1.2, ls="--", label=f"nominal N = {n_items}k")
    ax.set_xlabel("epochs per challenge (K)")
    ax.set_ylabel("effective N (independent Bernoulli samples)")
    ax.set_title(f"Effective sample size of one intercode-ctf run ({n_items} challenges)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(1, ks[-1] + 6)
    style_ax(ax)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_day_effects(idata, coords, path: Path):
    u = flat(idata, "run_effects")
    labels = [str(r) for r in coords["run"]]
    models = sorted({lab.split("|")[0] for lab in labels})
    cmap = {m: PALETTE[i] for i, m in enumerate(models)}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for y, lab in enumerate(labels):
        m, day = lab.split("|")
        lo, hi = hdi94(u[:, y])
        ax.hlines(y, lo, hi, color=cmap[m], lw=2)
        ax.plot(u[:, y].mean(), y, "o", color=cmap[m], ms=6)
    ax.set_yticks(range(len(labels)), [f"{SHORT.get(lab.split('|')[0], '?')}  {lab.split('|')[1][5:]}" for lab in labels], fontsize=8)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xlabel("day-batch effect on logit scale (posterior mean, 94% HDI)")
    ax.set_title("Frontier fan-out campaign (Sep 2025): day-to-day drift, 14 challenges")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    df, log = prep.load_clean()

    d1 = prep.build_d1(df)
    d1x = prep.build_d1x(df)
    d2 = prep.build_d2(df)
    log |= {
        "d1_cells": len(d1), "d1_runs": d1.run.nunique(), "d1_models": d1.model.nunique(),
        "d1_samples": int(d1.n_total.sum()),
        "d1x_cells": len(d1x), "d1x_runs": d1x.run.nunique(), "d1x_variants": d1x.variant.nunique(),
        "d1x_samples": int(d1x.n_total.sum()),
        "d2_cells": len(d2), "d2_batches": d2.run.nunique(), "d2_samples": int(d2.n_total.sum()),
    }
    (OUT / "data_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))

    # run registry (for plots + FINDINGS traceability)
    registry = (
        df[df.run_internal_id.astype(str).isin(d1.run.unique())]
        .groupby("run_internal_id")
        .agg(model=("model", "first"), start=("created", "min"), variant=("variant", "first"))
        .reset_index()
        .rename(columns={"run_internal_id": "run"})
    )
    registry["run"] = registry["run"].astype(str)
    registry.to_csv(OUT / "d1_run_registry.csv", index=False)

    # empirical, model-free checks
    disp = dispersion_check(d1)
    disp.to_csv(OUT / "dispersion_check.csv", index=False)
    print(disp.to_string())
    d1rows = df[df.run_internal_id.astype(str).isin(d1.run.unique())]
    epoch_check = d1rows.groupby("epoch")["score"].agg(["mean", "count"]).reset_index()
    epoch_check.to_csv(OUT / "epoch_check.csv", index=False)

    # ---- synthetic recovery (design copied from D1) ----
    truth_a = dict(mu=-1.0, sigma_model=0.8, sigma_item=1.8, sigma_inter=0.4, sigma_run=0.35)
    truth_b = truth_a | {"sigma_run": 0.0}
    # heterogeneous run noise mirroring the empirical dispersion check
    truth_c = truth_a | {"sigma_run": {
        "azureai/Mistral-Large-2411": 0.7,
        "gemma/gemma-3-27b-it": 0.1,
        "mistralazure/Mistral-Large-2411": 0.1,
        "mistralazure/Mistral-Large-2411-CAST": 0.7,
    }}
    rec = []
    for name, truth, seed, model_fn, het in [
        ("synth_a", truth_a, 42, crossed_var(**PRIORS), False),
        ("synth_b", truth_b, 43, crossed_var(**PRIORS), False),
        ("synth_c_pooled", truth_c, 44, crossed_var(**PRIORS), False),
        ("synth_c_het", truth_c, 44, crossed_var_hetrun(**PRIORS), True),
    ]:
        sim, realized = simulate_like(d1, truth, seed)
        idata, coords, _ = fit_and_save(sim, ["model", "item", "run"], model_fn, name, seed=1, het_run=het)
        rec += recovery_rows(idata, truth, realized, name)
        if het:  # per-model sigma_run recovery for the heterogeneous fit
            sr = flat(idata, "sigma_run")
            for m, mname in enumerate(coords["model"]):
                lo, hi = hdi94(sr[:, m])
                rec.append({"fit": name, "param": f"sigma_run[{SHORT.get(str(mname))}]",
                            "kind": "superpop", "target": truth_c["sigma_run"][str(mname)],
                            "post_mean": sr[:, m].mean(), "hdi3": lo, "hdi97": hi,
                            "in_hdi": bool(lo <= truth_c["sigma_run"][str(mname)] <= hi)})
    rec_df = pd.DataFrame(rec)
    rec_df.to_csv(OUT / "synthetic_recovery.csv", index=False)
    print(rec_df.to_string())

    # ---- real fits ----
    id1, c1, _ = fit_and_save(d1, ["model", "item", "run"], crossed_var(**PRIORS), "d1_primary")
    id1h, c1h, _ = fit_and_save(d1, ["model", "item", "run"], crossed_var_hetrun(**PRIORS),
                                "d1_hetrun", het_run=True)
    id1x, c1x, _ = fit_and_save(
        d1x, ["model", "variant", "item", "run"],
        crossed_var_variant(**(PRIORS | {"prior_sigma_variant": 0.75})), "d1x_variants",
    )
    id2, c2, _ = fit_and_save(d2, ["model", "item", "run"], crossed_var(**PRIORS), "d2_frontier_days")

    # per-model run-effect scales (heterogeneous fit)
    srh = flat(id1h, "sigma_run")
    per_model = pd.DataFrame([
        {"model": SHORT.get(str(m), str(m)), "sigma_run_mean": srh[:, i].mean(),
         "sigma_run_hdi3": hdi94(srh[:, i])[0], "sigma_run_hdi97": hdi94(srh[:, i])[1],
         "n_runs": int(d1[d1.model == str(m)].run.nunique())}
        for i, m in enumerate(c1h["model"])
    ])
    per_model.to_csv(OUT / "sigma_run_per_model.csv", index=False)
    print(per_model.to_string())

    # ---- variance shares ----
    shares = pd.concat([
        variance_shares(id1, "d1"),
        variance_shares(id1x, "d1x", has_variant=True),
        variance_shares(id2, "d2_frontier"),
    ])
    shares.to_csv(OUT / "variance_shares.csv", index=False)
    print(shares.to_string())

    # ---- design effects ----
    de = pd.concat([
        design_effects(id1, c1, "d1_pooled", k_list=(1, 10)),
        design_effects(id1h, c1h, "d1_hetrun", k_list=(1, 10)),
        design_effects(id2, c2, "d2_frontier", k_list=(1, 10)),
    ])
    de.to_csv(OUT / "design_effects.csv", index=False)
    print(de.to_string())

    # ---- plots ----
    plot_shares(shares, OUT / "variance_shares.png")
    plot_run_effects(id1h, c1h, registry, OUT / "run_effects_d1.png")
    ks, curves, n_items = neff_curves(id1h, c1h)
    plot_neff(ks, curves, n_items, OUT / "neff_vs_epochs_d1.png")
    plot_day_effects(id2, c2, OUT / "day_effects_d2.png")
    claims.main()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
