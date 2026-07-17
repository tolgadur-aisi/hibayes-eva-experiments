"""E5 — portfolio statistical-health audit of the public team_ru slice.

Idempotent entrypoint:
    cd ~/dev/hibayes-experiments && uv run python -m experiments.e5_portfolio_health.run

Order: data prep/accounting -> synthetic recovery -> real fits -> league table
-> plots. One MCMC fit at a time; all data aggregated to binomial counts or
eval-level normals.
"""

# shared.bridge must be imported first: it sets numpyro host device count.
from shared.bridge import fit, make_state, run_processors, save_outputs  # isort: skip

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hibayes.process import extract_features, extract_observed_feature
from scipy.stats import norm

from experiments.e5_portfolio_health import metrics as M
from experiments.e5_portfolio_health import prep, synth
from experiments.e5_portfolio_health.models import (
    boolq_normal,
    crossed_betabinomial,
    crossed_binomial,
    run_repro_binomial,
)

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# Okabe-Ito (colour-blind safe)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "grey": "#999999"}

FIT_KW = dict(samples=2000, warmup=1500, chains=4, target_accept=0.99)

# Per-parameter r_hat/ESS rows for every fit's HEADLINE (reported) parameters;
# written to outputs/headline_param_diagnostics.csv (see metrics.py docstring).
HEADLINE_DIAG_ROWS: list[dict] = []
HEADLINE_PARAMS = {
    "crossed": ["model_effects", "sigma_item"],
    "crossed_betabin": ["model_effects", "sigma_item", "rho_cell"],
    "repro": ["group_effects", "sigma_run"],
    "boolq": ["model_effects", "sigma_run", "sigma_char", "sigma_char_common",
              "split_effects_free"],
}


def _timed_fit(state, model, tag, seed, kind, **fit_overrides):
    t0 = time.time()
    mas = fit(state, model, tag=tag, seed=seed, **{**FIT_KW, **fit_overrides})
    diag = save_outputs(mas, OUT, tag)
    diag["wall_s"] = round(time.time() - t0, 1)
    HEADLINE_DIAG_ROWS.extend(
        M.headline_param_diagnostics(mas.inference_data, tag, HEADLINE_PARAMS[kind])
    )
    print(f"[fit] {tag}: {diag}")
    return mas, diag


def fit_crossed(cells: pd.DataFrame, tag: str, seed: int, model_fn=crossed_binomial,
                kind: str = "crossed", **fit_overrides):
    state = make_state(cells)
    state = run_processors(
        state,
        extract_features(categorical_features=["model", "item_id"],
                         continuous_features=["n_total"]),
        extract_observed_feature(feature_name="n_correct"),
    )
    mas, diag = _timed_fit(state, model_fn(), tag, seed, kind, **fit_overrides)
    return mas, diag, list(state.coords["model"])


def fit_repro(runs: pd.DataFrame, tag: str, seed: int, **fit_overrides):
    state = make_state(runs[["group", "n_total", "n_correct"]].copy())
    state = run_processors(
        state,
        extract_features(categorical_features=["group"], continuous_features=["n_total"]),
        extract_observed_feature(feature_name="n_correct"),
    )
    return _timed_fit(state, run_repro_binomial(), tag, seed, "repro", **fit_overrides)


def fit_boolq(df: pd.DataFrame, tag: str, seed: int, ref_split: str):
    state = make_state(df[["model", "charity", "split", "se", "value"]].copy())
    state = run_processors(
        state,
        extract_features(categorical_features=["model", "charity", "split"],
                         continuous_features=["se"],
                         reference_categories={"split": ref_split}),
        extract_observed_feature(feature_name="value"),
    )
    mas, diag = _timed_fit(state, boolq_normal(), tag, seed, "boolq")
    return mas, diag, list(state.coords["model"])


# ---------------------------------------------------------------- synthetic
def synthetic_recovery() -> pd.DataFrame:
    rows = []

    def row(check, true, est, lo, hi, extra=""):
        ok = bool(lo <= true <= hi) if np.isfinite(true) else False
        rows.append({"check": check, "true": round(float(true), 4),
                     "posterior_mean": round(float(est), 4),
                     "hdi_low": round(float(lo), 4), "hdi_high": round(float(hi), 4),
                     "in_hdi": ok, "note": extra})

    # S1 crossed
    df, truth = synth.sim_crossed()
    mas, _, names = fit_crossed(df, "synth_crossed", seed=101)
    me = mas.inference_data.posterior["model_effects"].values.reshape(-1, len(names))
    post_mean = me.mean(axis=0)
    corr = float(np.corrcoef(truth["model_effects"], post_mean)[0, 1])
    cover = 0
    for j in range(len(names)):
        lo, hi = M.hdi(me[:, j])
        cover += int(lo <= truth["model_effects"][j] <= hi)
    rows.append({"check": "crossed: corr(true, est) model effects", "true": 1.0,
                 "posterior_mean": round(corr, 4), "hdi_low": np.nan, "hdi_high": np.nan,
                 "in_hdi": corr >= 0.95, "note": "pass if >= 0.95"})
    rows.append({"check": "crossed: 94% HDI coverage of 8 model effects", "true": 0.94,
                 "posterior_mean": cover / 8, "hdi_low": np.nan, "hdi_high": np.nan,
                 "in_hdi": cover >= 6, "note": f"{cover}/8 covered, pass if >= 6"})
    si = mas.inference_data.posterior["sigma_item"].values.ravel()
    row("crossed: sigma_item", truth["sigma_item"], si.mean(), *M.hdi(si))
    mm = M.crossed_metrics(mas.inference_data, names, m_real=5.0)
    row("crossed: rho (intra-item corr)", truth["rho"], mm["summary"]["rho_mean"],
        *mm["summary"]["rho_hdi"])

    # S1b crossed, HARSH regime matching real cybench (45 models, bimodal item
    # pool with ~49% never-solved items, rho ~ 0.69, 70% missing cells) — the
    # regime that stresses the nuisance z_item tails on the real data. Checks
    # target the IDENTIFIED league-table estimands (see synth.sim_crossed_harsh
    # docstring): centred contrasts, item-marginal abilities, rho. Raw
    # intercepts carry an unidentified common offset (item-pool mean logit) in
    # this regime; the posterior-mean offset is reported in the note.
    df, truth = synth.sim_crossed_harsh()
    mas, _, names = fit_crossed(df, "synth_crossed_harsh", seed=105)
    me = mas.inference_data.posterior["model_effects"].values.reshape(-1, len(names))
    post_mean = me.mean(axis=0)
    corr = float(np.corrcoef(truth["model_effects"], post_mean)[0, 1])
    offset = float((post_mean - truth["model_effects"]).mean())
    rows.append({"check": "crossed-harsh: corr(true, est) model effects", "true": 1.0,
                 "posterior_mean": round(corr, 4), "hdi_low": np.nan, "hdi_high": np.nan,
                 "in_hdi": corr >= 0.95, "note": "45 models; pass if >= 0.95"})
    me_c = me - me.mean(axis=1, keepdims=True)
    cover_c = 0
    for j in range(len(names)):
        lo, hi = M.hdi(me_c[:, j])
        cover_c += int(lo <= truth["model_effects_centred"][j] <= hi)
    rows.append({"check": "crossed-harsh: 94% HDI coverage of 45 centred model effects",
                 "true": 0.94, "posterior_mean": round(cover_c / 45, 4),
                 "hdi_low": np.nan, "hdi_high": np.nan, "in_hdi": cover_c >= 39,
                 "note": (f"{cover_c}/45 covered, pass if >= 39; raw intercepts share an "
                          f"unidentified item-pool-mean offset in this regime "
                          f"(posterior-mean offset {offset:+.2f} logit)")})
    mm = M.crossed_metrics(mas.inference_data, names, m_real=1.0)
    ab = mm["abilities"]
    cover_p = int(sum(
        (ab["p_hdi_low"][j] <= truth["p_marginal"][j] <= ab["p_hdi_high"][j])
        for j in range(len(names))
    ))
    rows.append({"check": "crossed-harsh: 94% HDI coverage of 45 item-marginal abilities",
                 "true": 0.94, "posterior_mean": round(cover_p / 45, 4),
                 "hdi_low": np.nan, "hdi_high": np.nan, "in_hdi": cover_p >= 39,
                 "note": f"{cover_p}/45 covered (probability scale), pass if >= 39"})
    row("crossed-harsh: rho (intra-item corr)", truth["rho"], mm["summary"]["rho_mean"],
        *mm["summary"]["rho_hdi"],
        extra="bimodal item pool (~49% never solved), Gaussian item prior misspecified by design")

    # S2 run repro: alt + null
    for tag, sig, seed in [("synth_repro_alt", 0.20, 102), ("synth_repro_null", 0.0, 103)]:
        rdf, rtruth = synth.sim_run_repro(sig)
        mas, _ = fit_repro(rdf, tag, seed=seed)
        s = mas.inference_data.posterior["sigma_run"].values.ravel()
        lo, hi = M.hdi(s)
        if sig > 0:
            row("repro: sigma_run = 0.20", rtruth["sigma_run"], s.mean(), lo, hi)
        else:
            # Prior-vs-posterior concentration at the null: how much mass the
            # data move below eps = 0.05 (a practically-null run SD, ~1.2pp at
            # p = 0.5) relative to the HalfNormal(1) prior. A concentration
            # factor >= 10 is strong evidence for sigma_run ~ 0 (Jeffreys).
            eps = 0.05
            prior_mass = 2.0 * (norm.cdf(eps) - 0.5)  # HalfNormal(1) P(< eps)
            post_mass = float((s < eps).mean())
            conc = post_mass / prior_mass
            rows.append({"check": "repro: sigma_run = 0 (null, prior-vs-posterior)",
                         "true": 0.0,
                         "posterior_mean": round(float(s.mean()), 4),
                         "hdi_low": round(float(lo), 4), "hdi_high": round(float(hi), 4),
                         "in_hdi": conc >= 10.0,
                         "note": (f"concentration P(<{eps})_post/P(<{eps})_prior = "
                                  f"{post_mass:.3f}/{prior_mass:.3f} = {conc:.1f}, "
                                  "pass if >= 10")})

    # S3 boolq normal
    bdf, btruth = synth.sim_boolq()
    mas, _, names = fit_boolq(bdf, "synth_boolq", seed=104, ref_split="a_ref")
    th = mas.inference_data.posterior["model_effects"].values.reshape(-1, len(names))
    cover = 0
    for j in range(len(names)):
        lo, hi = M.hdi(th[:, j])
        cover += int(lo <= btruth["theta"][j] <= hi)
    rows.append({"check": "boolq: 94% HDI coverage of 5 thetas", "true": 0.94,
                 "posterior_mean": cover / 5, "hdi_low": np.nan, "hdi_high": np.nan,
                 "in_hdi": cover >= 4, "note": f"{cover}/5 covered, pass if >= 4"})
    s = mas.inference_data.posterior["sigma_run"].values.ravel()
    row("boolq: sigma_run", btruth["sigma_run"], s.mean(), *M.hdi(s))
    sc = mas.inference_data.posterior["sigma_char"].values.reshape(-1, len(names)).mean(axis=0)
    corr = float(np.corrcoef(btruth["sigma_char_m"], sc)[0, 1])
    rows.append({"check": "boolq: corr(true, est) sigma_char per model", "true": 1.0,
                 "posterior_mean": round(corr, 4), "hdi_low": np.nan, "hdi_high": np.nan,
                 "in_hdi": corr >= 0.8, "note": "pass if >= 0.8"})

    rec = pd.DataFrame(rows)
    rec.to_csv(OUT / "synthetic_recovery.csv", index=False)
    print(rec.to_string(index=False))
    return rec


def betabin_sensitivity(cells: pd.DataFrame, cmc: dict) -> pd.DataFrame:
    """S1 sensitivity fit: beta-binomial cells on cybench vs the binomial fit.

    Uses the p_floor-stabilised likelihood (see models.crossed_betabinomial):
    unfloored, the p -> 0 curvature on never-solved cells produced 197/8000
    step-size divergences at target_accept 0.99.
    """
    mas_bb, _, names_bb = fit_crossed(
        cells, "cybench_crossed_betabin", seed=500,
        model_fn=crossed_betabinomial, kind="crossed_betabin")
    bb = M.crossed_metrics(mas_bb.inference_data, names_bb, m_real=1.0)
    rc = mas_bb.inference_data.posterior["rho_cell"].values.ravel()
    sens = []

    def _sens_row(metric, b_mean, b_hdi, s_mean, s_hdi):
        sens.append({"metric": metric,
                     "binomial_mean": round(float(b_mean), 4),
                     "binomial_hdi_low": round(float(b_hdi[0]), 4),
                     "binomial_hdi_high": round(float(b_hdi[1]), 4),
                     "betabin_mean": round(float(s_mean), 4),
                     "betabin_hdi_low": round(float(s_hdi[0]), 4),
                     "betabin_hdi_high": round(float(s_hdi[1]), 4)})

    for k in ("disc_ratio", "p_best", "rho"):
        _sens_row(k, cmc["summary"][f"{k}_mean"], cmc["summary"][f"{k}_hdi"],
                  bb["summary"][f"{k}_mean"], bb["summary"][f"{k}_hdi"])
    _sens_row("median_model_logit_sd (disc denominator)",
              float(np.median(cmc["abilities"]["logit_sd"])), (np.nan, np.nan),
              float(np.median(bb["abilities"]["logit_sd"])), (np.nan, np.nan))
    _sens_row("rho_cell (betabin only)", np.nan, (np.nan, np.nan),
              float(rc.mean()), M.hdi(rc))
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(OUT / "sensitivity_betabin_cybench.csv", index=False)
    print("\n==== BETABIN SENSITIVITY (cybench) ====")
    print(sens_df.to_string(index=False))
    return sens_df


# --------------------------------------------------------------------- main
def main() -> None:
    keep_ids, _ = prep._keep_internal_ids()

    accounting: dict[str, dict] = {}
    league_rows = []
    crossed_results = {}
    repro_results = {}
    repro_tables = {}
    repro_diags = {}
    cells_by_task = {}
    derived_diag_rows = []

    # ---- B2a artifact: PRE-dedup replicate-group accounting (Finding 1).
    # Threshold is runs >= 10 samples here (the pre-dedup headline claim);
    # the post-dedup reproducibility fits use >= 20 (prep.MIN_RUN_N).
    pre = pd.concat(
        [prep.prededup_replicate_groups(t, min_run_n=10) for t in prep.SAMPLE_TASKS],
        ignore_index=True,
    )
    pre.to_csv(OUT / "run_repro_groups_prededup.csv", index=False)
    cyb = pre[pre["task"] == "cybench"]
    print(f"[prededup] cybench: {len(cyb)} groups / {int(cyb['n_runs'].sum())} runs "
          f"(runs >= 10 samples), sd_acc == 0 in {cyb['sd_zero'].mean():.1%} of groups")

    # ---- synthetic recovery first (trust gate)
    synthetic_recovery()

    # ---- sample-level benchmarks
    for i, task in enumerate(prep.SAMPLE_TASKS):
        td = prep.load_task(task, keep_ids)
        cells, inclusion = prep.crossed_cells(td)
        cells_by_task[task] = cells
        inclusion.to_csv(OUT / f"model_inclusion_{task}.csv", index=False)
        runs = prep.replicate_runs(td)
        accounting[task] = td.accounting

        mas, diag, names = fit_crossed(cells, f"{task}_crossed", seed=200 + i)
        derived_diag_rows.append(
            {"fit": f"{task}_crossed", **M.derived_diagnostics_crossed(mas.inference_data)}
        )
        m_real = float(td.accounting["median_epochs_per_item"])
        cm = M.crossed_metrics(mas.inference_data, names, m_real=m_real)
        crossed_results[task] = cm
        pd.DataFrame(cm["abilities"]).sort_values("p_mean", ascending=False).to_csv(
            OUT / f"abilities_{task}.csv", index=False)

        # empirical replicate-group table + model-based sigma_run
        emp = (runs.groupby(["group", "model"])
               .agg(n_runs=("acc", "size"), mean_acc=("acc", "mean"),
                    sd_acc=("acc", "std"), med_n=("n_total", "median"))
               .reset_index())
        emp["binom_sd"] = np.sqrt(emp["mean_acc"] * (1 - emp["mean_acc"]) / emp["med_n"])
        emp["ratio"] = emp["sd_acc"] / emp["binom_sd"]
        emp.to_csv(OUT / f"run_repro_groups_{task}.csv", index=False)
        repro_tables[task] = (runs, emp)

        masr, diagr = fit_repro(runs, f"{task}_repro", seed=300 + i)
        repro_diags[task] = diagr
        pbar_med = float(emp["mean_acc"].median())
        n_med = float(runs["n_total"].median())
        rm = M.repro_metrics(masr.inference_data, pbar_med, n_med)
        repro_results[task] = rm

        merged = {**cm["summary"], **rm}
        v, why = M.verdict(merged, n_models=len(names))
        league_rows.append({
            "benchmark": task, "n_models": len(names),
            "n_items": td.accounting["items_total"],
            "n_runs_dedup": td.accounting["runs_after_dedup"],
            "n_samples": td.accounting["rows_in_kept_models"],
            "disc_ratio_mean": merged["disc_ratio_mean"],
            "disc_ratio_hdi_low": merged["disc_ratio_hdi"][0],
            "disc_ratio_hdi_high": merged["disc_ratio_hdi"][1],
            "disc_ratio_scale": "logit (sample-level crossed model)",
            "best_model": merged["best_model"],
            "p_best_mean": merged["p_best_mean"],
            "p_best_hdi_low": merged["p_best_hdi"][0],
            "p_best_hdi_high": merged["p_best_hdi"][1],
            "p_sat_gt90": merged["p_sat_gt90"],
            "rho_mean": merged["rho_mean"],
            "rho_hdi_low": merged["rho_hdi"][0],
            "rho_hdi_high": merged["rho_hdi"][1],
            "deff_epochs10_mean": merged["deff_epochs10_mean"],
            "deff_realized_mean": merged["deff_realized_mean"],
            "median_epochs_per_item": m_real,
            "sigma_run_mean": merged["sigma_run_mean"],
            "sigma_run_hdi_low": merged["sigma_run_hdi"][0],
            "sigma_run_hdi_high": merged["sigma_run_hdi"][1],
            "run_sd_extra_pp_mean": merged["run_sd_extra_pp_mean"],
            "run_inflation_mean": merged["run_inflation_mean"],
            "run_inflation_hdi_low": merged["run_inflation_hdi"][0],
            "run_inflation_hdi_high": merged["run_inflation_hdi"][1],
            "run_eff_n_median": merged["run_eff_n_median"],
            "repro_groups": td.accounting["repro_groups"],
            "repro_runs": td.accounting["repro_runs"],
            "verdict": v, "justification": why,
        })

    # ---- B2b artifact: swe_repro target_accept sensitivity (supports the
    # funnel-tip explanation of its single divergence at target_accept 0.99)
    swe_runs, _ = repro_tables["swe_bench"]
    sd = repro_diags["swe_bench"]
    ta_rows = [{
        "target_accept": 0.99, "tag": "swe_bench_repro",
        "sigma_run_mean": round(repro_results["swe_bench"]["sigma_run_mean"], 4),
        "sigma_run_hdi_low": round(repro_results["swe_bench"]["sigma_run_hdi"][0], 4),
        "sigma_run_hdi_high": round(repro_results["swe_bench"]["sigma_run_hdi"][1], 4),
        "n_divergences": sd["n_divergences"], "max_r_hat": sd["max_r_hat"],
    }]
    for ta in (0.95, 0.97):
        tag = f"swe_bench_repro_ta{int(ta * 100)}"
        masr, diagr = fit_repro(swe_runs, tag, seed=302, target_accept=ta)
        s_ta = masr.inference_data.posterior["sigma_run"].values.ravel()
        lo, hi = M.hdi(s_ta)
        ta_rows.append({"target_accept": ta, "tag": tag,
                        "sigma_run_mean": round(float(s_ta.mean()), 4),
                        "sigma_run_hdi_low": round(lo, 4),
                        "sigma_run_hdi_high": round(hi, 4),
                        "n_divergences": diagr["n_divergences"],
                        "max_r_hat": diagr["max_r_hat"]})
    pd.DataFrame(ta_rows).to_csv(OUT / "swe_repro_target_accept_sensitivity.csv",
                                 index=False)

    # ---- S1 sensitivity: beta-binomial cells on cybench (the benchmark with
    # the largest measured run-level overdispersion, sigma_run ~ 0.57 logit,
    # which the binomial cell likelihood ignores). Compares the league-table
    # metrics under a likelihood that absorbs within-cell overdispersion.
    betabin_sensitivity(cells_by_task["cybench"], crossed_results["cybench"])

    # ---- boolq (eval-level, normal approximation)
    bq, bacct = prep.load_boolq()
    accounting["boolq_preference"] = bacct
    mas, diag, names = fit_boolq(bq, "boolq_normal", seed=400, ref_split="validation")
    derived_diag_rows.append(
        {"fit": "boolq_normal", **M.derived_diagnostics_boolq(mas.inference_data)}
    )
    se_med = float(bq["se"].median())
    bm = M.boolq_metrics(mas.inference_data, names, se_med)
    pd.DataFrame(bm["abilities"]).sort_values("p_mean", ascending=False).to_csv(
        OUT / "abilities_boolq_preference.csv", index=False)
    v, why = M.verdict(bm["summary"], n_models=len(names))
    s = bm["summary"]
    league_rows.append({
        "benchmark": "boolq_preference", "n_models": len(names),
        "n_items": np.nan, "n_runs_dedup": bacct["rows_final"],
        "n_samples": np.nan,
        "disc_ratio_mean": s["disc_ratio_mean"],
        "disc_ratio_hdi_low": s["disc_ratio_hdi"][0],
        "disc_ratio_hdi_high": s["disc_ratio_hdi"][1],
        "disc_ratio_scale": "probability (eval-level normal; tiny SEs, not comparable to logit rows)",
        "best_model": s["best_model"],
        "p_best_mean": s["p_best_mean"],
        "p_best_hdi_low": s["p_best_hdi"][0],
        "p_best_hdi_high": s["p_best_hdi"][1],
        "p_sat_gt90": s["p_sat_gt90"],
        "rho_mean": np.nan, "rho_hdi_low": np.nan, "rho_hdi_high": np.nan,
        "deff_epochs10_mean": np.nan, "deff_realized_mean": np.nan,
        "median_epochs_per_item": np.nan,
        "sigma_run_mean": s["sigma_run_mean"],
        "sigma_run_hdi_low": s["sigma_run_hdi"][0],
        "sigma_run_hdi_high": s["sigma_run_hdi"][1],
        "run_sd_extra_pp_mean": s["sigma_run_mean"] * 100,
        "run_inflation_mean": s["run_inflation_mean"],
        "run_inflation_hdi_low": s["run_inflation_hdi"][0],
        "run_inflation_hdi_high": s["run_inflation_hdi"][1],
        "run_eff_n_median": np.nan,
        "repro_groups": np.nan, "repro_runs": np.nan,
        "verdict": v, "justification": why,
    })

    league = pd.DataFrame(league_rows)
    league.to_csv(OUT / "health_league_table.csv", index=False)
    dd = pd.DataFrame(derived_diag_rows)
    dd.to_csv(OUT / "derived_diagnostics.csv", index=False)
    hd = pd.DataFrame(HEADLINE_DIAG_ROWS)
    hd.to_csv(OUT / "headline_param_diagnostics.csv", index=False)
    print("\n==== HEADLINE-PARAMETER CONVERGENCE ====")
    print(hd.to_string(index=False))
    print("\n==== DERIVED-METRIC CONVERGENCE ====")
    print(dd.to_string(index=False))
    (OUT / "data_accounting.json").write_text(json.dumps(accounting, indent=2))
    pd.DataFrame(accounting).T.reset_index(names="benchmark").to_csv(
        OUT / "data_accounting.csv", index=False)

    make_plots(crossed_results, repro_tables, repro_results, bm)
    print("\n==== LEAGUE TABLE ====")
    print(league[["benchmark", "n_models", "disc_ratio_mean", "p_best_mean",
                  "rho_mean", "run_inflation_mean", "verdict"]].to_string(index=False))


# -------------------------------------------------------------------- plots
def _short_labels(models: pd.Series) -> list[str]:
    """Strip provider prefixes but keep a disambiguator when the short name
    collides (e.g. two providers serving Mistral-Large-2411)."""
    parts = [str(m).split("/") for m in models]
    short = [p[-1] for p in parts]
    counts = pd.Series(short).value_counts()
    return [
        (f"{p[-2]}/{s}" if counts[s] > 1 and len(p) > 1 else s)[:40]
        for p, s in zip(parts, short)
    ]


def make_plots(crossed_results, repro_tables, repro_results, bm) -> None:
    tasks = list(crossed_results)

    # 1. model abilities forest (4 panels)
    fig, axes = plt.subplots(1, 4, figsize=(22, 10))
    panels = [(t, pd.DataFrame(crossed_results[t]["abilities"])) for t in tasks]
    panels.append(("boolq_preference (eval-level)", pd.DataFrame(bm["abilities"])))
    for ax, (t, ab) in zip(axes, panels[:4]):
        ab = ab.sort_values("p_mean")
        y = np.arange(len(ab))
        ax.errorbar(ab["p_mean"], y,
                    xerr=[ab["p_mean"] - ab["p_hdi_low"], ab["p_hdi_high"] - ab["p_mean"]],
                    fmt="o", color=C["blue"], ecolor=C["grey"], capsize=2, ms=4)
        ax.axvline(0.9, ls="--", color=C["red"], lw=1, label="saturation (0.90)")
        ax.set_yticks(y)
        ax.set_yticklabels(_short_labels(ab["model"]), fontsize=7)
        ax.set_xlabel("P(success), item-marginal" if "boolq" not in t else "score (typical condition)")
        ax.set_title(t, fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("E5: posterior model abilities with 94% HDIs (deduped public team_ru slice)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "abilities_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. item difficulty distributions + rho
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, t in zip(axes, tasks):
        cm = crossed_results[t]
        ax.hist(cm["item_p_mean"], bins=24, range=(0, 1), color=C["blue"], edgecolor="white")
        s = cm["summary"]
        ax.set_title(f"{t}\nrho = {s['rho_mean']:.2f} [{s['rho_hdi'][0]:.2f}, {s['rho_hdi'][1]:.2f}]"
                     f" | Deff(10 epochs) = {s['deff_epochs10_mean']:.1f}", fontsize=10)
        ax.set_xlabel("posterior mean P(success) per item, average model")
        ax.set_ylabel("items")
    fig.suptitle("E5: item difficulty spread drives overdispersion (beta-binomial-style rho)")
    fig.tight_layout()
    fig.savefig(OUT / "item_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. run reproducibility: observed same-config runs vs binomial envelope
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, t in zip(axes, tasks):
        runs, emp = repro_tables[t]
        emp = emp.sort_values("mean_acc").reset_index(drop=True)
        order = {g: i for i, g in enumerate(emp["group"])}
        lo = emp["mean_acc"] - 1.96 * emp["binom_sd"]
        hi = emp["mean_acc"] + 1.96 * emp["binom_sd"]
        ax.fill_between(np.arange(len(emp)), lo, hi, color=C["grey"], alpha=0.4,
                        label="binomial 95% envelope", step="mid")
        x = runs["group"].map(order)
        ax.scatter(x, runs["acc"], s=12, color=C["orange"], zorder=3, label="individual runs")
        rm = repro_results[t]
        ax.set_title(f"{t}\nsigma_run = {rm['sigma_run_mean']:.2f} "
                     f"[{rm['sigma_run_hdi'][0]:.2f}, {rm['sigma_run_hdi'][1]:.2f}] (logit)",
                     fontsize=10)
        ax.set_xlabel("replicate group (same model+config+items), sorted by mean")
        ax.set_ylabel("run accuracy")
        ax.legend(fontsize=8)
    fig.suptitle("E5: same-config repeat runs vs what binomial sampling alone predicts (deduped)")
    fig.tight_layout()
    fig.savefig(OUT / "run_reproducibility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
