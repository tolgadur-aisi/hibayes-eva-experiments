"""E1 — cybench: epochs-vs-samples + model comparison.

Single idempotent entrypoint:
    uv run python -m experiments.e1_cybench.run

Stages:
  0. data prep + audit
  1. synthetic recovery (3 fits matching the real design shapes)
  2. Q1 per-model difficulty spectrum (10 challenge-hierarchical fits)
  3. Q2 ICC + design curve (claude-3-7 crossed fit + validation fits)
  4. Q3 joint model comparison (1 crossed fit, 10 models)
All outputs under experiments/e1_cybench/outputs/. Seeds fixed throughout.
"""

import matplotlib

matplotlib.use("Agg")

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from shared.bridge import binomial_agg, save_outputs  # noqa: E402
from experiments.e1_cybench import design as dz  # noqa: E402
from experiments.e1_cybench import plots  # noqa: E402
from experiments.e1_cybench import synth  # noqa: E402
from experiments.e1_cybench.fitting import (  # noqa: E402
    CROSSED_REPORTED,
    JOINT_REPORTED,
    draws,
    fit_crossed,
    fit_challenge_only,
    fit_joint,
    hdi,
    reported_diagnostics,
)
from experiments.e1_cybench.prep import hard_focal, load_clean  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
CLAUDE37 = "anthropic/claude-3-7-sonnet-20250219"
Z94 = 1.8808  # normal quantile for a 94% Wald interval
PI2_3 = np.pi ** 2 / 3


HEAVY_FIT = {"samples": 2000, "warmup": 1500, "target_accept": 0.9}


def stage0_prep() -> tuple[pd.DataFrame, dict]:
    df, audit = load_clean()
    hard = hard_focal(df)
    audit["degenerate_harness_runs"] = hard.attrs["degenerate"]
    per_model = (
        hard.groupby("model")
        .agg(n_attempts=("score", "size"), n_challenges=("challenge", "nunique"),
             n_runs=("run", "nunique"), naive_acc=("score", "mean"))
        .reset_index()
    )
    per_model.to_csv(OUT / "data_summary.csv", index=False)
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2))
    return hard, audit


def stage1_synthetic(hard: pd.DataFrame, diag: dict) -> pd.DataFrame:
    rows: list = []
    mas = synth.synth_challenge_only(rows)
    diag["synth_challenge_only"] = save_outputs(mas, OUT, "synth_challenge_only")

    claude_cells = binomial_agg(hard[hard["model"] == CLAUDE37],
                                by=["challenge", "run"])
    mas = synth.synth_crossed(claude_cells[["challenge", "run", "n_total"]],
                              rows, **HEAVY_FIT)
    diag["synth_crossed"] = save_outputs(mas, OUT, "synth_crossed")
    diag["synth_crossed"].update(reported_diagnostics(mas, CROSSED_REPORTED))

    joint_cells = binomial_agg(hard, by=["model", "challenge", "run"])
    naive = hard.groupby("model")["score"].mean().clip(0.03, 0.97)
    abilities = {m: float(logit(p)) for m, p in naive.items()}
    mas = synth.synth_joint(
        joint_cells[["model", "challenge", "run", "n_total"]], abilities,
        rows, **HEAVY_FIT)
    diag["synth_joint"] = save_outputs(mas, OUT, "synth_joint")
    diag["synth_joint"].update(reported_diagnostics(mas, JOINT_REPORTED))

    rec = pd.DataFrame(rows)
    rec.to_csv(OUT / "synth_recovery.csv", index=False)
    return rec


def stage2_q1(hard: pd.DataFrame, diag: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    prob_rows, icc_rows = [], []
    for m in sorted(hard["model"].unique()):
        sub = hard[hard["model"] == m]
        agg = binomial_agg(sub, by=["challenge"])
        mas, chal_names = fit_challenge_only(agg, tag=f"q1_{m.split('/')[-1]}")
        tag = m.split("/")[-1].replace(":", "_")
        diag[f"q1_{tag}"] = save_outputs(mas, OUT, f"q1_{tag}")

        ge = draws(mas, "group_effects")  # (sample, challenge), logit scale
        p = expit(ge)
        agg_idx = agg.set_index("challenge")
        for i, c in enumerate(chal_names):
            lo, hi = hdi(p[:, i])
            prob_rows.append({
                "model": m, "challenge": c,
                "p_mean": float(p[:, i].mean()),
                "p_hdi_3%": lo, "p_hdi_97%": hi,
                "n_correct": int(agg_idx.loc[c, "n_correct"]),
                "n_total": int(agg_idx.loc[c, "n_total"]),
            })
        sg = draws(mas, "sigma_group")
        icc_lat = sg ** 2 / (sg ** 2 + PI2_3)
        sp2 = p.var(axis=1)
        wbar = (p * (1 - p)).mean(axis=1)
        rho_obs = sp2 / (sp2 + wbar)
        pmean = p.mean(axis=0)
        lo_s, hi_s = hdi(sg)
        lo_r, hi_r = hdi(rho_obs)
        icc_rows.append({
            "model": m,
            "sigma_challenge_mean": float(sg.mean()),
            "sigma_challenge_hdi_3%": lo_s, "sigma_challenge_hdi_97%": hi_s,
            "icc_latent_mean": float(icc_lat.mean()),
            "rho_obs_mean": float(rho_obs.mean()),
            "rho_obs_hdi_3%": lo_r, "rho_obs_hdi_97%": hi_r,
            "frac_p_lt_05": float((pmean < 0.05).mean()),
            "frac_p_lt_10": float((pmean < 0.10).mean()),
            "frac_coinflip_20_80": float(((pmean > 0.2) & (pmean < 0.8)).mean()),
            "frac_p_gt_90": float((pmean > 0.9).mean()),
        })
    q1 = pd.DataFrame(prob_rows)
    q1.to_csv(OUT / "q1_challenge_probs.csv", index=False)
    icc = pd.DataFrame(icc_rows)
    icc.to_csv(OUT / "q1_model_sigma_icc.csv", index=False)
    plots.plot_difficulty_spectrum(q1, OUT / "q1_difficulty_spectrum.png")
    return q1, icc


def stage3_q2(hard: pd.DataFrame, diag: dict) -> dict:
    claude_cells = binomial_agg(hard[hard["model"] == CLAUDE37],
                                by=["challenge", "run"])
    mas, _ = fit_crossed(claude_cells, tag="q2_claude37_crossed", **HEAVY_FIT)
    diag["q2_claude37_crossed"] = save_outputs(mas, OUT, "q2_claude37_crossed")
    diag["q2_claude37_crossed"].update(
        reported_diagnostics(mas, CROSSED_REPORTED))

    curves, summary = dz.design_curves(mas)
    curves.to_csv(OUT / "q2_design_curve.csv", index=False)

    # illustration: the opus-4-1-style design (2 challenges, ~424 epochs each)
    mu = draws(mas, "mu")
    sc = draws(mas, "sigma_challenge")
    sr = draws(mas, "sigma_run")
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 2000)
    p = expit(mu[:, None] + sc[:, None] * z[None, :])
    se_c2 = np.sqrt((424 * p.var(axis=1) + (p * (1 - p)).mean(axis=1)) / 848)
    summary["se_848_attempts_2_challenges_median"] = float(np.median(se_c2))

    # run-effect floor: SD of the benchmark-mean score across runs (a
    # single-run measurement inherits this in full, regardless of k and c)
    a = draws(mas, "challenge_effects")
    idx = rng.choice(len(mu), size=800, replace=False)
    r = rng.normal(0, 1, 400)
    bench_by_run = expit(
        mu[idx][:, None, None] + a[idx][:, None, :]
        + (sr[idx][:, None] * r[None, :])[:, :, None]
    ).mean(axis=2)  # (draw, run-sim)
    run_sd = bench_by_run.std(axis=1)
    summary["run_effect_sd_acc_scale_median"] = float(np.median(run_sd))
    summary["run_effect_sd_acc_scale_q3"] = float(np.quantile(run_sd, 0.03))
    summary["run_effect_sd_acc_scale_q97"] = float(np.quantile(run_sd, 0.97))

    mc = dz.mc_check_formula(mas)
    mc.to_csv(OUT / "q2_formula_mc_check.csv", index=False)
    val = dz.mcmc_design_validation(mas, OUT)
    summary["mcmc_post_sd_pop_acc_k10_c40"] = float(
        val.loc[val["k"] == 10, "post_sd_pop_acc"].mean())
    summary["mcmc_post_sd_pop_acc_k100_c4"] = float(
        val.loc[val["k"] == 100, "post_sd_pop_acc"].mean())
    (OUT / "q2_icc_summary.json").write_text(json.dumps(summary, indent=2))
    plots.plot_design_curve(curves, summary, OUT / "q2_design_curve.png")
    return summary


def stage4_q3(hard: pd.DataFrame, diag: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = binomial_agg(hard, by=["model", "challenge", "run"])
    mas, coords = fit_joint(cells, tag="q3_joint", **HEAVY_FIT)
    diag["q3_joint"] = save_outputs(mas, OUT, "q3_joint")
    diag["q3_joint"].update(reported_diagnostics(mas, JOINT_REPORTED))

    ab = draws(mas, "model_ability")        # (sample, model), logit scale
    a = draws(mas, "challenge_effects")     # (sample, challenge)
    models = coords["model"]

    naive = hard.groupby("model")["score"].agg(["mean", "size"])
    rows = []
    for i, m in enumerate(models):
        # benchmark-mean accuracy: average over the 40 fitted challenges,
        # at a typical run (run effect 0)
        bench = expit(ab[:, i][:, None] + a).mean(axis=1)
        blo, bhi = hdi(bench)
        alo, ahi = hdi(ab[:, i])
        p_hat, n = naive.loc[m, "mean"], naive.loc[m, "size"]
        wald = Z94 * np.sqrt(p_hat * (1 - p_hat) / n)
        rows.append({
            "model": m,
            "ability_logit_mean": float(ab[:, i].mean()),
            "ability_logit_hdi_3%": alo, "ability_logit_hdi_97%": ahi,
            "bench_acc_mean": float(bench.mean()),
            "bench_acc_hdi_3%": blo, "bench_acc_hdi_97%": bhi,
            "naive_acc": float(p_hat), "n_attempts": int(n),
            "naive_lo": float(p_hat - wald), "naive_hi": float(p_hat + wald),
            "n_runs": int(hard[hard["model"] == m]["run"].nunique()),
        })
    abdf = pd.DataFrame(rows)
    abdf.to_csv(OUT / "q3_model_abilities.csv", index=False)

    pb = pd.DataFrame(
        [[float((ab[:, i] > ab[:, j]).mean()) for j in range(len(models))]
         for i in range(len(models))],
        index=models, columns=models)
    pb.to_csv(OUT / "q3_p_model_row_beats_col.csv")

    plots.plot_forest(abdf, OUT / "q3_forest.png")
    return abdf, pb


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diag: dict = {}
    t0 = time.time()

    hard, audit = stage0_prep()
    print(f"[stage0] prep done ({time.time() - t0:.0f}s): "
          f"{len(hard)} hard-variant focal rows")

    t = time.time()
    rec = stage1_synthetic(hard, diag)
    n_fail = int((~rec["in_hdi"]).sum())
    print(f"[stage1] synthetic recovery done ({time.time() - t:.0f}s): "
          f"{len(rec)} checks, {n_fail} outside 94% HDI")

    t = time.time()
    q1, icc = stage2_q1(hard, diag)
    print(f"[stage2] Q1 done ({time.time() - t:.0f}s)")

    t = time.time()
    q2 = stage3_q2(hard, diag)
    print(f"[stage3] Q2 done ({time.time() - t:.0f}s): "
          f"rho_obs={q2['rho_obs_median']:.3f}")

    t = time.time()
    abdf, pb = stage4_q3(hard, diag)
    print(f"[stage4] Q3 done ({time.time() - t:.0f}s)")

    worst = {
        "max_r_hat": max(d["max_r_hat"] for d in diag.values()),
        "min_ess_bulk": min(d["min_ess_bulk"] for d in diag.values()),
        "total_divergences": sum(d["n_divergences"] for d in diag.values()),
        # for the centred crossed/joint fits, convergence over the quantities
        # actually reported (raw z latents carry a non-identified mean-mode)
        "reported_max_r_hat": max(
            d.get("reported_max_r_hat", d["max_r_hat"])
            for d in diag.values()),
        "reported_min_ess_bulk": min(
            d.get("reported_min_ess_bulk", d["min_ess_bulk"])
            for d in diag.values()),
        "n_fits": len(diag),
    }
    manifest = {"wall_seconds": round(time.time() - t0, 1),
                "convergence_worst_case": worst, "fits": diag}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {manifest['wall_seconds']}s total; worst-case: {worst}")


if __name__ == "__main__":
    main()
