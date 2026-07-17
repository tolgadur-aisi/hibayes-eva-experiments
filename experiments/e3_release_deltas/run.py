"""E3: release-over-release capability deltas with honest uncertainty.

Idempotent entrypoint:  uv run python -m experiments.e3_release_deltas.run

Pipeline: data hygiene -> synthetic recovery -> per-benchmark matched fits ->
cross-benchmark pooled fit (+ tau prior sensitivity) -> naive-vs-model
contrast tables -> forest plots. All artefacts land in outputs/.
"""

from shared.bridge import (  # noqa: I001  (bridge first: sets host device count)
    binomial_agg,
    fit,
    load_samples,
    make_state,
    run_processors,
    save_outputs,
)

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from hibayes.process import extract_features, extract_observed_feature

from experiments.e3_release_deltas.common import (
    BENCH_OF_CLASS,
    PAIRS,
    build_matched_classes,
    fixed_model_item_binomial,
    hdi_and_probs,
    pooled_pair_binomial,
    simulate_betabinom_design,
    simulate_fixed_design,
    simulate_pooled_design,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# dataviz reference palette (light mode)
C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_POST = "#2a78d6"  # slot 1 blue: posterior estimates
C_NAIVE = "#eb6834"  # slot 6 orange: naive point estimates
C_GRID = "#d9d8d4"

SHORT = {
    "openai/o1": "o1",
    "openai/o3": "o3",
    "openai/gpt-5": "gpt-5",
    "openai/o4-mini": "o4-mini",
    "openai/gpt-4o-2024-08-06": "gpt-4o",
    "anthropic/claude-opus-4-20250514": "opus-4",
    "anthropic/claude-sonnet-4-20250514": "sonnet-4",
    "anthropic/claude-opus-4-1-20250805": "opus-4-1",
    "anthropic/claude-3-7-sonnet-20250219": "sonnet-3-7",
}


def fit_fixed_class(agg: pd.DataFrame, tag: str, seed: int = 0,
                    samples: int = 1500, warmup: int = 1000, **model_kwargs):
    """Fit the fixed-model/item-effect binomial model on an aggregated frame."""
    state = make_state(agg)
    state = run_processors(
        state,
        extract_features(
            categorical_features=["model", "item_id"], continuous_features=["n_total"]
        ),
        extract_observed_feature(feature_name="n_correct"),
    )
    mas = fit(state, fixed_model_item_binomial(**model_kwargs), tag=tag,
              samples=samples, warmup=warmup, seed=seed)
    return mas, state


def model_effect_draws(mas, state) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(draws[n_draws, n_models], model coord names, item draws[n_draws, n_items])."""
    post = mas.inference_data.posterior
    beta = post["model_effects"].values.reshape(-1, post["model_effects"].shape[-1])
    gamma = post["item_id_effects"].values.reshape(-1, post["item_id_effects"].shape[-1])
    return beta, list(state.coords["model"]), gamma


def pair_summary(beta, models, gamma, older: str, newer: str, mas=None) -> dict:
    """Logit delta + implied equal-item-weight accuracy delta for one pair."""
    i_old, i_new = models.index(older), models.index(newer)
    delta = beta[:, i_new] - beta[:, i_old]
    acc = lambda i: (1.0 / (1.0 + np.exp(-(beta[:, [i]] + gamma)))).mean(axis=1)  # noqa: E731
    pp = acc(i_new) - acc(i_old)
    out = {f"delta_{k}": v for k, v in hdi_and_probs(delta).items()}
    out.update({f"pp_{k}": v for k, v in hdi_and_probs(pp).items() if k != "p_gt_0.5"})
    if mas is not None:
        # convergence of the *reported* contrast: the additive model/item
        # location ridge cancels in the delta, so diagnose the delta directly
        import arviz as az

        be = mas.inference_data.posterior["model_effects"]
        d = be.sel(model=newer) - be.sel(model=older)
        out["delta_r_hat"] = az.rhat(d).to_array().values.item()
        out["delta_ess_bulk"] = az.ess(d).to_array().values.item()
    return out


N_SIM = 5  # simulated datasets per design (calibration, not single-draw luck)


def check_recovery(name: str, sim: int, true_val: float, stats: dict, rows: list) -> None:
    rows.append(
        {"design": name, "sim": sim, "true": true_val, **stats,
         "bias": stats["mean"] - true_val,
         "in_94_hdi": stats["hdi_3"] <= true_val <= stats["hdi_97"]}
    )


def synthetic_recovery(rows: list) -> None:
    """Calibration-style recovery: N_SIM datasets per design shape.

    Pass criterion (asserted in main): >=80% of all true values inside the
    94% HDI (nominal 94%; binomial noise with 40 checks), and per-design
    mean bias on the key deltas < 0.2 logits.
    """
    for sim in range(N_SIM):
        # Design A: CY-HARD shape (4 models x 40 items x 10 epochs)
        betas = [-1.0, 0.0, 0.5, 1.5]
        df, _ = simulate_fixed_design(betas, n_items=40, n_per_cell=10,
                                      sigma_item=1.5, seed=100 + sim)
        mas, state = fit_fixed_class(df, tag=f"synthA{sim}", seed=1 + sim,
                                     samples=1000, warmup=800)
        beta, models, gamma = model_effect_draws(mas, state)
        if sim == 0:
            print("synthA diag:", save_outputs(mas, OUT, "synthA_cyhard_shape"))
        for m in range(1, 4):
            d = beta[:, m] - beta[:, 0]
            check_recovery(f"A: beta{m}-beta0", sim, betas[m] - betas[0],
                           hdi_and_probs(d), rows)

        # Design B: CY-EASY shape (5 models x 2 items, n=430/cell, fixed items)
        betas_b = [0.8, 1.2, 0.3, 1.0, 1.5]
        df, _ = simulate_fixed_design(betas_b, n_items=2, n_per_cell=430,
                                      sigma_item=1.0, seed=200 + sim)
        mas, state = fit_fixed_class(df, tag=f"synthB{sim}", seed=2 + sim,
                                     samples=1000, warmup=800, item_mode="fixed")
        beta, models, gamma = model_effect_draws(mas, state)
        if sim == 0:
            print("synthB diag:", save_outputs(mas, OUT, "synthB_cyeasy_shape"))
        check_recovery("B: beta1-beta0", sim, betas_b[1] - betas_b[0],
                       hdi_and_probs(beta[:, 1] - beta[:, 0]), rows)
        check_recovery("B: beta4-beta2", sim, betas_b[4] - betas_b[2],
                       hdi_and_probs(beta[:, 4] - beta[:, 2]), rows)

        # Design E: GDM-MAIN shape (5 models x 14 items, n=400/cell, fixed items)
        betas_e = [0.6, 0.9, 1.4, 0.1, 0.7]
        df, _ = simulate_fixed_design(betas_e, n_items=14, n_per_cell=400,
                                      sigma_item=1.0, seed=500 + sim)
        mas, state = fit_fixed_class(df, tag=f"synthE{sim}", seed=5 + sim,
                                     samples=1000, warmup=800, item_mode="fixed")
        beta, models, gamma = model_effect_draws(mas, state)
        if sim == 0:
            print("synthE diag:", save_outputs(mas, OUT, "synthE_gdm_shape"))
        check_recovery("E: beta1-beta0", sim, betas_e[1] - betas_e[0],
                       hdi_and_probs(beta[:, 1] - beta[:, 0]), rows)
        check_recovery("E: beta2-beta3", sim, betas_e[2] - betas_e[3],
                       hdi_and_probs(beta[:, 2] - beta[:, 3]), rows)

        # Design D: CY-HARD shape with true overdispersion, beta-binomial fit
        df = simulate_betabinom_design(betas, n_items=40, n_per_cell=10,
                                       sigma_item=1.5, kappa=15.0, seed=400 + sim)
        mas, state = fit_fixed_class(df, tag=f"synthD{sim}", seed=4 + sim,
                                     samples=1000, warmup=800,
                                     likelihood="betabinomial")
        beta, models, gamma = model_effect_draws(mas, state)
        if sim == 0:
            print("synthD diag:", save_outputs(mas, OUT, "synthD_betabinom_shape"))
        for m in range(1, 4):
            check_recovery(f"D: beta{m}-beta0", sim, betas[m] - betas[0],
                           hdi_and_probs(beta[:, m] - beta[:, 0]), rows)

        # Design C: pooled shape (bench0: 2 items n=430; bench1: 14 items n=350)
        df = simulate_pooled_design(
            alphas=[0.8, 0.6], deltas=[0.5, 0.7], n_items=[2, 14],
            n_per_cell=[430, 350], sigma_item=1.0, seed=300 + sim,
        )
        mas, state = fit_pooled(df, tag=f"synthC{sim}", seed=3 + sim,
                                samples=1000, warmup=800)
        if sim == 0:
            print("synthC diag:", save_outputs(mas, OUT, "synthC_pooled_shape"))
        post = mas.inference_data.posterior
        mu = post["mu_delta"].values.ravel()
        check_recovery("C: mu_delta", sim, 0.6, hdi_and_probs(mu), rows)
        db = post["benchmark_delta"].values.reshape(-1, 2)
        for b, true_d in enumerate([0.5, 0.7]):
            check_recovery(f"C: delta_bench{b}", sim, true_d,
                           hdi_and_probs(db[:, b]), rows)


def fit_pooled(df: pd.DataFrame, tag: str, seed: int = 0, prior_tau_scale: float = 0.5,
               samples: int = 1500, warmup: int = 1000, target_accept: float = 0.97):
    state = make_state(df)
    state = run_processors(
        state,
        extract_features(
            categorical_features=["benchmark", "item_uid"],
            continuous_features=["n_total", "is_new"],
        ),
        extract_observed_feature(feature_name="n_correct"),
    )
    # dummy-coding mask: the first item (sorted coord order) of each benchmark
    # is the reference level (gamma = 0), absorbed into that benchmark's alpha
    item_coord = list(state.coords["item_uid"])
    seen: set[str] = set()
    mask = []
    for u in item_coord:
        b = u.split("/", 1)[0]
        mask.append(1.0 if b not in seen else 0.0)
        seen.add(b)
    state.features["item_ref_mask"] = jnp.array(mask, dtype=jnp.float32)
    mas = fit(
        state,
        pooled_pair_binomial(prior_tau_scale=prior_tau_scale),
        tag=tag,
        samples=samples,
        warmup=warmup,
        seed=seed,
        target_accept=target_accept,
    )
    return mas, state


def naive_accuracies() -> pd.DataFrame:
    """All-config (unmatched) accuracies per model string, per benchmark."""
    rows = []
    for bench in ["cybench", "gdm_intercode_ctf"]:
        df = load_samples(bench, score_col="score_includes")
        df = df[df.score.isin([0.0, 1.0])]
        for m, sub in df.groupby("model"):
            if m in SHORT:
                rows.append(
                    {"benchmark": bench, "model": m, "n": len(sub), "acc": float(sub.score.mean()),
                     "n_configs": sub.task_args.nunique()}
                )
    return pd.DataFrame(rows)


def forest_plot(rows: list[dict], title: str, fname: str) -> None:
    """rows: dicts with label, mean, lo, hi, kind ('post'|'pooled'), naive (or None)."""
    fig, ax = plt.subplots(figsize=(7.6, 1.7 + 0.75 * len(rows)), dpi=160)
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    ys = np.arange(len(rows))[::-1]
    naive_plotted = False
    xs = [0.0, 0.5]
    for y, r in zip(ys, rows):
        lw = 2.6 if r["kind"] == "pooled" else 2.0
        ax.plot([r["lo"], r["hi"]], [y, y], color=C_POST, lw=lw, solid_capstyle="round",
                zorder=3)
        ax.plot(r["mean"], y, "o", color=C_POST, ms=8 if r["kind"] == "pooled" else 7, zorder=4)
        ax.annotate(f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]",
                    (r["hi"], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=C_TEXT2, zorder=5)
        xs += [r["lo"], r["hi"], r["mean"]]
        if r.get("naive") is not None:
            # naive marker offset below the interval so it never collides
            ax.plot(r["naive"], y - 0.22, "D", color=C_NAIVE, ms=6.5, zorder=4,
                    markeredgecolor=C_SURFACE, markeredgewidth=1.2)
            xs.append(r["naive"])
            naive_plotted = True
    ax.axvline(0.0, color=C_TEXT2, lw=1.0, ls="--", zorder=2)
    ax.axvline(0.5, color=C_GRID, lw=1.2, ls=":", zorder=2)
    span = max(xs) - min(xs)
    ax.set_xlim(min(xs) - 0.06 * span, max(xs) + 0.42 * span)  # annotation headroom
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5, color=C_TEXT)
    ax.set_xlabel("release delta, logit scale (posterior mean, 94% HDI)\n"
                  "dashed: no change; dotted: +0.5 logit 'meaningful' threshold",
                  fontsize=9, color=C_TEXT2)
    ax.set_title(title, fontsize=11, color=C_TEXT, loc="left", pad=12)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2)
    handles = [plt.Line2D([], [], marker="o", ls="-", color=C_POST,
                          label="matched-config model (94% HDI)")]
    if naive_plotted:
        handles.append(plt.Line2D([], [], marker="D", ls="", color=C_NAIVE,
                                  label="naive all-config log-odds diff"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               frameon=False, labelcolor=C_TEXT2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(OUT / fname, facecolor=C_SURFACE)
    plt.close(fig)


def main() -> None:
    # ---------------------------------------------------------- 0. data
    classes, hygiene = build_matched_classes()
    hygiene.to_csv(OUT / "data_hygiene.csv", index=False)
    print(hygiene.to_string())

    # ------------------------------------------- 1. synthetic recovery
    rec_rows: list[dict] = []
    synthetic_recovery(rec_rows)
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(OUT / "synthetic_recovery.csv", index=False)
    coverage = rec["in_94_hdi"].mean()
    bias = rec.groupby(rec["design"].str[0])["bias"].mean()
    print(f"recovery: coverage={coverage:.2f} ({int(rec.in_94_hdi.sum())}/{len(rec)}), "
          f"mean bias by design: {bias.to_dict()}")
    assert coverage >= 0.80, f"synthetic recovery calibration FAILED: {coverage=}"
    assert (bias.abs() < 0.2).all(), f"synthetic recovery bias FAILED: {bias.to_dict()}"

    # ------------------------------------- 2. per-class matched fits
    fits = {}
    for i, (cls, df) in enumerate(classes.items()):
        # fixed item effects unless there are many items with modest per-item n
        # (cy_easy: 2 items; gdm_main: 14 items with n>=600/item -- fixed effects
        # are fully identified and avoid the strong-data non-centred funnel;
        # cy_hard: 40 items x 40 obs -- partial pooling validated in design A)
        agg = binomial_agg(df, by=["model", "item_id"])
        mode = "fixed" if agg.item_id.nunique() < 20 else "hier"
        mas, state = fit_fixed_class(agg, tag=cls, seed=10 + i, item_mode=mode,
                                     samples=2000, warmup=1200)
        diag = save_outputs(mas, OUT, f"fit_{cls}")
        print(f"{cls} diag:", diag)
        fits[cls] = model_effect_draws(mas, state) + (mas,)

    # robustness: cy_hard epochs come from a single run per model -- refit with
    # beta-binomial overdispersion in case epochs are within-run correlated
    agg = binomial_agg(classes["cy_hard"], by=["model", "item_id"])
    mas, state = fit_fixed_class(agg, tag="cy_hard_bb", seed=20,
                                 samples=2000, warmup=1200,
                                 likelihood="betabinomial")
    print("cy_hard_bb diag:", save_outputs(mas, OUT, "fit_cy_hard_bb"))
    fits["cy_hard_bb"] = model_effect_draws(mas, state) + (mas,)

    # ------------------------------------------ 3. matched pair deltas
    naive = naive_accuracies()
    naive.to_csv(OUT / "naive_accuracies.csv", index=False)
    nacc = {(r.benchmark, r.model): (r.acc, r.n) for r in naive.itertuples()}

    def naive_stats(bench, older, newer):
        (a_o, n_o), (a_n, n_n) = nacc[(bench, older)], nacc[(bench, newer)]
        lo = lambda p: np.log(p / (1 - p))  # noqa: E731
        return {"naive_acc_old": a_o, "naive_acc_new": a_n, "naive_n_old": n_o,
                "naive_n_new": n_n, "naive_pp": a_n - a_o, "naive_logit": lo(a_n) - lo(a_o)}

    summary = []
    for pair, (older, newer, cls_list) in PAIRS.items():
        for cls in cls_list:
            beta, models, gamma, mas = fits[cls]
            s = pair_summary(beta, models, gamma, older, newer, mas=mas)
            df = classes[cls]
            matched = {
                f"matched_acc_{t}": float(df[df.model == m].score.mean())
                for t, m in [("old", older), ("new", newer)]
            }
            summary.append(
                {"pair": pair, "class": cls, "benchmark": BENCH_OF_CLASS[cls],
                 "older": older, "newer": newer, **s, **matched,
                 **naive_stats(BENCH_OF_CLASS[cls], older, newer)}
            )

    # beta-binomial robustness row for o1 -> o3
    beta, models, gamma, mas = fits["cy_hard_bb"]
    s = pair_summary(beta, models, gamma, "openai/o1", "openai/o3", mas=mas)
    summary.append(
        {"pair": "o1 -> o3", "class": "cy_hard_bb", "benchmark": "cybench",
         "older": "openai/o1", "newer": "openai/o3", **s,
         **naive_stats("cybench", "openai/o1", "openai/o3")}
    )

    # -------------------------------- 4. pooled sonnet-4 -> opus-4-1
    older, newer, cls_list = PAIRS["sonnet-4 -> opus-4-1"]
    pooled_frames = []
    for cls in cls_list:
        df = classes[cls]
        df = df[df.model.isin([older, newer])].copy()
        df["benchmark"] = BENCH_OF_CLASS[cls]
        df["item_uid"] = df["benchmark"] + "/" + df["item_id"].astype(str)
        df["is_new"] = (df.model == newer).astype(float)
        pooled_frames.append(binomial_agg(df, by=["benchmark", "item_uid", "is_new"]))
    pooled_df = pd.concat(pooled_frames, ignore_index=True)

    pooled_stats = {}
    for tau, tag in [(0.5, "pooled_tau05"), (1.0, "pooled_tau10")]:
        mas, state = fit_pooled(pooled_df, tag=tag, seed=42, prior_tau_scale=tau,
                                samples=2000, warmup=1200, target_accept=0.99)
        diag = save_outputs(mas, OUT, f"fit_{tag}")
        print(f"{tag} diag:", diag)
        post = mas.inference_data.posterior
        mu = post["mu_delta"].values.ravel()
        tau_d = post["tau"].values.ravel()
        st = hdi_and_probs(mu)
        st["tau_mean"] = float(tau_d.mean())
        st["tau_hdi_97"] = float(np.quantile(tau_d, 0.97))
        pooled_stats[tag] = st
        summary.append(
            {"pair": "sonnet-4 -> opus-4-1", "class": tag, "benchmark": "POOLED",
             "older": older, "newer": newer,
             **{f"delta_{k}": v for k, v in st.items() if not k.startswith("tau")},
             "tau_mean": st["tau_mean"], "tau_q97": st["tau_hdi_97"]}
        )

    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "summary_deltas.csv", index=False)
    print(sm.to_string())

    # ------------------------------------------------- 5. forest plots
    def row_of(rec, label, kind="post", with_naive=True):
        return {"label": label, "mean": rec["delta_mean"], "lo": rec["delta_hdi_3"],
                "hi": rec["delta_hdi_97"], "kind": kind,
                "naive": rec.get("naive_logit") if with_naive else None}

    recs = {(r["pair"], r["class"]): r for r in summary}
    forest_plot(
        [
            row_of(recs[("o1 -> o3", "cy_hard")],
                   "cybench (hard, k8s)\n40 challenges, binomial"),
            row_of(recs[("o1 -> o3", "cy_hard_bb")],
                   "cybench (hard, k8s)\nbeta-binomial robustness"),
        ],
        "OpenAI o1 -> o3: release delta on cybench", "forest_o1_o3.png",
    )
    forest_plot(
        [
            row_of(recs[("sonnet-4 -> opus-4-1", "cy_easy")],
                   "cybench (easy, docker)\n2 challenges, matched config"),
            row_of(recs[("sonnet-4 -> opus-4-1", "gdm_main")],
                   "gdm_intercode_ctf\n14 challenges, matched config"),
            row_of(recs[("sonnet-4 -> opus-4-1", "pooled_tau05")],
                   "POOLED (2 benchmarks)\ntau ~ HalfNormal(0.5)", kind="pooled",
                   with_naive=False),
        ],
        "Anthropic claude-sonnet-4 -> claude-opus-4-1: release deltas",
        "forest_sonnet4_opus41.png",
    )

    # --------------------------- 6. config-sensitivity demonstration
    # same model, same benchmark, different matched class -> accuracy shift
    demo = []
    for m in ["openai/gpt-5", "openai/o3"]:
        a_easy = float(classes["cy_easy"].query("model == @m").score.mean())
        a_hard = float(classes["cy_hard"].query("model == @m").score.mean()) \
            if m in set(classes["cy_hard"].model) else np.nan
        demo.append({"model": m, "cybench_easy_acc": a_easy, "cybench_hard_acc": a_hard,
                     "config_shift_pp": a_easy - a_hard})
    pd.DataFrame(demo).to_csv(OUT / "config_sensitivity_demo.csv", index=False)
    print(pd.DataFrame(demo).to_string())

    # gpt-4o -> gpt-5: zero config overlap; decompose the naive gdm delta
    gdm = load_samples("gdm_intercode_ctf", score_col="score_includes")
    gdm = gdm[gdm.score.isin([0.0, 1.0])]
    g4 = gdm[gdm.model == "openai/gpt-4o-2024-08-06"]
    modern_items = set(classes["gdm_main"].item_id)
    g4_shared = g4[g4.item_id.isin(modern_items)]
    g5_acc = float(classes["gdm_main"].query("model == 'openai/gpt-5'").score.mean())
    pd.DataFrame(
        [{
            "benchmark": "gdm_intercode_ctf",
            "gpt4o_acc_all79items": float(g4.score.mean()),
            "gpt4o_n_all": len(g4),
            "gpt4o_acc_14modernitems": float(g4_shared.score.mean()),
            "gpt4o_n_14items": len(g4_shared),
            "gpt5_acc_modern": g5_acc,
            "shared_task_args_configs": 0,
            "note": "gpt-4o ran vanilla_agent_setup (token_limit 250k); "
                    "gpt-5 ran built-in agent max_attempts=3/max_messages=50. "
                    "No scaffold-matched rows exist; remaining gap after item "
                    "restriction confounds scaffold with model.",
        }]
    ).to_csv(OUT / "naive_gpt4o_gpt5_decomposition.csv", index=False)

    (OUT / "pooled_stats.json").write_text(json.dumps(pooled_stats, indent=2))
    print("DONE. Outputs in", OUT)


if __name__ == "__main__":
    main()
