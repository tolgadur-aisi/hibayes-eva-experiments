"""E4: capability growth over time on AISI public benchmarks.

Per task, fits a logit-scale trend of pooled headline accuracy against MODEL
RELEASE DATE with a per-model residual ("model random effect": runs of one
model are pooled into a single binomial observation, so N runs of one model
are never treated as N independent pieces of trend evidence). Also fits a
frontier (running-best-model) trend and a quadratic saturation check.

Idempotent single entrypoint:
    cd ~/dev/hibayes-experiments && uv run python -m experiments.e4_growth_curves.run
"""

from shared.bridge import (  # isort: skip  (must be first: sets numpyro devices)
    fit,
    load_evals,
    make_state,
    run_processors,
    save_outputs,
)

import json
import zlib
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import matplotlib
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from hibayes.model import model as hb_model
from hibayes.process import extract_features, extract_observed_feature

from experiments.e4_growth_curves.dates import (
    CANONICAL,
    RELEASE_DATES,
    exclusion_reason,
)
from experiments.e4_growth_curves.selection import (
    observed_record_stats,
    simulate_record_null,
    summarise_selection_test,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
EPOCH = pd.Timestamp("2025-01-01", tz="UTC")
HDI = 0.94
FRONTIER_MIN_N = 20  # samples needed before a model can define the frontier
FIT_MIN_N = 20  # cells with fewer pooled samples are excluded from trend fits:
# they carry almost no information about the trend (binomial se > 0.5 logits)
# but their weakly-identified theta_j funnels the sampler (verified: swe-like
# synthetic runs with N=2/N=12 cells hit ~100 divergences; 0 after exclusion)
SEED = 20260716
# Cap the effective binomial N per model x task cell. At N=10k the sampling
# se on the logit scale is ~0.02, an order of magnitude below the
# between-model residual sigma (~0.4-0.9), so inference is unchanged; without
# the cap boolq cells reach N~3M, whose near-delta likelihood makes NUTS
# adaptation fail sporadically (observed r_hat > 3 in synthetic recovery).
N_CAP = 10_000

# dataviz palette (validated: scripts/validate_palette.js, all checks pass)
C_MEAN = "#2a78d6"  # series 1: mean trend
C_FRONTIER = "#eb6834"  # series 6: frontier trend
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e8e8e6"
C_SURFACE = "#fcfcfb"

TASKS = ["cybench", "gdm_intercode_ctf", "swe_bench", "boolq_preference"]
TASK_LABELS = {
    "cybench": "Cybench (cyber CTF)",
    "gdm_intercode_ctf": "GDM InterCode CTF (cyber)",
    "swe_bench": "SWE-bench (software eng.)",
    "boolq_preference": "BoolQ-preference (QA, pattern scorer)",
}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@hb_model
def release_trend_logit_normal(
    prior_intercept_scale: float = 2.0,
    prior_slope_scale: float = 2.0,
    prior_sigma_scale: float = 1.0,
    quadratic: bool = False,
    no_slope: bool = False,
):
    """Trend on release date with a per-model residual, latent logits
    marginalised out.

    Generative view: K_j ~ Binomial(N_j, sigmoid(theta_j)),
    theta_j ~ Normal(alpha + beta*t_j [+ beta2*t_j^2], sigma_model).
    For the fitted cells (N_j >= 20, in practice N_j >= 385) the binomial
    likelihood for theta_j is well approximated by
    y_j ~ Normal(theta_j, se_j) with y_j the empirical logit, so theta
    integrates out analytically:  y_j ~ Normal(mu_j, sqrt(se_j^2 + sigma^2)).
    The se^2 floor in the variance removes the sigma->0 funnel that made the
    exact latent-theta binomial model diverge for small J (verified in
    synthetic recovery: latent version hit 5-100+ divergences on the J=3
    swe-like design; this version has none). One row per model: the trend is
    identified across models, never across repeated runs of one model.
    """

    def _model(features) -> None:
        t = features["t"]
        se = features["se"]
        alpha = numpyro.sample("alpha", dist.Normal(0.0, prior_intercept_scale))
        if no_slope:  # null model for the frontier selection calibration
            mu = alpha + 0.0 * t
        else:
            beta = numpyro.sample("beta", dist.Normal(0.0, prior_slope_scale))
            mu = alpha + beta * t
        if quadratic:
            beta2 = numpyro.sample("beta2", dist.Normal(0.0, prior_slope_scale))
            mu = mu + beta2 * t**2
        sigma = numpyro.sample("sigma_model", dist.HalfNormal(prior_sigma_scale))
        numpyro.sample(
            "obs",
            dist.Normal(mu, jnp.sqrt(se**2 + sigma**2)),
            obs=features["obs"],
        )

    return _model


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def build_clean_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean run-level data + exclusions ledger."""
    df = load_evals()
    ledger: list[dict] = [{"step": "raw rows", "n_dropped": 0, "n_left": len(df)}]

    def note(step: str, before: int, after: int) -> None:
        ledger.append({"step": step, "n_dropped": before - after, "n_left": after})

    n0 = len(df)
    df = df[df["score_headline_value"].notna()].copy()
    note("drop NaN score_headline_value (incl. all boolq boolq_scorer rows)", n0, len(df))

    n0 = len(df)
    df = df[df["score_headline_metric"].isin(["accuracy", "mean"])]
    note("drop metric not in {accuracy, mean}", n0, len(df))

    # duplicate uploads: same .eval log ingested from >1 S3 location
    n0 = len(df)
    df = df.drop_duplicates(
        subset=[
            "eval_id",
            "task_name",
            "model",
            "created",
            "score_headline_value",
            "completed_samples",
        ]
    )
    note("dedupe re-uploaded eval logs (same eval_id+created+value)", n0, len(df))

    # boolq: keep only the pattern/accuracy scorer (boolq_scorer headline has
    # no numeric value in the warehouse; dropped above)
    n0 = len(df)
    df = df[
        (df["task_name"] != "boolq_preference")
        | (df["score_headline_name"] == "pattern")
    ]
    note("boolq: keep pattern scorer only", n0, len(df))

    # undatable models
    df["excl_reason"] = df["model"].map(lambda m: exclusion_reason(m))
    for reason, grp in df[df["excl_reason"].notna()].groupby("excl_reason"):
        ledger.append(
            {
                "step": f"exclude models: {reason} "
                f"({grp['model'].nunique()} model strings)",
                "n_dropped": len(grp),
                "n_left": None,
            }
        )
    df = df[df["excl_reason"].isna()].drop(columns=["excl_reason"])
    ledger.append({"step": "final run-level rows", "n_dropped": 0, "n_left": len(df)})

    df["canonical"] = df["model"].map(CANONICAL)
    df["release_date"] = pd.to_datetime(
        df["canonical"].map(lambda c: RELEASE_DATES[c][0]), utc=True
    )
    df["t"] = (df["release_date"] - EPOCH).dt.days / 365.25
    df["k_run"] = df["score_headline_value"] * df["completed_samples"]
    return df.reset_index(drop=True), pd.DataFrame(ledger)


def data_checks() -> dict:
    """Persist the Data-section verification numbers (REVIEW_ROUND_1 sugg. 1).

    Recomputed from the raw parquet with the same filters as
    build_clean_runs, so every number quoted in FINDINGS' Data section is
    traceable to outputs/data_checks.json.
    """
    df = load_evals()
    df = df[df["score_headline_value"].notna()]
    df = df[df["score_headline_metric"].isin(["accuracy", "mean"])]
    key = [
        "eval_id",
        "task_name",
        "model",
        "created",
        "score_headline_value",
        "completed_samples",
    ]
    sizes = df.groupby(key).size()
    dup = sizes[sizes > 1]

    # boolq split stability: weighted per-canonical-model accuracy by split
    b = df[
        (df["task_name"] == "boolq_preference")
        & (df["score_headline_name"] == "pattern")
    ].drop_duplicates(subset=key)
    b = b[b["model"].map(exclusion_reason).isna()].copy()
    b["canonical"] = b["model"].map(CANONICAL)
    b["split"] = b["task_args"].map(lambda s: json.loads(s).get("split"))
    b["k"] = b["score_headline_value"] * b["completed_samples"]
    by_split = (
        b.groupby(["canonical", "split"])
        .apply(
            lambda x: x["k"].sum() / x["completed_samples"].sum(),
            include_groups=False,
        )
        .unstack()
    )
    split_diff = (by_split["train"] - by_split["validation"]).abs()
    checks = {
        "dedupe_rows_dropped": int((sizes - 1).sum()),
        "dedupe_distinct_eval_ids": int(dup.reset_index()["eval_id"].nunique()),
        "dedupe_copies_per_key_min": int(dup.min()),
        "dedupe_copies_per_key_max": int(dup.max()),
        "boolq_split_weighted_acc": {
            m: {s: float(v) for s, v in row.items()}
            for m, row in by_split.iterrows()
        },
        "boolq_max_abs_train_validation_diff": float(split_diff.max()),
    }
    (OUT / "data_checks.json").write_text(json.dumps(checks, indent=2))
    return checks


def aggregate_models(runs: pd.DataFrame) -> pd.DataFrame:
    """One binomial observation per task x canonical model."""
    agg = (
        runs.groupby(["task_name", "canonical", "release_date", "t"])
        .agg(
            n_runs=("internal_id", "count"),
            n_total=("completed_samples", "sum"),
            k_float=("k_run", "sum"),
            run_acc_sd=("score_headline_value", "std"),
            first_run=("created", "min"),
            last_run=("created", "max"),
        )
        .reset_index()
    )
    agg["n_correct"] = agg["k_float"].round().astype(int)
    agg["n_total"] = agg["n_total"].round().astype(int)
    agg["n_correct"] = np.minimum(agg["n_correct"], agg["n_total"])
    agg["acc"] = agg["n_correct"] / agg["n_total"]
    agg["rounding_err"] = (agg["k_float"] - agg["n_correct"]).abs()
    agg["in_fit"] = agg["n_total"] >= FIT_MIN_N
    return agg.sort_values(["task_name", "release_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# fitting helpers
# --------------------------------------------------------------------------
def cap_counts(d: pd.DataFrame) -> pd.DataFrame:
    """Cap effective N per cell (see N_CAP comment); rescale successes."""
    d = d.copy()
    over = d["n_total"] > N_CAP
    scale = N_CAP / d.loc[over, "n_total"]
    d.loc[over, "n_correct"] = (d.loc[over, "n_correct"] * scale).round().astype(int)
    d.loc[over, "n_total"] = N_CAP
    return d


def fit_trend(
    sub: pd.DataFrame,
    tag: str,
    quadratic: bool = False,
    samples: int = 2000,
    warmup: int = 1500,
    seed: int = SEED,
    prior_slope_scale: float = 2.0,
    no_slope: bool = False,
):
    """Fit release_trend_binomial to a task subset (one row per model)."""
    d = sub[["canonical", "t", "n_correct", "n_total"]].copy()
    d = d.rename(columns={"canonical": "model"})
    d = d[d["n_total"] >= FIT_MIN_N]  # see FIT_MIN_N comment
    d = cap_counts(d)
    d["t_c"] = d["t"] - d["t"].mean()  # centre within task
    t_mean = float(d["t"].mean())
    d = d.drop(columns=["t"]).rename(columns={"t_c": "t"})
    assert not d["model"].duplicated().any()
    # empirical logit + its standard error (Haldane-Anscombe 0.5 correction)
    k, n = d["n_correct"], d["n_total"]
    d["y"] = np.log((k + 0.5) / (n - k + 0.5))
    d["se"] = np.sqrt(1.0 / (k + 0.5) + 1.0 / (n - k + 0.5))
    state = make_state(d)
    state = run_processors(
        state,
        extract_features(
            continuous_features=["t", "se"], categorical_features=["model"]
        ),
        extract_observed_feature(feature_name="y"),
    )
    mas = fit(
        state,
        release_trend_logit_normal(
            quadratic=quadratic,
            prior_slope_scale=prior_slope_scale,
            no_slope=no_slope,
        ),
        tag=tag,
        samples=samples,
        warmup=warmup,
        chains=4,
        seed=seed,
        target_accept=0.99,
        max_tree_depth=12,
    )
    return mas, t_mean


def summarise_fit(mas, var: str = "beta") -> dict:
    post = mas.inference_data.posterior[var].values.ravel()
    lo, hi = az.hdi(post, hdi_prob=HDI)
    return {
        f"{var}_mean": float(post.mean()),
        f"{var}_hdi3": float(lo),
        f"{var}_hdi97": float(hi),
        f"P({var}>0)": float((post > 0).mean()),
    }


# --------------------------------------------------------------------------
# synthetic recovery
# --------------------------------------------------------------------------
def synthetic_recovery(agg: pd.DataFrame, n_reps: int = 10) -> pd.DataFrame:
    """Simulate from the fitted design (real dates/Ns) and check recovery."""
    scenarios = {
        # (task design to copy, true alpha, true beta, true sigma)
        "cybench_like": ("cybench", -1.0, 1.5, 0.5),
        "boolq_like": ("boolq_preference", 1.0, 0.8, 0.4),
        "swe_like_narrow": ("swe_bench", -1.0, 1.5, 0.4),
    }
    rows = []
    for name, (task, a_true, b_true, s_true) in scenarios.items():
        design = agg[agg["task_name"] == task][["canonical", "t", "n_total"]].copy()
        t_c = design["t"] - design["t"].mean()
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        for rep in range(n_reps):
            u = rng.normal(0.0, s_true, size=len(design))
            theta_true = a_true + b_true * t_c.values + u
            p = 1.0 / (1.0 + np.exp(-theta_true))
            k = rng.binomial(design["n_total"].values, p)
            # oracle: OLS slope of the realised latent logits (fitted cells) —
            # the best any estimator could do given this u-draw
            mask = design["n_total"].values >= FIT_MIN_N
            oracle = float(np.polyfit(t_c.values[mask], theta_true[mask], 1)[0])
            sim = design.copy()
            sim["n_correct"] = k
            sim["task_name"] = task
            mas, _ = fit_trend(
                sim,
                tag=f"synth_{name}_{rep}",
                samples=1000,
                seed=SEED + rep,
            )
            diag_summ = az.summary(mas.inference_data, round_to=5)
            div = int(mas.inference_data.sample_stats["diverging"].values.sum())
            s = summarise_fit(mas, "beta")
            sig = summarise_fit(mas, "sigma_model")
            rows.append(
                {
                    "scenario": name,
                    "rep": rep,
                    "J": int((design["n_total"] >= FIT_MIN_N).sum()),
                    "true_beta": b_true,
                    "beta_mean": s["beta_mean"],
                    "beta_hdi3": s["beta_hdi3"],
                    "beta_hdi97": s["beta_hdi97"],
                    "beta_covered": s["beta_hdi3"] <= b_true <= s["beta_hdi97"],
                    "oracle_ols_slope": oracle,
                    "beta_minus_oracle": s["beta_mean"] - oracle,
                    "true_sigma": s_true,
                    "sigma_mean": sig["sigma_model_mean"],
                    "max_r_hat": float(diag_summ["r_hat"].max()),
                    "min_ess_bulk": float(diag_summ["ess_bulk"].min()),
                    "divergences": div,
                }
            )
            print(
                f"[synth] {name} rep {rep}: beta {s['beta_mean']:.2f} "
                f"[{s['beta_hdi3']:.2f},{s['beta_hdi97']:.2f}] "
                f"(true {b_true}) cov={rows[-1]['beta_covered']}"
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# frontier
# --------------------------------------------------------------------------
def frontier_subset(sub: pd.DataFrame) -> pd.DataFrame:
    """Models that raised the running-best accuracy at their release date.

    NOTE (REVIEW_ROUND_1): this subset is monotonically increasing by
    construction — trends fitted to it are DESCRIPTIVE record-path
    summaries, never evidence of a trend. Selection-aware inference lives in
    frontier_selection_tests / outputs/frontier_null.csv.
    """
    eligible = sub[sub["n_total"] >= FRONTIER_MIN_N].sort_values("release_date")
    best = -np.inf
    keep = []
    for _, row in eligible.iterrows():
        if row["acc"] > best:
            keep.append(row["canonical"])
            best = row["acc"]
    return sub[sub["canonical"].isin(keep)].copy()


N_NULL_REPS = 4000


def frontier_selection_tests(agg: pd.DataFrame, fits: dict) -> pd.DataFrame:
    """Selection-aware frontier statistics (fixes REVIEW_ROUND_1 blocker).

    Per task with an estimable frontier: (i) record-count test against the
    exact exchangeable null; (ii) record-path OLS slope calibrated against a
    flat-null posterior predictive (no-slope refit of the same design);
    (iii) record-path quadratic coefficient calibrated against the fitted
    LINEAR model's posterior predictive (the saturation null). See
    selection.py for the machinery and rationale.
    """
    rows = []
    null_draws: dict = {}
    for task in ["cybench", "gdm_intercode_ctf"]:
        sub = cap_counts(agg[(agg["task_name"] == task) & agg["in_fit"]])
        sub = sub.sort_values("release_date").reset_index(drop=True)
        sub["n_cap"] = sub["n_total"]
        sub["k_cap"] = sub["n_correct"]
        sub["t"] = sub["t"] - sub["t"].mean()  # centre as in fit_trend

        # flat-null (alpha, sigma): posterior of a no-slope refit
        nmas, _ = fit_trend(
            agg[agg["task_name"] == task], tag=f"null_{task}", no_slope=True
        )
        ndiag = save_outputs(nmas, OUT, f"null_{task}")
        npost = nmas.inference_data.posterior
        a0 = npost["alpha"].values.ravel()
        s0 = npost["sigma_model"].values.ravel()

        # linear ppc (alpha, beta, sigma): posterior of the real mean-trend fit
        lpost = fits[task][0].inference_data.posterior
        a1 = lpost["alpha"].values.ravel()
        b1 = lpost["beta"].values.ravel()
        s1 = lpost["sigma_model"].values.ravel()

        obs = observed_record_stats(sub)
        flat = simulate_record_null(sub, a0, None, s0, N_NULL_REPS, seed=SEED)
        linear = simulate_record_null(
            sub, a1, b1, s1, N_NULL_REPS, seed=SEED + 1
        )
        row = summarise_selection_test(task, obs, flat, linear, j=len(sub))
        row["null_fit_max_r_hat"] = ndiag["max_r_hat"]
        row["null_fit_divergences"] = ndiag["n_divergences"]
        rows.append(row)
        null_draws[task] = (obs, flat, linear)
        print(f"[selection] {row}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "frontier_null.csv", index=False)
    plot_frontier_null(null_draws, out, OUT / "frontier_null.png")
    return out


def plot_frontier_null(null_draws: dict, summ: pd.DataFrame, path: Path) -> None:
    """Null distributions of record-path statistics vs observed values."""
    tasks = list(null_draws)
    fig, axes = plt.subplots(
        len(tasks), 3, figsize=(13.5, 3.9 * len(tasks)), facecolor=C_SURFACE
    )
    for r, task in enumerate(tasks):
        obs, flat, linear = null_draws[task]
        srow = summ[summ["task"] == task].iloc[0]
        panels = [
            (
                "records",
                flat["n_records"],
                obs["n_records"],
                f"# records (flat null)\nP(K >= {int(srow['obs_records_min_over_tie_orders'])}) "
                f"exact = {srow['p_records_exact_exchangeable']:.4f}",
                dict(bins=np.arange(0.5, flat["n_records"].max() + 1.5)),
            ),
            (
                "slope",
                flat["slope"].dropna(),
                obs["slope"],
                "record-path OLS slope under flat null (logits/yr)\n"
                f"P(null >= obs) = {srow['p_slope_ge_obs_flat_null']:.3f}",
                dict(bins=40),
            ),
            (
                "quad",
                linear["quad"].dropna(),
                obs["quad"],
                "record-path quad coeff under LINEAR ppc (logits/yr$^2$)\n"
                f"P(null <= obs) = {srow['p_quad_le_obs_linear_ppc']:.3f}",
                dict(bins=40),
            ),
        ]
        for c, (name, null_vals, obs_vals, title, hkw) in enumerate(panels):
            ax = axes[r, c]
            ax.set_facecolor(C_SURFACE)
            ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            ax.hist(
                null_vals, color=C_MEAN, alpha=0.75, zorder=2,
                label="null / ppc simulation", **hkw,
            )
            for v_i, v in enumerate(np.unique(np.round(obs_vals, 6))):
                ax.axvline(
                    v, color=C_FRONTIER, lw=2, zorder=3,
                    label="observed (tie orders)" if v_i == 0 else None,
                )
            ax.set_title(f"{TASK_LABELS[task]}\n{title}", fontsize=9.5,
                         color=C_TEXT, loc="left")
            ax.set_xlabel(name, fontsize=9, color=C_TEXT2)
            ax.set_ylabel("simulated reps", fontsize=9, color=C_TEXT2)
            ax.tick_params(labelsize=8, colors=C_TEXT2)
            ax.legend(fontsize=7.5)
    fig.suptitle(
        "E4 — record ('frontier') path statistics vs selection-aware nulls: "
        "the record subset rises by construction, so only these calibrated "
        "tests carry evidence",
        fontsize=12, color=C_TEXT, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _trend_band(idata, t_mean: float, t_grid: np.ndarray, quadratic: bool = False):
    post = idata.posterior
    a = post["alpha"].values.ravel()[:, None]
    b = post["beta"].values.ravel()[:, None]
    tc = (t_grid - t_mean)[None, :]
    logit = a + b * tc
    if quadratic:
        b2 = post["beta2"].values.ravel()[:, None]
        logit = logit + b2 * tc**2
    p = 1.0 / (1.0 + np.exp(-logit))
    lo, hi = np.quantile(p, [(1 - HDI) / 2, 1 - (1 - HDI) / 2], axis=0)
    return p.mean(axis=0), lo, hi


def plot_growth(agg, fits, frontier_fits, frontier_sets, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), facecolor=C_SURFACE)
    for ax, task in zip(axes.ravel(), TASKS):
        sub = agg[agg["task_name"] == task]
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(C_GRID)

        fit_sub = sub[sub["in_fit"]]
        t_grid = np.linspace(fit_sub["t"].min(), fit_sub["t"].max(), 120)
        date_grid = EPOCH.tz_localize(None) + pd.to_timedelta(t_grid * 365.25, "D")

        # mean trend + band
        mas, t_mean = fits[task]
        mean, lo, hi = _trend_band(mas.inference_data, t_mean, t_grid)
        ax.fill_between(
            date_grid, lo, hi, color=C_MEAN, alpha=0.16, lw=0, zorder=1,
            label=f"mean trend, {int(HDI*100)}% band",
        )
        ax.plot(date_grid, mean, color=C_MEAN, lw=2, zorder=3, label="mean trend")

        # frontier trend
        if task in frontier_fits:
            fmas, ft_mean = frontier_fits[task]
            fsub = frontier_sets[task]
            fmean, _, _ = _trend_band(fmas.inference_data, ft_mean, t_grid)
            ax.plot(
                date_grid, fmean, color=C_FRONTIER, lw=2, ls="--", zorder=3,
                label="record path (descriptive; rises by construction)",
            )
            ax.scatter(
                fsub["release_date"].dt.tz_localize(None), fsub["acc"],
                s=130, facecolors="none", edgecolors=C_FRONTIER, lw=1.6, zorder=4,
                label="record-setting models",
            )

        # model points, area ~ sample count
        size = 18 + 22 * np.log10(fit_sub["n_total"].clip(lower=1))
        ax.scatter(
            fit_sub["release_date"].dt.tz_localize(None), fit_sub["acc"], s=size,
            color=C_MEAN, edgecolors="white", lw=0.8, zorder=5,
            label="models (area ~ log N samples)",
        )
        excl = sub[~sub["in_fit"]]
        if len(excl):
            ax.scatter(
                excl["release_date"].dt.tz_localize(None), excl["acc"], s=26,
                facecolors="none", edgecolors=C_TEXT2, lw=1.0, zorder=4,
                label="excluded from fit (N < 20)",
            )

        # direct labels: frontier + first/last released, max ~6
        to_label = set(sub.iloc[[0, -1]]["canonical"])
        if task in frontier_sets:
            to_label |= set(frontier_sets[task]["canonical"])
        lab = sub[sub["canonical"].isin(to_label)].nlargest(6, "n_total")
        for _, r in lab.iterrows():
            ax.annotate(
                r["canonical"],
                (r["release_date"].tz_localize(None), r["acc"]),
                textcoords="offset points", xytext=(6, 7),
                fontsize=7.5, color=C_TEXT2,
            )

        s = summarise_fit(mas, "beta")
        ax.set_title(
            f"{TASK_LABELS[task]}\n"
            f"velocity {s['beta_mean']:+.2f} logits/yr "
            f"[{s['beta_hdi3']:+.2f}, {s['beta_hdi97']:+.2f}] {int(HDI*100)}% HDI",
            fontsize=10.5, color=C_TEXT, loc="left",
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Accuracy (pooled over runs)", fontsize=9, color=C_TEXT2)
        ax.set_xlabel("Model release date", fontsize=9, color=C_TEXT2)
        ax.tick_params(labelsize=8, colors=C_TEXT2)
        ax.legend(fontsize=7.5, loc="best", framealpha=0.9)

    fig.suptitle(
        "E4 — capability trend vs model release date (AISI public-benchmark runs; "
        "observational)", fontsize=13, color=C_TEXT, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def plot_recovery(rec: pd.DataFrame, path: Path) -> None:
    scen = list(rec["scenario"].unique())
    fig, axes = plt.subplots(1, len(scen), figsize=(4.2 * len(scen), 3.6),
                             facecolor=C_SURFACE, sharey=False)
    for ax, name in zip(np.atleast_1d(axes), scen):
        r = rec[rec["scenario"] == name]
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.axhline(r["true_beta"].iloc[0], color=C_FRONTIER, lw=2,
                   label=f"true beta = {r['true_beta'].iloc[0]}")
        ax.errorbar(
            r["rep"], r["beta_mean"],
            yerr=[r["beta_mean"] - r["beta_hdi3"], r["beta_hdi97"] - r["beta_mean"]],
            fmt="o", color=C_MEAN, ms=5, capsize=3, lw=1.4,
            label="posterior mean, 94% HDI",
        )
        cov = r["beta_covered"].mean()
        ax.set_title(f"{name} (J={r['J'].iloc[0]})\nHDI coverage {cov:.0%}",
                     fontsize=10, color=C_TEXT, loc="left")
        ax.set_xlabel("replicate", fontsize=9, color=C_TEXT2)
        ax.set_ylabel("slope beta (logits/yr)", fontsize=9, color=C_TEXT2)
        ax.tick_params(labelsize=8, colors=C_TEXT2)
        ax.legend(fontsize=7.5)
    fig.suptitle("E4 — synthetic recovery of the release-date slope", fontsize=12,
                 color=C_TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. model dates table + persisted data-verification numbers
    checks = data_checks()
    print(f"[data_checks] {checks['dedupe_distinct_eval_ids']} dup eval_ids, "
          f"copies {checks['dedupe_copies_per_key_min']}-"
          f"{checks['dedupe_copies_per_key_max']}, boolq split diff "
          f"{checks['boolq_max_abs_train_validation_diff']:.4f}")
    runs, ledger = build_clean_runs()
    raw_models = sorted(load_evals()["model"].unique())
    dates_rows = []
    for m in raw_models:
        reason = exclusion_reason(m)
        if reason is None:
            c = CANONICAL[m]
            d, src = RELEASE_DATES[c]
            dates_rows.append(
                {"model_raw": m, "canonical": c, "release_date": d, "source": src}
            )
        else:
            dates_rows.append(
                {"model_raw": m, "canonical": "EXCLUDED", "release_date": "",
                 "source": reason}
            )
    pd.DataFrame(dates_rows).to_csv(OUT / "model_dates.csv", index=False)
    ledger.to_csv(OUT / "exclusions.csv", index=False)

    # 2. aggregate
    agg = aggregate_models(runs)
    agg.to_csv(OUT / "model_task_agg.csv", index=False)
    print(agg[["task_name", "canonical", "release_date", "n_runs", "n_total", "acc"]]
          .to_string())
    assert not agg.duplicated(["task_name", "canonical"]).any()
    assert (agg["rounding_err"] < 30).all(), "count reconstruction drifted"

    # 3. synthetic recovery
    rec = synthetic_recovery(agg, n_reps=10)
    rec.to_csv(OUT / "synthetic_recovery.csv", index=False)
    plot_recovery(rec, OUT / "recovery.png")
    print(rec.groupby("scenario")[["beta_covered", "max_r_hat", "divergences"]]
          .agg({"beta_covered": "mean", "max_r_hat": "max", "divergences": "sum"}))

    # 4. real fits: mean trend per task
    fits: dict = {}
    velocity_rows = []
    for task in TASKS:
        sub = agg[agg["task_name"] == task]
        mas, t_mean = fit_trend(sub, tag=f"trend_{task}")
        diag = save_outputs(mas, OUT, f"trend_{task}")
        fits[task] = (mas, t_mean)
        s = summarise_fit(mas, "beta")
        sig = summarise_fit(mas, "sigma_model")
        # trend-line accuracy at first/last fitted release date (effect size)
        fit_sub = sub[sub["in_fit"]]
        t_lo, t_hi = fit_sub["t"].min(), fit_sub["t"].max()
        mean_line, lo_line, hi_line = _trend_band(
            mas.inference_data, t_mean, np.array([t_lo, t_hi])
        )
        velocity_rows.append(
            {
                "task": task, "n_models": int(sub["in_fit"].sum()),
                "n_models_excluded_smallN": int((~sub["in_fit"]).sum()),
                "release_span_yr": round(t_hi - t_lo, 3),
                **s,
                "sigma_model_mean": sig["sigma_model_mean"],
                "sigma_model_hdi3": sig["sigma_model_hdi3"],
                "sigma_model_hdi97": sig["sigma_model_hdi97"],
                "trend_acc_at_first_release": float(mean_line[0]),
                "trend_acc_at_last_release": float(mean_line[1]),
                **{f"diag_{k}": v for k, v in diag.items() if k != "model"},
            }
        )
        print(f"[fit] {task}: {velocity_rows[-1]}")

    pd.DataFrame(velocity_rows).to_csv(OUT / "velocity_summary.csv", index=False)

    # 4b. prior sensitivity: slope prior N(0,2) -> N(0,4)
    sens_rows = []
    for task in TASKS:
        sub = agg[agg["task_name"] == task]
        smas, _ = fit_trend(sub, tag=f"sens_{task}", prior_slope_scale=4.0)
        s2 = summarise_fit(smas, "beta")
        sens_rows.append(
            {
                "task": task,
                "prior_slope_scale": 4.0,
                **s2,
                "beta_mean_base_prior": next(
                    r["beta_mean"] for r in velocity_rows if r["task"] == task
                ),
            }
        )
        print(f"[sens] {task}: {s2}")
    pd.DataFrame(sens_rows).to_csv(OUT / "velocity_prior_sensitivity.csv", index=False)

    # 5. record-path ("frontier") fits — DESCRIPTIVE ONLY (REVIEW_ROUND_1):
    # the subset rises by construction, so P(beta>0)/P(beta2<0) are dropped
    # from the summary; selection-aware inference is step 5a.
    SELECTION_NOTE = (
        "descriptive record path: subset is monotone by construction, "
        "P-values would be ~1 under any truth incl. no trend; "
        "see frontier_null.csv for selection-aware tests"
    )
    frontier_fits: dict = {}
    frontier_sets: dict = {}
    frontier_rows = []
    for task in TASKS:
        sub = agg[agg["task_name"] == task]
        fsub = frontier_subset(sub)
        frontier_sets[task] = fsub
        row = {"task": task, "n_frontier": len(fsub),
               "frontier_models": ";".join(fsub["canonical"]),
               "selection_note": SELECTION_NOTE}
        if len(fsub) >= 4:
            fmas, ft_mean = fit_trend(fsub, tag=f"frontier_{task}")
            fdiag = save_outputs(fmas, OUT, f"frontier_{task}")
            frontier_fits[task] = (fmas, ft_mean)
            s = summarise_fit(fmas, "beta")
            s.pop("P(beta>0)")  # vacuous under record selection
            row.update({f"descr_{k}": v for k, v in s.items()})
            row.update({f"diag_{k}": v for k, v in fdiag.items() if k != "model"})
        if len(fsub) >= 5:
            qmas, _ = fit_trend(fsub, tag=f"frontier_quad_{task}", quadratic=True)
            qdiag = save_outputs(qmas, OUT, f"frontier_quad_{task}")
            qpost = qmas.inference_data.posterior["beta2"].values.ravel()
            qlo, qhi = az.hdi(qpost, hdi_prob=HDI)
            row.update(
                {
                    "descr_beta2_mean": float(qpost.mean()),
                    "descr_beta2_hdi3": float(qlo),
                    "descr_beta2_hdi97": float(qhi),
                    "quad_max_r_hat": qdiag["max_r_hat"],
                    "quad_divergences": qdiag["n_divergences"],
                }
            )
        frontier_rows.append(row)
        print(f"[frontier] {row}")
    pd.DataFrame(frontier_rows).to_csv(OUT / "frontier_summary.csv", index=False)

    # 5a. selection-aware frontier inference (fixes REVIEW_ROUND_1 blocker)
    selection_summary = frontier_selection_tests(agg, fits)

    # 5b. cross-task velocity comparison (cyber hard vs cyber easy), using the
    # same posterior draws that are saved to the .idata.nc files
    def _beta(mas) -> np.ndarray:
        return mas.inference_data.posterior["beta"].values.ravel()

    cross_rows = []
    b_cy, b_ic = _beta(fits["cybench"][0]), _beta(fits["gdm_intercode_ctf"][0])
    for label, a, b in [
        ("mean_trend", b_cy, b_ic),
        (
            "frontier_trend",
            _beta(frontier_fits["cybench"][0]),
            _beta(frontier_fits["gdm_intercode_ctf"][0]),
        ),
    ]:
        diff = a - b
        lo, hi = az.hdi(diff, hdi_prob=HDI)
        descriptive = label == "frontier_trend"  # record-selected: no P quoted
        cross_rows.append(
            {
                "comparison": f"{label}: beta_cybench - beta_gdm_intercode_ctf",
                # record-path betas are selection-biased, so their difference
                # carries no calibrated probability (REVIEW_ROUND_1)
                "P(cybench_faster)": (
                    None if descriptive else float((diff > 0).mean())
                ),
                "diff_mean": float(diff.mean()),
                "diff_hdi3": float(lo),
                "diff_hdi97": float(hi),
                "note": SELECTION_NOTE if descriptive else "inferential",
            }
        )
    pd.DataFrame(cross_rows).to_csv(OUT / "cross_task.csv", index=False)
    print(cross_rows)

    # 6. plot
    plot_growth(agg, fits, frontier_fits, frontier_sets, OUT / "growth_curves.png")

    # 7. machine-readable headline
    headline = {
        "velocity": velocity_rows,
        "frontier_descriptive_record_paths": frontier_rows,
        "frontier_selection_tests": selection_summary.to_dict("records"),
        "data_checks": checks,
        "synthetic_coverage": rec.groupby("scenario")["beta_covered"].mean().to_dict(),
    }
    (OUT / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()
