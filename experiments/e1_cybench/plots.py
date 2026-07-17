"""Plots for E1 (matplotlib, Agg). Palette/marks follow the dataviz method:
validated reference palette, thin marks, hairline solid grid, text in ink."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"   # categorical slot 1
GREEN = "#008300"  # categorical slot 2

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 9,
})

SHORT = {
    "anthropic/claude-3-7-sonnet-20250219": "claude-3-7-sonnet",
    "anthropic/claude-opus-4-20250514": "claude-opus-4",
    "gemma/gemma-3-27b-it": "gemma-3-27b",
    "mistralazure/Mistral-Large-2411": "mistral-large-2411",
    "openai/gpt-4o-2024-08-06": "gpt-4o",
    "openai/gpt-5": "gpt-5",
    "openai/o1": "o1",
    "openai/o1-2024-12-17": "o1-2024-12-17",
    "openai/o3": "o3",
    "openai/o3-mini": "o3-mini",
}


def plot_difficulty_spectrum(q1: pd.DataFrame, path) -> None:
    """Small multiples: sorted per-challenge solve probabilities per model."""
    models = (q1.groupby("model")["p_mean"].mean()
              .sort_values(ascending=False).index.tolist())
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharey=True)
    for ax, m in zip(axes.flat, models):
        d = q1[q1["model"] == m].sort_values("p_mean").reset_index(drop=True)
        x = np.arange(len(d))
        ax.vlines(x, d["p_hdi_3%"], d["p_hdi_97%"], color=BLUE, alpha=0.45,
                  linewidth=1.4)
        ax.plot(x, d["p_mean"], "o", color=BLUE, markersize=3.4,
                markeredgecolor=SURFACE, markeredgewidth=0.7)
        ax.set_title(SHORT[m], fontsize=9, color=INK)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color=BASE, linewidth=1.0)
        n_coin = int(((d["p_mean"] > 0.2) & (d["p_mean"] < 0.8)).sum())
        n_zero = int((d["p_mean"] < 0.05).sum())
        ax.text(0.03, 0.95, f"{n_zero} @ p<.05 | {n_coin} coin-flip",
                transform=ax.transAxes, fontsize=7.2, color=INK2, va="top")
    for ax in axes[:, 0]:
        ax.set_ylabel("posterior solve probability")
    for ax in axes[1, :]:
        ax.set_xlabel("challenge (sorted by difficulty)")
    fig.suptitle(
        "cybench (hard variant): per-challenge solve probability, "
        "posterior mean and 94% HDI (hierarchical binomial per model)",
        fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_design_curve(curves: pd.DataFrame, summary: dict, path) -> None:
    a = curves[curves["curve"] == "A_superpopulation"]
    b = curves[curves["curve"] == "B_fixed_benchmark_40"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(a["k"], a["se_q3"], a["se_q97"], color=BLUE, alpha=0.10,
                    linewidth=0)
    ax.plot(a["k"], a["se_median"], color=BLUE, linewidth=2,
            label="unlimited challenge pool (superpopulation)")
    ax.fill_between(b["k"], b["se_q3"], b["se_q97"], color=GREEN, alpha=0.10,
                    linewidth=0)
    ax.plot(b["k"], b["se_median"], color=GREEN, linewidth=2,
            label="fixed 40-challenge benchmark (c = 400/k)")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xlabel("epochs per challenge k  (challenges c = 400/k)")
    ax.set_ylabel("SE of benchmark score (accuracy scale)")
    ax.set_title(
        "Fixed budget of 400 attempts: fewer epochs on more challenges wins\n"
        f"(claude-3-7-sonnet posterior; observed-scale ICC "
        f"$\\rho$ = {summary['rho_obs_median']:.2f})",
        fontsize=10, color=INK)
    k10 = a.loc[a["k"] == 10, "se_median"].iloc[0]
    k100 = a.loc[a["k"] == 100, "se_median"].iloc[0]
    ax.annotate(f"k=10, c=40: SE {k10:.3f}", (10, k10),
                textcoords="offset points", xytext=(-8, 12),
                fontsize=8, color=INK2, ha="right")
    ax.annotate(f"k=100, c=4: SE {k100:.3f}", (100, k100),
                textcoords="offset points", xytext=(-14, 22),
                fontsize=8, color=INK2, ha="right")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_forest(ab: pd.DataFrame, path) -> None:
    """Posterior benchmark-mean accuracy vs naive pooled accuracy."""
    d = ab.sort_values("bench_acc_mean").reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.hlines(y + 0.16, d["naive_lo"], d["naive_hi"], color=MUTED,
              linewidth=1.6)
    ax.plot(d["naive_acc"], y + 0.16, "o", color=MUTED, markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1,
            label="naive pooled accuracy ± Wald 94% CI")
    ax.hlines(y - 0.16, d["bench_acc_hdi_3%"], d["bench_acc_hdi_97%"],
              color=BLUE, linewidth=2.2)
    ax.plot(d["bench_acc_mean"], y - 0.16, "o", color=BLUE, markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1,
            label="hierarchical posterior (benchmark-mean) 94% HDI")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in d["model"]], color=INK)
    ax.set_xlabel("accuracy on cybench hard variant (40 challenges)")
    ax.set_title(
        "Model comparison with honest uncertainty: challenge + run random "
        "effects\nvs naive pooled accuracy (attempt-weighted)",
        fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, None)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
