"""Raw-data look: every individual trial, before any statistical model.

Three jittered scatter subplots from the trial-level snapshot the pipeline
saves just before aggregation (snapshot_trials in config.yaml):

    score vs benchmark | score vs model | score vs scaffold (harness)

One dot per trial, no aggregation. Scores are binary, so dots are jittered
in both axes to show density; alpha adapts to the trial count.

    uv run python -m modeling.plot_raw
    uv run python -m modeling.plot_raw --trials modeling/synth/.output/trials.parquet
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOT_COLOR = "#4269d0"
INK = "#3d4551"
MUTED = "#767d86"
JITTER_X = 0.32
JITTER_Y = 0.06
MAX_LABEL_CHARS = 38


def jitter_strip(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    rng: np.random.Generator,
    alpha: float,
) -> None:
    levels = sorted(df[column].astype(str).unique())
    positions = {level: i for i, level in enumerate(levels)}
    x = df[column].astype(str).map(positions).to_numpy(dtype=float)
    x += rng.uniform(-JITTER_X, JITTER_X, size=len(x))
    y = df["score"].to_numpy(dtype=float) + rng.uniform(-JITTER_Y, JITTER_Y, size=len(df))

    ax.scatter(x, y, s=4, c=DOT_COLOR, alpha=alpha, linewidths=0, rasterized=True)
    labels = [
        lvl if len(lvl) <= MAX_LABEL_CHARS else lvl[: MAX_LABEL_CHARS - 1] + "…"
        for lvl in levels
    ]
    ax.set_xticks(range(len(levels)), labels, rotation=45, ha="right", fontsize=7, color=INK)
    ax.set_xlabel(column, color=INK)
    ax.set_yticks([0, 1], ["0 (fail)", "1 (pass)"], color=INK)
    ax.set_ylim(-0.25, 1.25)
    ax.grid(axis="y", color="#e3e5e8", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", default="modeling/.output/trials.parquet")
    parser.add_argument("--out", default=None, help="Defaults to raw_scores.png next to the trials file")
    args = parser.parse_args()

    trials_path = Path(args.trials)
    df = pd.read_parquet(trials_path)
    out = Path(args.out) if args.out else trials_path.parent / "raw_scores.png"

    rng = np.random.default_rng(0)
    # keep dense strips readable: more trials -> fainter dots
    alpha = float(np.clip(30_000 / max(len(df), 1), 0.02, 0.5))

    # scaffold (harness) usually has the most levels; give it more width
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), sharey=True, width_ratios=[1, 1.4, 1.6]
    )
    for ax, column in zip(axes, ["benchmark", "model", "scaffold"]):
        jitter_strip(ax, df, column, rng, alpha)
    axes[0].set_ylabel("score (jittered)", color=INK)
    axes[2].set_xlabel("scaffold (harness)", color=INK)
    fig.suptitle(
        f"Raw trial outcomes before modeling — one dot per trial, n={len(df):,}",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[plot_raw] {len(df):,} trials -> {out}")


if __name__ == "__main__":
    main()
