"""Side check (not part of run.py): is sigma_run systematically
under-recovered in the claude-3-7 cell design? 3 seeds."""

import numpy as np
import pandas as pd
from scipy.special import expit

from shared.bridge import binomial_agg
from experiments.e1_cybench.fitting import draws, fit_crossed, hdi
from experiments.e1_cybench.prep import hard_focal, load_clean


def main() -> None:
    df, _ = load_clean(verbose=False)
    hard = hard_focal(df)
    design = binomial_agg(
        hard[hard["model"] == "anthropic/claude-3-7-sonnet-20250219"],
        by=["challenge", "run"])[["challenge", "run", "n_total"]]
    print("cells:", len(design), "runs:", design['run'].nunique())
    mu, sigma_c, sigma_r = -1.0, 2.5, 0.4
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        chal = sorted(design["challenge"].unique())
        runs = sorted(design["run"].unique())
        a = rng.normal(0, sigma_c, len(chal))
        b = rng.normal(0, sigma_r, len(runs))
        cells = design.copy()
        p = expit(mu + cells["challenge"].map(dict(zip(chal, a))).values
                  + cells["run"].map(dict(zip(runs, b))).values)
        cells["n_correct"] = rng.binomial(cells["n_total"].values, p)
        mas, _ = fit_crossed(cells, tag=f"sr_check_{seed}",
                             samples=800, warmup=800)
        sr = draws(mas, "sigma_run")
        lo, hi = hdi(sr)
        print(f"seed {seed}: realized sd(b)={b.std(ddof=1):.3f} "
              f"post mean={sr.mean():.3f} hdi=[{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
