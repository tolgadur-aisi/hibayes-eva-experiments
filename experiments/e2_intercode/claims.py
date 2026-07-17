"""Posterior probabilities / contrasts for the headline claims (from saved idata).

    uv run python -m experiments.e2_intercode.claims
"""

import json

import arviz as az
import numpy as np
from scipy.special import expit

from experiments.e2_intercode.common import OUT, flat, hdi94


def main():
    claims = {}

    h = az.from_netcdf(OUT / "d1_hetrun.idata.nc")
    models = [str(v) for v in h.posterior.coords["model"].values]
    sr = flat(h, "sigma_run")  # (draws, M)
    gi = models.index("gemma/gemma-3-27b-it")
    for m, name in enumerate(models):
        if m == gi:
            continue
        claims[f"P(sigma_run[{name}] > sigma_run[gemma])"] = float((sr[:, m] > sr[:, gi]).mean())
    claims["P(sigma_run[gemma] < 0.2)"] = float((sr[:, gi] < 0.2).mean())

    d1 = az.from_netcdf(OUT / "d1_primary.idata.nc")
    s = flat(d1, "sigma_run")
    claims["d1 pooled P(sigma_run > 0.3)"] = float((s > 0.3).mean())

    d2 = az.from_netcdf(OUT / "d2_frontier_days.idata.nc")
    s2 = flat(d2, "sigma_run")
    claims["d2 P(sigma_day > 0.2)"] = float((s2 > 0.2).mean())

    # gpt-5 day contrast on the accuracy scale (14 fixed items)
    runs = [str(v) for v in d2.posterior.coords["run"].values]
    m2 = [str(v) for v in d2.posterior.coords["model"].values]
    mu, a, b, g, u = (flat(d2, v) for v in ["mu", "model_effects", "item_effects", "inter_effects", "run_effects"])
    mi = m2.index("openai/gpt-5")
    lo_day, hi_day = "openai/gpt-5|2025-09-17", "openai/gpt-5|2025-09-20"
    logit = mu[:, None] + a[:, mi, None] + b + g[:, mi]  # (draws, I)
    acc = {d: expit(logit + u[:, runs.index(d), None]).mean(axis=1) for d in (lo_day, hi_day)}
    diff = acc[hi_day] - acc[lo_day]
    claims["gpt-5 accuracy Sep20 minus Sep17 (pp)"] = {
        "mean": float(diff.mean() * 100),
        "hdi94": [x * 100 for x in hdi94(diff)],
        "P(>0)": float((diff > 0).mean()),
    }

    (OUT / "claims.json").write_text(json.dumps(claims, indent=2))
    print(json.dumps(claims, indent=2))


if __name__ == "__main__":
    main()
