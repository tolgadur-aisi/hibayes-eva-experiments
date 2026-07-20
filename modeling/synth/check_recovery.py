"""Compare fitted posteriors against the synthetic ground truth.

Reads the inference data saved by the hibayes pipeline and reports, for every
generating parameter, the posterior mean, 94% HDI, and whether the truth is
inside it. Fails (exit 1) if coverage or item-effect correlation is poor, so
run_synth.py doubles as a regression test for the whole pipeline.
"""

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO / "modeling" / "synth" / ".output"
TRUTH_PATH = REPO / "data" / "synth" / "truth.json"

COVERAGE_THRESHOLD = 0.85
ITEM_CORR_THRESHOLD = 0.9


def load_idata() -> az.InferenceData:
    candidates = sorted(OUTPUT_DIR.glob("models/*/inference_data.nc"))
    if not candidates:
        raise FileNotFoundError(f"no inference_data.nc under {OUTPUT_DIR}/models/")
    if len(candidates) > 1:
        print(f"[warn] multiple fitted models, using {candidates[0]}")
    return az.from_netcdf(candidates[0])


def hdi_bounds(idata: az.InferenceData, var: str) -> tuple[np.ndarray, np.ndarray]:
    hdi = az.hdi(idata, var_names=[var], hdi_prob=0.94)[var].values
    hdi = np.atleast_2d(hdi)
    return hdi[..., 0].ravel(), hdi[..., 1].ravel()


def check_vector(
    idata: az.InferenceData, var: str, coord: str, truth_by_name: dict
) -> list[dict]:
    post = idata.posterior[var]
    names = [str(n) for n in post[coord].values]
    means = post.mean(dim=("chain", "draw")).values.ravel()
    lo, hi = hdi_bounds(idata, var)
    rows = []
    for name, mean, l, h in zip(names, means, lo, hi):
        if name not in truth_by_name:
            raise KeyError(
                f"{var}: fitted level {name!r} not in truth "
                f"(have {sorted(truth_by_name)}) -- label drift between "
                f"generator and pipeline"
            )
        true = truth_by_name[name]
        rows.append(
            {
                "param": f"{var}[{name}]",
                "truth": true,
                "mean": float(mean),
                "hdi_low": float(l),
                "hdi_high": float(h),
                "covered": bool(l <= true <= h),
            }
        )
    return rows


def main() -> int:
    truth = json.loads(TRUTH_PATH.read_text())
    idata = load_idata()

    rows = []

    post_intercept = idata.posterior["intercept"]
    lo, hi = hdi_bounds(idata, "intercept")
    rows.append(
        {
            "param": "intercept",
            "truth": truth["intercept"],
            "mean": float(post_intercept.mean().values),
            "hdi_low": float(lo[0]),
            "hdi_high": float(hi[0]),
            "covered": bool(lo[0] <= truth["intercept"] <= hi[0]),
        }
    )
    rows += check_vector(idata, "model_effects", "model", truth["model_effects"])
    rows += check_vector(idata, "scaffold_effects", "scaffold", truth["scaffold_effects"])
    rows += check_vector(idata, "benchmark_effects", "benchmark", truth["benchmark_effects"])
    rows += check_vector(
        idata, "benchmark_item_sigma", "benchmark", truth["benchmark_item_sigma"]
    )

    width = max(len(r["param"]) for r in rows)
    print(f"\n{'parameter':<{width}}  {'truth':>7}  {'mean':>7}  {'94% HDI':>18}  covered")
    for r in rows:
        print(
            f"{r['param']:<{width}}  {r['truth']:>7.2f}  {r['mean']:>7.2f}  "
            f"[{r['hdi_low']:>7.2f}, {r['hdi_high']:>7.2f}]  {'yes' if r['covered'] else 'NO'}"
        )

    coverage = float(np.mean([r["covered"] for r in rows]))

    # item deviations: correlation of posterior means with truth
    item_post = idata.posterior["benchmark_item_effects"]
    names = [str(n) for n in item_post["benchmark_item"].values]
    means = item_post.mean(dim=("chain", "draw")).values.ravel()
    true_devs = np.array([truth["item_deviations"][n] for n in names])
    item_corr = float(np.corrcoef(means, true_devs)[0, 1])

    print(f"\ncoverage: {coverage:.0%} of {len(rows)} parameters in 94% HDI")
    print(f"item deviation correlation (n={len(names)}): {item_corr:.3f}")

    ok = coverage >= COVERAGE_THRESHOLD and item_corr >= ITEM_CORR_THRESHOLD
    print("PARAMETER RECOVERY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
