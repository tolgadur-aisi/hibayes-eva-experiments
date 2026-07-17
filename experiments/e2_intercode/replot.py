"""Regenerate plots from saved idata without refitting (cosmetic tweaks only).

    uv run python -m experiments.e2_intercode.replot
"""

import arviz as az

from experiments.e2_intercode.common import OUT
from experiments.e2_intercode.run import neff_curves, plot_neff


def main():
    idata = az.from_netcdf(OUT / "d1_hetrun.idata.nc")
    coords = {
        "model": [str(v) for v in idata.posterior.coords["model"].values],
        "run": [str(v) for v in idata.posterior.coords["run"].values],
    }
    ks, curves, n_items = neff_curves(idata, coords)
    plot_neff(ks, curves, n_items, OUT / "neff_vs_epochs_d1.png")
    print("regenerated", OUT / "neff_vs_epochs_d1.png")


if __name__ == "__main__":
    main()
