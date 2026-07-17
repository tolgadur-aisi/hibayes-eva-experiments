"""Selection-aware frontier statistics for E4 (fixes REVIEW_ROUND_1 blocker).

The frontier ("running best") subset is monotonically increasing BY
CONSTRUCTION, so any trend fitted to it has P(beta>0) ~ 1 under any truth,
including no trend at all. Record-path slopes/HDIs are therefore descriptive
only. This module provides statistics whose null distributions account for
the record selection:

1. **Record-count test** (selection-free). Among J exchangeable values in
   release order, P(model i sets a record) = 1/i independently, so the number
   of records K is Poisson-binomial; `record_count_pmf_exact` gives the exact
   null pmf. Observing many more records than E[K] = H_J is evidence of a
   trend that does not condition on the selected subset.
2. **Flat-null record-slope calibration.** Simulate the real design (same
   release dates, same capped Ns) under beta = 0 with (alpha, sigma) drawn
   from the posterior of a no-slope refit, apply the same record selection,
   and compute the record-path OLS slope on empirical logits. The observed
   statistic is computed identically; P(null >= observed) is a
   selection-aware test of the record-path trend.
3. **Linear posterior-predictive saturation check.** Simulate under the
   fitted *linear* mean-trend posterior (beta2 = 0), apply selection, and
   compute the record-path quadratic coefficient. Record paths decelerate
   mechanically (records get rarer), so the observed quad coefficient is
   evidence of saturation only if it is more negative than this null.

Release-date ties (e.g. o3 / o4-mini, both 2025-04-16) make the record scan
order-dependent: observed statistics are reported over all tie-consistent
orderings, and simulations randomise the order within tie groups per rep.
"""

from itertools import permutations, product

import numpy as np
import pandas as pd

RNG_SEED = 20260716


def empirical_logit(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Haldane-Anscombe empirical logit (matches fit_trend's observation)."""
    return np.log((k + 0.5) / (n - k + 0.5))


def record_mask(acc: np.ndarray) -> np.ndarray:
    """True where acc strictly exceeds the running maximum (a 'record')."""
    out = np.zeros(len(acc), dtype=bool)
    best = -np.inf
    for i, a in enumerate(acc):
        if a > best:
            out[i] = True
            best = a
    return out


def record_count_pmf_exact(j: int) -> np.ndarray:
    """Exact null pmf of #records among j exchangeable values (p_i = 1/i)."""
    pmf = np.array([1.0])
    for i in range(1, j + 1):
        p = 1.0 / i
        new = np.zeros(len(pmf) + 1)
        new[:-1] += pmf * (1 - p)
        new[1:] += pmf * p
        pmf = new
    return pmf


def tie_orderings(dates: np.ndarray, max_variants: int = 64) -> list[np.ndarray]:
    """All row orderings consistent with the release dates (permute ties)."""
    groups: list[list[int]] = []
    seen: dict = {}
    for i, d in enumerate(dates):
        if d in seen:
            groups[seen[d]].append(i)
        else:
            seen[d] = len(groups)
            groups.append([i])
    variants = []
    for combo in product(*[list(permutations(g)) for g in groups]):
        order = np.array([i for g in combo for i in g])
        variants.append(order)
        if len(variants) >= max_variants:
            break
    return variants


def _shuffled_tie_order(dates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random ordering consistent with the release dates."""
    jitter = rng.random(len(dates))
    return np.lexsort((jitter, dates))


def _path_stats(t: np.ndarray, y: np.ndarray, acc: np.ndarray) -> dict:
    """Record count + record-path OLS slope and quadratic coefficient."""
    rec = record_mask(acc)
    k = int(rec.sum())
    slope = np.nan
    quad = np.nan
    tr, yr = t[rec], y[rec]
    if k >= 2 and len(np.unique(tr)) >= 2:
        slope = float(np.polyfit(tr, yr, 1)[0])
    if k >= 3 and len(np.unique(tr)) >= 3:
        quad = float(np.polyfit(tr, yr, 2)[0])
    return {"n_records": k, "slope": slope, "quad": quad}


def observed_record_stats(sub: pd.DataFrame) -> pd.DataFrame:
    """Observed record stats for every tie-consistent ordering.

    `sub`: eligible cells sorted by release date with columns
    t (centred years), n_cap, k_cap, acc, release_date.
    """
    dates = sub["release_date"].values
    t = sub["t"].to_numpy()
    y = empirical_logit(sub["k_cap"].to_numpy(), sub["n_cap"].to_numpy())
    acc = sub["acc"].to_numpy()
    rows = []
    for v, order in enumerate(tie_orderings(dates)):
        rows.append(
            {"tie_variant": v, **_path_stats(t[order], y[order], acc[order])}
        )
    return pd.DataFrame(rows)


def simulate_record_null(
    sub: pd.DataFrame,
    alpha_draws: np.ndarray,
    beta_draws: np.ndarray | None,
    sigma_draws: np.ndarray,
    n_reps: int,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Posterior-predictive record-path stats for the design in `sub`.

    beta_draws=None simulates the flat null (beta = 0); otherwise the fitted
    linear model. Parameters are drawn per rep from the posterior, latent
    logits get a fresh N(0, sigma) model residual, counts are Binomial, and
    the record scan uses a random tie-consistent order.
    """
    rng = np.random.default_rng(seed)
    dates = sub["release_date"].values
    t = sub["t"].to_numpy()
    n = sub["n_cap"].to_numpy()
    n_draws = len(alpha_draws)
    rows = []
    for _ in range(n_reps):
        i = int(rng.integers(n_draws))
        mu = alpha_draws[i] + (0.0 if beta_draws is None else beta_draws[i] * t)
        theta = mu + rng.normal(0.0, sigma_draws[i], size=len(t))
        k = rng.binomial(n, 1.0 / (1.0 + np.exp(-theta)))
        order = _shuffled_tie_order(dates, rng)
        y = empirical_logit(k, n)
        acc = k / n
        rows.append(_path_stats(t[order], y[order], acc[order]))
    return pd.DataFrame(rows)


def summarise_selection_test(
    task: str,
    obs: pd.DataFrame,
    flat: pd.DataFrame,
    linear: pd.DataFrame,
    j: int,
    hdi: float = 0.94,
) -> dict:
    """One row for frontier_null.csv: observed vs null record-path stats.

    p-values use the conservative tie ordering (min observed record count,
    min observed slope, max observed quad); slope/quad p-values are
    conditional on the null rep having enough records to define the
    statistic, with the defined fraction reported.
    """
    qlo, qhi = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    k_min, k_max = int(obs["n_records"].min()), int(obs["n_records"].max())
    pmf = record_count_pmf_exact(j)
    p_rec_exact = float(pmf[k_min:].sum())
    p_rec_sim = float((flat["n_records"] >= k_min).mean())

    s_min, s_max = float(obs["slope"].min()), float(obs["slope"].max())
    fs = flat["slope"].dropna()
    p_slope = float((fs >= s_min).mean())

    q_min, q_max = float(obs["quad"].min()), float(obs["quad"].max())
    lq = linear["quad"].dropna()
    p_quad = float((lq <= q_max).mean())

    return {
        "task": task,
        "n_eligible_models": j,
        "obs_records_min_over_tie_orders": k_min,
        "obs_records_max_over_tie_orders": k_max,
        "expected_records_null": float(sum(1.0 / i for i in range(1, j + 1))),
        "p_records_exact_exchangeable": p_rec_exact,  # conservative tie order
        "p_records_exact_at_max_tie_order": float(pmf[k_max:].sum()),
        "p_records_sim_flat_null": p_rec_sim,
        "obs_record_slope_min": s_min,
        "obs_record_slope_max": s_max,
        "flat_null_slope_mean": float(fs.mean()),
        "flat_null_slope_q3": float(fs.quantile(qlo)),
        "flat_null_slope_q97": float(fs.quantile(qhi)),
        "flat_null_frac_slope_defined": float(flat["slope"].notna().mean()),
        "p_slope_ge_obs_flat_null": p_slope,
        "obs_record_quad_min": q_min,
        "obs_record_quad_max": q_max,
        "linear_ppc_quad_mean": float(lq.mean()),
        "linear_ppc_quad_q3": float(lq.quantile(qlo)),
        "linear_ppc_quad_q97": float(lq.quantile(qhi)),
        "linear_ppc_frac_quad_defined": float(linear["quad"].notna().mean()),
        "p_quad_le_obs_linear_ppc": p_quad,
        "n_reps": len(flat),
    }
