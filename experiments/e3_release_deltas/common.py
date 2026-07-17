"""E3 release-over-release deltas: data builders + numpyro models.

Comparable-config ("matched") analysis classes, chosen after auditing
task_args heterogeneity (96 distinct configs among focus models on cybench,
16 on gdm_intercode_ctf):

- CY-HARD  cybench, default agent, sandbox k8s, variant_names=hard.
           Models: o1, o3, gpt-5, claude-opus-4. 40 challenges x 10 epochs,
           one run per model, all runs 2026-01-31..2026-02-02, task_version 1.0.0.
- CY-EASY  cybench, default agent, sandbox docker, variant_names=easy.
           Models: claude-sonnet-4, claude-opus-4-1, gpt-5, o3, o4-mini.
           Only 2 challenges (dynastic, primary_knowledge), 1 sample per run,
           848-2075 runs per model, 2025-09-17..2025-09-20.
- GDM-MAIN gdm_intercode_ctf, default agent, max_attempts=3, max_messages=50.
           Same 5 models, 14 challenges (sample_ids subsets per run),
           1 sample per run, 2025-09-17..2025-09-21.

Exclusion policy (documented in FINDINGS.md):
- rows with a non-empty `error` string are infra/API failures -> excluded;
- rows with limit == 'operator' (human stopped the run) -> excluded;
- rows with limit in {token, message, time} are genuine failures (the agent
  exhausted its task-defined budget) -> KEPT with their recorded score;
- non-binary scores are excluded by binomial_agg (bridge prints counts).
"""

from shared.bridge import load_samples  # noqa: F401  (sets numpyro host devices first)

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from hibayes.model import model
from hibayes.model.models import check_features

FOCUS_MODELS = {
    "cy_hard": [
        "openai/o1",
        "openai/o3",
        "openai/gpt-5",
        "anthropic/claude-opus-4-20250514",
    ],
    "cy_easy": [
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-opus-4-1-20250805",
        "openai/gpt-5",
        "openai/o3",
        "openai/o4-mini",
    ],
    "gdm_main": [
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-opus-4-1-20250805",
        "openai/gpt-5",
        "openai/o3",
        "openai/o4-mini",
    ],
}

# family pairs (older, newer) -> which matched classes cover them
PAIRS = {
    "o1 -> o3": ("openai/o1", "openai/o3", ["cy_hard"]),
    "sonnet-4 -> opus-4-1": (
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-opus-4-1-20250805",
        ["cy_easy", "gdm_main"],
    ),
}

BENCH_OF_CLASS = {"cy_hard": "cybench", "cy_easy": "cybench", "gdm_main": "gdm_intercode_ctf"}


def _cy_scaffold(task_args: str | None) -> str:
    d = json.loads(task_args) if isinstance(task_args, str) else {}
    agent = d.get("agent")
    agent_name = None
    if isinstance(agent, str):
        try:
            agent_name = json.loads(agent).get("name")
        except (json.JSONDecodeError, AttributeError):
            agent_name = agent
    variant = d.get("variant_names") or d.get("variants")
    return f"agent={agent_name}|sandbox={d.get('sandbox_type')}|variant={variant}"


def _gdm_is_modern(task_args: str | None) -> bool:
    d = json.loads(task_args) if isinstance(task_args, str) else {}
    return d.get("max_attempts") == 3 and d.get("max_messages") == 50 and d.get("solver") is None


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the documented row-exclusion policy; return (df, counts)."""
    n0 = len(df)
    real_err = df["error"].fillna("").str.len() > 0
    df = df[~real_err]
    n_err = n0 - len(df)
    op = df["limit"] == "operator"
    df = df[~op]
    n_op = int(op.sum())
    return df.reset_index(drop=True), {"n_input": n0, "excl_real_error": n_err, "excl_operator_limit": n_op}


def build_matched_classes() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return {class_name: sample-level df} + a hygiene summary table."""
    cy = load_samples("cybench", score_col="score_includes")
    cy["scaffold"] = cy["task_args"].map(_cy_scaffold)
    gdm = load_samples("gdm_intercode_ctf", score_col="score_includes")
    gdm["modern"] = gdm["task_args"].map(_gdm_is_modern)

    raw = {
        "cy_hard": cy[
            (cy.scaffold == "agent=None|sandbox=k8s|variant=hard")
            & cy.model.isin(FOCUS_MODELS["cy_hard"])
        ],
        "cy_easy": cy[
            (cy.scaffold == "agent=None|sandbox=docker|variant=easy")
            & cy.model.isin(FOCUS_MODELS["cy_easy"])
        ],
        "gdm_main": gdm[gdm.modern & gdm.model.isin(FOCUS_MODELS["gdm_main"])],
    }
    out: dict[str, pd.DataFrame] = {}
    rows = []
    for name, df in raw.items():
        df, excl = apply_exclusions(df.copy())
        assert df.duplicated(["run_internal_id", "item_id", "epoch"]).sum() == 0, name
        out[name] = df
        for m, sub in df.groupby("model"):
            rows.append(
                {
                    "class": name,
                    "benchmark": BENCH_OF_CLASS[name],
                    "model": m,
                    "n_rows": len(sub),
                    "n_items": sub.item_id.nunique(),
                    "n_runs": sub.run_internal_id.nunique(),
                    "n_limit_kept": int(sub.limit.notna().sum()),
                    "acc": round(float(sub.score.mean()), 4),
                    "first": str(sub.created.min()),
                    "last": str(sub.created.max()),
                    **excl,
                }
            )
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------- models


@model
def fixed_model_item_binomial(
    prior_model_scale: float = 2.0,
    prior_sigma_item_scale: float = 1.5,
    item_mode: str = "hier",
    likelihood: str = "binomial",
    prior_log_kappa_loc: float = 3.0,
    prior_log_kappa_scale: float = 1.5,
):
    """Fixed model effects + item (challenge) effects.

    logit p_{mj} = beta_model[m] + gamma_item[j]
    beta_model[m] ~ Normal(0, prior_model_scale)          (fixed: no pooling
        across models -- avoids the few-group over-shrinkage trap)

    item_mode:
      "hier"  -- gamma_j = sigma_item * z_j (non-centred partial pooling);
                 use when there are enough items (>= ~10).
      "fixed" -- dummy-coded fixed effects, gamma_0 = 0 (fully identified;
                 use for very few items where the hierarchy funnels).

    likelihood:
      "binomial"     -- standard.
      "betabinomial" -- adds cell-level overdispersion via concentration
                 kappa ~ LogNormal (robustness for correlated epochs).

    Model-vs-model deltas are invariant to the beta/gamma location trade-off.
    """

    def _model(features):
        check_features(
            features,
            ["obs", "n_total", "model_index", "num_model", "item_id_index", "num_item_id"],
        )
        beta = numpyro.sample(
            "model_effects", dist.Normal(0.0, prior_model_scale).expand([features["num_model"]])
        )
        n_items = features["num_item_id"]
        if item_mode == "hier":
            sigma_item = numpyro.sample("sigma_item", dist.HalfNormal(prior_sigma_item_scale))
            z_item = numpyro.sample("z_item", dist.Normal(0, 1).expand([n_items]))
            gamma = numpyro.deterministic("item_id_effects", sigma_item * z_item)
        elif item_mode == "fixed":
            g_free = numpyro.sample("g_free", dist.Normal(0.0, 2.0).expand([n_items - 1]))
            gamma = numpyro.deterministic(
                "item_id_effects", jnp.concatenate([jnp.zeros(1), g_free])
            )
        else:  # pragma: no cover
            raise ValueError(item_mode)
        logit_p = beta[features["model_index"]] + gamma[features["item_id_index"]]
        p = jax.nn.sigmoid(logit_p)
        if likelihood == "binomial":
            numpyro.sample(
                "obs",
                dist.Binomial(total_count=features["n_total"], probs=p),
                obs=features["obs"],
            )
        elif likelihood == "betabinomial":
            kappa = numpyro.sample(
                "kappa", dist.LogNormal(prior_log_kappa_loc, prior_log_kappa_scale)
            )
            numpyro.sample(
                "obs",
                dist.BetaBinomial(
                    concentration1=p * kappa + 1e-4,
                    concentration0=(1.0 - p) * kappa + 1e-4,
                    total_count=features["n_total"],
                ),
                obs=features["obs"],
            )
        else:  # pragma: no cover
            raise ValueError(likelihood)

    return _model


@model
def pooled_pair_binomial(
    prior_alpha_scale: float = 2.0,
    prior_mu_delta_scale: float = 1.5,
    prior_tau_scale: float = 0.5,
    prior_sigma_item_scale: float = 1.5,
):
    """Cross-benchmark pooled release delta for ONE (older, newer) pair.

    logit p = alpha_b + delta_b * is_new + gamma_item
    delta_b = mu_delta + tau * z_b   (benchmark-level partial pooling)
    gamma_item: dummy-coded FIXED effects, the first item of each benchmark is
    the reference (gamma=0, absorbed by alpha_b). Fixed (not hierarchical)
    because one benchmark has only 2 items -- a per-benchmark sigma funnels.
    """

    del prior_sigma_item_scale  # kept in signature for config compatibility

    def _model(features):
        check_features(
            features,
            [
                "obs",
                "n_total",
                "is_new",
                "benchmark_index",
                "num_benchmark",
                "item_uid_index",
                "num_item_uid",
                "item_ref_mask",
            ],
        )
        nb = features["num_benchmark"]
        alpha = numpyro.sample("alpha", dist.Normal(0.0, prior_alpha_scale).expand([nb]))
        mu_delta = numpyro.sample("mu_delta", dist.Normal(0.0, prior_mu_delta_scale))
        tau = numpyro.sample("tau", dist.HalfNormal(prior_tau_scale))
        z_b = numpyro.sample("z_b", dist.Normal(0, 1).expand([nb]))
        delta_b = numpyro.deterministic("benchmark_delta", mu_delta + tau * z_b)
        g_raw = numpyro.sample(
            "g_raw", dist.Normal(0.0, 2.0).expand([features["num_item_uid"]])
        )
        gamma = numpyro.deterministic(
            "item_uid_effects", g_raw * (1.0 - features["item_ref_mask"])
        )
        b = features["benchmark_index"]
        logit_p = alpha[b] + delta_b[b] * features["is_new"] + gamma[features["item_uid_index"]]
        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)),
            obs=features["obs"],
        )

    return _model


# ---------------------------------------------------------- synthetic data


def simulate_fixed_design(
    betas: list[float], n_items: int, n_per_cell: int, sigma_item: float, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Binomial counts for the fixed-effects design; returns (df, item_effects)."""
    rng = np.random.default_rng(seed)
    gamma = rng.normal(0.0, sigma_item, n_items)
    rows = []
    for m, beta in enumerate(betas):
        for j in range(n_items):
            p = 1.0 / (1.0 + np.exp(-(beta + gamma[j])))
            rows.append(
                {
                    "model": f"model_{m}",
                    "item_id": f"item_{j:03d}",
                    "n_correct": int(rng.binomial(n_per_cell, p)),
                    "n_total": n_per_cell,
                }
            )
    return pd.DataFrame(rows), gamma


def simulate_betabinom_design(
    betas: list[float], n_items: int, n_per_cell: int, sigma_item: float,
    kappa: float, seed: int,
) -> pd.DataFrame:
    """Fixed design with cell-level overdispersion (p_cell ~ Beta(p*k,(1-p)*k))."""
    rng = np.random.default_rng(seed)
    gamma = rng.normal(0.0, sigma_item, n_items)
    rows = []
    for m, beta in enumerate(betas):
        for j in range(n_items):
            p = 1.0 / (1.0 + np.exp(-(beta + gamma[j])))
            p_cell = rng.beta(p * kappa, (1 - p) * kappa)
            rows.append(
                {
                    "model": f"model_{m}",
                    "item_id": f"item_{j:03d}",
                    "n_correct": int(rng.binomial(n_per_cell, p_cell)),
                    "n_total": n_per_cell,
                }
            )
    return pd.DataFrame(rows)


def simulate_pooled_design(
    alphas: list[float],
    deltas: list[float],
    n_items: list[int],
    n_per_cell: list[int],
    sigma_item: float,
    seed: int,
) -> pd.DataFrame:
    """Two-benchmark pooled-pair design (is_new in {0,1})."""
    rng = np.random.default_rng(seed)
    rows = []
    for b, (alpha, delta) in enumerate(zip(alphas, deltas)):
        gamma = rng.normal(0.0, sigma_item, n_items[b])
        for j in range(n_items[b]):
            for is_new in (0, 1):
                p = 1.0 / (1.0 + np.exp(-(alpha + delta * is_new + gamma[j])))
                rows.append(
                    {
                        "benchmark": f"bench_{b}",
                        "item_uid": f"bench_{b}/item_{j:03d}",
                        "is_new": float(is_new),
                        "n_correct": int(rng.binomial(n_per_cell[b], p)),
                        "n_total": n_per_cell[b],
                    }
                )
    return pd.DataFrame(rows)


def hdi_and_probs(draws: np.ndarray) -> dict:
    """Posterior summary for a 1-d array of delta draws."""
    import arviz as az

    lo, hi = az.hdi(draws, hdi_prob=0.94)
    return {
        "mean": float(np.mean(draws)),
        "hdi_3": float(lo),
        "hdi_97": float(hi),
        "p_gt_0": float((draws > 0).mean()),
        "p_gt_0.5": float((draws > 0.5).mean()),
    }
