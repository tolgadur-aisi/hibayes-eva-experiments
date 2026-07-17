"""Custom numpyro models for E5 (via the hibayes @model decorator).

Design notes:
- Model effects are FIXED (weak Normal priors), not partially pooled: with as
  few as 4 models per benchmark, hierarchical shrinkage badly biases the
  spread (the known two_level_group_binomial trap). Item/run/condition effects
  are hierarchical (many levels, non-centred).
- Observed site must be named "obs" (hibayes contract).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from hibayes.model import Model, model
from hibayes.model.models import check_features


@model
def crossed_binomial(
    prior_model_scale: float = 2.0,
    prior_sigma_item_scale: float = 3.0,
) -> Model:
    """Crossed fixed-model x random-item binomial on (model, item) cell counts.

    logit p = model_effects[m] + item_effects[i];
    item effects ~ N(0, sigma_item) (non-centred), which soft-identifies the
    model intercepts as ability on the average item.

    Identifiability note (established by the harsh synthetic scenario): when a
    large share of items is never solved, the item-pool MEAN logit — and hence
    the raw model intercepts — is only weakly identified (a common offset can
    trade off against the unbounded depths of never-solved items; the
    Normal(0, prior_model_scale) prior regularises it). All reported league
    metrics are offset-invariant: the discrimination spread is an SD across
    models, and abilities/rho are item-marginal probability-scale quantities
    where never-solved items saturate at 0 regardless of depth.
    """

    def model(features) -> None:  # noqa: A001 - hibayes convention
        check_features(
            features,
            ["obs", "num_model", "model_index", "num_item_id", "item_id_index", "n_total"],
        )
        model_effects = numpyro.sample(
            "model_effects",
            dist.Normal(0.0, prior_model_scale).expand([features["num_model"]]),
        )
        sigma_item = numpyro.sample("sigma_item", dist.HalfNormal(prior_sigma_item_scale))
        z_item = numpyro.sample(
            "z_item", dist.Normal(0.0, 1.0).expand([features["num_item_id"]])
        )
        # sum-to-zero centring removes the flat direction between the mean of
        # the item effects and the model intercepts (identifies model_effects
        # as ability at the item-pool average logit).
        item_effects = numpyro.deterministic(
            "item_id_effects", sigma_item * (z_item - jnp.mean(z_item))
        )

        logit_p = (
            model_effects[features["model_index"]]
            + item_effects[features["item_id_index"]]
        )
        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)),
            obs=features["obs"],
        )

    return model


@model
def crossed_betabinomial(
    prior_model_scale: float = 2.0,
    prior_sigma_item_scale: float = 3.0,
    prior_rho_cell_a: float = 1.0,
    prior_rho_cell_b: float = 3.0,
    p_floor: float = 1e-3,
) -> Model:
    """Sensitivity variant of ``crossed_binomial`` with beta-binomial cells.

    Identical mean structure (fixed model effects + centred random item
    effects) but each (model, item) cell is BetaBinomial with intra-cell
    correlation rho_cell (shared across cells): kappa = (1 - rho)/rho,
    k ~ BetaBinomial(n, p * kappa, (1 - p) * kappa). This absorbs the
    run-level overdispersion (config drift, provider updates) that the
    binomial cell likelihood ignores, widening ability HDIs accordingly.
    rho_cell ~ Beta(1, 3) (weakly favours small; identified by multi-sample
    cells only — single-sample cells reduce to Bernoulli).

    ``p_floor`` smoothly maps p into [p_floor, 1 - p_floor]: the beta-binomial
    log-density's curvature grows like 1/(p * kappa) as p -> 0 on never-solved
    cells, which caused ~2.5% step-size divergences unfloored; below the floor
    the likelihood is nearly flat (a 0/130 cell moves by < 1.5% probability
    between p = 1e-3 and p -> 0 at kappa ~ 3), so the floor bounds curvature
    at negligible inferential cost.
    """

    def model(features) -> None:  # noqa: A001
        check_features(
            features,
            ["obs", "num_model", "model_index", "num_item_id", "item_id_index", "n_total"],
        )
        model_effects = numpyro.sample(
            "model_effects",
            dist.Normal(0.0, prior_model_scale).expand([features["num_model"]]),
        )
        sigma_item = numpyro.sample("sigma_item", dist.HalfNormal(prior_sigma_item_scale))
        z_item = numpyro.sample(
            "z_item", dist.Normal(0.0, 1.0).expand([features["num_item_id"]])
        )
        item_effects = numpyro.deterministic(
            "item_id_effects", sigma_item * (z_item - jnp.mean(z_item))
        )
        rho_cell = numpyro.sample(
            "rho_cell", dist.Beta(prior_rho_cell_a, prior_rho_cell_b)
        )
        kappa = (1.0 - rho_cell) / rho_cell

        logit_p = (
            model_effects[features["model_index"]]
            + item_effects[features["item_id_index"]]
        )
        p = jax.nn.sigmoid(logit_p) * (1.0 - 2.0 * p_floor) + p_floor
        numpyro.sample(
            "obs",
            dist.BetaBinomial(
                concentration1=p * kappa,
                concentration0=(1.0 - p) * kappa,
                total_count=features["n_total"],
            ),
            obs=features["obs"],
        )

    return model


@model
def run_repro_binomial(
    prior_group_scale: float = 2.5,
    prior_sigma_run_scale: float = 1.0,
) -> Model:
    """Run-level reproducibility: fixed replicate-group means + run effects.

    k_r ~ Binomial(n_r, sigmoid(mu_g[group(r)] + sigma_run * z_r)).
    sigma_run > 0 means same-config repeat runs swing more than binomial.
    """

    def model(features) -> None:  # noqa: A001
        check_features(features, ["obs", "num_group", "group_index", "n_total"])
        mu_g = numpyro.sample(
            "group_effects",
            dist.Normal(0.0, prior_group_scale).expand([features["num_group"]]),
        )
        sigma_run = numpyro.sample("sigma_run", dist.HalfNormal(prior_sigma_run_scale))
        n_runs = features["obs"].shape[0]
        z_run = numpyro.sample("z_run", dist.Normal(0.0, 1.0).expand([n_runs]))
        logit_p = mu_g[features["group_index"]] + sigma_run * z_run
        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)),
            obs=features["obs"],
        )

    return model


@model
def boolq_normal(
    prior_theta_loc: float = 0.7,
    prior_theta_scale: float = 0.3,
    prior_split_scale: float = 0.2,
    prior_sigma_char_common_scale: float = 0.2,
    prior_sigma_char_scale: float = 0.3,
    prior_sigma_run_scale: float = 0.1,
) -> Model:
    """Normal-approximation hierarchical model for eval-level boolq scores.

    value_e ~ N(theta_m + b_split + delta_c + gamma_{m,c}, se_e^2 + sigma_run^2)
    on the probability scale. delta_c: shared charity-condition effects;
    gamma_{m,c}: model-specific condition sensitivity with per-model scale
    sigma_char_m. Split uses reference coding (first level = 0). theta_m is
    ability in the typical condition (delta = gamma = 0) at the reference split.
    """

    def model(features) -> None:  # noqa: A001
        check_features(
            features,
            [
                "obs", "num_model", "model_index", "num_charity", "charity_index",
                "num_split", "split_index", "se",
            ],
        )
        n_model = features["num_model"]
        n_char = features["num_charity"]
        theta = numpyro.sample(
            "model_effects",
            dist.Normal(prior_theta_loc, prior_theta_scale).expand([n_model]),
        )
        b_free = numpyro.sample(
            "split_effects_free",
            dist.Normal(0.0, prior_split_scale).expand([features["num_split"] - 1]),
        )
        b_split = jnp.concatenate([jnp.zeros(1), b_free])

        # Condition effects are sum-to-zero centred (delta over charities,
        # gamma per model row) so theta_m is identified as the model's mean
        # over the charity pool at the reference split.
        sigma_char_common = numpyro.sample(
            "sigma_char_common", dist.HalfNormal(prior_sigma_char_common_scale)
        )
        z_char = numpyro.sample("z_char", dist.Normal(0.0, 1.0).expand([n_char]))
        delta = sigma_char_common * (z_char - jnp.mean(z_char))

        sigma_char = numpyro.sample(
            "sigma_char", dist.HalfNormal(prior_sigma_char_scale).expand([n_model])
        )
        z_int = numpyro.sample("z_int", dist.Normal(0.0, 1.0).expand([n_model, n_char]))
        gamma = (z_int - jnp.mean(z_int, axis=1, keepdims=True)) * sigma_char[:, None]

        sigma_run = numpyro.sample("sigma_run", dist.HalfNormal(prior_sigma_run_scale))

        m_idx = features["model_index"]
        c_idx = features["charity_index"]
        mu = theta[m_idx] + b_split[features["split_index"]] + delta[c_idx] + gamma[m_idx, c_idx]
        sd = jnp.sqrt(features["se"] ** 2 + sigma_run**2)
        numpyro.sample("obs", dist.Normal(mu, sd), obs=features["obs"])

    return model
