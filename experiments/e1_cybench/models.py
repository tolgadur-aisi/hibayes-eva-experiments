"""Custom numpyro models for E1 (crossed random effects).

Built-in hibayes hierarchies are nested; cybench needs crossed effects
(challenge x run, and model + challenge + run). Both models use non-centred
parameterisation and Binomial(logits=...) for stability.

Prior choices (see FINDINGS.md / synthetic recovery):
- challenge sd: HalfNormal(2.0) — challenge difficulty spans several logits;
  the hibayes default (0.1) over-shrinks badly.
- run sd: HalfNormal(1.0) — run/config drift is real but smaller.
- ability: Normal(0, 2) — accuracy 2%..98% a priori plausible.
"""

import jax.numpy as jnp  # noqa: F401  (kept for parity with hibayes models)
import numpyro
import numpyro.distributions as dist
from hibayes.model import model
from hibayes.model.models import check_features
from hibayes.process import Features


@model
def challenge_run_binomial(
    prior_mu_loc: float = 0.0,
    prior_mu_scale: float = 2.0,
    prior_sigma_challenge_scale: float = 2.0,
    prior_sigma_run_scale: float = 1.0,
):
    """Single-model crossed effects: logit p = mu + a_challenge + b_run."""

    def _model(features: Features) -> None:
        check_features(
            features,
            ["obs", "n_total", "challenge_index", "num_challenge",
             "run_index", "num_run"],
        )
        mu = numpyro.sample("mu", dist.Normal(prior_mu_loc, prior_mu_scale))

        sigma_c = numpyro.sample(
            "sigma_challenge", dist.HalfNormal(prior_sigma_challenge_scale))
        z_c = numpyro.sample(
            "z_challenge", dist.Normal(0, 1).expand([features["num_challenge"]]))
        # centred so mu is identified as the panel-mean latent difficulty
        a = sigma_c * (z_c - jnp.mean(z_c))
        numpyro.deterministic("challenge_effects", a)

        sigma_r = numpyro.sample(
            "sigma_run", dist.HalfNormal(prior_sigma_run_scale))
        z_r = numpyro.sample(
            "z_run", dist.Normal(0, 1).expand([features["num_run"]]))
        r = sigma_r * z_r
        numpyro.deterministic("run_effects", r)

        logit_p = (mu + a[features["challenge_index"]]
                   + r[features["run_index"]])
        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], logits=logit_p),
            obs=features["obs"],
        )

    return _model


@model
def model_challenge_run_binomial(
    prior_ability_loc: float = 0.0,
    prior_ability_scale: float = 2.0,
    prior_sigma_challenge_scale: float = 2.0,
    prior_sigma_run_scale: float = 1.0,
):
    """Joint comparison: logit p = ability_model + a_challenge + b_run.

    ability_model are independent (fixed-effect-style) intercepts — no
    partial pooling across models, so 10 groups cannot over-shrink; challenge
    and run effects are zero-centred random effects shared across models.
    """

    def _model(features: Features) -> None:
        check_features(
            features,
            ["obs", "n_total", "model_index", "num_model",
             "challenge_index", "num_challenge", "run_index", "num_run"],
        )
        ability = numpyro.sample(
            "model_ability",
            dist.Normal(prior_ability_loc, prior_ability_scale)
            .expand([features["num_model"]]),
        )

        sigma_c = numpyro.sample(
            "sigma_challenge", dist.HalfNormal(prior_sigma_challenge_scale))
        z_c = numpyro.sample(
            "z_challenge", dist.Normal(0, 1).expand([features["num_challenge"]]))
        # centred so abilities are identified as panel-mean latent ability
        a = sigma_c * (z_c - jnp.mean(z_c))
        numpyro.deterministic("challenge_effects", a)

        sigma_r = numpyro.sample(
            "sigma_run", dist.HalfNormal(prior_sigma_run_scale))
        z_r = numpyro.sample(
            "z_run", dist.Normal(0, 1).expand([features["num_run"]]))
        r = sigma_r * z_r
        numpyro.deterministic("run_effects", r)

        logit_p = (ability[features["model_index"]]
                   + a[features["challenge_index"]]
                   + r[features["run_index"]])
        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], logits=logit_p),
            obs=features["obs"],
        )

    return _model
