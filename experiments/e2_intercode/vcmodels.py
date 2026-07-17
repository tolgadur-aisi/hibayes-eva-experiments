"""Custom numpyro variance-component models for E2.

Cross-classified logistic random effects on binomial cell counts:

    crossed_var:          logit p = mu + a_model + b_item + g_model.item + u_run
    crossed_var_variant:  ... + v_variant (scaffold variant within model), run within variant

All effects non-centred. Epoch-level residual is the latent-logistic variance
pi^2/3 (standard threshold formulation for binary outcomes).
"""

import jax
import numpyro
import numpyro.distributions as dist
from hibayes.model import model
from hibayes.model.models import check_features
from hibayes.process import Features


@model
def crossed_var(
    prior_mu_scale: float = 1.5,
    prior_sigma_model: float = 1.5,
    prior_sigma_item: float = 2.0,
    prior_sigma_inter: float = 0.75,
    prior_sigma_run: float = 0.75,
):
    """Model + item + model.item + run(model) variance components."""

    def _model(features: Features) -> None:
        check_features(
            features,
            ["obs", "n_total", "model_index", "num_model", "item_index", "num_item",
             "run_index", "num_run"],
        )
        mu = numpyro.sample("mu", dist.Normal(0.0, prior_mu_scale))

        sigma_model = numpyro.sample("sigma_model", dist.HalfNormal(prior_sigma_model))
        z_m = numpyro.sample("z_model", dist.Normal(0, 1).expand([features["num_model"]]))
        a = numpyro.deterministic("model_effects", sigma_model * z_m)

        sigma_item = numpyro.sample("sigma_item", dist.HalfNormal(prior_sigma_item))
        z_i = numpyro.sample("z_item", dist.Normal(0, 1).expand([features["num_item"]]))
        b = numpyro.deterministic("item_effects", sigma_item * z_i)

        sigma_inter = numpyro.sample("sigma_inter", dist.HalfNormal(prior_sigma_inter))
        z_g = numpyro.sample(
            "z_inter",
            dist.Normal(0, 1).expand([features["num_model"], features["num_item"]]),
        )
        g = numpyro.deterministic("inter_effects", sigma_inter * z_g)

        sigma_run = numpyro.sample("sigma_run", dist.HalfNormal(prior_sigma_run))
        z_r = numpyro.sample("z_run", dist.Normal(0, 1).expand([features["num_run"]]))
        u = numpyro.deterministic("run_effects", sigma_run * z_r)

        mi, ii, ri = features["model_index"], features["item_index"], features["run_index"]
        logit_p = mu + a[mi] + b[ii] + g[mi, ii] + u[ri]

        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)),
            obs=features["obs"],
        )

    return _model


@model
def crossed_var_hetrun(
    prior_mu_scale: float = 1.5,
    prior_sigma_model: float = 1.5,
    prior_sigma_item: float = 2.0,
    prior_sigma_inter: float = 0.75,
    prior_sigma_run: float = 0.75,
):
    """As crossed_var but with a separate run-effect scale per model.

    Motivated by the model-free dispersion check: run overdispersion differs
    strongly by model/provider. Requires extra feature `run_model_index`
    (length num_run, mapping each run to its model).
    """

    def _model(features: Features) -> None:
        check_features(
            features,
            ["obs", "n_total", "model_index", "num_model", "item_index", "num_item",
             "run_index", "num_run", "run_model_index"],
        )
        mu = numpyro.sample("mu", dist.Normal(0.0, prior_mu_scale))

        sigma_model = numpyro.sample("sigma_model", dist.HalfNormal(prior_sigma_model))
        z_m = numpyro.sample("z_model", dist.Normal(0, 1).expand([features["num_model"]]))
        a = numpyro.deterministic("model_effects", sigma_model * z_m)

        sigma_item = numpyro.sample("sigma_item", dist.HalfNormal(prior_sigma_item))
        z_i = numpyro.sample("z_item", dist.Normal(0, 1).expand([features["num_item"]]))
        b = numpyro.deterministic("item_effects", sigma_item * z_i)

        sigma_inter = numpyro.sample("sigma_inter", dist.HalfNormal(prior_sigma_inter))
        z_g = numpyro.sample(
            "z_inter",
            dist.Normal(0, 1).expand([features["num_model"], features["num_item"]]),
        )
        g = numpyro.deterministic("inter_effects", sigma_inter * z_g)

        sigma_run = numpyro.sample(
            "sigma_run", dist.HalfNormal(prior_sigma_run).expand([features["num_model"]])
        )
        z_r = numpyro.sample("z_run", dist.Normal(0, 1).expand([features["num_run"]]))
        u = numpyro.deterministic(
            "run_effects", sigma_run[features["run_model_index"]] * z_r
        )

        mi, ii, ri = features["model_index"], features["item_index"], features["run_index"]
        logit_p = mu + a[mi] + b[ii] + g[mi, ii] + u[ri]

        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)),
            obs=features["obs"],
        )

    return _model


@model
def crossed_var_variant(
    prior_mu_scale: float = 1.5,
    prior_sigma_model: float = 1.5,
    prior_sigma_variant: float = 0.75,
    prior_sigma_item: float = 2.0,
    prior_sigma_inter: float = 0.75,
    prior_sigma_run: float = 0.75,
):
    """As crossed_var plus a scaffold-variant component (variant within model)."""

    def _model(features: Features) -> None:
        check_features(
            features,
            ["obs", "n_total", "model_index", "num_model", "item_index", "num_item",
             "run_index", "num_run", "variant_index", "num_variant"],
        )
        mu = numpyro.sample("mu", dist.Normal(0.0, prior_mu_scale))

        sigma_model = numpyro.sample("sigma_model", dist.HalfNormal(prior_sigma_model))
        z_m = numpyro.sample("z_model", dist.Normal(0, 1).expand([features["num_model"]]))
        a = numpyro.deterministic("model_effects", sigma_model * z_m)

        sigma_variant = numpyro.sample("sigma_variant", dist.HalfNormal(prior_sigma_variant))
        z_v = numpyro.sample("z_variant", dist.Normal(0, 1).expand([features["num_variant"]]))
        v = numpyro.deterministic("variant_effects", sigma_variant * z_v)

        sigma_item = numpyro.sample("sigma_item", dist.HalfNormal(prior_sigma_item))
        z_i = numpyro.sample("z_item", dist.Normal(0, 1).expand([features["num_item"]]))
        b = numpyro.deterministic("item_effects", sigma_item * z_i)

        sigma_inter = numpyro.sample("sigma_inter", dist.HalfNormal(prior_sigma_inter))
        z_g = numpyro.sample(
            "z_inter",
            dist.Normal(0, 1).expand([features["num_model"], features["num_item"]]),
        )
        g = numpyro.deterministic("inter_effects", sigma_inter * z_g)

        sigma_run = numpyro.sample("sigma_run", dist.HalfNormal(prior_sigma_run))
        z_r = numpyro.sample("z_run", dist.Normal(0, 1).expand([features["num_run"]]))
        u = numpyro.deterministic("run_effects", sigma_run * z_r)

        mi, ii, ri, vi = (
            features["model_index"],
            features["item_index"],
            features["run_index"],
            features["variant_index"],
        )
        logit_p = mu + a[mi] + v[vi] + b[ii] + g[mi, ii] + u[ri]

        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], probs=jax.nn.sigmoid(logit_p)),
            obs=features["obs"],
        )

    return _model
