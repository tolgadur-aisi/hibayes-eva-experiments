"""Hierarchical binomial model for the eva unified scaffold analysis.

The statistical model, in her words: pass/fail ~ model + scaffold +
benchmark_item, with benchmark_item inheriting from benchmark. Concretely:

    logit P(pass) = intercept
                    + model_effect[model]          (sum-to-zero)
                    + scaffold_effect[scaffold]    (sum-to-zero)
                    + benchmark_effect[b(item)]    (sum-to-zero)
                    + item_deviation[item],  item_deviation ~ N(0, sigma[b(item)])

plus any config-declared main effects (e.g. token_given, the token budget a
run was granted) and categorical interactions (e.g. benchmark x token_given),
all effect-coded sum-to-zero.

    n_correct ~ Binomial(n_total, logit P(pass))

Each benchmark gets its own mean difficulty and its own item-difficulty spread
(sigma), and item effects are partially pooled within their benchmark. The
item deviations are non-centred for sampler geometry.

Structure mirrors syco-at-t's hierarchical_ordered_logistic_model so the
config reads the same way; only the likelihood and the nesting differ.
"""

from typing import List, Optional

import jax.numpy as jnp
import numpyro
from hibayes.model import Model, check_features, model
from hibayes.process import Features
from numpyro import distributions as dist


def _interaction_effects(
    name1: str, name2: str, features: Features, prior: dist.Distribution
) -> jnp.ndarray:
    """Effect-coded interaction matrix: rows and columns each sum to zero.

    Same construction as syco-at-t's _create_interaction_effects: sample the
    (n1-1)x(n2-1) free block, complete the last row/column from the
    constraints.
    """
    n1 = features[f"num_{name1}"]
    n2 = features[f"num_{name2}"]
    if n1 == 1 or n2 == 1:
        return jnp.zeros((n1, n2))

    raw = numpyro.sample(
        f"{name1}_{name2}_effects_constrained", prior.expand([(n1 - 1) * (n2 - 1)])
    ).reshape((n1 - 1, n2 - 1))
    b_full = jnp.zeros((n1, n2))
    b_full = b_full.at[: n1 - 1, : n2 - 1].set(raw)
    b_full = b_full.at[n1 - 1, : n2 - 1].set(-jnp.sum(raw, axis=0))
    b_full = b_full.at[:, n2 - 1].set(-jnp.sum(b_full[:, : n2 - 1], axis=1))
    numpyro.deterministic(f"{name1}_{name2}_effects", b_full)
    return b_full


@model
def item_nested_binomial(
    main_effects: Optional[List[str]] = None,
    interactions: Optional[List[List[str]]] = None,
    item_effect: str = "benchmark_item",
    nest_within: str = "benchmark",
    prior_intercept_loc: float = 0.0,
    prior_intercept_scale: float = 1.5,
    prior_main_effects_loc: float = 0.0,
    prior_main_effects_scale: float = 1.0,
    prior_interaction_scale: float = 0.5,
    prior_nest_effects_scale: float = 2.0,
    prior_item_sigma_scale: float = 2.0,
) -> Model:
    """Binomial GLM with fixed main effects and item effects nested in benchmarks.

    Args:
        main_effects: Categorical fixed effects (sum-to-zero coded), e.g.
            ["model", "scaffold"]. Each needs {name}_index / num_{name} features
            from extract_features with effect_coding_for_main_effects=True.
        interactions: Pairs of categorical features, e.g.
            [["benchmark", "token_given"]]. Each pair gets a doubly
            sum-to-zero effect matrix (rows and columns sum to zero).
        item_effect: Categorical feature holding the item identity.
        nest_within: Categorical feature the item effects are nested inside.
            Requires the item_benchmark_index feature from the
            extract_item_benchmark_index processor.
        prior_intercept_loc: Mean of the normal prior on the intercept.
        prior_intercept_scale: Scale of the normal prior on the intercept.
        prior_main_effects_loc: Mean of the normal prior on main effects.
        prior_main_effects_scale: Scale of the normal prior on main effects.
        prior_nest_effects_scale: Scale of the normal prior on the nesting
            variable's mean effects (benchmark difficulty differences are
            large on the logit scale, so this default is wide).
        prior_item_sigma_scale: Scale of the half-normal prior on each
            benchmark's item-difficulty spread. Agentic benchmarks show item
            spreads of 2-3 logits, hence the generous default.
    """
    main_effects = list(main_effects or [])
    interactions = [tuple(pair) for pair in (interactions or [])]

    def _model(features: Features) -> None:
        required = ["obs", "n_total", "item_benchmark_index"]
        interacting = [name for pair in interactions for name in pair]
        for effect in main_effects + interacting + [item_effect, nest_within]:
            required.extend([f"{effect}_index", f"num_{effect}"])
        check_features(features, required)

        intercept = numpyro.sample(
            "intercept", dist.Normal(prior_intercept_loc, prior_intercept_scale)
        )
        eta = intercept

        # Fixed main effects, sum-to-zero coded (n-1 free parameters)
        prior_main = dist.Normal(prior_main_effects_loc, prior_main_effects_scale)
        for effect in main_effects:
            n_levels = features[f"num_{effect}"]
            idx = features[f"{effect}_index"]
            if n_levels > 1:
                free = numpyro.sample(
                    f"{effect}_effects_constrained", prior_main.expand([n_levels - 1])
                )
                coefs = jnp.concatenate([free, -jnp.sum(free, keepdims=True)])
            else:
                coefs = jnp.zeros(n_levels)
            numpyro.deterministic(f"{effect}_effects", coefs)
            eta = eta + coefs[idx]

        # Interaction effects, doubly sum-to-zero
        prior_interaction = dist.Normal(0.0, prior_interaction_scale)
        for name1, name2 in interactions:
            b_full = _interaction_effects(name1, name2, features, prior_interaction)
            eta = eta + b_full[features[f"{name1}_index"], features[f"{name2}_index"]]

        # Nesting variable mean effects (benchmark difficulty), sum-to-zero
        n_nests = features[f"num_{nest_within}"]
        if n_nests > 1:
            nest_free = numpyro.sample(
                f"{nest_within}_effects_constrained",
                dist.Normal(0.0, prior_nest_effects_scale).expand([n_nests - 1]),
            )
            nest_effects = jnp.concatenate(
                [nest_free, -jnp.sum(nest_free, keepdims=True)]
            )
        else:
            nest_effects = jnp.zeros(n_nests)
        numpyro.deterministic(f"{nest_within}_effects", nest_effects)

        # Item effects nested within benchmarks, non-centred:
        # item_effect ~ Normal(0, sigma[benchmark of item]).
        # Deviations are hard sum-to-zero centred *within* each benchmark --
        # otherwise a benchmark's mean trades off against the average of its
        # item deviations (identified only through the prior), which shows up
        # as poor mixing on benchmark_effects.
        n_items = features[f"num_{item_effect}"]
        item_nest = features["item_benchmark_index"]  # (n_items,) -> nest level
        sigma_item = numpyro.sample(
            f"{item_effect}_sigma",
            dist.HalfNormal(prior_item_sigma_scale).expand([n_nests]),
        )
        z_item = numpyro.sample(f"{item_effect}_z", dist.Normal(0, 1).expand([n_items]))
        raw_deviation = z_item * sigma_item[item_nest]
        nest_sums = jnp.zeros(n_nests).at[item_nest].add(raw_deviation)
        nest_counts = jnp.zeros(n_nests).at[item_nest].add(1.0)
        item_deviation = raw_deviation - (nest_sums / nest_counts)[item_nest]
        numpyro.deterministic(f"{item_effect}_effects", item_deviation)

        item_idx = features[f"{item_effect}_index"]
        eta = eta + nest_effects[item_nest[item_idx]] + item_deviation[item_idx]

        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], logits=eta),
            obs=features["obs"],
        )

    return _model
