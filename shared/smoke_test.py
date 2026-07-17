"""End-to-end smoke test on synthetic data: bridge -> processors -> NUTS -> diagnostics.

    cd ~/dev/hibayes-experiments && uv run python shared/smoke_test.py
"""

import numpy as np
import pandas as pd

from shared.bridge import binomial_agg, diagnostics, fit, make_state, run_processors
from hibayes.model.models import two_level_group_binomial
from hibayes.process import extract_features, extract_observed_feature

rng = np.random.default_rng(0)
true_p = {"model_a": 0.7, "model_b": 0.5, "model_c": 0.3}
rows = [
    {"model": m, "item": i, "score": float(rng.random() < p)}
    for m, p in true_p.items()
    for i in range(40)
]
df = pd.DataFrame(rows)

agg = binomial_agg(df, by=["model"])
agg = agg.rename(columns={"model": "group"})
state = make_state(agg)
state = run_processors(
    state,
    extract_features(continuous_features=["n_total"], categorical_features=["group"]),
    extract_observed_feature(feature_name="n_correct"),
)
mas = fit(state, two_level_group_binomial(), tag="smoke", samples=500, warmup=500, chains=2)
diag = diagnostics(mas)
print(diag)
assert diag["max_r_hat"] < 1.05, f"r_hat too high: {diag['max_r_hat']}"
assert diag["n_divergences"] == 0, f"divergences: {diag['n_divergences']}"

post = mas.inference_data.posterior["group_effects"]
import jax

means = jax.nn.sigmoid(post.mean(dim=("chain", "draw")).values)
print("recovered p:", dict(zip(state.coords["group"], [round(float(x), 2) for x in means])))
print("SMOKE TEST OK")
