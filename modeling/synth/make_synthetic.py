"""Generate synthetic eva-shaped extracts with known parameters.

Produces parquet files matching modeling/extract.py's output schema (one per
benchmark) plus a truth.json with the generating parameters, so the full
config-driven pipeline can be validated end to end by parameter recovery
before it ever touches real data. Invoked by modeling/synth/run_synth.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.processors import canonical_args

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "synth"

# Ground truth, all effect sets sum-to-zero to match the model's coding.
INTERCEPT = -0.4

MODELS = {
    "model-alpha": -1.5,
    "model-bravo": -0.8,
    "model-charlie": -0.2,
    "model-delta": 0.3,
    "model-echo": 0.9,
    "model-foxtrot": 1.3,
}

# Scaffolds as concrete (solver, task_args, solver_args) triples, the way they
# appear in eva. Labels are derived with the SAME canonicalisation the
# pipeline uses, so recovery keys can't drift from pipeline labels.
EXCLUDED_TASK_ARGS = {"variants", "sample_ids", "samples"}
SCAFFOLD_SPECS = [
    # (solver, task_args, solver_args, true effect)
    ("basic_agent", {"max_attempts": 1, "variants": "hard"}, {}, -0.4),
    ("basic_agent", {"max_attempts": 3, "variants": "hard"}, {"tools": "full"}, 0.1),
    ("react_agent", {"max_attempts": 3, "variants": "hard"}, {"tools": "full"}, 0.3),
]

BENCHMARKS = {
    # name: (n_items, benchmark effect, item sigma, score column, letter scores?)
    "cybench": (40, -1.2, 2.0, "score_includes", True),
    "gdm_intercode_ctf": (79, 0.9, 1.2, "score_includes", True),
    "swe_bench": (50, 0.3, 0.8, "score_swe_bench_scorer", False),
}

# Token budgets the runs were GIVEN (None = unlimited -> level "none").
# Main effects sum to zero; the benchmark x token interaction matrix has
# rows and columns summing to zero, matching the model's coding.
TOKEN_BUDGETS = {
    200_000: -0.35,
    1_000_000: 0.05,
    None: 0.30,
}
TOKEN_LEVEL = {200_000: "200000", 1_000_000: "1000000", None: "none"}
BENCH_TOKEN_INTERACTION = {
    #                 200k   1M    none
    "cybench":            [0.20, -0.05, -0.15],
    "gdm_intercode_ctf":  [-0.25, 0.10, 0.15],
    "swe_bench":          [0.05, -0.05, 0.00],
}

EPOCHS = 6
P_PARTIAL = 0.005  # sprinkle partial credit to exercise the drop logic
P_MISSING = 0.005


def scaffold_label(solver: str, task_args: dict, solver_args: dict) -> str:
    """The label derive_scaffold will produce for this triple."""
    label = "|".join(
        [
            solver,
            canonical_args(json.dumps(task_args), EXCLUDED_TASK_ARGS),
            canonical_args(json.dumps(solver_args)),
        ]
    )
    return label.rstrip("|")


def generate(seed: int = 0) -> dict:
    """Write synthetic parquets to data/synth/ and return the truth dict."""
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    truth = {
        "seed": seed,
        "intercept": INTERCEPT,
        "model_effects": MODELS,
        "scaffold_effects": {
            scaffold_label(solver, ta, sa): eff for solver, ta, sa, eff in SCAFFOLD_SPECS
        },
        "benchmark_effects": {b: spec[1] for b, spec in BENCHMARKS.items()},
        "benchmark_item_sigma": {b: spec[2] for b, spec in BENCHMARKS.items()},
        "token_given_effects": {
            TOKEN_LEVEL[budget]: eff for budget, eff in TOKEN_BUDGETS.items()
        },
        "benchmark_token_given_effects": {
            bench: dict(zip([TOKEN_LEVEL[b] for b in TOKEN_BUDGETS], row))
            for bench, row in BENCH_TOKEN_INTERACTION.items()
        },
        "item_deviations": {},
    }

    for benchmark, (n_items, bench_eff, sigma, score_col, letters) in BENCHMARKS.items():
        items = [f"item_{i:03d}" for i in range(n_items)]
        # centred within benchmark, matching the model's sum-to-zero item coding
        item_dev = rng.normal(0.0, sigma, size=n_items)
        item_dev -= item_dev.mean()
        truth["item_deviations"].update(
            {f"{benchmark}/{it}": float(d) for it, d in zip(items, item_dev)}
        )

        rows = []
        for model, m_eff in MODELS.items():
            for solver, task_args, solver_args, s_eff in SCAFFOLD_SPECS:
                for t_idx, (budget, t_eff) in enumerate(TOKEN_BUDGETS.items()):
                    bt_eff = BENCH_TOKEN_INTERACTION[benchmark][t_idx]
                    eta = INTERCEPT + m_eff + s_eff + t_eff + bt_eff + bench_eff + item_dev
                    p = 1.0 / (1.0 + np.exp(-eta))
                    for epoch in range(1, EPOCHS + 1):
                        outcome = rng.random(n_items) < p
                        for item, ok in zip(items, outcome):
                            u = rng.random()
                            if letters:
                                score = "C" if ok else "I"
                                if u < P_PARTIAL:
                                    score = "P"
                                elif u < P_PARTIAL + P_MISSING:
                                    score = None
                            else:
                                score = 1.0 if ok else 0.0
                                if u < P_MISSING:
                                    score = None
                            rows.append(
                                {
                                    "task_name": benchmark,
                                    "model": model,
                                    "solver": solver,
                                    "task_args": json.dumps(task_args),
                                    "solver_args": json.dumps(solver_args),
                                    "model_generate_config": None,
                                    "sandbox_type": "docker",
                                    "message_limit": 50,
                                    "token_limit": budget,
                                    "item_id": item,
                                    "epoch": epoch,
                                    score_col: score,
                                }
                            )
        df = pd.DataFrame(rows)
        df.to_parquet(DATA_DIR / f"{benchmark}.samples.parquet", index=False)
        print(f"[synth] {benchmark}: {len(df)} rows -> {DATA_DIR / f'{benchmark}.samples.parquet'}")

    (DATA_DIR / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"[synth] truth -> {DATA_DIR / 'truth.json'}")
    return truth


if __name__ == "__main__":
    generate()
