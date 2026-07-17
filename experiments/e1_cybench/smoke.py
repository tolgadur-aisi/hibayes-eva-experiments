"""Tiny smoke test of the E1 plumbing (not part of run.py)."""

from shared.bridge import binomial_agg, diagnostics

from experiments.e1_cybench import synth
from experiments.e1_cybench.fitting import draws, fit_crossed, fit_joint
from experiments.e1_cybench.prep import hard_focal, load_clean


def main() -> None:
    rows: list = []
    mas = synth.synth_challenge_only(rows, samples=200, warmup=200)
    print(diagnostics(mas))
    print(rows)

    df, _ = load_clean(verbose=False)
    hard = hard_focal(df)
    sub = hard[hard["model"] == "mistralazure/Mistral-Large-2411"]
    cells = binomial_agg(sub, by=["challenge", "run"])
    mas, coords = fit_crossed(cells, tag="smoke", samples=200, warmup=200)
    print(diagnostics(mas))
    print("sigma_challenge mean:", draws(mas, "sigma_challenge").mean())

    two = hard[hard["model"].isin(
        ["mistralazure/Mistral-Large-2411", "openai/gpt-5"])]
    jc = binomial_agg(two, by=["model", "challenge", "run"])
    mas, coords = fit_joint(jc, tag="smoke_joint", samples=200, warmup=200)
    print(diagnostics(mas))
    print("models:", coords["model"])
    print("ability means:", draws(mas, "model_ability").mean(axis=0))


if __name__ == "__main__":
    main()
