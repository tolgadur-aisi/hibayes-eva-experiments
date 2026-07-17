"""Side check: is claude-3-7's huge sigma_run real config drift?
Look at per-run accuracy on the same challenges."""

import pandas as pd

from experiments.e1_cybench.prep import _task_args_field, hard_focal, load_clean

pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 300)


def main() -> None:
    df, _ = load_clean(verbose=False)
    hard = hard_focal(df)
    c = hard[hard["model"] == "anthropic/claude-3-7-sonnet-20250219"].copy()
    c["max_messages"] = c["task_args"].map(
        lambda s: _task_args_field(s, "max_messages"))
    c["created_day"] = c["created"].dt.date

    runs = (c.groupby("run")
            .agg(rows=("score", "size"), chal=("challenge", "nunique"),
                 acc=("score", "mean"), day=("created_day", "first"),
                 mm=("max_messages", "first"))
            .sort_values("acc"))
    print(runs.to_string())
    print()
    # challenges covered by >= 5 runs: per-run accuracy spread
    wide = (c.groupby(["challenge", "run"])["score"].mean().unstack())
    multi = wide[wide.notna().sum(axis=1) >= 5]
    print("per-challenge run-to-run accuracy (challenges in >=5 runs):")
    stats = pd.DataFrame({
        "n_runs": multi.notna().sum(axis=1),
        "min": multi.min(axis=1), "max": multi.max(axis=1),
        "mean": multi.mean(axis=1),
    })
    print(stats.sort_values("mean").to_string())


if __name__ == "__main__":
    main()
