"""Side check: diagnose the all-zero claude-3-7 runs of 2025-06-23/24,
and scan all focal models for similar degenerate broad-coverage runs."""

import pandas as pd

from experiments.e1_cybench.prep import _task_args_field, hard_focal, load_clean

pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 100)


def main() -> None:
    df, _ = load_clean(verbose=False)
    hard = hard_focal(df)

    zero = hard[hard["run"].isin(["124615", "131722", "128720"])].copy()
    print(zero.groupby(["run", "limit"], dropna=False)
          .agg(n=("score", "size"), msg=("message_count", "mean"),
               tok=("total_tokens", "mean"), t=("total_time", "mean"))
          .to_string())
    print()
    print(zero.groupby("run")[["message_count", "total_tokens"]]
          .describe().to_string())
    print()
    for r in ["124615", "131722"]:
        ta = hard.loc[hard["run"] == r, "task_args"].iloc[0]
        print(r, "max_messages:", _task_args_field(ta, "max_messages"),
              "| agent cfg hash:", hash(_task_args_field(ta, "agent")) % 10**6)
    print("128720 (normal):", _task_args_field(
        hard.loc[hard["run"] == "128720", "task_args"].iloc[0], "max_messages"))
    print()

    # scan: runs with >=20 attempts on >=10 challenges and 0 solves, where the
    # model's other runs solve >15% overall
    for m in sorted(hard["model"].unique()):
        sub = hard[hard["model"] == m]
        overall = sub["score"].mean()
        g = sub.groupby("run").agg(n=("score", "size"), acc=("score", "mean"),
                                   chal=("challenge", "nunique"))
        deg = g[(g["n"] >= 20) & (g["chal"] >= 10) & (g["acc"] == 0.0)]
        if len(deg) and overall > 0.15:
            print(m, "overall acc", round(overall, 3), "degenerate runs:")
            print(deg.to_string())


if __name__ == "__main__":
    main()
