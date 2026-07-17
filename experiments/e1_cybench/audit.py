"""One-off audit: challenge overlap, run-level dedupe check, config heterogeneity."""

import hashlib
import json

import pandas as pd

from experiments.e1_cybench.prep import FOCAL_MODELS, load_clean, _task_args_field

pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 300)


def main() -> None:
    df, _ = load_clean()
    hard = df[(df["variant"] == "hard") & df["model"].isin(FOCAL_MODELS)]

    # challenge coverage matrix
    cov = hard.groupby(["challenge", "model"]).size().unstack(fill_value=0)
    n_models_per_chal = (cov > 0).sum(axis=1)
    print("challenges by number of models covering them:")
    print(n_models_per_chal.value_counts().sort_index())
    print()
    print("challenges NOT covered by all models:")
    print(cov[n_models_per_chal < cov.shape[1]].to_string())
    print()

    # run-level dedupe check on RAW data: hash each run's content
    raw = pd.read_parquet("data/cybench.samples.parquet")
    def run_hash(g: pd.DataFrame) -> str:
        content = g.sort_values(["item_id", "epoch"])[
            ["item_id", "epoch", "score_includes", "total_tokens", "message_count"]
        ].to_csv(index=False)
        return hashlib.md5(content.encode()).hexdigest()

    rh = raw.groupby(["model", "run_internal_id"]).apply(run_hash, include_groups=False)
    rh = rh.reset_index(name="h")
    dup_runs = rh[rh.duplicated(["model", "h"], keep="first")]
    n_dup_rows = raw[raw["run_internal_id"].isin(dup_runs["run_internal_id"])].shape[0]
    print(f"whole runs that are content-identical to an earlier run: "
          f"{len(dup_runs)}/{len(rh)} runs, {n_dup_rows} rows")
    print()

    # config heterogeneity within model (hard variant): max_messages / solver / sandbox
    hard = hard.copy()
    hard["max_messages"] = hard["task_args"].map(
        lambda s: _task_args_field(s, "max_messages"))
    hard["sandbox"] = hard["task_args"].map(
        lambda s: _task_args_field(s, "sandbox_type"))
    for col in ["max_messages", "solver", "sandbox"]:
        print(f"--- {col} by model (hard variant) ---")
        print(hard.groupby(["model", hard[col].fillna("none").astype(str)])
              .size().unstack(fill_value=0).to_string())
        print()

    # per (model, challenge, run) cell sizes for joint model
    cells = hard.groupby(["model", "run_internal_id", "challenge"]).agg(
        n=("score", "size"), k=("score", "sum"))
    print("joint model: n cells =", len(cells), "| runs =",
          hard.groupby(["model", "run_internal_id"]).ngroups)
    # epochs available per run
    ep = hard.groupby(["model", "run_internal_id", "challenge"]).size()
    print("attempts per (model,run,challenge) cell distribution:")
    print(ep.describe())


if __name__ == "__main__":
    main()
