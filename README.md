# eva × hibayes: how much does scaffold matter?

One question, one config, one model: **how well do models perform across
benchmarks, and how much of that depends on their scaffold?**

    logit P(pass) = model + scaffold + benchmark_item
    (item effects nested within benchmark)

Fitted as a hierarchical binomial GLM with [hibayes](https://github.com/UKGovernmentBEIS/hibayes)'
config-driven pipeline, on pass/fail benchmark data from the
[eva](https://github.com/AI-Safety-Institute/eva) warehouse. The whole
analysis is [`modeling/config.yaml`](modeling/config.yaml); everything else is
small custom pieces the config references.

## Layout

```
modeling/
├── config.yaml        the analysis: load → process → model → check → communicate
├── extract.py         eva prod → data/modeling/*.parquet (run on a platform dev VM)
├── processors.py      score coercion, scaffold derivation, item↔benchmark nesting
├── custom_model.py    hierarchical binomial (main effects + nested item effects)
├── discovery/         step 0: which evals are pass/fail, which variables have coverage
└── synth/             parameter-recovery validation of the full pipeline
```

## Run it

Requirements: an AISI platform dev VM (VPC access to the eva Aurora cluster),
AWS credentials, and team_ru membership. `cd ~/dev/eva/client && uv run
eva-check` verifies the setup. eva enforces row-level security: without the
right team role, queries silently return only the rows you can see.

```bash
# 0. discovery — which tasks are pass/fail, which covariates have coverage
uv run python modeling/discovery/list_evals.py
uv run python modeling/discovery/list_variables.py --pass-fail-only

# 1. extract sample-level data for the selected pass/fail tasks
uv run python modeling/extract.py

# 2. fit + check + plot, all from the config
uv run hibayes-full --config modeling/config.yaml --out modeling/.output --no-tui
```

Outputs land in `modeling/.output/`: convergence and predictive checks under
`models/*/diagnostics/`, the forest plot and summary table under
`communicate/`. Iterate by editing the config (priors, variables, fit
settings) and re-running; the staged CLIs (`hibayes-load` / `hibayes-process`
/ `hibayes-model` / `hibayes-comm`) avoid re-loading data while you iterate on
the model.

## What "scaffold" means here

eva has no scaffold column. We derive one:
`solver | canonical(task_args − data-selection keys) | canonical(solver_args)`
(see `derive_scaffold` in [`modeling/processors.py`](modeling/processors.py)).
Keys that select *items* rather than configure the agent (e.g. cybench
variants) are excluded via config and belong to item identity instead. If the
scaffold definition changes, only `config.yaml` changes.

## Validation

The full pipeline — real config, synthetic data — is exercised by:

```bash
uv run python -m modeling.synth.run_synth
```

This generates eva-shaped data with known parameters, runs `hibayes-full` on
it, and checks the posteriors recover the truth (94% HDI coverage across all
16 generating parameters, item-effect correlation ≥ 0.9). Run it after any
change to the processors, model, or config.

## Data notes

- `data/` is gitignored: eval results are AISI-internal (the benchmarks are
  public, the results are not) and extracts are ~100s of MB. Regenerate with
  `modeling/extract.py`; `data/modeling/MANIFEST.json` records the snapshot.
- Every fitted number is pinned to its extraction snapshot — the warehouse
  moves, so re-extracting can shift results.
- The five earlier bespoke analyses (epoch design curves, variance
  decomposition, release deltas, growth curves, portfolio health) live in git
  history up to commit `1362644` — including their `FINDINGS.md` — and
  informed this model's priors and its run-effect caveats.

## Known limitations (v1)

- **No run effect.** Same-config reruns drift by ~1 logit on some
  provider/benchmark pairs (see E2/E5 in the historical findings), so the
  binomial likelihood is overdispersed and intervals are somewhat optimistic.
  v2 should add a run-level random effect — a config + small model change.
- **model × scaffold overlap.** Additive effects are only identified where
  models share scaffolds. Check the crossing table (`.output/processed_data.parquet`)
  before reading the forest plot causally; historically some model pairs had
  zero shared configs.
- **Three benchmarks** on the public team_ru slice. The discovery scripts are
  the path to widening this once broader eva access is in scope.
