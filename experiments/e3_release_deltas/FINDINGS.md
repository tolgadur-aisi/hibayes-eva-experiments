# E3 — Release-over-release capability deltas with honest uncertainty

## Question
For within-family model release pairs present in team_ru's eval warehouse data
(cybench, gdm_intercode_ctf, swe_bench), what is the release-over-release ability
delta on the logit scale under a hierarchical model with challenge effects, does it
replicate across benchmarks, and where does the naive "accuracy went up by X points"
reading mislead?

## Data
Parquet extracts (team_ru, status=success). Sample-level rows: cybench 66,694;
gdm_intercode_ctf 83,078; swe_bench 22,063. Bridge score coercion dropped 3,016
cybench and 180 gdm rows with unparseable scores (`load_samples` log).

**Config heterogeneity is severe** (96 distinct task_args configs among focus models
on cybench). Matched-config analysis classes (see `data_hygiene.csv`):

| class | benchmark | config | models (n rows) |
|---|---|---|---|
| CY-HARD | cybench | default agent, k8s, variant=hard; 40 challenges x 10 epochs, 1 run/model, 2026-01-31..02-02 | o1 (400), o3 (400), gpt-5 (400), opus-4 (400) |
| CY-EASY | cybench | default agent, docker, variant=easy; only 2 challenges, 1 sample/run, 2025-09-17..20 | sonnet-4 (904), opus-4-1 (848), gpt-5 (1405), o3 (2075), o4-mini (1074) |
| GDM-MAIN | gdm_intercode_ctf | built-in agent, max_attempts=3, max_messages=50; 14 challenges, 1 sample/run, 2025-09-17..21 | sonnet-4 (5581), opus-4-1 (4744), gpt-5 (6545), o3 (11816), o4-mini (7128) |

Item overlap between pair members is 100% within every class. Zero duplicate
(run, item, epoch) rows (asserted in `common.build_matched_classes`).

**Analysable family pairs:** o1→o3 (CY-HARD); sonnet-4→opus-4-1 (CY-EASY + GDM-MAIN,
pooled). **Not analysable at matched config (0 shared task_args configs):**
gpt-4o→gpt-5 (also 0 shared items on cybench), sonnet-3-7→sonnet-4,
o3-mini→o4-mini, opus-4→opus-4-1. swe_bench contains no within-family pair
(o3-mini, o1-2024-12-17, claude-3-5-sonnet only) and is excluded.

**Exclusions/policy:** rows with non-empty `error` (infra/API failures) and
`limit='operator'` excluded — both counts are 0 inside the matched classes.
Rows hitting token/message/time limits are genuine failures (budget is part of the
task) and are KEPT with their recorded score: 555 limit rows in CY-HARD, 1,892 in
CY-EASY, 13,028 in GDM-MAIN (`data_hygiene.csv`, n_limit_kept).

## Models (specs + priors)
1. **Per-class** (`fixed_model_item_binomial`): n_correct_mj ~ Binomial(n_mj, logit⁻¹(β_m + γ_j)).
   β_m ~ Normal(0, 2) fixed effects (no pooling across 4–5 models — avoids the
   few-group shrinkage trap). Items: CY-HARD (40 items) non-centred partial pooling
   γ_j = σ·z_j, σ ~ HalfNormal(1.5); CY-EASY (2) and GDM-MAIN (14) dummy-coded fixed
   effects γ ~ Normal(0, 2) (strong data; avoids non-centred funnel). Delta = β_new − β_old.
2. **CY-HARD robustness** (`likelihood="betabinomial"`): BetaBinomial with
   κ ~ LogNormal(3, 1.5) — the 10 epochs/item come from a single run per model and
   may be correlated.
3. **Pooled pair** (`pooled_pair_binomial`): logit p = α_b + δ_b·is_new + γ_item;
   δ_b = μ_δ + τ·z_b, μ_δ ~ Normal(0, 1.5), τ ~ HalfNormal(0.5) (sensitivity: 1.0);
   fixed item effects, first item per benchmark as reference.
All fits: NUTS, 4 chains, 2,000 samples, 1,200 warmup (synthetics 1,000/800).

## Validation
- **Synthetic recovery** (`synthetic_recovery.csv`): 5 simulated datasets per design
  shape (A=CY-HARD hier, B=CY-EASY fixed, E=GDM fixed, D=beta-binomial with true
  κ=15, C=pooled 2-benchmark). 63/65 true deltas inside the 94% HDI (96.9%; nominal
  94%), mean bias per design: A +0.038, B +0.015, C +0.007, D −0.013, E −0.008 logits.
  An earlier single-seed check flagged a 2.2σ miss — recovery is calibration across
  seeds, not single-draw luck.
- **Convergence** (`fit_*.diagnostics.json`): all real fits 0 divergences.
  Full-parameter max r_hat ≤ 1.0105; the 1.0105 (fit_cy_hard) sits on the additive
  model/item location ridge which cancels exactly in the reported contrast: every
  headline delta has r_hat ≤ 1.0006 and ESS_bulk ≥ 9,412 (`summary_deltas.csv`,
  delta_r_hat / delta_ess_bulk). Pooled μ_δ: r_hat 1.0004, ESS > 1,400.

## Findings
All rows in `summary_deltas.csv`; plots `forest_o1_o3.png`, `forest_sonnet4_opus41.png`.
Deltas are conditional (within-challenge) log-odds differences; "pp" = accuracy
change averaged over the class's challenges (equal weight). Observational data —
no causal claims.

1. **o1 → o3 (cybench hard, matched config): a large, unambiguous jump.**
   δ = +2.86 logits [94% HDI +2.23, +3.46], P(δ>0) = 1.00, P(δ>+0.5) = 1.00
   (binomial). Beta-binomial robustness (headline): δ = +2.66 [+1.74, +3.59] — same
   conclusion with honestly wider intervals; κ ≈ 3.6 [1.5, 5.9] indicates strong
   within-cell epoch correlation (ICC ≈ 0.22), so the binomial HDI is too narrow.
   Implied accuracy: 0.282 → 0.468, +19.1pp [+13.5, +25.1] (beta-binomial row).
2. **sonnet-4 → opus-4-1: real but small, and it replicates in direction.**
   cybench-easy δ = +0.27 [+0.05, +0.47], P(δ>0) = 0.99; gdm δ = +0.44 [+0.35, +0.52],
   P(δ>0) = 1.00. Both P(δ>+0.5) ≤ 0.08 — confidently *below* the +0.5-logit
   "meaningful" threshold per benchmark. Accuracy terms: +5.2pp [+1.2, +9.2] and
   +8.0pp [+6.4, +9.4].
3. **Cross-benchmark pooling is honest about generalisation: 2 benchmarks bound
   little.** Pooled μ_δ = +0.48 [−0.16, +1.09], P(μ_δ>0) = 0.94 (τ ~ HalfNormal(0.5));
   sensitivity τ ~ HalfNormal(1.0): μ_δ = +0.47 [−0.45, +1.41], P = 0.89
   (`pooled_stats.json`). The point estimate is stable but the interval quadruples
   vs the per-benchmark fits: with 2 benchmarks, τ is prior-dominated (posterior
   mean 0.40 vs 0.60 under the two priors) and "does it transfer to a new
   benchmark?" is genuinely unresolved even when each benchmark's delta is precise.
4. **Naive reads mis-state o1 → o3 in both directions at once.** Naive all-config
   accuracies (`naive_accuracies.csv`): o1 0.336 (n=1,570; inflated by 1,200 rows
   using a bespoke `best_agent_model_setup_o1` scaffold) → o3 0.606 (n=2,475; 2,075
   rows on the 2-challenge easy variant). Naive read: "+27.1pp" — overclaims the
   matched marginal effect (+18.3pp binomial / +19.1pp beta-binomial) by ~40%
   because o3's mix is easier. Meanwhile the naive log-odds diff (+1.12) understates
   the within-challenge capability shift (+2.66) by ~2.4x because o1's scaffold
   advantage masks it. Same number, two opposite errors.
5. **gpt-4o → gpt-5 cannot be honestly estimated from this warehouse, and the naive
   delta is mostly composition.** Zero shared configs on both benchmarks; zero shared
   items on cybench (naive: 0.132 → 0.765, +63.3pp across different scaffolds,
   sandboxes and challenge sets). On gdm (`naive_gpt4o_gpt5_decomposition.csv`):
   naive +27.8pp (0.534 → 0.812); restricting gpt-4o to the 14 modern-class items
   moves it to 0.708 — item mix alone accounts for +17.4pp (63%) of the naive delta;
   the remaining +10.4pp still confounds scaffold (vanilla_agent vs built-in agent)
   with the model upgrade.
6. **Config shifts rival release deltas** (`config_sensitivity_demo.csv`): the same
   gpt-5 scores 0.832 on CY-EASY vs 0.530 on CY-HARD (+30.2pp), o3 0.633 vs 0.468
   (+16.6pp) — larger than either sonnet-4→opus-4-1 delta and comparable to o1→o3.

## Caveats
- Observational warehouse data: run configs, dates and scaffolds are chosen by eval
  teams, not randomised. Matched-config classes remove the scaffold confound within
  a pair but cannot rule out e.g. inference-stack drift between run dates
  (CY-HARD runs span 2 days, CY-EASY/GDM-MAIN 3–4 days — small window).
- CY-HARD has exactly one run per model; run-level variance is not estimable and is
  confounded with the model effect. The beta-binomial absorbs within-run epoch
  correlation but not a run-level offset.
- CY-EASY covers 2 challenges — its delta is "on these two challenges", not
  "on cybench". The pooled model treats it as one benchmark-level draw.
- Deltas are conditional log-odds; marginal accuracy changes (also reported) are
  smaller by non-collapsibility. Do not compare conditional deltas across classes
  with different item-difficulty spreads.
- "o1" = `openai/o1` rows only; `openai/o1-2024-12-17` (n=1,282, bespoke agents
  only) has no matched-config rows and is excluded from the matched fit.
- P(δ > +0.5 logits) uses an arbitrary but pre-specified threshold; μ_δ pooling
  uses N=2 benchmarks and its τ posterior is prior-sensitive (documented above).
