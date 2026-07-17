# E5 — Portfolio statistical-health audit of the public team_ru slice

## Question
For each of cybench, gdm_intercode_ctf, swe_bench, boolq_preference: does the benchmark
(1) discriminate models beyond noise, (2) have frontier headroom, (3) behave like iid binomial
across challenges (overdispersion), (4) reproduce across same-config repeat runs — and what is
the per-benchmark verdict? Also: is an LLM-judge variance analysis feasible on this slice?

## Data (rows, filters, exclusions)
Parquet extracts of 2026-07-16 (team_ru, status=success, no replay/mock models). All counts in
`outputs/data_accounting.csv`; per-model inclusion in `outputs/model_inclusion_*.csv`.

**Duplicate ingests (headline hygiene finding).** The warehouse holds the same eval log under
multiple `internal_id`s (same `eval_id`/`run_id`/`created`; the log file was ingested once per
S3 key it was copied to). Deduped by keeping the lowest `internal_id` per (task, `eval_id`):
cybench −26,179/66,694 sample rows (39.3%); gdm_intercode_ctf −25,449/83,078 (30.6%);
swe_bench −11,171/22,063 (50.6%, runs 232→109); boolq_preference −77/1,967 eval rows.
Pre-dedup replicate-group accounting (used by Finding 1): `outputs/run_repro_groups_prededup.csv`.

After dedup: rows with non-empty `error` (harness/API failures, no score) dropped as missing
data — cybench 1,509, gdm 90, swe_bench 1,051. Rows that hit a limit but were scored are KEPT
as genuine failures (budget exhaustion ≠ missing): cybench 24,579 (63% of clean rows, mostly
token), gdm 17,683, swe_bench 2,470. No non-binary scores; no duplicate (run,item,epoch) rows.
Final: cybench 39,006 rows / 6,507 runs / 49 models / 134 items; gdm 57,539 / 35,848 / 15 / 79;
swe_bench 9,841 / 109 / 4 / 61. Models with <50 samples excluded from fits (cybench 49→45,
swe_bench 4→3). boolq_preference is eval-level only and only the `pattern`-scorer runs carry
value+stderr → 1,890 runs, 7 models, 74 charity-framing conditions × 2 splits.
Config heterogeneity (cybench: 101 distinct task_args) is pooled by the crossed model and
isolated for reproducibility via exact-config replicate groups (same model + task_args + solver
+ run_epochs + item set, runs ≥20 samples): cybench 12 groups/38 runs, gdm 5/21, swe 4/9.

## Models (specs + priors)
One fit at a time; 4 chains × 2000 samples (1500 warmup), target_accept 0.99; binomial
aggregation throughout. Custom numpyro models (`models.py`):
1. `crossed_binomial` (per benchmark): (model,item) cells; logit p = model_m + item_i;
   model effects FIXED Normal(0,2) (few models → partial pooling would over-shrink, the known
   two_level trap); item effects non-centred N(0, sigma_item), sigma_item ~ HalfNormal(3),
   sum-to-zero centred (identifies model effects at the item-pool average). Identifiability:
   with many never-solved items the pool-mean logit — a common intercept offset — is only
   weakly identified; all reported metrics are offset-invariant (spread across models,
   probability-scale abilities, rho), established by the harsh synthetic scenario below.
2. `run_repro_binomial`: replicate-group fixed means Normal(0,2.5) + run effects
   N(0, sigma_run), sigma_run ~ HalfNormal(1) on the logit scale.
3. `boolq_normal` (eval-level normal approximation, documented): value_e ~
   N(theta_m + b_split + delta_c + gamma_{m,c}, se_e² + sigma_run²); delta_c ~ N(0, σ_common),
   gamma row-centred with per-model scale sigma_char_m ~ HalfNormal(0.3); theta = ability in the
   typical condition at the reference (validation) split. Values ≤0.91 with se≈0.005 → boundary
   truncation negligible.
4. `crossed_betabinomial` (sensitivity only, cybench): same mean structure, beta-binomial cells
   with shared intra-cell correlation rho_cell ~ Beta(1,3), absorbing the run-level
   overdispersion the binomial cells ignore. p smoothly floored to [1e-3, 1−1e-3] to bound the
   p→0 curvature that caused 197/8000 divergences unfloored; the floored fit matches the
   unfloored one (disc ratio 3.69 vs 3.72, both in run.log) with 0 divergences.

## Validation
**Synthetic recovery — 13/13 checks pass** (`outputs/synthetic_recovery.csv`):
- Crossed, mild regime (8 models, Gaussian items, rho 0.21): effect corr 0.995, 8/8 HDI
  coverage, sigma_item and rho in 94% HDI.
- Crossed, HARSH regime matching real cybench (45 models, bimodal item pool, ~49% items never
  solved, true rho 0.69, 70% missing cells — the regime stressing the nuisance z_item tails):
  effect corr 0.994; centred model effects 39/45 covered; item-marginal abilities 41/45; rho
  0.681 [0.670, 0.691] vs true 0.688. Raw intercepts are not an identified estimand here
  (never-solved item depths trade off against a common offset; posterior-mean offset +0.22
  logit) — all reported estimands are offset-invariant.
- Run repro: sigma_run=0.20 recovered (0.232 [0.175, 0.294]); the null (sigma_run=0) passes a
  prior-vs-posterior concentration criterion, P(σ<0.05)_post / P(σ<0.05)_prior = 0.429/0.040
  = 10.8 ≥ 10 (posterior mean 0.060 [0.000, 0.121]).
- boolq: thetas 5/5 covered; sigma_run 0.0195 [0.0182, 0.0210] vs true 0.02; per-model
  condition-scale corr 0.98.

**Convergence** (global: `outputs/*.diagnostics.json`; per-parameter headline stats:
`outputs/headline_param_diagnostics.csv`; derived metrics: `outputs/derived_diagnostics.csv`):
- All real-data repro fits (incl. the two target_accept refits) pass globally across every
  parameter: max r_hat ≤ 1.0028, min ESS ≥ 2,181. Synth fits: global max r_hat ≤ 1.0051
  except the harsh scenario (below).
- All four crossed/boolq fits (including swe) exceed the 1.01 gate in their GLOBAL worst case,
  confined to nuisance non-centred z's for items/conditions solved by all or no models
  (unbounded logits): global max r_hat 1.017 (cybench), 1.037 (gdm), 1.012 (swe), 1.013
  (boolq). Headline parameters pass: model_effects max r_hat/min ESS 1.0094/463 (cybench),
  1.0048/656 (gdm), 1.0016/1564 (swe), 1.0010/8828 (boolq); swe sigma_item 1.0033/780; boolq
  sigma_char 1.0093/614, sigma_run 1.0005/8226. Exception disclosed: gdm sigma_item r_hat
  1.019, ESS 337 — not a reported quantity. Every league-table derived metric has r_hat ≤
  1.0063 and ESS ≥ 497.
- swe_repro had 1 divergence in 8,000 draws (sigma_run→0 funnel tip). Artifact-backed
  sensitivity (`outputs/swe_repro_target_accept_sensitivity.csv`): target_accept 0.99/0.95/0.97
  give sigma_run 0.118/0.119/0.120 with HDIs ≈ [0.00, 0.31] in all three (divergences 1/1/0).
  Every other reported fit has 0 divergences, including the stabilised betabin fit (global max
  r_hat 1.0082, min ESS 544).
- Harsh synth fit: global max r_hat 1.031 (nuisance z_item, by design); headline model_effects
  1.0033/1023.

## Findings (94% HDIs; league table: `outputs/health_league_table.csv`)
1. **Duplicate ingestion corrupts naive warehouse analyses.** 31–51% of public sample rows are
   literal copies. Pre-dedup (`outputs/run_repro_groups_prededup.csv`; note that table uses
   runs ≥10 samples, vs ≥20 for the post-dedup reproducibility fits), cybench had 87 "replicate
   groups" (222 runs) with sd_acc = 0 in 86.2% of groups — copies, not reproducibility;
   swe_bench 50 groups/117 runs, 92% sd=0. Post-dedup only 12 genuine cybench groups (38 runs)
   remain. Any consumer averaging the public slice without deduping by `eval_id` double-counts
   half of swe_bench.
2. **Discrimination — all four benchmarks separate models beyond single-model noise; the
   ratios are model-conditional upper bounds.** Posterior spread of model effects ÷ median
   single-model posterior SD (scale annotated per row in the league-table CSV): cybench 7.6
   [6.7, 8.5]; gdm 14.4 [13.3, 15.6]; swe_bench 5.7 [4.9, 6.4] (logit scale); boolq 112
   [112, 113] (probability scale — eval-level SEs are tiny; not comparable to the logit rows).
   These condition on the binomial cell likelihood, which ignores the run-level overdispersion
   measured in Finding 5. Sensitivity on cybench (`outputs/sensitivity_betabin_cybench.csv`):
   beta-binomial cells (rho_cell 0.255 [0.215, 0.290]) widen the median per-model ability SD
   0.335 → 0.460 logit and halve the ratio to 3.7 [3.0, 4.3] — still ≫ 1, so the qualitative
   conclusion stands.
3. **Saturation — none.** Best item-marginal ability: cybench gpt-5 0.617 [0.578, 0.656];
   gdm gpt-5 0.677 [0.667, 0.687]; swe_bench o1-2024-12-17 0.444 [0.415, 0.473]; boolq
   gpt-4o-2024-08-06 0.893 [0.891, 0.895]. P(best > 0.90) = 0.00 everywhere (robust to the
   cybench betabin sensitivity: 0.567 [0.481, 0.651]); boolq is within ~1pp of the 0.90 flag
   and its two top models sit at 0.87–0.89 — watch-listed, not saturated.
4. **Overdispersion is severe on all agentic benchmarks.** Intra-item correlation rho:
   cybench 0.68 [0.64, 0.72] (betabin sensitivity 0.61 [0.56, 0.66]); gdm 0.41 [0.40, 0.42];
   swe_bench 0.56 [0.53, 0.59]. Naive SE = sqrt(p(1−p)/N) is only valid at 1 epoch/item; at 10
   epochs the design effect is cybench 7.1, gdm 4.7, swe 6.0 — e.g. a 10-epoch cybench run
   (N=1,340) has effective N ≈ 190. Repeating epochs buys far less precision than adding items.
5. **Run reproducibility — cybench and gdm runs swing ~4× more than binomial.** Same-config
   sigma_run (logit): cybench 0.57 [0.39, 0.78] → ±6.2pp extra run-level SD, total swing
   3.9× [2.8, 5.2] the binomial SE at the median replicate run (n=400, acc≈0.13; effective
   n ≈ 28); gdm 0.27 [0.17, 0.37] → ±6.7pp, 3.9× [2.6, 5.3] (n=790; effective n ≈ 55) —
   driven partly by two Mistral-Large/CAST groups with 11–14pp swings (empirical ratios 7.3–7.9,
   `outputs/run_repro_groups_*.csv`). swe_bench is consistent with binomial: sigma_run 0.12
   [0.00, 0.31], inflation 1.27 [1.00, 1.98], but only 9 runs in 4 groups.
6. **boolq run noise is 3.3× the reported stderr.** sigma_run 0.0170 [0.0163, 0.0177] vs median
   reported se 0.0054 → a single run's naive CI understates run-to-run variance by 3.3×
   [3.2, 3.4]. Separately, models differ enormously in charity-framing sensitivity
   (condition SD, all 74 conditions covered per model): claude-3-7-sonnet-20250219 ≈ 0.22,
   claude-sonnet-4 0.11, opus-4-1 0.07 vs gpt-4o-2024-08-06 0.002 — the eval measures a
   model-dependent mixture of capability and framing susceptibility.
7. **Verdicts** — cybench: **noisy** (great discrimination and headroom, but 3.9× run noise and
   rho 0.68); gdm_intercode_ctf: **noisy** (same pattern at 3.9×); swe_bench: **underpowered**
   (3 usable models, 109 runs after 50% dedup, 9 replicate runs; binomial-consistent otherwise);
   boolq_preference: **noisy** (run noise 3.3× stated stderr; near-ceiling for GPT-family;
   strong condition confound).
8. **Judge-effects analysis is NOT feasible on this slice — confirmed.** The scorer inventory
   (`data/scorer_inventory.json`) shows every public-task scorer is rule-based: `includes`
   (cybench, gdm), `swe_bench_scorer`, `boolq_scorer`/`pattern` (+ a few null-scorer runs).
   There is no model-graded scorer, hence zero identifiable judge variance. Enabling it needs
   internal long-form tasks scored by `rubric_scorer/<judge-model>` (or any model_graded_*
   scorer) with the judge identity recorded, ideally with the same transcripts re-scored by ≥2
   judges.

## Caveats
- Observational warehouse data: configs, scaffolds, solvers and time all vary across runs;
  model "abilities" are as-run portfolio averages, not controlled comparisons — no causal
  claims. Within-cell overdispersion beyond the item effect is unmodelled in the binomial
  fits: ability HDIs and disc-ratio denominators (Findings 2–3) are optimistic — ~2× on
  cybench per the betabin sensitivity — so read league disc ratios as model-conditional upper
  bounds. gdm/swe/boolq got no sensitivity fit; their headroom over 1 (14.4/5.7/112) is large
  enough that a comparable halving changes no verdict. sigma_run absorbs config-invisible
  drift (temperature, provider updates) only within exact-config groups.
- Replicate-group counts are small post-dedup (4–12 groups); sigma_run pools across groups and
  is dominated by the models teams happened to re-run (gemma, Mistral, claude-3-7, o3-mini).
- Item-marginal abilities are counterfactual means over each benchmark's full observed item
  pool; models evaluated on subsets are extrapolated via item effects. Raw logit-scale
  intercepts (`abilities_*.csv` logit_mean) additionally carry the weakly identified pool-mean
  offset discussed under Models — compare them only as contrasts.
- boolq "pattern" accuracy under charity framings is a persuasion-style metric; the normal
  approximation ignores [0,1] truncation (max fitted mean 0.89, se ~0.005 — negligible).
- deff_realized uses the median realised epochs/item (1.0 for cybench/gdm, 2.54 for swe);
  the Deff=10-epoch column is a scenario, teams do run 10-epoch cybench sweeps.
- 45 cybench "models" include 30+ uk-dsit fine-tune variants of gpt-4o clustered at p≈0.1–0.25;
  they widen the discrimination spread modestly but are genuinely distinct checkpoints.
