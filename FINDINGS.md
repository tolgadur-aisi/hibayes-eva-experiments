# hibayes × eva: Bayesian analysis of the public team_ru eval slice

**Run 2026-07-16.** Five experiments on eva prod extracts (team_ru, public benchmarks:
cybench, gdm_intercode_ctf, swe_bench, boolq_preference), each independently reviewed by a
skeptical-statistician agent and iterated until approved (E1–E3: approved round 1;
E4, E5: approved round 2 after one revision each). Per-experiment details, code, and
reviewer verdicts in `experiments/e*/{FINDINGS.md,run.py,outputs/,REVIEW_ROUND_*.json}`.
Data: one-pass prod extract cached in `data/` (`shared/extract.py`); no experiment
touches the DB. Reproduce any experiment: `uv run python -m experiments.<id>.run`.

## Cross-cutting findings

### 1. Duplicate ingestion corrupts naive warehouse reads (eva product bug)
31–51% of public sample rows are byte-identical re-ingestions of the same eval log stored
under multiple S3 keys (same `eval_id`/`run_id`/`created`, new `internal_id`): cybench 39%,
intercode 31% (52% of multi-sample runs), swe_bench 51%. Any consumer averaging the
warehouse without deduping by `eval_id` double-counts half of swe_bench; pre-dedup, 86–92%
of "replicate run groups" have exactly zero score variance (copies, not reproducibility).
**Action for eva**: dedup at ETL (upsert key currently `evals.log` = S3 key) or expose a
deduped view. (E1 F8, E2 F6, E5 F1.)

### 2. Epochs are massively redundant on agentic benchmarks
Repeated attempts at the same challenge correlate strongly: ICC ρ = 0.78 [0.72, 0.82] on
cybench (per-model 0.63–0.87); intra-item ρ: intercode 0.41, swe_bench 0.56. Design effect
at 10 epochs ≈ 5–8: a 10-epoch cybench run of N=1,340 attempts carries effective N ≈ 190.
Concrete design rule: **for ability estimation, minimise epochs-per-challenge until every
challenge is covered; only then add epochs.** The warehouse's 848-attempts-on-2-challenges
designs leave ability SE at ±27 accuracy points — the same budget spread 40×10 gives ±6.
(E1 F2–F4, E5 F4.)

### 3. Run-to-run drift is real, provider-dependent, and ~4× binomial noise
Same model, same config, different launch: σ_run ≈ 1.0–1.1 logits on cybench (±2.7pp on
the benchmark score, 1 s.d.); same-config replicate runs swing 3.9× [2.8, 5.2] what
binomial noise predicts (cybench and intercode alike). It is provider-dependent: Azure-served
Mistral setups drift at σ_run ≈ 0.9 logits while gemma serving is stable at ≈ 0.08 — a 50×
difference in a run's effective sample size (28 vs 1,274 of nominal 790). gpt-5 rose
+12.2pp [7.7, 16.8] on a fixed 14-challenge set within a single September week. Single-run
scores inherit this floor regardless of sample count; boolq's reported stderr understates
run-to-run variance 3.3×. (E1 F5, E2 F2–F5, E5 F5–F6.)

### 4. Naive accuracy comparisons mislead in both directions at once
- o1's apparent cybench lead over claude-3-7 (non-overlapping-ish Wald CIs) collapses to
  P(o1 > claude-3-7) = 0.57 under challenge+run structure; single-run Wald CIs are ~2.5×
  too narrow. Attempt-weighting biases pooled accuracy by −3.6pp when epochs target hard
  challenges. (E1 F7.)
- The naive o1→o3 delta (+27.1pp) overstates the matched-config effect (+19.1pp) by ~40%
  (o3 ran an easier mix) while simultaneously understating the within-challenge capability
  shift 2.4× (o1's bespoke scaffold masks it). 63% of the naive gpt-4o→gpt-5 intercode
  delta is item-mix composition; the remainder still confounds scaffold. (E3 F4–F5.)
- Config shifts rival release deltas: the same gpt-5 scores +30.2pp higher on the easy
  cybench variant than the hard one — bigger than most release-over-release jumps. (E3 F6.)

### 5. The capability signal itself (with honest uncertainty)
- **o1 → o3 on cybench-hard (matched config): +2.86 logits [+2.23, +3.46]**; beta-binomial
  robustness +2.66 [+1.74, +3.59], P(δ>0.5 logits) = 1.00. A genuine, large jump. (E3 F1.)
- **sonnet-4 → opus-4-1: real but small** — +0.27 (cybench-easy) / +0.44 (intercode) logits,
  P(δ>0)≥0.99 both, but confidently *below* the +0.5-logit threshold. With only 2
  benchmarks, "does it transfer?" is genuinely unresolved (pooled μ_δ HDI −0.16..+1.09). (E3 F2–F3.)
- **Cyber capability velocity: +2.94 logits/yr [+1.22, +4.60] on cybench** (P=0.997;
  trend accuracy 0.109→0.662 over Aug-2024→Aug-2025), +0.92 [+0.00, +1.93] on the easier
  intercode; hard-cyber outpaces easy-cyber with P=0.98. Selection-free record-count tests
  agree (P=0.034/0.009). boolq-preference is flat (−0.30 [−2.25, +1.91]) — it measures
  model-idiosyncratic framing susceptibility, not monotone capability. swe_bench velocity
  is unidentifiable from this slice (3 models, 0.28yr span). (E4 F1–F5.)
- **No benchmark is saturated**: P(best ability > 0.90) = 0.00 on all four; best abilities
  0.44–0.68 on the agentic three; boolq watch-listed at 0.89. No deceleration evidence
  once record-selection artifacts are removed. (E5 F3, E4 F6.)

### Portfolio verdicts (E5 league table)
| benchmark | discrimination | headroom | overdispersion ρ | run repro | verdict |
|---|---|---|---|---|---|
| cybench | 7.6× (3.7× under betabin) | best 0.62 | 0.68 | 3.9× binomial | **noisy** |
| gdm_intercode_ctf | 14.4× | best 0.68 | 0.41 | 3.9× | **noisy** |
| swe_bench | 5.7× | best 0.44 | 0.56 | ~binomial (9 runs) | **underpowered** |
| boolq_preference | high (eval-level) | best 0.89 | — | 3.3× stated se | **noisy** |

### Judge-effects analysis: not feasible on the public slice (confirmed)
Every public-task scorer is rule-based (includes, pattern, swe_bench_scorer, boolq_scorer).
Zero identifiable judge variance. Needs internal long-form tasks with `rubric_scorer/<judge>`
headline scorers — ideally same transcripts re-scored by ≥2 judges. (E5 F8.)

## Method/meta notes
- hibayes default variance priors (σ scale ~0.1) over-shrink badly at the challenge spreads
  agentic benchmarks actually have (σ_c ≈ 3–9 logits): every experiment had to widen priors
  and validate via synthetic recovery. Worth upstream defaults discussion + an eva loader.
- The adversarial review loop caught one genuine statistical error before it shipped:
  round-1 frontier "P>0.999" trends were selection-biased by construction (record paths
  rise under any null); replaced with exact record-count tests. This is a concrete
  argument for check-gated automation in any eva statistical layer.
- All headline fits pass r_hat ≤ 1.01 / ESS ≥ 400 / 0-divergence gates on reported
  quantities, with documented, verified justifications where global worst-cases sit on
  non-identified nuisance modes. 100+ MCMC fits total, incl. ~60 synthetic-recovery fits.

## Standing caveats
Observational warehouse data throughout: scaffolds, solvers, sandboxes, limits and dates
vary with the model; "model" effects are model+scaffold packages; run effects quantify but
do not attribute drift; no causal claims. Public slice only (team_ru); RLS means other
teams' data is invisible here. Extraction snapshot: 2026-07-16.
