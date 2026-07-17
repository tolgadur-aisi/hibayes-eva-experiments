"""FINDINGS.md content for E1 (required by the experiment output contract).

Run `python -m experiments.e1_cybench.findings_text` to (re)write FINDINGS.md.
Kept as code so the deliverable is reproducible alongside run.py outputs.
"""

from pathlib import Path

FINDINGS = """\
# E1 — cybench: epochs-vs-samples + model comparison

Reproduce: `uv run python -m experiments.e1_cybench.run` (~7 min, 15 MCMC fits + 8 validation
fits), then `uv run python -m experiments.e1_cybench.findings_text` for this file.
All numbers below are traceable to `outputs/*` (file named per finding).

## Question

On AISI's cybench results: (1) what does the per-challenge difficulty spectrum look like per
model; (2) how reliable are repeated epochs (ICC / design effect), and for a fixed budget of N
attempts, what epochs-per-challenge k minimises uncertainty about model ability; (3) what does a
partial-pooling model comparison say that naive accuracy ± Wald does not?

## Data

`data/cybench.samples.parquet` (66,694 rows; team_ru, status=success). Filters, in order
(`outputs/data_audit.json`):

- Dropped 3,016 rows with no parseable score — these coincide with infra errors (2,814 Helm
  install failures, mostly one o3-mini batch; environment never started, so missing data, not
  model failure). 2 rows have an error string *and* a score → kept.
- **Kept** all 41,468 scored rows that hit a limit (token/message/time/operator): the scorer
  graded them; hitting the budget without the flag is a genuine failure, not missing data.
- item_id encodes challenge+variant in 3 conventions (`avatar (hard)`, `avatar-hard`,
  `avatar (cybench_hard)`) with 5 spelling aliases (`lock_talk`=`locktalk`, `emaze`=`ezmaze`, …)
  — normalised in `prep.py`; 45 challenges total.
- **Dedupe**: 185 whole runs (24,713 rows, 39% of scored rows!) are byte-identical re-ingestions
  of another run (same model/created/scores/tokens; e.g. run 297551 ≡ 123360) — dropped.
- **Degenerate harness runs**: claude-3-7 runs 124615 & 131722 (440 attempts, 2025-06-23) solved
  0/40 challenges incl. ones it solves at ~88% elsewhere; every sample died at message_count=2
  with a token limit — broken agent loop, excluded (rule + mechanism in `prep.py::hard_focal`).
  No other model triggers the rule.
- Headline set: **hard variant, 10 focal models, 19,677 attempts, 40 challenges × all 10
  models** (`outputs/data_summary.csv`). Easy-variant-only models (opus-4-1, sonnet-4, o4-mini:
  2–8 challenges, single-epoch repeat runs) are excluded from fits; used as a design
  illustration. o1 and o1-2024-12-17 are kept separate (different scaffolds/dates).

## Models (specs + priors)

All binomial GLMMs on aggregated counts, NUTS, 4 chains (heavy fits: 2000 draws / 1500 warmup /
target_accept 0.9; Q1 fits: 1500/1000).

- **Q1, per model** (`hibayes` `simplified_group_binomial_exponential`): logit p = μ + a_chal,
  a ~ N(0, σ_c); μ ~ N(0, 2), σ_c ~ Exponential(0.5). hibayes defaults (σ scale 0.1)
  over-shrink badly at the true σ_c ≈ 5–9 — priors widened deliberately.
- **Q2** (`models.py::challenge_run_binomial`): crossed logit p = μ + a_chal + b_run on
  claude-3-7 (702 cells, 54 runs); a centred (Σa=0), σ_c ~ HalfNormal(2), σ_r ~ HalfNormal(1).
- **Q3** (`models.py::model_challenge_run_binomial`): logit p = ability_model + a_chal + b_run,
  10 independent ability ~ N(0,2) (no pooling across models, so no over-shrink), shared centred
  challenge effects and run effects (2,568 cells, 127 runs).

## Validation

- **Synthetic recovery** on the real cell designs (`outputs/synth_recovery.csv`, 28 checks,
  fixed seeds): 27/28 pass. Q1 shape (40×10): μ, σ_c, effect corr 0.87 (the information ceiling
  at k=10), 34/40 effects in 94% HDI. Q2 shape ×3 replicates: σ_run recovered in 3/3 (single
  realizations can miss — weakly identified; hence replicates). Q3 shape: 9/10 abilities in HDI
  (1 miss by 0.06 logits from the unweighted-run-mean target approximation), σ_c, σ_r in HDI,
  Spearman 0.95, 31/32 big-gap (>0.5 logit) pairs recovered with P>0.8, **0 confidently wrong**.
- **Convergence** (`outputs/run_manifest.json`, per-fit `*.diagnostics.json`): 0 divergences in
  all 15 fits. Every *reported* quantity: r_hat ≤ 1.008, ESS ≥ 450. Raw z_challenge latents in
  the centred fits carry a non-identified mean-mode (cancels in all reported quantities);
  including them, worst r_hat = 1.018 / ESS 376 — see `fitting.py::reported_diagnostics`.
- **Design-curve validation**: closed-form SE matches 200k-rep Monte Carlo at k=10 and k=100
  (0.062/0.062, 0.193/0.192; `outputs/q2_formula_mc_check.csv`); 8 MCMC refits on synthetic
  designs give posterior SDs consistent in size and ordering (0.060 at k=10/c=40 vs 0.133 at
  k=100/c=4; `outputs/q2_design_mcmc_validation.csv`).

## Findings

Intervals are 94% HDIs; SEs are on the accuracy scale.

1. **cybench is nearly all-or-nothing per model.** Latent challenge spread σ_c = 4.9–9.1 logits
   per model; 27.5% (gpt-5) to 82.5% (gemma) of the 40 challenges sit at solve prob < 0.05, and
   only 2–11 challenges per model are "coin-flips" (0.2<p<0.8). Frontier models have the
   flattest spectrum (o3: 11 coin-flips; gpt-5: 7 + 14 at p>0.9).
   (`outputs/q1_model_sigma_icc.csv`, `q1_challenge_probs.csv`, `q1_difficulty_spectrum.png`.)
2. **Epochs on the same challenge are highly redundant.** Observed-scale ICC ρ = 0.78
   [0.72, 0.82] (claude-3-7 crossed model; per-model Q1 range 0.63–0.87). Design effect
   1+(k−1)ρ: k=10 epochs → 8.0; k=100 → 78. (`outputs/q2_icc_summary.json`.)
3. **For a fixed budget, minimise k.** At N=400 (superpopulation of cybench-like challenges):
   SE = 0.022 at k=1/c=400, 0.062 at k=10/c=40, 0.194 at k=100/c=4. For the *fixed 40-challenge
   benchmark score*, cover all 40 (k=10): SE 0.0075 vs 0.204 at k=100/c=4 — a 27× penalty for
   concentrating the same budget on 4 challenges. Optimal k is the smallest that covers the
   pool; k>1 only pays once every challenge is covered. (`outputs/q2_design_curve.csv`, `.png`.)
4. **The 2-challenge mega-epoch designs in the warehouse are uninformative about ability.**
   An opus-4-1-style design (848 attempts on 2 challenges) has ability SE ≈ 0.27 — after 848
   attempts the general-ability estimate is still ±27 accuracy points (1 s.d.); 400 attempts
   spread 40×10 give ±6.2. (`outputs/q2_icc_summary.json:
   se_848_attempts_2_challenges_median`.)
5. **Run-to-run drift is real and non-trivial.** σ_run = 1.10 [0.84, 1.37] logits (claude-3-7,
   54 runs spanning Feb–Jun 2025 configs); joint-model σ_run = 1.03 [0.83, 1.23]. On the
   accuracy scale the benchmark score moves ±2.7 [2.1, 3.6] points (1 s.d.) between runs of the
   same model — single-run scores inherit this floor on top of sampling noise.
   (`outputs/q2_claude37_crossed.idata.nc`, `q2_icc_summary.json`, `q3_joint.idata.nc`.)
6. **Ranking (benchmark-mean accuracy, joint model)**: gpt-5 0.52 [0.40, 0.64] > o3 0.47
   [0.35, 0.58] > claude-opus-4 0.39 [0.30, 0.49] > o1 0.33 [0.26, 0.39] ≈ claude-3-7 0.32
   [0.31, 0.34] > o1-2024-12-17 0.25 [0.23, 0.28] > o3-mini 0.22 [0.20, 0.25] > gpt-4o 0.14 >
   mistral-large 0.11 > gemma-3-27b 0.09. P(gpt-5 > o3) = 0.71, P(gpt-5 > claude-opus-4) =
   0.94, P(o3 > claude-opus-4) = 0.84. (`outputs/q3_model_abilities.csv`,
   `q3_p_model_row_beats_col.csv`, `q3_forest.png`.)
7. **Naive accuracy ± Wald misleads in both directions.** (a) Overconfident: single-run models'
   honest HDIs are ~2.5× the Wald CI (gpt-5: ±0.12 vs ±0.047), and o1's apparent lead over
   claude-3-7 (0.325 [0.297, 0.353] vs 0.288 [0.278, 0.298], barely overlapping) collapses to
   P(o1 > claude-3-7) = 0.57 once challenge and run structure is modelled. (b) Biased:
   claude-3-7's pooled accuracy is 0.288 but its benchmark-mean is 0.324 [0.31, 0.34] —
   attempt-weighting drags it 3.6 points low because its 100-epoch runs targeted its hardest
   challenges. Meanwhile claude-3-7 > o1-2024-12-17 with P = 1.00 despite overlapping Wald CIs.
8. **Warehouse data-quality by-catch**: 39% of scored cybench rows are exact re-ingestion
   duplicates (185 runs), and one broken claude-3-7 batch (440 zero-solve attempts,
   message_count=2) would bias any unfiltered analysis. (`outputs/data_audit.json`.)

## Caveats

- Observational warehouse data: scaffolds, limits and sandboxes differ across runs and models
  (gemma/mistral ran a different agent than the OpenAI/Anthropic runs; max_messages varies
  30–10000). "Model" effects = model+scaffold packages; no causal claims. Run effects absorb
  within-model drift only.
- gpt-5, o3 and claude-opus-4 have a single hard-variant run each — their intervals are honest
  but their point estimates are unprotected against a bad/good run draw (finding 5 sizes this).
- The joint model is additive on the logit scale (no model×challenge interaction); Q1 per-model
  fits show spectra differ by model, so ranking summaries are panel-average statements.
- Q1 per-model fits ignore run structure; for multi-config models (claude-3-7, o3-mini) some
  run variance loads onto σ_c (joint σ_c 4.9 vs claude-only Q1 σ_c 9.1 brackets this).
- σ_run partially confounds with challenge subsetting in epoch-heavy small-panel runs;
  synthetic replicates recover it, but single realizations can under-cover (see Validation).
- Latent-scale ICC (0.92) is prior-sensitive when many challenges are near-deterministic; the
  observed-scale ρ (0.78) and the accuracy-scale curves are the robust quantities.
- Design curves are computed from claude-3-7's difficulty distribution at a typical run; other
  models' ρ ranges 0.63–0.87, shifting the curves but not the ordering in k.
"""


def main() -> None:
    path = Path(__file__).resolve().parent / "FINDINGS.md"
    path.write_text(FINDINGS)
    print(f"wrote {path} ({len(FINDINGS.splitlines())} lines)")


if __name__ == "__main__":
    main()
