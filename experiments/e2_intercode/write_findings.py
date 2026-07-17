"""Emit FINDINGS.md (required by the experiment output contract).

    uv run python -m experiments.e2_intercode.write_findings
"""

from pathlib import Path

CONTENT = """\
# E2 — gdm_intercode_ctf: variance decomposition at warehouse scale

## Question

Decompose intercode-ctf outcome variance into model / challenge / run-within-model /
epoch-residual components; quantify run-to-run drift beyond binomial noise ("is our infra a
noise source?"); derive the effective sample size of a standard run. All estimates from eva
warehouse extracts (team_ru, status=success), observational — no causal claims.

## Data (`data/gdm_intercode_ctf.samples.parquet`; counts in `outputs/data_log.json`)

- Raw: 83,078 sample rows, 15 models, 79 challenges, Apr–Sep 2025.
- **Excluded**: 180 rows with a sample `error` (all openai/gpt-4o 429 quota errors, unscored).
- **Kept**: 21,685 rows with a `limit` hit (token/message/time/operator), scored as recorded —
  exhausting the task resource budget is a genuine failure mode, not missing data (394 of these
  still scored C: flag found before the limit).
- **Duplicate ingestion**: 33 of 64 multi-sample runs are byte-identical re-ingestions of the
  same eval log under a new `run_internal_id` (identical per-(item,epoch) scores AND token
  counts, asserted in `prep.py`). Collapsed 64 -> 31 unique runs. The 35,814 single-sample
  fan-out runs are clean. Clean rows: 57,539.
- Scores: `score_includes` C/I -> 1/0. No duplicate (run, item, epoch) rows. Accuracy is flat
  in epoch number (0.556–0.582, no trend; `outputs/epoch_check.csv`).
- **D1 (primary)**: the one homogeneous config (`task_args={}`, solver `intercode_agent`):
  4 models (gemma-3-27b, Mistral-Large-2411 via two Azure providers, ML-2411-CAST), 12 runs
  (5/2/2/3), 79 challenges x 10 epochs = 948 cells, 9,480 samples (Jun 2025).
- **D1X (sensitivity)**: + all scaffold variants (max_messages 50/99999, prompt_var_2/3,
  gpt-4o vanilla_agent): 5 models, 9 model x variant levels, 26 runs, 19,829 samples.
- **D2 (frontier)**: fan-out campaign Sep 17–21 2025 (claude-opus-4.1, claude-sonnet-4, gpt-5,
  o3, o4-mini; 14 challenges, 1 sample per run) aggregated to 22 model x UTC-day batches,
  210 cells, 35,814 samples. Day-batch is the run analogue (run ids are per-attempt there).
- Excluded from D1/D1X: 5 vllm models (1 run each, `chain` solver — run confounded with model).

## Models (`vcmodels.py`; priors in `run.py::PRIORS`)

Cross-classified logistic-binomial random effects on cell counts, all non-centred:
`logit p = mu + a_model + b_challenge + g_model.challenge + u_run`; `n_correct ~ Binomial(n, p)`.
Priors: mu~N(0,1.5); HalfNormal sigmas: model 1.5, challenge 2.0, interaction 0.75, run 0.75
(variant 0.75 in D1X's extra `v_variant` term; per-model sigma_run in the `hetrun` variant).
Epoch residual = latent-logistic pi^2/3. Shares are finite-population (var of realized effects
per draw, ddof=1) over total. NUTS: 4 chains x 1500 draws, warmup 1500, target_accept 0.99.

## Validation

- **Synthetic recovery** (`outputs/synthetic_recovery.csv`), design copied from D1:
  - A (sigma_run=0.35): all components recovered (finite-pop SDs in 94% HDI) except run SD
    marginally shrunk (realized 0.315 vs HDI [0.12, 0.31]) — sigma_run estimates are, if
    anything, slightly conservative.
  - B (sigma_run=0): no phantom run variance (posterior 0.07, HDI upper 0.16; HDI cannot
    contain the boundary 0 exactly — mass piles at 0 as it should).
  - C (heterogeneous run SD 0.7/0.1/0.1/0.7): pooled fit recovers the pooled finite-pop run SD
    and leaves other components unbiased; per-model fit recovers all four sigma_run in HDI.
- **Convergence** (`outputs/*.diagnostics.json`): d1_primary r_hat<=1.006, ess_bulk>=753,
  1 divergence/6000; d1_hetrun 1.008/743/0; d2 1.003/2000/0; synths 0 divergences.
  d1x_variants: r_hat 1.027 / ess 175 on the global intercept `mu` only (location trade-off
  with unconstrained random effects); every variance component has r_hat<=1.007, ess>975, and
  shares are location-invariant — accepted for this sensitivity fit.
- **Model-free cross-check** (`outputs/dispersion_check.csv`): between-run variance of per-item
  correct counts vs exact binomial null (MC, 2000 reps): ML-2411(azureai) ratio 3.15 (p<5e-4),
  ML-2411-CAST 2.94 (p<5e-4), gemma 1.10 (p=0.17), ML-2411(mistralazure) 0.91 (p=0.64).

## Findings (posterior mean [94% HDI]; files: `variance_shares.csv`, `sigma_run_per_model.csv`, `design_effects.csv`, `claims.json`)

1. **Challenge identity dominates.** D1 logit-variance shares: challenge 67.4% [62.6, 72.0]
   (sigma_item 3.05 [2.54, 3.59]); epoch residual 24.1% [20.7, 27.4]; model x challenge 3.9%
   [2.9, 5.1]; run 3.2% [1.9, 4.6]; model 1.4% [0.0, 4.3] (these 4 models are similar-strength;
   not a general claim about models). Frontier (D2): challenge 47.8% [37.5, 59.7], model 5.9%
   [1.8, 10.4], model x challenge 7.7% [5.0, 10.4], day-batch 1.3% [0.5, 2.2], residual 37.4%.
2. **Run-to-run drift is real and provider-dependent.** Pooled sigma_run = 0.70 [0.40, 1.01]
   logits, P(sigma_run>0.3)=1.00. Per-model (hetrun fit): gemma 0.08 [0.00, 0.22]
   (P(<0.2)=0.93, 5 runs); ML-2411(azureai) 0.92 [0.37, 1.55] (2 runs); ML-2411-CAST 0.95
   [0.46, 1.55] (3 runs); ML-2411(mistralazure) 0.53 [0.00, 1.24] (2 runs — posterior close to
   the HN(0.75) prior, i.e. weakly identified; the model-free check for it is binomial-consistent).
   P(sigma_run > gemma's): azureai 0.999, CAST 0.999. Same benchmark, same config, days apart:
   the Azure-served Mistral setups moved by ~0.9 logits (~8–17pp on run mean score,
   `sd_between_pp` in design_effects.csv) while gemma serving was stable.
3. **Known scaffold changes ~ same order as unexplained drift.** D1X: sigma_variant 0.37
   [0.00, 0.80] (share 1.1% [0.0, 3.1]) vs run share 1.9% [1.1, 2.7]. Message-limit and prompt
   variants move scores about as much as an unexplained relaunch does.
4. **Frontier campaign drifts within a single week.** sigma_day = 0.35 [0.20, 0.51],
   P(>0.2)=0.99 (D2). Sharpest case: gpt-5 accuracy on the fixed 14-challenge set rose
   +12.2pp [7.7, 16.8] from Sep 17 to Sep 20 (P>0 = 1.00) inside one campaign — same model id,
   same task config in the warehouse.
5. **Effective N of a run varies ~50x with serving stability** (fixed-item estimand: precision
   of a fresh run's mean score vs iid Bernoulli). Stable gemma: one 79x10 run is worth
   N_eff = 1274 [370, 1866] — MORE than its nominal 790 (heterogeneous fixed items act as
   stratification; at K=1, 168 [133, 191] vs nominal 79). Drifting setups: azureai 28 [4, 60],
   CAST 26 [5, 53] of nominal 790 — a 790-sample run carries the information of ~25 iid samples,
   and the run-noise ceiling (~22, K->inf) means epochs beyond ~3 buy nothing. Frontier
   day-batches (14 items, K=10): opus-4.1 70 [33, 108], sonnet-4 70 [32, 104], gpt-5 54
   [26, 80], o3 36 [16, 57], o4-mini 49 [21, 73] of nominal 140.
6. **Warehouse hygiene**: 52% of multi-sample intercode runs in eva are duplicate ingestions
   (33/64). Any variance analysis skipping dedup would badly understate run variance and
   overstate reliability.

## Caveats

- Observational warehouse data. "Run effect" bundles endpoint version, infra load, sandbox
  image drift, and anything else that changed between launches — we quantify, not attribute.
- D1 rests on 12 unique runs across 4 models; 2-run models give prior-influenced per-model
  sigma_run (flagged above). sigma_model from 4 groups is weakly identified (share HDI ~[0, 4%]).
- Synthetic recovery shows mild downward shrinkage of the run SD -> drift estimates are more
  likely under- than over-stated.
- N_eff is for the fixed-79-item estimand; generalising to a challenge population would add the
  dominant challenge variance and shrink every N_eff drastically. N_eff > nominal is the
  stratification effect, not free information.
- Epoch residual uses the latent-logistic pi^2/3 convention; shares depend on that scale choice.
- D2 day-batches are a UTC-date proxy for logical launches; within-day batch structure is unmodelled.
- Frontier models ran only 14 of 79 challenges with a different scaffold (max_attempts=3,
  max_messages=50) — D1 and D2 numbers are not directly comparable.

Reproduce: `uv run python -m experiments.e2_intercode.run` (rebuilds all of `outputs/`; ~25 min).
"""


def main():
    path = Path(__file__).resolve().parent / "FINDINGS.md"
    path.write_text(CONTENT)
    print("wrote", path)


if __name__ == "__main__":
    main()
