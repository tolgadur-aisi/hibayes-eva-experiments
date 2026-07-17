"""FINDINGS.md source for E4 (kept in code so the deliverable is regenerable).

Every number below is traceable to a file in experiments/e4_growth_curves/outputs/.
Round 2: frontier inference reworked after REVIEW_ROUND_1 (record-selection blocker).
Regenerate with:
    uv run python -m experiments.e4_growth_curves.findings_doc
"""

from pathlib import Path

FINDINGS = """\
# E4 — Capability growth over time on AISI public benchmarks

## Question

How fast does headline benchmark performance grow with **model release date** (logits/year),
per benchmark? Is the frontier decelerating near the ceiling anywhere? Observational data from
the eva warehouse; no causal claims. (Round 2: round 1's frontier fits were record-selection-
biased — REVIEW_ROUND_1 blocker — now descriptive only, replaced by selection-aware tests.)

## Data

`data/public_evals.parquet` (eval-level, one row per eval run). Filters, in order
(`outputs/exclusions.csv`; dedupe and boolq verification numbers persisted in
`outputs/data_checks.json`):

| step | dropped | left |
|---|---|---|
| raw rows (run dates 2025-01-13 .. 2026-02-02) | — | 53,041 |
| NaN `score_headline_value` (incl. all 8,280 boolq `boolq_scorer` rows — no numeric headline in warehouse; 94 swe, 18 cybench, 6 intercode) | 8,398 | 44,643 |
| duplicate uploads: same `eval_id`+`created`+value ingested from 2-5 S3 locations (298 eval_ids; `data_checks.json`) | 340 | 44,303 |
| excluded models: 31 `ft:gpt-4o-*` uk-dsit fine-tunes (44 rows); 5 `vllm/*` locally hosted checkpoints (5 rows — the base models have public release dates, e.g. Llama-3.1-8B-Instruct 2024-07-23, but served-weight provenance is unverifiable); `Mistral-Large-2411-CAST` deployment variant (8) | 57 | 44,246 |

boolq_preference uses the `pattern`/accuracy scorer only. Its per-model weighted accuracies are
stable across the train/validation splits (max |train − validation| = 0.032 over the 5 fitted
models; `data_checks.json`); the finer task_args (73 charity framings, 249 full configs) have
small per-config N, so per-config stability is **not** claimed — configs are pooled.

44,246 runs -> **33 model x task binomial cells** (`outputs/model_task_agg.csv`):
`n_correct = round(sum(value * completed_samples))`, `n_total = sum(completed_samples)` per
task x canonical model (rounding error <= 0.5 per cell). Aliases canonicalised (e.g.
`openai/o3` -> `o3-2025-04-16`); 16 dated models, dates + per-row sources in
`outputs/model_dates.csv` (embedded snapshot dates; web-verified for undated strings: o3/o4-mini
2025-04-16, o3-mini 2025-01-31, o1-mini 2024-09-12, gpt-5 2025-08-07, Gemma-3 2025-03-12,
Mistral-Large-2411 2024-11-19). Cells with `n_total < 20` are plotted but excluded from fits
(5 cells: 3 cybench, 2 swe). `error`/`limit` are not observable at eval level; the headline
accuracy already scores limit-hit samples as failures (Inspect default), accepted as genuine
failures. Runs with `epochs > 1` contribute epoch-replicates as extra Bernoulli trials.

Fitted cells per task: cybench 12, gdm_intercode_ctf 8, boolq_preference 5, swe_bench 3.
Fitted release spans: 2024-08-06..2025-08-07 (~1.0 yr) except swe (2024-10-22..2025-01-31, 0.28 yr).

## Models

`release_trend_logit_normal` (custom hibayes/numpyro model, `run.py`):
one observation per model j;
`y_j = logit((k_j+0.5)/(n_j-k_j+0.5))`, `se_j` = Haldane-Anscombe se;
`y_j ~ Normal(alpha + beta*t_j [+ beta2*t_j^2], sqrt(se_j^2 + sigma_model^2))`,
t = release date (years, centred per task). Priors: alpha, beta, beta2 ~ N(0,2);
sigma_model ~ HalfNormal(1).
This marginalises a latent-logit binomial GLMM (`K_j ~ Binom(N_j, sigmoid(theta_j))`,
`theta_j ~ N(mu_j, sigma)`) under a normal approximation to the per-cell likelihood — excellent
at our min fitted N=385. One pooled cell per model is the "model random effect": repeated runs
of one model never count as independent trend evidence. Effective N capped at 10,000/cell
(se floor 0.02 logits << sigma_model). NUTS, 4 chains x 2000 (warmup 1500), seed 20260716.

**Record path ("frontier") — descriptive only.** The same model refit on the models that raised
the running-best accuracy is reported with `descr_*` columns (`frontier_summary.csv`): that
subset is monotone **by construction**, so P(beta>0) ~ 1 under any truth including no trend, and
its sigma/HDIs are selection-biased (round-1 blocker). Selection-aware frontier inference
(`selection.py`, `outputs/frontier_null.csv`, 4,000 reps per null): (i) **record-count test** —
among J exchangeable accuracies P(model i sets a record) = 1/i, exact Poisson-binomial null;
(ii) **flat-null slope calibration** — simulate the real design (dates, capped Ns) from the
posterior of a no-slope refit, apply the same record selection, compare record-path OLS slopes;
(iii) **saturation check** — record-path quadratic coefficient against the posterior predictive
of the fitted *linear* model (record paths decelerate mechanically as records get rarer).
Release-date ties (o3/o4-mini; sonnet-4/opus-4) flip the cybench record count between 6 and 7:
observed statistics span all tie-consistent orderings, simulations randomise tie order per rep,
p-values use the conservative ordering.

## Validation

**Synthetic recovery** (`outputs/synthetic_recovery.csv`, `recovery.png`): binomial data simulated
from the real designs (same release dates and N) with known (alpha, beta, sigma), 10 reps each:

| scenario | J | 94% HDI coverage of true beta | mean abs(posterior − oracle OLS) | divergences |
|---|---|---|---|---|
| cybench_like (beta=1.5, sigma=0.5) | 12 | 9/10 | 0.09 | 0 |
| boolq_like (beta=0.8, sigma=0.4) | 5 | 10/10 | 0.09 | 0 |
| swe_like_narrow (beta=1.5, sigma=0.4) | 3 | 10/10 | 1.26 | 0 |

Posterior slope tracks the oracle (OLS on the realised latent logits) to ~0.09 logits/yr for
J>=5. The swe-like design is prior-dominated (mean HDI width 6.1 logits/yr): **swe velocity is
not identifiable from this warehouse**. 2/30 swe-like reps fall below the ess_bulk=400 bar
(327, 387; r_hat <= 1.011, 0 divergences) — validation-only fits for the scenario already
declared unidentifiable. Two sampler failure modes caught and fixed during validation are
documented in `run.py` (alpha<->u ridge at N~3M cells; sigma funnel at N<20 or J=3).

**Frontier selection artifact quantified** (`outputs/frontier_null.csv`, `frontier_null.png`):
under a **no-trend** null the record-path OLS slope averages +3.95 logits/yr on the cybench
design (94% interval [0.31, 11.24]) and +1.43 [0.09, 4.98] on intercode — same order as the
observed record slopes (3.19, 1.01), so no P-values are attached to record-path quantities.

**Convergence (real fits)** (`outputs/*.diagnostics.json`, 10 fits: 4 trend, 2 record-path,
2 record-quad, 2 no-slope null): max r_hat <= 1.0039, min ess_bulk >= 1,489, divergences 0 —
except trend_swe_bench with 3/8,000 divergent draws (non-headline, reported as unidentified).
**Prior sensitivity** (`outputs/velocity_prior_sensitivity.csv`): slope prior N(0,2) -> N(0,4)
moves cybench beta 2.94 -> 3.43 (HDI still excludes 0), intercode 0.92 -> 0.97, boolq -0.30 ->
-0.37, swe -1.02 -> -1.43 (still spans 0). Conclusions unchanged; quote cybench as "about 3".

## Findings

All from `outputs/{velocity_summary,frontier_summary,frontier_null,cross_task,model_task_agg}.csv`
(intervals: 94% HDIs; simulation quantiles for nulls); plots: `growth_curves.png`, `frontier_null.png`.

1. **Cybench mean velocity is strongly positive: +2.94 logits/yr [+1.22, +4.60],
   P(beta>0)=0.997** (12 models, releases 2024-08 -> 2025-08). Trend-line accuracy
   0.109 -> 0.662 over that year. Selection-free record evidence agrees: 6-7 of 12 models set
   accuracy records (tie-order dependent) vs 3.1 expected under no trend — exact
   P(K>=6)=0.034, P(K>=7)=0.006.
2. **InterCode-CTF velocity is positive but ~3x slower: +0.92 logits/yr [+0.004, +1.93],
   P(beta>0)=0.961**; record evidence: 6 of 8 models set records vs 2.7 expected, exact
   P(K>=6)=0.009. The task started much easier (trend acc 0.493 at 2024-08 vs cybench 0.109).
3. **Hard-cyber outpaces easy-cyber on the mean trend: P(beta_cybench > beta_intercode) = 0.98**,
   diff +2.02 logits/yr [+0.26, +3.56] (`cross_task.csv`). The record-path difference
   (+2.14 [+1.34, +2.90]) is descriptive only — record paths are selection-biased, no P attached.
4. **BoolQ-preference shows no capability growth: -0.30 logits/yr [-2.25, +1.91],
   P(beta>0)=0.378.** The record path never advanced: gpt-4o-2024-08-06 (pooled acc 0.892) is
   unbeaten by anything through gpt-5 (0.874); claude-sonnet-4 scores 0.504. Between-model spread
   is huge (sigma_model 1.07 [0.50, 1.75]) — behaviour on this preference-framed task is
   model-idiosyncratic, not a monotone capability.
5. **SWE-bench velocity is unidentified**: -1.02 [-3.95, +2.03], P(beta>0)=0.24, from 3 models
   over 0.28 yr; the synthetic check shows this design cannot recover even a large true slope.
   The record path never advanced (claude-3-5-sonnet-20241022 pooled 0.434; o3-mini 0.290 — but
   o3-mini ran different subset configs; not evidence of regression).
6. **No saturation evidence once record selection is accounted for.** The cybench record-path
   curvature (descriptive beta2 -0.92 [-2.61, +0.77]) is typical of what a *purely linear* trend
   produces mechanically after record selection: P(ppc quad <= observed) = 0.43 (cybench), 0.72
   (intercode). Round 1's "suggestive deceleration" (P(beta2<0)=0.875) is **withdrawn** as a
   selection artifact; best observed accuracies (0.765 / 0.812) are below ceiling.
7. **Record-path slopes carry ~no trend information** (`frontier_null.csv`): observed record
   slopes (cybench 3.19, intercode 1.01 logits/yr) sit mid-null, P(no-trend null >= observed) =
   0.51 / 0.52. Growth evidence comes from the mean-trend fits (findings 1-2) and record counts;
   round 1's frontier P-values (>0.999, 0.976, 0.9995) were vacuous-by-construction — retracted.
8. **Between-model residual spread is large everywhere** (sigma_model: cybench 0.91 [0.55, 1.33],
   intercode 0.45 [0.23, 0.74], boolq 1.07 [0.50, 1.75]): a same-date model can sit ~1 logit off
   the trend, so release date explains the direction of travel, not individual model quality.

## Caveats

- **Observational.** Release date is a proxy for "newer model"; scaffolds, solvers, task versions
  and elicitation practice drift over the same period. Velocities describe *measured* growth in
  this warehouse, not a causal effect of model releases.
- **Record selection**: any quantity computed on the running-best subset (record-path slopes,
  curvatures, HDIs, the plotted dashed lines) conditions on a subset that rises by construction
  and describes the realised record path only; frontier inference uses only `frontier_null.csv`.
- **Eval-mix confound**: cybench/intercode cells pool runs over whatever challenge subsets teams
  chose to run (most runs are single-challenge, `dataset_samples=1`); per-model challenge mixes
  differ and are not observable at eval level. sigma_model absorbs static mix differences, but a
  mix shift correlated with release date would bias beta.
- **Survivorship/selection**: teams run the models they expect to matter (o3: 13.9k runs,
  Mistral-Large: 33), and older low scorers are less likely to be re-run on new task versions.
- Epoch replicates are treated as independent trials (inflates N, not the trend; the N-cap makes
  this nearly irrelevant). Alias resolution (`-latest`, undated `openai/*`) is best-effort with
  documented sources; snapshot dates can precede public availability by <= 8 days.
- boolq_preference is a preference-framing variant of BoolQ, not plain QA; finding 4 is about
  this task, not QA in general.
"""


def main() -> None:
    out = Path(__file__).resolve().parent / "FINDINGS.md"
    out.write_text(FINDINGS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
