# hibayes × eva experiments

Five Bayesian analyses of AISI's public team_ru eval data (cybench, gdm_intercode_ctf,
swe_bench, boolq_preference), run on 2026-07-16 by Claude agents against parquet extracts
from the eva warehouse, using [hibayes](https://github.com/UKGovernmentBEIS/hibayes) /
NumPyro. Each experiment was independently checked by a skeptical-reviewer agent and
iterated until approved.

> **Note:** the outlines below are **Claude's interpretation** of the experiments, written
> as a narrative summary. The authoritative, reviewer-approved write-up is
> [FINDINGS.md](FINDINGS.md) (consolidated synthesis, including the data-hygiene findings
> omitted here), and each experiment's own `experiments/e*/FINDINGS.md` with code, plots,
> diagnostics and reviewer verdicts. Reproduce any experiment with
> `uv run python -m experiments.<id>.run`.
>
> **No stats background?** [FINDINGS_FOR_DUMMIES.md](FINDINGS_FOR_DUMMIES.md) is a
> plain-language walkthrough of the same five experiments (also Claude's interpretation) —
> no logits, no credible intervals, just the intuition and the takeaways.

All numbers are posterior means with 94% HDIs, and every experiment validated itself with
synthetic-parameter-recovery fits before touching real data.

## Data

The analyses run off local parquet extracts in `data/`, which are **gitignored**: they are
~200MB, and although the benchmarks themselves are public, the eval *results* are
AISI-internal — don't publish them. The fitted posteriors
(`experiments/*/outputs/*.idata.nc`, ~800MB) are gitignored for the same size reason;
small outputs (plots, CSVs, diagnostics JSONs) are tracked, as are `data/MANIFEST.json`
and `data/scorer_inventory.json` (snapshot provenance).

Regenerating the data is a single script, run from the eva client project on an AISI
platform dev VM:

```bash
cd ~/dev/eva/client && uv run --with pyarrow python <this-repo>/shared/extract.py
```

Requirements: VPC connectivity to the eva Aurora cluster (any platform dev VM has it),
AWS credentials, `AISI_PLATFORM_USER`/`AISI_PLATFORM_PROJECT` set, and membership of
`team_ru` — eva enforces Postgres row-level security, so without that role the queries
succeed but silently return only the rows your roles can see. `uv run eva-check` (in
`eva/client`) verifies the whole setup first.

The script pulls sample-level rows for cybench / gdm_intercode_ctf / swe_bench and
eval-level rows for all four public tasks (filters: `team='team_ru'`,
`status='success'`, no `replay/*`/mock/none models), chunked by quarter to stay under
statement timeouts. It takes a few minutes, skips files that already exist, and writes
`data/MANIFEST.json` with the extraction timestamp.

**Snapshot caveat:** every number in this repo is pinned to the **2026-07-16** extraction.
The warehouse moves (new runs, re-ingestions), so a fresh extract is a new snapshot and
regenerated numbers can differ. After re-extracting, rebuild any experiment's outputs and
posteriors with `uv run python -m experiments.<id>.run`.

## E1 — cybench: epochs vs samples, and what honest uncertainty changes

**The experiment.** Cleaned cybench down to a comparable core: 19,677 attempts, 40
hard-variant challenges × 10 models. Three binomial models on aggregated counts: per-model
hierarchical challenge effects (to get each model's difficulty spectrum), a crossed
challenge×run model on claude-3-7's 54 runs — it has a 100-epoch run, which is what makes
the epoch-reliability question answerable — and a joint model (ability + challenge + run
effects) for cross-model ranking. The epochs-vs-samples answer comes from turning the
estimated intraclass correlation into a design curve: posterior uncertainty about ability
as a function of how a fixed attempt budget is split between challenges and epochs,
cross-checked against 200k-rep Monte Carlo.

**Findings.**

- cybench is nearly all-or-nothing: challenge spread is 5–9 logits per model; 27–82% of
  challenges sit below a 5% solve rate and only 2–11 per model are genuine coin-flips.
- Repeated epochs on the same challenge are highly redundant: ICC ρ = 0.78 [0.72, 0.82],
  i.e. design effect 8 at 10 epochs, 78 at 100. For a fixed budget, minimise epochs until
  every challenge is covered — concentrating 400 attempts on 4 challenges instead of 40
  costs 27× in standard error.
- The warehouse contains genuinely uninformative designs: 848 attempts on 2 challenges
  still leaves ability uncertain to ±27 accuracy points.
- Run-to-run drift: σ_run ≈ 1.1 logits, which moves a benchmark score ±2.7pp between
  identical runs of the same model.
- Ranking with honest uncertainty: gpt-5 0.52 > o3 0.47 > opus-4 0.39 > … but
  P(gpt-5 > o3) is only 0.71. Naive Wald intervals are ~2.5× too narrow, and o1's apparent
  lead over claude-3-7 (barely-overlapping naive CIs) collapses to P = 0.57 once challenge
  and run structure is modeled.

## E2 — intercode-ctf: where does the variance actually come from?

**The experiment.** A cross-classified random-effects model —
logit P(solve) = model + challenge + model×challenge + run — fitted on the one fully
homogeneous config in the data (4 models, 12 runs, 79 challenges × 10 epochs), plus a
sensitivity fit including scaffold variants and a separate fit on the frontier fan-out
campaign (where day-of-launch plays the run role). The point of the decomposition is to
attribute outcome variance to its sources on a common scale. Included a model-free
cross-check: between-run dispersion of per-item counts against an exact binomial null, so
the drift claim doesn't rest on the prior.

**Findings.**

- Challenge identity dominates everything: 67% [63, 72] of variance. Epoch residual 24%,
  model×challenge 4%, run 3%, model ~1% (these four models are similar-strength — that
  last share isn't a general claim).
- Run drift is provider-dependent: Azure-served Mistral setups drift at σ_run ≈ 0.9 logits
  (8–17pp swings in run means) while gemma serving is stable at ≈ 0.08. The model-free
  dispersion check confirms it (ratios ~3, p < 5e-4).
- Deliberate scaffold changes (message limits, prompt variants) move scores about as much
  as an unexplained relaunch does — "we changed nothing" and "we changed the prompt" are
  similar-sized perturbations.
- Even one week is enough to drift: the September frontier campaign shows σ_day = 0.35,
  with gpt-5 gaining +12.2pp [7.7, 16.8] on a fixed 14-challenge set between Sep 17 and
  Sep 20.
- Consequence for effective sample size: a stable 790-sample run is worth N_eff ≈ 1,274
  (fixed-item stratification actually helps), a drifting one ≈ 26–28 — a ~50× range — and
  drift puts a ceiling (~22) on what any number of epochs can buy.

## E3 — release-over-release deltas at matched config

**The experiment.** Rather than compare raw accuracies, build matched-config classes —
same task_args, same solver, same challenge set, runs days apart — so the scaffold
confound is removed within each pair, then fit binomial models with fixed model effects
and (hierarchical) item effects; a beta-binomial variant handles within-run epoch
correlation where a pair has only one run per model. A pooled two-benchmark model asks
whether a delta generalises. Pairs with zero shared configs are declared non-analysable
rather than forced.

**Findings.**

- o1 → o3 (cybench hard, matched): +2.86 logits [+2.23, +3.46]; the beta-binomial
  robustness check gives +2.66 [+1.74, +3.59]. P(δ > 0.5 logits) = 1.00 — a large,
  unambiguous jump (+19pp accuracy).
- sonnet-4 → opus-4-1: real but small — +0.27 logits on cybench-easy, +0.44 on intercode,
  both P(δ>0) ≥ 0.99, both confidently *below* the 0.5-logit "meaningful" threshold.
- Two benchmarks can't settle generalisation: the pooled delta is +0.48 [−0.16, +1.09] —
  each benchmark's delta is precise, but "does it transfer to a new benchmark?" stays
  genuinely open with N=2.
- The naive o1→o3 read (+27.1pp) errs in both directions at once: it overstates the
  matched effect (~+19pp) because o3 ran an easier mix, while understating the
  within-challenge capability shift 2.4× because o1's bespoke scaffold masks it.
- gpt-4o → gpt-5 is not honestly estimable from this warehouse: zero shared configs; 63%
  of the naive intercode delta is item composition, and the remainder still confounds
  scaffold with model.
- Perspective: the same gpt-5 scores +30pp higher on the easy cybench variant than the
  hard one — config choice rivals a major release jump.

## E4 — capability growth over time

**The experiment.** Eval-level data collapsed to 33 model×task cells, with each model
dated by *release date* (embedded snapshot dates plus web-verified ones), one pooled cell
per model so that re-running a model a thousand times never counts as extra trend
evidence. Per task: a measurement-error logit trend (slope in logits/year, with a
model-level residual). The interesting methodological part: the first-round "frontier is
accelerating" result was rejected by the reviewer as selection bias — a running-best
record path rises by construction under *any* truth — and was replaced by selection-aware
tests: an exact record-count null (P(model i sets a record) = 1/i under exchangeability),
flat-null slope calibration, and a posterior-predictive curvature check.

**Findings.**

- Cybench capability velocity: +2.94 logits/yr [+1.22, +4.60], P = 0.997 — trend accuracy
  went 0.11 → 0.66 in the year to Aug 2025. The selection-free record-count test agrees
  (6–7 record-setters among 12 models vs 3.1 expected; exact P = 0.034/0.006).
- Intercode grows ~3× slower (+0.92 [+0.00, +1.93]); hard-cyber outpaces easy-cyber with
  P = 0.98.
- boolq-preference is flat (−0.30 [−2.25, +1.91]) with a huge model-idiosyncratic spread —
  it behaves like a framing-susceptibility measure, not a monotone capability;
  gpt-4o-2024-08-06 is still unbeaten on it through gpt-5.
- swe_bench velocity is unidentifiable here (3 models over 0.28 years — the synthetic
  check shows even a large true slope couldn't be recovered from that design).
- No saturation: the apparent frontier deceleration is exactly what a *purely linear*
  trend produces mechanically after record selection (PPC p = 0.43/0.72), so the round-1
  deceleration hint was withdrawn; best observed accuracies are well below ceiling.
- Record-path slopes themselves carry ~no trend information (observed values sit mid-null,
  p ≈ 0.5) — a useful negative result for anyone tempted to fit lines through "best model
  so far" plots.

## E5 — portfolio statistical-health audit

**The experiment.** The same four questions asked uniformly of every benchmark:
discrimination (posterior spread of model abilities ÷ a single model's posterior
uncertainty), saturation (P(best ability > 0.90)), overdispersion (intra-item correlation,
i.e. how far from iid binomial), and run reproducibility (exact-config replicate groups,
run-level σ vs binomial prediction). Crossed binomial models with fixed model effects; an
eval-level normal model for boolq (which adds charity-framing condition effects); a
beta-binomial sensitivity fit on cybench to check how much the binomial likelihood
flatters discrimination.

**Findings.**

- Discrimination: all four benchmarks separate models well beyond noise — ratios 7.6
  (cybench; 3.7 under the more honest beta-binomial), 14.4 (intercode), 5.7 (swe_bench).
- Saturation: none. Best abilities 0.62 / 0.68 / 0.44 on the agentic three; boolq's 0.89
  is watch-listed but P(>0.90) = 0 everywhere.
- Overdispersion is severe on all agentic benchmarks (ρ = 0.68 / 0.41 / 0.56), which is
  the same phenomenon as E1's epoch redundancy seen portfolio-wide: 10-epoch design
  effects of ~5–7.
- Run reproducibility: cybench and intercode same-config runs swing 3.9× [2.8, 5.2] what
  binomial noise predicts; boolq's run-to-run noise is 3.3× its own reported stderr;
  swe_bench is binomial-consistent but from only 9 replicate runs.
- boolq measures a model-dependent mixture: framing-condition sensitivity ranges from
  SD ≈ 0.22 (claude-3-7) to 0.002 (gpt-4o) — two models with the same mean score can be
  measuring different things.
- Verdicts: cybench **noisy**, intercode **noisy**, swe_bench **underpowered**,
  boolq-preference **noisy**. And the judge-effects question is formally confirmed
  infeasible on the public slice — every scorer is rule-based — so that analysis needs the
  internal `rubric_scorer/<judge>` long-form tasks, ideally with the same transcripts
  scored by two judges.

---

If you're picking what leads a paper: E1's design curve + E5's portfolio-wide ρ (the same
story at two zoom levels), E2's provider-drift decomposition, and E3's o1→o3
matched-vs-naive contrast are the four figures to build it around.
