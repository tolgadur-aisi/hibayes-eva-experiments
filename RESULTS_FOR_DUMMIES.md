# The results, in plain English

*A no-stats walkthrough of the first real fit (2026-07-21 data snapshot). The
technical version, with figures, is in [README.md](README.md). This file is
Claude's plain-language interpretation of the same numbers.*

## What we asked

Every eval result in the warehouse is a model attempting a task item and
passing or failing. But the pass rate isn't just about the model: it also
depends on **which items** it faced, **which benchmark** they came from, **how
much token budget** the run was given, and **the harness** it ran inside (the
solver, prompts, attempt limits, tools — what we call the *scaffold*).

So we fitted one statistical model to all 166,511 attempts at once, asking it
to split the credit: how much of passing is the model, how much is the
scaffold, how much is the token budget, and how much is just which items you
happened to face?

## The five takeaways

**1. The harness matters as much as the model — probably more.**
The gap between the least and most helpful scaffold is *larger* than the gap
between the weakest and strongest of all 57 models. In success-chance terms:
the same middling model could look nearly hopeless in one harness and look
like a frontier system in another. If you remember one thing, remember this:
a benchmark number without its harness attached is close to meaningless.

**2. We caught the problem red-handed.**
The warehouse contains `openai/o1` and `openai/o1-2024-12-17` — the *same
model* under two names, run in different eras with different harnesses. One
label lands far below the average model, the other well above. Same brain,
different plumbing, wildly different score. That gap is exactly the
model-vs-scaffold entanglement this project exists to measure.

**3. But don't over-trust any single scaffold's number (yet).**
Of the 160 distinct scaffolds we found, 139 were only ever used with a single
model. That's like rating race car drivers when each one only ever drove their own
car — you can tell the car+driver combos apart, but you can't fully separate
driver skill from car quality. The *direction* (harness matters a lot) is
solid; individual scaffold rankings are not. Fixing this is the top item on
the to-do list.

**4. Which items you face matters more than anything else.**
Within every benchmark, the spread between easy and hard items dwarfs the
spread between models. Cybench is the extreme case: it's mostly items that
almost nothing solves plus items that almost everything solves, and it's by
far the hardest benchmark overall. Practical consequence: two runs of the
same model on different item subsets can differ more than two different
models on the same subset.

**5. Token budgets showed no clear effect.**
Bigger budgets *look* slightly better, but almost none of the differences are
distinguishable from noise. One oddity proves the data isn't an experiment:
runs with *no* token limit at all scored the *worst* — not because unlimited
tokens hurt, but because "no limit set" is simply what older runs did. Nobody
varied budgets on purpose, so the data can't really answer this question.

## How much should you trust this?

The machinery itself ran cleanly — the standard convergence checks all passed,
which means these numbers are what the model genuinely concluded from the
data, not a computation glitch. The real limits are in the data, not the math:

- scaffolds tangled with models (point 3),
- repeated runs of identical setups drift more than coin-flip noise allows,
  which the current model doesn't yet account for — so the uncertainty bands
  are, if anything, a bit too confident.

## Reading the plots (in `modeling/figures/`)

- **raw_scores.png** — every single attempt as one dot, pass (top row) or
  fail (bottom row), before any statistics. Thicker top stripe = more passing.
- **forest_*.png** — each row is one model / scaffold / budget. The dot is
  the best guess of its effect, the line is the uncertainty. The red dashed
  line at zero means "exactly average": right of it = better than average,
  left = worse, and if the line crosses zero we can't rule out "no effect".

## What happens next

In order of value: collapse the 160 scaffolds down to the few knobs that
actually differ (so scaffold effects stop absorbing model identity); merge
the duplicate model names (fine-tune checkpoints currently count as separate
models); teach the model that repeated runs drift; and drop the token-budget
variable unless someone actually runs a budget experiment. All of these are
config changes, not rewrites.
