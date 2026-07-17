# Findings, in plain language

A no-heavy-stats walkthrough of the five experiments, written by Claude. If you know what a
normal distribution is, you're overqualified. The technical, reviewer-approved version is
[FINDINGS.md](FINDINGS.md); the per-experiment detail is in `experiments/e*/FINDINGS.md`.

## The core idea: an eval score is a poll, and polls have margins of error

When you see "model X scored 42% on cybench," that 42% feels like a fact. It isn't — it's an
*estimate*, the way a political poll is an estimate. If you polled 20 people you'd trust the
result less than if you polled 20,000. Eval scores work the same way: they're built from a
limited number of noisy attempts, so every score really has a hidden "give or take" attached.

The whole project is: **for AISI's real evals, figure out how big that "give or take" actually
is** — because people quote these numbers to two decimal places as if they were exact, and
they're often much shakier than they look. We used hibayes (the SoE team's tool) because it's
built to produce honest give-or-take ranges instead of a single misleadingly-precise number.

A couple of plain-language terms used below:

- **Credible range** (the technical docs call it "94% HDI"): the band the true value is 94%
  likely to fall in. "0.42, give or take 0.10" means the true skill is very probably between
  0.32 and 0.52.
- **"70% chance A beats B"**: instead of flatly declaring "A is better," we say how *sure* we
  are. 99% means basically certain; 55% means it's nearly a coin toss even though A's raw
  score was higher.

And the villain that keeps coming back: **eval scores are noisier than they look, for three
reasons that a plain average completely hides.** The five experiments each pin down one of
those reasons and measure it.

## E1 — cybench: does running an eval more times actually help?

**What we did.** Took the cybench hacking benchmark (40 challenges, ~10 models) and asked two
things: how *reliable* is a score, and does the common practice of re-running each challenge
many times ("epochs") actually buy you a better measurement?

**What we found.**

The single most useful result: **re-running the same challenge is nearly worthless compared to
testing a new challenge.** Here's the intuition. Imagine measuring how good someone is at
trivia. Asking them the *same question* 100 times tells you almost nothing new after the first
couple of tries — you just learn whether they know that one fact. Asking 100 *different*
questions tells you far more. Eval epochs are the "same question again" case. We measured this:
attempts on the same challenge are ~78% redundant, so a run that looks like it has 1,340 data
points is really worth about 190 independent ones.

The practical punchline for the eval teams: **spend your compute budget covering more
challenges, not re-running the ones you have.** We found runs in the warehouse that spent 848
attempts on just 2 challenges — after all that compute, the model's true skill was still only
pinned down to "somewhere in a ±27 percentage-point window." The same budget spread across 40
challenges would have nailed it to ±6.

And once we account for all this honestly, some published-looking rankings dissolve. Model o1
*looked* clearly ahead of claude-3-7 on the raw numbers — but properly measured, it's a 57%
chance o1 is better, i.e. basically a coin flip. The raw error bars were about 2.5× too small.

## E2 — intercode: where does the noise actually come from?

**What we did.** Same kind of hacking benchmark, but here the question was a breakdown: when
scores bounce around, *what's* causing the bounce? We split the total variation into buckets —
is it differences between models? Between challenges? Between one run and another of the
identical setup?

**What we found.**

Two things. First, **by far the biggest driver of the score is just which challenges are in
the set** (about two-thirds of all the variation). The challenges differ wildly in difficulty,
so which ones you happen to include swamps almost everything else. This is why comparing two
models on different challenge subsets is so treacherous.

Second, and more surprising: **running the exact same eval on a different day gives a
meaningfully different answer — and how much depends on where the model is hosted.** Models
served through one Azure setup drifted a lot between runs; the same benchmark on a
locally-served model was rock stable. The gap was enormous: a stable run was worth ~50× more
(as a reliable measurement) than a drifty one. This is basically "our measuring instrument
itself is wobbling, and the wobble depends on the plumbing" — and now it's quantified rather
than a vague suspicion. We even caught gpt-5's score jumping 12 points on a fixed challenge set
within a single week, same model, same config.

## E3 — comparing model releases fairly (o1 → o3, etc.)

**What we did.** When a new model comes out, everyone wants "how much better is it?" The
problem: the new and old models were usually tested under *different conditions* (different
challenge subsets, different tool setups). So a raw comparison mixes up "the model got better"
with "the new one had an easier test." We hunted for cases where the old and new model were run
under genuinely matched conditions, and only compared those.

**What we found.**

- **o1 → o3 was a big, real jump** on hacking — we're essentially certain it's a genuine
  improvement (roughly +19 percentage points, honestly measured).
- **claude sonnet-4 → opus-4-1 was real but small** — a clear, reliable improvement, but
  confidently a *modest* one, not a leap.
- The most important lesson is about how raw comparisons lie. The naive "o1 → o3" number was
  +27 points. But that single number was wrong in *two opposite ways at once*: it overstated
  the gain by ~40% (because o3 happened to get an easier mix of challenges) while
  *understating* the model's actual capability jump (because o1 had been given a better tool
  setup that was flattering it). When we controlled conditions properly, we also found
  something sobering: **just changing the eval's difficulty setting moved gpt-5's score by 30
  points — bigger than most actual model upgrades.** The setup can matter more than the model.

## E4 — how fast are capabilities growing over time?

**What we did.** Plotted headline scores against each model's *release date* (not the date we
happened to run it) and fitted a trend line — with honest error bands — to ask "how fast is the
field improving, per benchmark?"

**What we found.**

- **Cyber capability is rising fast and the trend is solid**: on cybench the trend goes from
  ~11% to ~66% over the year to August 2025.
- **Nothing is maxed out** — no benchmark is close to its ceiling, so these evals still have
  room to measure future models.
- **Not everything is "capability."** The boolq benchmark showed *no* upward trend and huge
  model-to-model scatter — it turns out to measure something more like "how susceptible is this
  model to a particular framing" than raw ability, so newer isn't better there.
- The methodology lesson worth telling SoE: the reviewer agent caught our first attempt
  cheating itself. We'd drawn a trend line through the "best model so far at each date," which
  *always* slopes upward no matter what — even with random data — because you're only ever
  plotting record-breakers. We threw that out and replaced it with a test that doesn't have
  that built-in bias. (This is exactly the kind of subtle self-deception that argues for having
  automated statistical checks in any future eva feature.)

## E5 — a health check across the whole public portfolio

**What we did.** Ran the *same four diagnostic questions* on every public benchmark, like a
standardized blood panel: (1) does it actually tell models apart, or is everyone in a blur?
(2) is it near its ceiling (saturated)? (3) how much noisier than ideal is it? (4) do repeat
runs agree?

**What we found.** A per-benchmark verdict:

- All four benchmarks *do* separate models well — good news, they're informative.
- None are saturated — they've all got headroom.
- But the agentic ones (cybench, intercode) are **noisy**: repeat runs swing ~4× more than pure
  luck would predict — the same drift story from E2, confirmed portfolio-wide.
- swe_bench is **underpowered** — too little clean data to say much yet.
- boolq is **noisy** and its "score" partly measures framing-sensitivity, not skill.
- And we confirmed the one analysis we *couldn't* do here: measuring "does it matter which AI
  grades the answers?" All the public benchmarks grade with fixed rules, not an AI judge, so
  there's nothing to measure — that question needs AISI's internal AI-graded evals, which is a
  natural follow-up.

## The one-sentence version

**Eval numbers look precise but hide three kinds of noise — which questions you picked,
re-running the same question, and day-to-day drift — and once you measure them honestly, some
rankings and "improvements" shrink or vanish.** E1 and E5 tell that story at two zoom levels
(one benchmark vs the whole portfolio), E2 explains the mechanism, and E3 shows it biting on
real release comparisons.
