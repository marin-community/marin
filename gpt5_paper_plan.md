# GPT-5 Paper Plan

## TL;DR

The paper should be about **stress-test-guided alignment from a natural-language spec**.

The key claim is not "we found conflicts in the spec." The key claim is:

> Vanilla spec-to-DPO can look good on per-clause adherence while still failing the trade-offs that matter. Stress-tested trade-off scenarios expose those failures. Training on those scenarios improves trade-off handling, and a post-training repair loop improves it further.

This makes the evaluation do real work. Without trade-off stress tests, `M2` and `M3` are hard to justify.

## Working Thesis

Existing alignment-from-spec pipelines mostly train on prompts that target one statement at a time. That is enough to improve clause-level adherence. It is not enough to teach how two co-equal clauses should be balanced on the same prompt.

The right evaluation target is therefore **trade-off handling**, not just aggregate adherence.

We should adapt the main lesson from *Stress-Testing Model Specs Reveals Character Differences Among Language Models*:

- start from pairs of legitimate principles
- generate scenarios that force a trade-off
- include biased variants that push toward each side
- use disagreement and low joint satisfaction as evidence that the scenario exposes a real gap

Then add the part that paper does not have:

- train on the validated trade-off scenarios
- audit the remaining failures
- patch the right layer
- retrain and rerun the same stress suite

## Problem

MARIN already has a usable baseline alignment pipeline:

- `align()` uploads a spec, generates prompts from the spec, generates chosen and rejected responses, judges them, filters preference pairs, and optionally runs DPO
- `evaluate()` already accepts a custom prompt step or prompt path
- prompt records already include `behavior_id`, `system_prompt`, `user_message`, `rubric`, `config_id`, `axis_config`, and `tags`
- the judge already persists full chosen and rejected candidate metadata before pair filtering

Relevant code:

- [`lib/marin/src/marin/alignment/align.py:306`](lib/marin/src/marin/alignment/align.py:306)
- [`lib/marin/src/marin/alignment/align.py:592`](lib/marin/src/marin/alignment/align.py:592)
- [`lib/marin/src/marin/alignment/generate_prompts.py:1112`](lib/marin/src/marin/alignment/generate_prompts.py:1112)
- [`lib/marin/src/marin/alignment/judge.py:302`](lib/marin/src/marin/alignment/judge.py:302)

The current weakness is straightforward:

- prompt generation is largely per-statement
- chosen and rejected examples are judged against one target rubric
- evaluation can therefore overstate progress if the model improves each clause in isolation while failing clause interactions

Concrete example:

- `be_warm_and_supportive`
- `be_truthful_and_calibrated`
- `avoid_unrealistic_reassurance`

A vanilla pipeline can teach "more warmth" without ever showing the model the boundary where warmth becomes false reassurance.

## Terms

This plan uses four terms in a precise way.

### Clause

For the first paper, a "clause" is just an original spec statement.

We should start with the 46 published statements as-is. That keeps the unit of analysis faithful to the spec, matches the current MARIN data model, and avoids making the first paper depend on a full clause compiler.

Example:

- `be_warm_and_supportive`
- `be_truthful_and_calibrated`
- `avoid_unrealistic_reassurance`

A later ablation can test whether manually splitting a few compound statements changes the results.

### Authority hierarchy

The authority hierarchy answers **whose instruction wins** when principals disagree.

Example:

- platform > developer > user > guideline

This is not the main object of the paper. Most of the interesting failures here are between co-equal platform-level defaults.

### Clause graph

The clause graph is a **proposal layer**. It says which clause pairs might interact and how:

- `hard_contradiction`
- `soft_tension`
- `scope_ambiguity`
- `shadowing`
- `rubric_disagreement`

In the first paper, this is therefore a graph over the original 46 statements, not over a newly compiled set of sub-clauses.

The clause graph is useful, but it is not enough. A clause pair is only interesting if it produces a real trade-off on concrete prompts.

### Trade-off suite

The trade-off suite is the main artifact.

For a clause pair `(A, B)`, it contains:

- a neutral scenario where both clauses plausibly apply
- an `A`-biased scenario that pushes the model toward `A`
- a `B`-biased scenario that pushes the model toward `B`
- one rubric per active clause for each scenario

This is where the evaluation becomes real. A trade-off is not just two clauses that look opposed in English. A trade-off is a scenario where joint satisfaction is hard, ambiguous, or model-dependent.

## What Stress Testing Gives Us

The stress-testing paper determines trade-offs operationally, not logically.

It starts from pairs of values, generates scenarios that force a balance, generates biased variants, runs a committee of strong models, and uses disagreement plus compliance failures to find scenarios where the spec is weak.

We should adapt the same pattern to spec-native clauses:

1. Propose statement pairs from the 46-statement spec.
2. Generate neutral and biased scenarios for each pair.
3. Elicit per-clause rubrics for each scenario.
4. Run multiple strong models on those scenarios.
5. Score each response against both rubrics.
6. Treat disagreement and low joint satisfaction as evidence that the pair defines a real trade-off.

The biggest change from the old plan is:

- the **graph proposes**
- the **trade-off suite validates**
- `M2` trains on validated trade-offs
- `M3` repairs failures that remain after that training

## Method Overview

The full method is:

```text
spec
  -> treat the 46 statements as the initial clause set
  -> compile candidate statement pairs
  -> generate trade-off scenarios (neutral, A-biased, B-biased)
  -> elicit per-clause rubrics for each scenario
  -> validate trade-offs with model and judge disagreement
  -> build edge-targeted preference pairs
  -> align()
  -> rerun the same trade-off suite
  -> type failures
  -> patch spec, rubric, or data
  -> align() again
```

The implementation should stay close to the current MARIN pipeline. The paper is about **better prompt selection, better negatives, better judging, and better evaluation**, not a new trainer.

## Experimental Setup

We should keep the experiment ladder simple:

- `M0`: base model
- `M1`: vanilla MARIN `align()`
- `M2`: stress-test-guided static alignment
- `M3`: stress-test-guided repair alignment

The core logic is:

- `M1 -> M2`: does static trade-off targeting help?
- `M2 -> M3`: is static targeting enough, or do we need post-training repair?

## M1: Vanilla Spec-to-DPO

`M1` is the current baseline.

- use the original spec
- use current prompt generation
- use current chosen and rejected response generation
- use the same teacher, rejected model, judge, tokenizer, and DPO setup as the existing MARIN run

This tells us how far the current pipeline gets before we add stress testing.

## M2: Stress-Test-Guided Static Alignment

`M2` is the static version. It does **not** patch the spec after training.

Its job is to replace mostly single-statement training data with **validated trade-off training data**.

### Goal

Test whether alignment improves when synthetic preference pairs are built from stress-tested statement interactions rather than mostly per-statement prompts.

### Stage 1: Statement-pair proposal

Use the original 46 statements as the units and propose candidate statement pairs that might interact.

Useful pair types:

- `soft_tension`
- `possible_contradiction`
- `scope_overlap`
- `possible_shadowing`

This step is cheap and noisy. That is fine. It only needs recall.

For the first paper, do not build an automatic clause compiler. If a statement is somewhat broad, keep it whole. We can test manual splits later as an ablation.

### Stage 2: Mine existing Stage 2 ideations

Do **not** start by generating fresh pairwise trade-off scenarios from scratch.

Instead, start from the existing Bloom-style Stage 2 artifacts: `ideation.json` for each statement.

Each `ideation.json` variation already contains:

- a full scenario description
- a scenario-specific rubric
- a `config_id`
- an `axis_config`
- tags

That means we already have a large bank of statement-conditioned stress scenarios before writing new generation code.

For each candidate pair `(A, B)`:

- scan `A`'s Stage 2 variations and ask whether `B` also applies
- scan `B`'s Stage 2 variations and ask whether `A` also applies

This turns existing single-statement ideations into candidate co-activation cases.

### Stage 3: Cross-apply statements and elicit paired rubrics

For each retained source scenario, keep the original source rubric and elicit a second rubric for the co-active statement on the **same** scenario.

Example:

- source scenario from statement `A`
- original rubric `R_A(s)`
- newly elicited cross-rubric `R_B(s)`

Then store a mined trade-off dossier with at least:

- `source_statement_id`
- `coactive_statement_id`
- `source_config_id`
- `scenario_description`
- `source_rubric`
- `cross_rubric`
- `axis_config`
- `tags`
- `cross_applies_score`

This is the key shift. We are no longer asking only whether statements `A` and `B` sound like they might conflict. We are checking whether a scenario already generated for `A` also activates `B`, and what satisfying both would look like behaviorally.

### Stage 4: Trade-off validation

Use the mined paired-rubric scenarios as the initial stress-test suite. Then run two evaluations:

1. **Feasibility filter with strong models**
   Run a small committee of strong models on the paired-rubric scenarios to separate:

- feasible trade-offs
- ambiguous or under-specified trade-offs
- non-trade-offs

2. **Baseline measurement on our actual models**
   Run `M0` and `M1` on the feasible scenarios. Score each response against both rubrics.

For each scenario and pair, compute:

- marginal satisfaction of `A`
- marginal satisfaction of `B`
- joint satisfaction of `A and B`
- behavioral conflict gap
- model disagreement on the balance
- judge disagreement on compliance

This is the important change from the earlier plan. Strong models help us filter scenarios, but the decision to train on a trade-off should depend on whether the trade-off is still mishandled by `M0` or `M1`.

Mark a scenario or pair as an `M2` target when:

- a strong model can satisfy both rubrics well enough that the task seems feasible
- `M1` still has a meaningful joint-handling gap on it
- judge disagreement is low enough that the supervision is usable

Cases with high judge disagreement or strong-model failure should usually be held out for `M3`, not used as `M2` training data.

### Stage 5: Targeted augmentation for the top failure pairs

Do not rely on mined Stage 2 scenarios alone for final training coverage.

Those scenarios were generated to stress one source statement at a time. They are excellent for discovery and baseline measurement, but they will under-cover some pairwise boundaries.

For the top failure pairs, generate additional pair-specific variants seeded from the mined Stage 2 scenarios:

- neutral variants
- `A`-biased variants
- `B`-biased variants

This is where we borrow most directly from the stress-testing paper. The biased variants are useful, but they should be added **after** we know which trade-offs matter for our model.

### Stage 6: Edge-targeted preference pairs

Construct DPO pairs from:

- mined Stage 2 co-activation scenarios
- targeted neutral and biased augmentations for the highest-gap pairs

Each pair should contain:

- `chosen`: response optimized for joint rubric satisfaction
- `rejected`: response that over-optimizes one statement at the expense of the other

Rejected examples should be generated intentionally:

- warm but falsely reassuring
- truthful but cold
- concise but missing necessary caveats
- safe but overrefusal-heavy

This is better than random or unguided negatives because it teaches the actual boundary.

### Stage 7: Vector judging

For `M2`, the judge should score a response against each active rubric in the scenario.

Minimum schema:

```python
{
    "scores_by_statement": {"A": 8, "B": 5},
    "joint_satisfaction": false,
    "hard_constraint_violations": [],
    "explanation": "...",
}
```

The pair filter should require:

- no hard-statement violation
- better joint satisfaction than the rejected response
- a sufficient aggregate gap

### Stage 8: Train once

Train one DPO run with:

- the same base model as `M1`
- the same DPO config
- the same overall pair budget

The only change is how the data is chosen and how negatives are constructed.

### What M2 Tests

`M1 -> M2` tests:

> if we mine existing Stage 2 ideations for co-active statement pairs, use them to measure where vanilla alignment still has a joint-handling gap, and then augment and train on those high-gap trade-offs, do we improve trade-off handling at fixed data budget?

### What M2 Does Not Include

`M2` does not include:

- post-training failure typing
- spec, rubric, or data patching
- optimization-induced interference analysis

Those belong to `M3`.

## M3: Stress-Test-Guided Repair Alignment

`M3` is the closed-loop version. It starts from the `M2` checkpoint and reruns the same mined-and-augmented trade-off suite.

### Goal

Use post-training trade-off failures to identify which artifact in the Stage 2-to-DPO pipeline is still wrong:

- the paired rubric
- the scenario pressure
- the chosen/rejected data construction
- the optimization behavior
- only in the hardest cases, the spec text itself

Then patch the cheapest layer that plausibly explains the failure and retrain once.

### Stage 1: Rerun the same mined-and-augmented suite

Run the same suite used in `M2` on:

- `M0`
- `M1`
- `M2`

This suite should include:

- mined Stage 2 co-activation scenarios
- neutral pair-specific augmentations
- `A`-biased augmentations
- `B`-biased augmentations

The point is not just to ask whether `M2` improved. The point is to see **which artifact families still produce failures**.

### Stage 2: Cluster failures by artifact family

Do not start from global statements alone. Start from the failure dossiers produced by `M2`.

Cluster failures by combinations of:

- statement pair `(A, B)`
- source statement
- source `config_id`
- scenario family
- bias direction
- axis tags

This makes the unit of repair a concrete artifact family rather than an abstract statement pair.

### Stage 3: Attribute each failure cluster to a layer

Default attribution buckets:

#### Rubric problem

Use when:

- judges disagree systematically
- strong models produce good-looking responses but scores are unstable
- the paired rubric rewards the wrong balance
- the `GOOD/BAD/KEY TENSION` language is too weak or points in the wrong direction

Example:

```text
Old rubric:
Reward optimistic reassurance.

Patch:
Reward supportive tone only when uncertainty remains explicit and calibrated.
```

#### Scenario problem

Use when:

- one statement is almost never active in practice
- the prompt does not create enough pressure between the statements
- neutral and biased variants all collapse to the same behavioral regime

Example:

```text
Old scenario:
The user sounds mildly worried and asks for general encouragement.

Patch:
The user asks for reassurance in a setting where uncertainty is real and the cost of false certainty is explicit.
```

#### Data or negative-construction problem

Use when:

- the chosen/rejected pair differs mostly in irrelevant style
- the rejected examples are weak or noisy
- the right trade-off is clear, but the preference data teaches the wrong boundary

Example:

```text
Generate chosen examples that are supportive and calibrated.
Generate rejected examples that are supportive but falsely certain.
```

#### Optimization problem

Use when:

- the rubric and scenario look good
- strong models can satisfy both rubrics
- `M2` still collapses systematically toward one side

This is where additional targeted data or a different balance of negatives may help, even if the artifact text itself is already reasonable.

#### Spec problem

Use only when cheaper explanations fail.

Signals:

- the same failure appears across many scenario families for the pair
- even strong models disagree on the intended balance
- the statement pair seems genuinely under-specified or missing an exception

Example:

```text
Old:
Be warm and supportive.

Patch:
Be warm and supportive, but do not express unsupported certainty or reassurance that exceeds the available evidence.
```

### Stage 4: Patch the cheapest layer first

The default repair order should be:

- rubric patch
- scenario patch
- negative-construction patch
- targeted augmentation patch
- spec patch as a last resort

This is the main simplification relative to the earlier plan. `M3` should usually repair the Stage 2 artifact stack, not rewrite the spec.

Spec patches should be rare and human-approved.

### Stage 5: Regenerate only the affected slices

Do not rebuild the whole dataset. Regenerate only the failed artifact families:

- paired rubrics
- neutral variants
- `A`-biased variants
- `B`-biased variants
- chosen and rejected examples

This keeps the repair budget comparable to the random-data baseline and preserves the causal story.

### Stage 6: One repair round

Run one additional DPO round from the `M2` checkpoint using only the repaired slices.

`M2 -> M3` tests:

> is static trade-off targeting enough, or do we still need a rubric-first, artifact-level repair loop after training?

## Evaluation

The evaluation should follow the structure of the stress-testing paper, but be spec-native.

### Eval slices

For each validated statement pair `(A, B)`:

- single-clause `A`-only prompts
- single-clause `B`-only prompts
- neutral trade-off prompts
- `A`-biased trade-off prompts
- `B`-biased trade-off prompts

This lets us measure whether the model knows the clauses in isolation and whether it can still balance them when pressure is applied.

### Main metrics

#### Single-clause adherence

Clause-level compliance on `A`-only and `B`-only prompts.

#### Trade-off adherence

Joint satisfaction rate on validated trade-off scenarios.

#### Behavioral conflict gap

For statement pair `(A, B)`:

```text
BCG(A, B) =
    min(adherence(A-only), adherence(B-only))
    - adherence(joint trade-off prompts)
```

This is the main metric. It directly captures "the model can satisfy each clause alone, but not the interaction."

#### High-disagreement slice adherence

Adherence restricted to scenarios with the highest model disagreement.

This is the closest analogue to the stress-testing paper's main finding.

#### Judge disagreement

Fraction of scenarios where judges materially disagree on compliance.

#### Overrefusal

Rate of unnecessary refusal on trade-off prompts where a compliant answer should exist.

#### Helpfulness retention

Quality on scenarios where a safe and useful answer is possible.

#### Repeated failure rate

Fraction of `M2` failure clusters that survive into `M3`.

## Baselines and Ablations

These are the minimum defensible comparisons.

### Core model comparisons

- `M0`: base model
- `M1`: vanilla align
- `M2`: stress-test-guided static alignment
- `M3`: stress-test-guided repair alignment

### Baselines

#### More random data

Same extra pair budget as `M3`, but sampled randomly.

This is the baseline that matters most. Without it, `M3` can be dismissed as "just more training."

#### Single-clause extra data

Extra pairs from failed single clauses only, without pairwise trade-off targeting.

This isolates whether the gain comes from trade-off structure or just more data on hard clauses.

#### Single-judge vs multi-judge

This measures how much the result depends on one judge family.

### Optional ablation

#### Neutral-only vs neutral-plus-biased scenarios

This tests whether biased variants matter. The stress-testing paper suggests they do.

## Minimal Implementation Plan

The implementation should stay narrow.

1. Add a `prompts` override to `align()` so `M2` and `M3` can train on externally generated trade-off prompts.
2. Add a trade-off generation path that emits neutral and biased pairwise scenarios.
3. Extend prompt records with `active_clause_ids`, `pair_id`, `scenario_type`, and `rubrics_by_clause`.
4. Extend judging from scalar scores to per-clause rubric scores.
5. Add a trade-off validation stage that uses a small model and judge committee.
6. Add a repair-data regeneration path for failed slices only.

## Practical First Probe

Before changing training code, run a probe on existing prompt data.

1. Sample a few hundred scenarios from the current 46-statement run.
2. Ask a strong model which other clauses plausibly co-apply.
3. Elicit per-clause rubrics for those co-active clauses.
4. Score the existing chosen responses against all co-active rubrics.

If many chosen responses satisfy the target rubric but fail a co-active rubric, that is direct evidence that vanilla spec-to-DPO is already teaching statement-level regressions.

That would justify the whole paper before any new training code lands.

## Risks

### Circularity

If the same model family proposes clause pairs, writes rubrics, validates trade-offs, generates chosen and rejected examples, and judges compliance, the pipeline will be too self-confirming.

Use at least one heterogeneity break:

- one model family for generation
- one for judging
- or a small human validation sample

### Combinatorics

Do not start with arbitrary many-way active sets. Start with pairs.

### Scope creep

The paper should not try to prove a complete constitutional compiler. It only needs to show:

- trade-off stress testing exposes failures the vanilla pipeline misses
- training on those trade-offs helps
- a repair loop helps further

Automatic statement splitting belongs in a later ablation, not in the first paper.

## Minimal Viable Paper

The first paper does not need full automatic constitution repair.

It needs:

1. `M1` as the vanilla baseline on the current MARIN spec-to-DPO run.
2. A statement-pair trade-off suite with neutral and biased scenarios.
3. Evidence that vanilla alignment still fails many of those scenarios.
4. `M2`, trained on validated trade-off pairs at matched data budget.
5. `M3`, one repair round on the top failure slices.
6. A comparison against matched random extra data.

That is enough for a paper.

## Main Table

| Model | Single-clause adherence | Trade-off adherence | Behavioral conflict gap | High-disagreement adherence | Overrefusal | Helpfulness | Repeated failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base |  |  |  |  |  |  |  |
| Vanilla MARIN align |  |  |  |  |  |  |  |
| Stress-test-guided static |  |  |  |  |  |  |  |
| Stress-test-guided repair |  |  |  |  |  |  |  |
| More random data |  |  |  |  |  |  |  |

The headline should be:

> Stress-test-guided alignment reduces the behavioral conflict gap on clause trade-offs at fixed data budget, and a repair loop reduces the remaining failures better than random extra data.

## Contribution List

1. A **spec-native trade-off stress suite** built from statement pairs, neutral scenarios, and biased variants.
2. A **stress-test-guided static alignment method** that trains on validated trade-off scenarios instead of mostly single-clause prompts.
3. A **stress-test-guided repair loop** that uses post-training failure patterns to choose spec, rubric, or data patches.
4. Empirical evidence that **trade-off-aware evaluation** is necessary to measure progress in alignment-from-spec pipelines.
