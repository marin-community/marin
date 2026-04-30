# Logbook: Spec-Driven Alignment / Tension-Atlas Project

Working logbook for the current `gpt5_paper_plan.md` line of work on
spec-driven alignment, trade-off stress testing, and the `M1 -> M2 -> M3`
ladder.

This file records the current state after:

- reviewing the Bloom synthetic-data pipeline artifacts
- reading the related-work notes in the Obsidian vault
- reading the stress-testing-model-specs paper notes
- inspecting the implemented `experiments/posttrain/` pipeline
- auditing the BCG metric against the full-atlas results
- regenerating the comparison tables and plots offline with `JSR/BJS`

---

## Project status

The project is alive, but the framing changed materially.

The old version of the story was:

- build a conflict graph
- define Behavioral Conflict Gap (BCG)
- show DPO is a high-variance intervention on trade-offs
- train `M2` on high-BCG points

That version is no longer the right one.

The current version is:

- build a **tension atlas** over the 46 spec statements
- evaluate trade-offs using **Joint Satisfaction Rate (JSR)** and
  **Balanced Joint Score (BJS)**
- always condition results on **oracle feasibility slices**
- treat `M1` as the broad-coverage baseline
- treat `M2` as **broad + targeted tension data**
- treat `M3` as **artifact-level repair**, mostly on tension/rubric data,
  not spec rewriting

The metric correction matters a lot. It changed the empirical story from
"DPO is a near wash with many regressions" to
"DPO improves feasible trade-offs broadly, but still leaves a clear residual
gap and some sharp local failures."

---

## Core decisions we have made

### 1. Use the 46 statements as-is

For the first paper, use the original 46 spec statements as the unit of
analysis.

We are **not** making a clause compiler a dependency for v1.

Possible later ablation:

- manually split obviously compound statements

But that is not part of the core path.

### 2. Keep broad coverage from the original pipeline

The original MARIN/Bloom-style synthetic data pipeline is still valuable.
It gives broad coverage of:

- single-statement adherence
- ordinary helpfulness / tone / style behavior
- non-corner cases

The tension atlas does something different:

- identifies **where** co-equal statements sharply interact
- gives concrete prompts at those corners
- makes better chosen/rejected trade-off data possible

So the right framing is:

- `M1`: broad baseline
- `M2`: broad + tension
- `M3`: broad + tension + repair

Not:

- throw away the broad pipeline and train only on tensions

### 3. Use the implemented tension-atlas path, not ideation mining, as the main frontend

We initially considered mining Bloom Stage 2 `ideation.json` artifacts for
co-active scenarios.

That is still useful context, but the implemented path is better as the main
discovery/eval frontend:

- [experiments/posttrain/stage1_pair_propose.py](../experiments/posttrain/stage1_pair_propose.py)
- [experiments/posttrain/stage2_tension_atlas.py](../experiments/posttrain/stage2_tension_atlas.py)
- [experiments/posttrain/stage3_paired_rubrics.py](../experiments/posttrain/stage3_paired_rubrics.py)

Why it is better:

- it is pair-native
- it targets specific cross-axis corners
- it avoids sparse-coverage dependence on existing single-statement scenario pools
- it gives a much more interpretable object than "this scenario also seems to activate statement B"

The tension atlas should exclude demographic axes. Those are useful for fairness
evaluation, but not as the primary source of trade-off structure.

### 4. `M3` should be rubric-first / artifact-first

The default repair order for `M3` should be:

1. paired-rubric patch
2. tension-corner patch
3. prompt-variant patch
4. negative-construction patch
5. spec patch only as last resort

This is important. Most remaining failures should be repaired in the
artifact stack, not by rewriting the spec.

---

## What we learned about the original synthetic-data pipeline

Bloom's original synthetic data stages are:

- Stage 1: `understanding.json`
- Stage 2: `ideation.json`
- Stage 3: `eval_prompts.json`

That original pipeline still matters because it tells us what artifacts already
exist and what "broad coverage" means in practice.

Important clarification:

- the current stress-testing pipeline has its own stage numbering
- its "Stage 2" is **not** Bloom Stage 2

Current stress-testing stages are:

- Stage 1: pair proposal
- Stage 2: tension atlas
- Stage 3: paired rubrics
- Stage 4: evaluation

This naming collision caused confusion and should be documented clearly in any
future writeup.

---

## What we learned about coverage

The original per-statement prompt generation does **not** give full coverage of
cross-statement trade-off space.

The core problem is not concretization loss. Concretization is fine.

The real issue is:

- each statement has its own axis space
- the scenario set is a thin sample of that space
- there is no cross-statement coverage guarantee

This is why the tension-atlas idea is important.

The better object is:

- a pair of statements
- a set of behavior-specific axes
- a small number of **named high-tension corners**

This lets us generate directly at the corner instead of hoping sparse scenario
sampling happened to hit it.

---

## Metric correction: BCG was a mistake

After reading `.agents/logbooks/claude_stress_testing.md` Experiment 13 and
auditing the current outputs, BCG should not be the headline metric.

### Why BCG is bad

BCG in code was:

```text
BCG = min(mean_A_score, mean_B_score) - joint_rate * 10
```

This is bad for four reasons:

1. It mixes incompatible quantities.
2. It is dominated by coarse threshold jumps when `N=3`.
3. It produces many strongly negative values even when trade-off handling is good.
4. It hides global-failure points where the model is just bad at both rubrics.

Most importantly, it distorted the project's main empirical story.

The old BCG framing suggested:

- DPO is a near wash
- DPO is highly bimodal / high-variance

The corrected framing says:

- DPO helps substantially more than it hurts
- the gain is especially clear on oracle-feasible points
- there is still large residual headroom to the oracle

### Replacement metrics

Use:

- **JSR**: joint satisfaction rate
- **BJS**: balanced joint score = harmonic mean of the marginals / 10

And always decompose by oracle feasibility:

- **Feasible**: oracle JSR `>= 2/3`
- **Marginal**: oracle JSR `= 1/3`
- **Infeasible**: oracle JSR `= 0`

BCG may remain in tables as a deprecated diagnostic only.

---

## Current numbers after offline recompute

These were regenerated locally from cached Stage 4 summaries only. No new API
calls were needed.

Artifacts:

- [experiments/posttrain/stage4_output/comparison_full.md](../experiments/posttrain/stage4_output/comparison_full.md)
- [experiments/posttrain/stage4_output/comparison_full.png](../experiments/posttrain/stage4_output/comparison_full.png)
- [experiments/posttrain/stage4_output/comparison_radar_full.png](../experiments/posttrain/stage4_output/comparison_radar_full.png)
- [experiments/posttrain/stage4_output/comparison.md](../experiments/posttrain/stage4_output/comparison.md)
- [experiments/posttrain/stage4_output/comparison.png](../experiments/posttrain/stage4_output/comparison.png)
- [experiments/posttrain/stage4_output/comparison_radar.png](../experiments/posttrain/stage4_output/comparison_radar.png)

### Full atlas

From `comparison_full.md`:

- oracle overall JSR: `0.523`
- `M0` overall JSR: `0.196`
- `M1` overall JSR: `0.316`

Oracle feasibility decomposition:

- Feasible: `1346 / 2547` = `52.8%`
- Marginal: `296 / 2547` = `11.6%`
- Infeasible: `905 / 2547` = `35.5%`

Feasible slice:

- oracle JSR: `0.917`
- `M0` JSR: `0.303`
- `M1` JSR: `0.473`

Feasible-slice DPO effect on shared points:

- JSR improvements: `486`
- JSR regressions: `184`
- win/loss ratio: `2.64x`
- mean delta JSR: `+0.170`

This is the current headline.

### Probe (50-point sample)

Probe numbers are noisier but directionally consistent.

From `comparison.md`:

- Feasible points: `18 / 48`
- Feasible-slice `M0` JSR: `0.361`
- Feasible-slice `M1` JSR: `0.500`

This is now useful as a sanity check only. The full-atlas outputs should drive
the paper story.

---

## Current interpretation of the plots

The corrected radar plots now make sense:

- oracle is high and fairly flat across families
- `M1` is above `M0` in every family on the feasible slice
- `Style / Structure` is the weakest family for `M1`
- the gap to oracle remains large everywhere

Important caveat:

Family means do **not** imply absence of local regressions.

There are still sharp pair-level failures, but they are no longer the main
aggregate story.

---

## Current M2 plan

### What `M2` is

`M2` is the static alignment method.

It should:

- keep broad coverage from the original pipeline
- add targeted trade-off data at validated tension corners
- focus on oracle-feasible corners where `M1` still underperforms

### Data recipe

Main paper comparison currently leans toward:

- `M1`: 100% `D_broad`
- `M2`: fixed-budget mixture of `D_broad + D_tension`

Suggested first recipe:

- `70–80% D_broad`
- `20–30% D_tension`

This is a research-design choice, not a product rule.

If the all-data variant tells a cleaner or more useful story, we should follow
the results.

### M2 target definition

The current target filter should be:

- `feasibility_slice == feasible`
- large `oracle JSR - M1 JSR`
- `weakest_marginal_score >= 3`
- low enough judge disagreement

Rank by:

1. JSR drop from oracle to `M1`
2. BJS as a secondary tie-break

Not by BCG.

### M2 supervision object

The supervision object is a **tension-point dossier**:

- statement pair
- tension name
- peak corner
- example prompt
- paired rubrics
- optional neutral / `A`-biased / `B`-biased prompt variants

Chosen/rejected pairs should be generated directly from these retained points.

---

## Current M3 plan

`M3` is the repair stage after `M2`.

It should:

- rerun the same tension-atlas evaluation
- cluster remaining failures by artifact family
- repair the cheapest plausible layer first
- retrain once with repaired slices and broad replay

High-disagreement and oracle-infeasible points are mainly `M3` / analysis
targets, not `M2` training targets.

The oracle-infeasible slice is a separate paper finding:

> some statement interactions are structurally unsatisfiable for a frontier
> oracle and therefore not patchable by post-training alone.

---

## Current writeup state

`gpt5_paper_plan.md` has been updated to reflect the corrected direction:

- tension atlas instead of ideation-mining-first
- broad + tension mixture for `M2`
- rubric/tension repair for `M3`
- `JSR/BJS` instead of BCG
- feasibility decomposition
- explicit note that fixed-budget is a current leaning, not a locked decision

The supporting scripts were also updated so the repo is not telling two
different metric stories:

- `stage4_bcg_eval.py` now documents BCG as deprecated and emits BJS
- `stage4_compare.py` and `stage4_full_plots.py` now write JSR/BJS-framed outputs
- the Stage 1 docstring now says Stage 2 is the tension atlas, not ideation mining

---

## Immediate next steps

The next step is **not** `M3`.

The next step is to convert the corrected full-atlas evaluation into an
actual retained `M2` target list.

### Immediate plan

1. Read `comparison_full.json`.
2. Keep only oracle-feasible points.
3. Rank by `oracle JSR - M1 JSR`.
4. Filter out low-competence global-failure points via weakest marginal.
5. Group retained points by statement pair.
6. Manually review the top `30–50` points for rubric/pathology issues.
7. Freeze the first `D_tension` slice.

Only after that should we:

8. generate chosen/rejected data
9. train `M2`
10. rerun the same evaluation

### What would count as success

For `M2`, success means:

- better feasible-slice JSR than `M1`
- better feasible-slice BJS than `M1`
- no large single-statement regressions
- a clean story against the matched random-data baseline

---

## Open questions

1. **Fixed-budget vs all-data main comparison**
   The current leaning is fixed-budget for the paper, all-data as a practical
   variant. This is still open.

2. **How much manual review before M2**
   The current instinct is to manually inspect the top retained points before
   using them for training.

3. **When to increase N**
   `N=3` is fine for wide screening. A later pass on retained points should use
   `N=5` or `N=10`, but only after the retained set is defined.

4. **What to do with the infeasible slice**
   It should likely become a secondary paper result rather than a training
   target.

---

## Current one-paragraph project summary

The project should now be framed as a tension-atlas alignment paper, not a BCG
paper. We use the 46 statements as-is, build a pair-native atlas of
high-pressure cross-axis corners, score those corners with paired rubrics, and
measure trade-off handling with JSR/BJS conditioned on oracle feasibility.
`M1` provides broad baseline coverage. `M2` adds targeted tension-corner data
to that broad pool. `M3` repairs the remaining failures mostly in the
tension/rubric artifact stack rather than by rewriting the spec. The immediate
next task is to turn the corrected full-atlas evaluation into a reviewed
retained tension set for the first `M2` run.

---

## 2026-04-20 — M2 target-selection pass from corrected full-atlas outputs

### Goal

Turn the corrected `comparison_full.json` output into:

- a broad candidate pool of feasible residual trade-offs
- a review set for manual inspection
- a provisional first `D_tension` slice

### Inputs

- [experiments/posttrain/stage4_output/comparison_full.json](../experiments/posttrain/stage4_output/comparison_full.json)

### Commands

```bash
python3 -m py_compile experiments/posttrain/select_m2_targets.py
python3 experiments/posttrain/select_m2_targets.py
```

### Selection rule used

Candidate pool:

- `feasibility_slice == feasible`
- `weakest_marginal_score >= 3.0`
- `oracle JSR - M1 JSR >= 2/3`

Ranking:

1. larger oracle-vs-`M1` JSR gap
2. lower `M1` JSR
3. lower `M1` BJS

Provisional seed exclusions:

- `support_programmatic_use`
- `formatting`
- `letter_and_spirit`
- `no_agenda`

These were excluded from the first seed slice because they are more likely to
be dominated by format/meta artifacts on a first pass.

### Outputs

- [experiments/posttrain/select_m2_targets.py](../experiments/posttrain/select_m2_targets.py)
- [experiments/posttrain/stage4_output/m2_candidate_pool.json](../experiments/posttrain/stage4_output/m2_candidate_pool.json)
- [experiments/posttrain/stage4_output/m2_candidate_pool.csv](../experiments/posttrain/stage4_output/m2_candidate_pool.csv)
- [experiments/posttrain/stage4_output/m2_pair_summary.csv](../experiments/posttrain/stage4_output/m2_pair_summary.csv)
- [experiments/posttrain/stage4_output/m2_target_review.md](../experiments/posttrain/stage4_output/m2_target_review.md)
- [experiments/posttrain/stage4_output/m2_seed_slice.json](../experiments/posttrain/stage4_output/m2_seed_slice.json)

### Results

Candidate pool:

- `452` tension points
- `249` statement pairs

Review set:

- top `50` points by the ranking above

Provisional seed slice:

- `40` points
- `38` unique statement pairs
- capped at `2` points per pair

The top-ranked retained points all have:

- oracle JSR `= 1.0`
- `M1` JSR `= 0.0`
- weakest marginal `>= 3`

So the retained set is not being driven by global incompetence or infeasible
oracle cases. These are exactly the "oracle can do it, `M1` cannot jointly do
it" corners we wanted.

### Interpretation

This pass confirms that the corrected eval leaves a large, nontrivial residual
`M2` target set. We are not scraping together a few edge cases.

The main trade-off now is not "do we have enough targets?" It is "how
conservative should the first seed slice be?"

Current read:

- the broad candidate pool is large enough for multiple `M2` variants
- the first seed slice should stay semantically clean
- artifact-heavy statement families can be layered back in later if needed

### Limitation

This review used the scored full-atlas summary, not the full paired-rubric
JSONL. The checked-in artifact contains:

- `pair_id`
- `tension_name`
- JSR/BJS/marginals

but not the full `example_prompt` and rubric text.

So this is a **summary-level review**, not the final hand audit. Before
generating `D_tension`, we should inspect the underlying prompt/rubric dossiers
for the provisional seed slice.

### Next step after this pass

Load the underlying dossiers for the `m2_seed_slice.json` points, inspect
prompt/rubric quality, then generate chosen/rejected examples for the first
`M2` training run.
