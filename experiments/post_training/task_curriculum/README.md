# Task curriculum pilot

This directory contains the first task-level curriculum catalog and the code used to validate it. It separates five records:

1. `macro_areas.json` is the broad coverage inventory. It vendors the active hierarchy from the September 3 TaskTrove competency audit: 34 active macro areas and 151 active micro areas. The source vocabulary has 35 macros; one has no active child in the published chart. Credited task counts are source-level, overlap across competencies, and are not task labels.
2. `micro_extensions_v0.json` records project-owned additions without rewriting that source snapshot. Geometry and combinatorics are provisional because repeated blind gaps support them, but they have not received human domain review. Evidence rows include the TaskTrove release, source identity, content hashes, frozen split, parent macro, and blind micro-gap status.
3. `math_v0.json` contains six candidate units derived from 24 TaskTrove Clean discovery tasks. A unit states an observable outcome, its boundaries, and representative examples. It does not select a verifier.
4. A semantic task key contains a summary, subject hint, hardest operation, required operations, and answer form. This remains independent of any curriculum version. TaskCompendium's `TaskSpec` remains the owner of task semantics and private correctness contracts.
5. A versioned assignment records macro and unit candidates, distances, thresholds, and status. Changing a curriculum re-embeds changed unit text and searches the cached task vectors. It does not rerun task annotation.

The broad inventory is a navigation layer. Macro and micro names seed coverage work; they are not trainable units. `math_v0.json` maps every unit to a macro area and uses either a published micro area or an evidence-backed local extension. The source snapshot remains unchanged when local task evidence exposes a gap.

Broad and local assignment use different projections of the same semantic key. Macro routing embeds subject, summary, and hardest operation. Unit routing embeds the hardest operation alone. In the pilot, adding subject and summary improved macro top-one accuracy from 35/58 to 45/58 and mathematics top-one accuracy from 32/48 to 40/48. Mathematics reached 48/48 top-three recall. A conservative threshold assigned only 2 of 17 exact-macro holdout tasks, so label-name anchors are not adequate for production macro routing.

A second blind sample drew eight tasks from each of six sources. The review found 42 exact macro matches, two ambiguous matches, and four inventory gaps. Calendar tasks all mapped to C21 and competitive-programming tasks all mapped to C13. The other four sources crossed macro boundaries; open-domain QA also exposed three biology or astrophysics gaps. Source mappings are therefore useful priors, but only calendar and competitive programming supported a single source-level macro in this sample.

The second sample also compared published label names with examples from the calibration half. On 22 exact-macro holdout tasks, the label names reached 10 top-one and 18 top-three matches. Task-derived nearest anchors covered 18 of the 22 tasks and reached 16 top-one and 18 top-three matches within that covered set. C01 and C28 had no calibration anchors. Add reviewed examples to every macro before fitting assignment thresholds.

## Assignment states

`assignment.py` performs hierarchical nearest-anchor assignment:

- `assigned`: macro and unit pass their distance and margin thresholds;
- `ambiguous`: the macro or unit winner is too close to its runner-up;
- `coverage_gap`: the task fits a known macro, but no local unit fits;
- `outside_inventory`: no macro anchor is close enough.

The distinction between `coverage_gap` and `outside_inventory` matters. Most held-out mathematics tasks missed by the six-unit pilot are evidence that the local catalog is incomplete, not evidence that they lie outside a global curriculum.

The blind macro review found 24 micro-vocabulary gaps in 64 tasks. Twenty were mathematics tasks involving geometry, combinatorics, number theory, or abstract algebra. Repeated examples across sources and splits support provisional geometry and combinatorics extensions. Chemistry, physics, ecology, abstract algebra, legal reasoning, and sequence interpretation remain recorded gaps because the pilot evidence is sparse. The inventory also mixes domain and workflow axes: `Knowledge Application & QA` overlaps clinical and scientific reasoning.

The broad sample stores `subject_domain` and `task_mechanic` as separate annotation fields. Calendar and competitive-programming sources were coherent by mechanic; open-domain QA was not. A two-facet catalog remains an experiment until a balanced cross-domain sample shows that both projections improve held-out routing.

## Rubric

`rubric_v1.json` scores each unit from 0 to 2 on observable outcome, boundary quality, evidence support, generation feasibility, generated-task validity, and difficulty gradient. Treat this score as a diagnostic. Its 10-of-12 threshold combines independent failure modes and uses pilot-scale sample counts as policy.

The next rubric revision uses three gate results: `pass`, `fail`, and `insufficient_evidence`. Units advance through `candidate`, `boundary_validated`, `generation_validated`, `difficulty_qualified`, and `ready_for_training_trial`. Boundary, generation, and difficulty gates require frozen evidence artifacts and an explicit sufficiency policy. Those policies are pending; the pilot does not define sample-size defaults.

All six units currently pass only the candidate gate. No unit has a frozen balanced boundary fixture. Five 3/3 generated-task reviews are insufficient to establish a 95% validity rate. Geometric optimization fails generation validity because one of three generated tasks had a wrong gold. Five units fail the current Luna difficulty batch because every valid task was solved, the ordering was implausible, or both. Modular exponentiation produced mixed outcomes with plausible ordering and remains insufficient pending a difficulty sufficiency policy.

Run the checked-in evaluation with:

```bash
uv run python -m experiments.post_training.task_curriculum.evaluate
```

The next evaluation should freeze per-unit fixtures with exact members, close non-members, ambiguous cases, coverage gaps, and outside-inventory tasks. Catalog metrics are covered-task top-two recall, confident assignment precision, covered-task confident recall, coverage-gap recall, and outside-inventory recall. Prerequisite edges require a separate transfer experiment and remain empty in `math_v0`.

## Provenance

- [TaskTrove competency coverage audit](https://storage.googleapis.com/marin-public/benjaminfeuer/tasktrove-competency-coverage/2026.09.03/index.html)
- [Coverage issue #8879](https://github.com/marin-community/marin/issues/8879)
- [TaskCompendium PR #9187](https://github.com/marin-community/marin/pull/9187)
