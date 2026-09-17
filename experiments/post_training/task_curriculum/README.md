# Task curriculum pilot

This directory contains the first task-level curriculum catalog and the code used to validate it. It separates four records:

1. `macro_areas.json` is the broad coverage inventory. It vendors the active hierarchy from the September 3 TaskTrove competency audit: 34 active macro areas and 151 active micro areas. The source vocabulary has 35 macros; one has no active child in the published chart. Credited task counts are source-level, overlap across competencies, and are not task labels.
2. `math_v0.json` contains six candidate units derived from 24 TaskTrove Clean discovery tasks. A unit states an observable outcome, its boundaries, and representative examples. It does not select a verifier.
3. A semantic task key contains a summary, subject hint, hardest operation, required operations, and answer form. This remains independent of any curriculum version. TaskCompendium's `TaskSpec` remains the owner of task semantics and private correctness contracts.
4. A versioned assignment records macro and unit candidates, distances, thresholds, and status. Changing a curriculum re-embeds changed unit text and searches the cached task vectors. It does not rerun task annotation.

The broad inventory is a navigation layer. Macro and micro names seed coverage work; they are not trainable units. `math_v0.json` maps every unit to a macro area and uses a micro area only when the published label fits. Triangle metrics and pairing/involution counting intentionally have no micro mapping because the current mathematics vocabulary has no geometry or combinatorics label.

Broad and local assignment use different projections of the same semantic key. Macro routing embeds subject, summary, and hardest operation. Unit routing embeds the hardest operation alone. In the pilot, adding subject and summary improved macro top-one accuracy from 35/58 to 45/58 and mathematics top-one accuracy from 32/48 to 40/48. Mathematics reached 48/48 top-three recall. A conservative threshold assigned only 2 of 17 exact-macro holdout tasks, so label-name anchors are not adequate for production macro routing.

## Assignment states

`assignment.py` performs hierarchical nearest-anchor assignment:

- `assigned`: macro and unit pass their distance and margin thresholds;
- `ambiguous`: the macro or unit winner is too close to its runner-up;
- `coverage_gap`: the task fits a known macro, but no local unit fits;
- `outside_inventory`: no macro anchor is close enough.

The distinction between `coverage_gap` and `outside_inventory` matters. Most held-out mathematics tasks missed by the six-unit pilot are evidence that the local catalog is incomplete, not evidence that they lie outside a global curriculum.

The blind macro review found 24 micro-vocabulary gaps in 64 tasks. Twenty were mathematics tasks involving geometry, combinatorics, number theory, or abstract algebra. Five contrast tasks exposed macro gaps for chemistry, physics/materials science, and ecology. The inventory also mixes domain and workflow axes: `Knowledge Application & QA` overlaps clinical and scientific reasoning. Keep these as explicit gaps until a broader sample supports additions.

## Rubric

`rubric_v1.json` scores each unit from 0 to 2 on observable outcome, boundary quality, evidence support, generation feasibility, generated-task validity, and difficulty gradient. A ready unit needs at least 10 of 12 points, three reviewed examples from at least two sources, at least 95% valid generated tasks, and no fatal flaw.

The initial evidence in `pilot_evaluation.json` leaves all six units as candidates. Three lack cross-source discovery support, the other three have only two discovery examples, five generated-task sets were solved perfectly by Luna, and geometric optimization generated one wrong gold.

Run the checked-in evaluation with:

```bash
uv run python -m experiments.post_training.task_curriculum.evaluate
```

The next evaluation should add a balanced sample of exact members and close non-members. Catalog metrics are covered-task top-two recall, confident assignment precision, covered-task confident recall, coverage-gap recall, and outside-inventory recall. Prerequisite edges require a separate transfer experiment and remain empty in `math_v0`.

## Provenance

- [TaskTrove competency coverage audit](https://storage.googleapis.com/marin-public/benjaminfeuer/tasktrove-competency-coverage/2026.09.03/index.html)
- [Coverage issue #8879](https://github.com/marin-community/marin/issues/8879)
- [TaskCompendium PR #9187](https://github.com/marin-community/marin/pull/9187)
