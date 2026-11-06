# BioTasks frozen-reference audit (preliminary)

Scope: 305 combined-v1 frozen TaskSpecs and the GLM-5.3 OpenAI-compatible
ShellSim-v2 trial. This is **not** a repaired bank or a scientific release
review. Frozen TaskSpecs and calibration scores must not be modified in place.

## Solver failures versus task defects

The 305 trials included 231 normally graded submissions, 60 completed
submission-format failures, 13 ungraded submission failures, and one unavailable
trial. Regrading JSON fragments extracted from the 60 completed format failures
found parseable candidate JSON in 59 and exact frozen-reference matches in 33.
This is diagnostic only: the frozen bare-JSON policy still scores malformed
submissions zero. The 49 normally graded but wrong responses and the remaining
format failures must not automatically be attributed to defective tasks.

Static checks identified 42 instructions without a literal mention of every
top-level expected key and 114 without a literal mention of every nested key or
label; many nested names are naturally data-derived, so these are review
queues, not defect counts. No serialized expected answer was found verbatim in
solver-visible resources.

## Independent recomputation is not self-validating

First-pass GLM recomputation returned 245 `confirmed`, 55 `incorrect`, and five
incomplete audits. A second GLM pass on the 60 unresolved records returned 20
`confirmed`, 35 `incorrect`, and five errors. These numbers **must not** be
interpreted as verified defect counts: the two passes share the same model,
and both make elementary errors and sometimes contradict their own corrected
answer JSON. Artifacts:

- `audits/reference-recomputation-glm-v1/results.jsonl` (SHA-256
  `da1901178dc343357e686dd287de0e9dca34985b97c8eef86ee337767433a02e`).
- `audits/reference-recomputation-glm-v1/adjudication.jsonl` (SHA-256
  `0fca4faf518d8911228f02ab1617f35653120ccd8924ff62d65701a4088c8394`).

Root: `s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/combined/v1/`.
These audit outputs include private solver-visible task data and must stay private.

### Deterministically checked examples

- **Confirmed defective reference:** `biotasks-read_counting_modes-78a95b29f211f753`.
  Read `e` occupies chr1:4000–4299, wholly inside *both* features E
  (4000–4499) and F (4000–4899). The instruction explicitly discards a read
  satisfying `IntersectionStrict` for more than one feature. Frozen reference
  has `intersection_strict.F=1`; the correct value is **0**. The GLM trial
  actually returned 0 and was scored wrong, so at least one measured model
  failure is demonstrably task-side.
- **Confirmed defective reference:** `biotasks-loss_function_computation-771b920f400a1b98`.
  For ET, probability sum 2.08, target sum 2 and probability–target dot
  product 1.4 yield Dice `2.8/4.08 = 0.6862745098...`, hence **0.686275**
  at six decimals, not the frozen **0.677966**. Correct loss, using *unrounded*
  Dice values, is **0.295571**, not the frozen **0.298341**. The GLM trial
  returned both correct values and was scored wrong. Notably both auditing
  GLM passes also computed the loss incorrectly as 0.295572: their proposed
  replacement answer must **not** be adopted.
- **False audit alarm:** `biotasks-signal_enrichment_scoring-51b7193a32ba9c5e`.
  Both audits claimed G1 has nonzero `w13` and a score of 4.6. Its actual CSV
  row has `w13=0`, so the frozen G1 score of **3.8** and top gene G5 are
  correct. The adjudicator's JSON also contradicted its own claimed ranking.
- **False audit alarm:** `biotasks-microbiome_diversity_from_biom_table-f210bcdd0707eea4`.
  Sample2 has counts 8, 8, 4 (proportions 0.4, 0.4, 0.2), not three equal
  proportions; Shannon entropy is **1.0549**, as frozen, not `ln(3)`.

## Required before recalibration or IRT

1. Build deterministic, task-family-specific recomputations for the disputed
   references, and compare **every** output field, including derived rankings,
   unrounded intermediates, indexing and rounding. Prioritize the 35 second-pass
   `incorrect` flags and five incomplete records, but do not assume the other
   265 are correct.
2. Separate `verified_reference_error`, `instruction_ambiguity`,
   `schema_or_format_friction`, `model_error`, and `unresolved` per task. Inspect
   original GLM submissions in each category; model-authored audit claims are
   only leads until independently recomputed.
3. Version any fixes with new frozen task IDs/digests and a manifest linking
   old/new tasks and verdict evidence. Do not silently amend combined-v1 or
   retroactively alter its calibration responses. Exclude unverified or
   defective items from official IRT until independently adjudicated.
