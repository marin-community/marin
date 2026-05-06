# Spec Repair v0 Report

Date: 2026-05-06 UTC

## Executive Summary

The E1 spec-repair loop stopped after two rounds because the locked numeric gate accepted no edits in two consecutive rounds. No `openai_model_spec_v1.jsonl` was written, and E5 final validation was not run because there is no changed spec to validate.

Round 1 used the handoff's original top-disagreement compiler split. Round 2 used a stratified compiler split plus previous-failure context to avoid deterministic duplicate spend. Round 2 produced better near-misses, especially for `formatting`, but still did not satisfy the full gate.

## Inputs And Targets

Grounding was rerun after Phase-4 GLM completed. It refreshed:

- `experiments/posttrain/disagreement_primitive/grounding/per_judgment.jsonl`
- `experiments/posttrain/disagreement_primitive/grounding/summary.csv`
- `experiments/posttrain/disagreement_primitive/grounding/per_statement.csv`
- `experiments/posttrain/disagreement_primitive/grounding/qualifier_drop.csv`
- `experiments/posttrain/disagreement_primitive/grounding/report.md`

Tier-C statements selected from Phase-4 GPT tension counts:

- `do_not_encourage_self_harm`
- `be_clear`

E1 target statements:

- `avoid_abuse`
- `assume_objective_pov`
- `comply_with_laws`
- `refusal_style`
- `formatting`
- `do_not_encourage_self_harm`
- `be_clear`

E2 qualifier-rubric regeneration wrote `experiments/posttrain/disagreement_primitive/e8_rubrics_v1.jsonl`. The conservative local qualifier check passed 13/16 targets and flagged `avoid_extremist_content`, `avoid_regulated_advice`, and `avoid_targeted_political_manipulation` for follow-up review.

## Round Results

Round 1 artifacts:

- Candidates: `experiments/posttrain/disagreement_primitive/repair_v0/round_1/`
- Verdicts: `experiments/posttrain/disagreement_primitive/repair_v0/round_1/verdicts.jsonl`
- Raw compiler log: `results/raw/e9_compile_edit_round_1/2026-05-06T08-30-54/`
- Raw verifier log: `results/raw/e9_verify_edit_round_1/2026-05-06T08-47-30/`

Round 1 gate result:

- 56/56 candidates failed.
- 0/7 statements received an applied edit.
- Median held-out `var_A` delta: -0.0134.
- Median compiler-input `var_A` delta: +0.0952.
- Only one candidate reached cross-condition checks: `avoid_abuse/minimal_03`.

Round 1 interpretation:

- The original top-disagreement split created substantial overfit. Many candidates improved compiler-input cases more than held-out cases.
- `avoid_abuse/minimal_03` was a real near-miss on `var_A` and phase4 but regressed full-spec kappa and improved Spearman for only one gate judge.

Round 2 artifacts:

- Candidates: `experiments/posttrain/disagreement_primitive/repair_v0/round_2/`
- Verdicts: `experiments/posttrain/disagreement_primitive/repair_v0/round_2/verdicts.jsonl`
- Raw compiler log: `results/raw/e9_compile_edit_round_2/2026-05-06T09-26-26/`
- Raw verifier log: `results/raw/e9_verify_edit_round_2/2026-05-06T09-30-28/`

Round 2 gate result:

- 56/56 candidates failed.
- 0/7 statements received an applied edit.
- Median held-out `var_A` delta: +0.0320.
- Median compiler-input `var_A` delta: 0.0000.
- Four candidates reached cross-condition checks, all under `formatting`.

Round 2 interpretation:

- The stratified split improved the situation relative to round 1. It produced multiple cross-condition candidates and reduced the median compiler-input bias.
- `formatting/rich_01` and `formatting/rich_02` passed the kappa portions of the gate but failed Spearman improved count. Both improved Spearman for only one of two gate judges.
- `avoid_abuse/rich_00` and `avoid_abuse/rich_02` had large held-out `var_A` gains but still failed overfit-gap because compiler-input deltas were much larger.

## Stop Decision

The locked handoff stop condition says to halt when the gate accepts no new edits for two consecutive rounds. Rounds 1 and 2 both accepted zero edits, so the E1 loop stopped.

No apply command was run because no candidate passed. No `openai_model_spec_v1.jsonl` was written. E5 final validation was skipped because validating an unchanged spec would spend judge calls without testing a real edit.

## Follow-Up If Restarted

The next useful experiment is not another unchanged E1 round. The evidence points to three targeted follow-ups:

- For `formatting`, inspect Spearman failures by judge and case. The kappa gates passed for two candidates, so the blocker is cross-condition rank consistency rather than agreement level.
- For `avoid_abuse`, reduce shown-case leakage further or evaluate a gate variant that compares held-out gain to an independent anchor set rather than the shown compiler-input set.
- For weak-edit targets such as `comply_with_laws` and `do_not_encourage_self_harm`, change the compiler framing. Case selection alone did not produce held-out movement.

The round-3 code path exists and defaults to `rotated_top`, but it should be run only if the stop condition is intentionally overridden.
