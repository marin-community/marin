# Debugging log for 300M raw metric matrix audit

## Overview

Audit the refreshed 300M raw metric matrix against a previously exported copy at
`/Users/calvinxu/Downloads/raw_metric_matrix_300m.csv`, identify value/schema differences, and explain
the `baseline_olmix_loglinear_uncheatable_bpb` status change.

## Initial Status

The refreshed matrix and downloaded matrix both had 242 rows, but initially differed in column count:

- Downloaded matrix: 1,201 columns.
- Refreshed matrix: 1,197 columns.

The downloaded Olmix row had `status=failed` and `source_cohort=original_swarm_300m`; the refreshed row
had `status=completed` and `source_cohort=signal`.

## MMLU Category BPB Columns

The refreshed matrix initially dropped four columns:

- `lm_eval/mmlu_humanities_5shot/bpb`
- `lm_eval/mmlu_other_5shot/bpb`
- `lm_eval/mmlu_social_sciences_5shot/bpb`
- `lm_eval/mmlu_stem_5shot/bpb`

Root cause: `_add_mmlu_category_smooth_metrics` skipped derivation when the output column already existed,
even if the existing column was all null. `metrics_wide.csv` had empty category-level BPB columns, while
subject-level BPB columns existed.

Fix: derive category smooth metrics and fill existing null category columns with subject-category means.

Result after rebuilding:

- Both matrices have 242 rows and 1,201 columns.
- No column set differences remain.
- No numeric metric or weight values differ on common rows/columns.

## Olmix Status Change

The metric values for `baseline_olmix_loglinear_uncheatable_bpb` did not change. The metadata changed:

- Downloaded matrix: `status=failed`, from `local_qsplit240_300m_6b_partial_results`.
- Refreshed matrix: `status=completed`, from the higher-priority downstream-eval overlay
  `local_300m_gsm8k_humaneval_completion`.

The operational run registry still records the original 300M Olmix training parent as failed, while the
checkpoint exists and downstream eval artifacts have completed. Therefore the matrix `status` column is
not a pure training-status field; it is inherited from the metric registry's canonical run row. The raw
matrix README now documents this caveat.

## Results

The previous matrix was stale/wrong in provenance fields:

- It used simplified `registry_run_key` values without `source_experiment`.
- It retained `source_cohort=original_swarm_300m` instead of the normalized `signal` cohort for Olmix.

The refreshed matrix is numerically equivalent after the MMLU category BPB fix, and has better canonical
registry keys. The only remaining semantic caveat is the overloaded `status` column.
