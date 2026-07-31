# Debugging log for fixed-subset 300M noise estimates

## Overview

The goal was to repair missing noise estimates for the existing fixed-subset `run_00097` 300M/6B baseline. The raw metric matrix and SNR outputs should use the ten fixed-subset seed rows as the noise baseline until the variable-subset baseline is complete.

## Initial Status

The fixed-subset matrix existed, but it only had 17 complete BPB columns. Core metrics such as `eval/uncheatable_eval/bpb`, `eval/paloma/bpb`, and `lm_eval/mmlu_5shot/bpb` were missing for all ten fixed-subset rows in the local matrix. The 300M signal matrix itself was not missing these metrics.

## Hypothesis: Fixed-Subset Collect Results Were Not Overlaid

The fixed-subset runs had complete metrics in the original GCS collect output:

`gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_fixed_subset/collect_results-605e6a/results.csv`

That file has 10 seed rows and 84 complete BPB columns, including Paloma, uncheatable, and MMLU subject BPBs. The local SNR loader only fell back to this file when fixed-subset rows were absent from `metrics_wide.csv`. Since sparse fixed-subset provenance rows were present, the fallback never ran.

## Changes

- Updated `build_eval_signal_to_noise_all_metrics_300m.py` so fixed-subset rows from `metrics_wide.csv` are always overlaid with the fixed-subset GCS collect metrics by `wandb_run_id`.
- Updated `metric_registry/build_raw_metric_matrix_300m.py` so the fixed-subset raw matrix uses the enriched fixed-noise loader instead of the sparse `metrics_wide.csv` rows.
- Fixed the SNR signal-side population to use only the 242 `cohort == signal` 300M mixture rows. The previous path loaded all 300M rows, including seed-sweep and other non-signal rows, which made `signal_n` drift above 242.

## Results

Regenerated:

- `metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv`
- `metric_registry/raw_metric_matrix_300m/noise_baseline_run00097_fixed_subset_300m.csv`
- `metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m_with_noise.csv`
- `eval_signal_to_noise_all_metrics_300m_current.csv`
- `eval_signal_to_noise_all_metrics_300m_current_task_summary.csv`
- `eval_signal_to_noise_all_metrics_300m_current_keep_drop.csv`
- `eval_signal_to_noise_all_metrics_300m_current_summary.json`

Post-fix audit:

- Fixed-subset rows: 10.
- Fixed-subset metric columns: 640.
- Fixed-subset complete BPB columns: 105, up from 17.
- `eval/uncheatable_eval/bpb`, `eval/paloma/bpb`, `lm_eval/mmlu_5shot/bpb`, teacher-forced GSM8K BPB, and MCQ-smooth MedMCQA BPB all have 10/10 fixed-subset values.
- Current SNR summary uses `signal_rows = 242`, `noise_rows = 10`, `noise_subset_mode = fixed`, and every emitted SNR row has `noise_n = 10`.
- Of the 43 BPB columns used in the earlier IRT reproduction, 39 now have complete fixed-subset noise values. The remaining exact aliases are not in the fixed-subset collect output: `lm_eval/hellaswag_0shot/bpb`, `lm_eval/piqa/bpb`, `lm_eval/arc_easy/bpb`, and `lm_eval/mmlu_sl_verb_5shot/bpb`.

## Future Work

- [ ] Once the variable-subset baseline finishes and has full downstream metrics, switch `--noise-subset-mode auto` to prefer it for operational SNR while preserving fixed-subset for trainer-only noise decomposition.
- [ ] Consider backfilling fixed-subset MMLU-SL-Verb if it becomes important; the current fixed-subset collect output does not include `lm_eval/mmlu_sl_verb_5shot/bpb`.
- [ ] For the IRT input specifically, either drop the four exact aliases without fixed-subset noise or run targeted fixed-subset evals for those aliases.

## Variable-Subset Follow-Up

The variable-subset baseline completed on GCS with 10/10 final `step-22887` checkpoints and HF exports under:

`gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_variable_subset`

Added a direct-GCS loader for this case because the heavy registry builder was not usable in the local environment and the metric registry had not yet ingested these rows. The loader reads each child `checkpoints/eval_metrics.jsonl`, extracts the exact `step=22887` row, and tags the rows as `cohort=seed_sweep`, `source_experiment=pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_variable_subset`.

Regenerated raw matrix outputs now have:

- Signal rows: 242.
- Fixed-subset noise rows: 10.
- Variable-subset noise rows: 10.
- `raw_metric_matrix_300m_with_noise.csv` rows: 262.

Variable-subset metric coverage is validation/perplexity-only for now:

- Variable-subset metric columns: 60.
- Variable-subset complete BPB columns: 26.
- No downstream `lm_eval`, `teacher_forced`, or `mcq_smooth` metrics yet for the variable-subset rows.

The default SNR builder now uses `noise_subset_mode=variable` when all 10 variable rows are available. This produces 60 SNR rows, all validation/perplexity metrics, with `signal_n=242` and `noise_n=10`.

Key fixed-vs-variable noise comparison CSV:

`reference_outputs/variable_subset_noise_20260501/fixed_vs_variable_noise_key_metrics.csv`
