# Debugging log for metric registry missing values

## Initial Status

The metric registry was generated successfully, but downstream analysis needed an
audit for missing row fields and empty columns before using it as the canonical
source for fit datasets and W&B backfills.

## Hypothesis 1: Missingness Is Mostly Provenance Sparsity

The initial audit found no all-null columns in `runs.csv`, `metrics_wide.csv`,
`metrics_long.csv.gz`, or `eval_artifacts.csv`. The empty `backfills.csv` file
was all-null because there were no manual backfills yet.

Several high-null fields were expected:

- `trainer_seed` and `source_run_name` only apply to run_00097 seed/replay rows.
- `checkpoint_root` is missing for 300M rows whose current status is `not_found`.
- `num_fewshot` is missing for non-lm-eval metrics.

## Hypothesis 2: Some Source Metadata Should Have Been Filled at Ingestion

The audit found three accidental gaps:

- GCS collector rows had completed metrics but no `status`.
- Some rows had `checkpoint_root` but no `wandb_run_id`.
- The 300M partial-results source and some run_00097 replay rows lacked a
  `source_experiment` field in the registry despite having a known source.

## Changes Made

- Added ingestion-time defaults for `source_experiment` and `status`.
- Derived missing `wandb_run_id` from `checkpoint_root`.
- Preserved `candidate_source_experiment` separately from registry
  `source_experiment`.
- Dropped all-null columns when materializing fit datasets so legacy collapsed
  topology weight columns do not appear in 39-domain fit tables.

## Results

After rebuilding:

- `runs.csv`, `metrics_wide.csv`, `metrics_long.csv.gz`, and
  `eval_artifacts.csv` have no all-null columns.
- `source_experiment`, `status`, and derivable `wandb_run_id` are complete in
  canonical run/metric facts.
- Remaining missing `checkpoint_root` values correspond to 300M `not_found`
  rows without any checkpoint/W&B ID in the partial-results CSV. 60M local-only
  validation rows now derive checkpoint roots from `source_experiment` and
  `wandb_run_id`; sampled derived paths exist in GCS.
- Fit datasets for macro BPB and PIQA no longer contain all-null columns.
- Unsuffixed `lm_eval/piqa/*` metrics are intentionally not merged with
  `lm_eval/piqa_5shot/*` because current 300M launch configs use 10-shot PIQA,
  while olmo-base/easy-overlap explicitly uses `piqa_5shot`.

## Hypothesis 3: Empty 300M Rows Were From a Stale Partial Snapshot

The registry initially used `qsplit240_300m_6b_partial_results.csv` for 300M
run status. That file had only 77 `completed` rows and 139 `not_found` rows, so
the registry showed 95 300M rows with no checkpoint or W&B ID. The strict GCS
success table `qsplit240_300m_6b_completed_vs_60m.csv` already had 238
`SUCCESS` rows.

## Changes Made

- Refreshed `qsplit240_300m_6b_completed_vs_60m.csv` from GCS.
- Added the strict-success table as a higher-priority 300M metric source in the
  registry builder.
- Mapped `bpb_300m_6b` to canonical `eval/uncheatable_eval/bpb`.

## Results

After rebuilding, the 300M registry has 241 signal rows:

- 238 completed rows with checkpoint root, W&B run ID, and
  `eval/uncheatable_eval/bpb`.
- 3 remaining `not_found` rows: `run_00119`, `run_00149`, `run_00209`.
