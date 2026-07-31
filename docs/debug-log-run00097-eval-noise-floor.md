# Debugging Log For run_00097 Eval Noise Floor

Investigate why some `eval/*` metrics in the `run_00097` seed-sweep noise baseline only show `noise_n = 8` instead of `10`, and determine whether the gap is caused by missing raw eval results or a collector bug.

## Initial status

- The ranked SNR table reports `noise_n = 8` for historical `eval/*` rows sourced from:
  - `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study/collect_results-ca72ba/results.csv`
- The `run_00097` overlap rerun has a clean `noise_n = 10`, so the gap appears specific to the older seed-study collector path or raw results.

## Hypothesis 1

The historical `collect_results` CSV dropped or failed to flatten two seeds even though raw per-run artifacts exist for all ten seeds.

## Changes to make

- Inspect the aggregate CSV and identify which seeds/metrics are missing.
- Inspect the historical seed-study launcher and per-run artifact layout.
- Compare aggregate CSV coverage against raw per-run outputs in GCS.

## Future Work

- [ ] If this is a collector bug, add a regression check so incomplete aggregate coverage is caught earlier.
- [ ] If raw data is missing, identify whether rerunning only the missing seeds is possible.

## Results

- The historical `noise_n = 8` gap was not caused by missing checkpoints.
- All four previously missing rows had successful checkpoint roots and `checkpoints/eval_metrics.jsonl` available:
  - `trainer_seed_10000`
  - `trainer_seed_10006`
  - `trainer_seed_10008`
  - `exact_replay_control_a`
- The old aggregate CSV was a stale historical collection artifact, and the SNR summary was also using the whole aggregate frame instead of filtering to `cohort == "seed_sweep"`.
- `collect_manifest_results` now backfills missing `eval/*` metrics from `checkpoints/eval_metrics.jsonl` when W&B lookup misses a row.
- A local corrected baseline was rebuilt at:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/run00097_seed_study_backfill/results.csv`
- The ranked SNR table was regenerated from the corrected 10-seed baseline, and every row now has:
  - `signal_n = 240`
  - `noise_n = 10`
