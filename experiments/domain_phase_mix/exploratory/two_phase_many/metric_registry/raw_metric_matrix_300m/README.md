# 300M Raw Metric Matrix

This directory contains the wide raw metric matrix for the 300M/6B mixture swarm.

## Files

- `raw_metric_matrix_300m.csv`: 242 rows, including qsplit-core rows, OLMix, and the stratified/uniform baseline.
- `raw_metric_matrix_300m_with_noise.csv`: the same 242 signal rows plus completed `run_00097` noise rows.
- `noise_baseline_run00097_fixed_subset_300m.csv`: trainer-seed noise with the simulated-epoch subset fixed
  to `run_00097`.
- `noise_baseline_run00097_variable_subset_300m.csv`: trainer-seed noise with the simulated-epoch subset
  left variable, matching the original swarm sampling more closely.
- `summary.json`: row/column counts and rows whose known mixture weights were hydrated.

## Columns

- Provenance columns identify the run, checkpoint, source experiment, and cohort.
- `status` is inherited from the metric registry's canonical run row. It may describe the highest-priority
  metric/eval source rather than the original training parent status. Use `checkpoint_root` plus metric
  non-nullness for matrix usability, and consult `run_registry/logical_runs.csv` for operational training
  status.
- `phase_0_*` columns are the first-phase mixture weights.
- `phase_1_*` columns are the second-phase mixture weights.
- `exposure_80_20_*` columns are the exposure-matched average mixture:
  `0.8 * phase_0 + 0.2 * phase_1`.
- Noise rows use the `run_00097` phase weights. Fixed-subset noise rows vary `noise_trainer_seed` while
  using `noise_simulated_epoch_subset_seed=97`; variable-subset noise rows vary `noise_trainer_seed` while
  leaving `noise_simulated_epoch_subset_seed` blank.
- `eval/*`, `lm_eval/*`, `teacher_forced/*`, and `mcq_smooth/*` columns are raw metric values.

The current metric registry row for `baseline_stratified` lacks phase-weight columns, so this builder
hydrates it as the known uniform stratified baseline: equal weight over the 39 active domains in both
phases.
