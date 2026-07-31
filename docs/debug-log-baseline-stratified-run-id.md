# Debugging log for baseline_stratified run_id provenance

## Initial status

`build_raw_metric_matrix_300m.py` failed validation because the 300M signal matrix had one empty required provenance field: `run_id` for `baseline_stratified`.

## Hypothesis: local-only registry rebuild dropped static baseline metadata

The current metric registry summary was generated with `include_gcs=false`. In that mode, the strict 300M success CSV source that maps `baseline_stratified` to `run_id=3` is skipped. Local raw-PPL eval rows still include the `baseline_stratified` checkpoint and metrics, but not `run_id`, so the canonical `metrics_wide.csv` row ended up with `run_id` blank.

## Changes made

- Added `_fill_known_run_metadata` in `build_metric_registry.py`.
- The normalizer now fills stable static baseline metadata for eval-only source rows, currently `baseline_stratified -> run_id=3`.

## Results

- `uv run python -m py_compile` passed for the touched registry/matrix builders.
- `build_metric_registry.py --no-gcs` now preserves `baseline_stratified` with `run_id=3` and zero null `run_id` values in the 242-row 300M signal cohort.
- Local-only registry rebuilds now keep proportional variable-subset noise rows as run-registry provenance rows, so local eval overlays can hydrate them without requiring a full GCS registry refresh.
- `build_proportional_variable_subset_noise_matrices.py` now directly hydrates final-step training `eval/*` metrics from the corresponding checkpoint `eval_metrics.jsonl` files, restoring the 60M and 300M proportional-noise exports.
- `build_raw_metric_matrix_300m.py` now completes successfully; `raw_metric_matrix_300m.csv` has 242 signal rows, and `raw_metric_matrix_300m_with_proportional_noise.csv` has 252 rows.
