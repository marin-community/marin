# Debugging log for ppert noise parity backfill

## Initial status

The proportional perturbation downstream eval overlays have 112 candidate rows: 55 perturbations at
`60m_1p2b`, 55 perturbations at `300m_6b`, plus proportional baseline anchors at both scales. Most
downstream eval families are locally collected, but the parity lm-eval overlay
`ppert_noise_parity_eval_results.csv` has 15 rows with no useful metrics.

## Hypothesis: missing parity rows are a collection bug

Checked `ppert_noise_parity_retry4_state.csv`, which contains the 15 remaining parity rows. For each
row, the exact `result_path` on `gs://marin-us-east5/.../ngd3dm2_ppert3_noise_parity_20260509/...`
has `.executor_status = FAILED` and no `results.json`.

Result: this is not a simple local collection miss for those exact rows. Some similarly named rows at
the other scale succeeded, so broad run-name matching can be misleading. The exact failed eval keys are
still missing artifacts.

## Required backfill

Run one more failure-only parity eval retry using
`experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_perturbation_scale_transfer/ppert_noise_parity_retry4_state.csv`.
The retry should launch only 15 eval children, each covering:

- `mmlu_5shot`
- `mmlu_sl_verb_5shot`
- `mmlu_pro_5shot`
- `arc_easy`
- `piqa`
- `hellaswag_0shot`

This does not require new training, only downstream lm-eval jobs against existing exact HF checkpoints.

## Future Work

- [ ] Upload the retry4 state CSV to GCS before submitting, or ensure the local state file is bundled.
- [ ] Submit with `max_concurrent=15` on `v5p-8` in `us-east5-a`.
- [ ] Collect from both the original prefix and the new retry prefix.
- [ ] Rebuild the metric registry after collection so local perturbation eval overlays are visible in
      `metrics_wide.csv`.
