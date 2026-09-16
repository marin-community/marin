# Debugging log for piqa-choice-logprob missing metric

Investigate why the completed `baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_piqa_5shot_choice_logprob` run appears to be missing `lm_eval/piqa_5shot/choice_logprob` in our local registry, despite the deployment succeeding.

## Initial status

The run completed successfully at:

`gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_metric_objective_grp_no_l2_raw_optima/baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_piqa_5shot_choice_logprob-31e31d`

Symptoms:

- `.executor_status` is `SUCCESS`
- `checkpoints/eval_metrics.jsonl` does not contain any `lm_eval/*` keys
- the run-registry row therefore showed a missing objective metric for `lm_eval/piqa_5shot/choice_logprob`

## Hypothesis 1

The PIQA lm-eval task ran successfully, but we only read `checkpoints/eval_metrics.jsonl`, while lm-eval metrics are actually persisted elsewhere.

## Changes to make

- Inspect `lm_eval_artifacts/` and `tracker_metrics.jsonl` under the checkpoint root.
- Confirm whether the W&B replicate summary contains the missing metric.
- Patch provenance readers to fall back to `tracker_metrics.jsonl` for metrics absent from `eval_metrics.jsonl`.

## Results

Findings:

- `lm_eval_artifacts/lm_eval_harness_results.4576.json` exists and contains:
  - `piqa_5shot -> choice_logprob,none = -16.26244656274181`
- `tracker_metrics.jsonl` exists and contains the flattened W&B summary keys:
  - `lm_eval/piqa_5shot/choice_logprob = -16.26244656274181`
  - `lm_eval/piqa_5shot/bpb = 4.984975757123305`
  - other `lm_eval/averages/*` values as well
- The replicated W&B metadata is present:
  - run id: `baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_piqa_5shot_choice_logprob-31e31d`
  - url: `https://wandb.ai/marin-community/marin/runs/baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_piqa_5shot_choice_logprob-31e31d`

Conclusion:

The lm-eval task finished and was logged to W&B. The bug is that `eval_metrics.jsonl` only captures the standard eval callback outputs, while lm-eval outputs are replicated separately via `tracker_metrics.jsonl` and `lm_eval_artifacts`. Our local provenance code was reading only `eval_metrics.jsonl`, so it incorrectly treated the target lm-eval metric as missing.

## Future Work

- [ ] Add tracker-summary fallback to other checkpoint metric readers that currently only read `eval_metrics.jsonl`
- [ ] Audit whether analysis scripts should prefer `tracker_metrics.jsonl` for all `lm_eval/*` metrics
- [ ] Consider writing lm-eval results into `eval_metrics.jsonl` as well, if we want a single authoritative checkpoint metric file
