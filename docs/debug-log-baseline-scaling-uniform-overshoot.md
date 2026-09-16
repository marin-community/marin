# Debugging log for baseline scaling Uniform overshoot

Investigate why the central baseline-scaling plot showed suspicious Uniform values at corrected `340M/10.4B` and `900M/24B`.

## Initial status

The paper plot showed:

- Uniform `340M/10.4B`: `eval/uncheatable_eval/bpb = 0.900022`, marked diagnostic.
- Uniform `900M/24B`: `eval/uncheatable_eval/bpb = 1.048645`, marked diagnostic.

Both points came from `run_registry_latest_metric`, not exact target-step labels.

## Hypothesis 1: the plotted points are overshot metrics, not 1x target metrics

The registry rows for both cells had `has_target_eval = False` and `target_eval_step = NaN`.

## Results

Confirmed.

Uniform `340M/10.4B`:

- Checkpoint root: `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_520m_10p4b/baseline_stratified-652f5a`
- Target final step: `19835`
- Only saved checkpoint: `step-39671`
- Latest plotted BPB: `0.900022` at `step-39671`
- No eval record at `step-19835`
- Nearest evals around target: `step-19000` BPB `1.079984`, `step-20000` BPB `1.076850`

This is not a valid 1x two-phase point. The run/config was resumed from an already-overshot checkpoint, and the available metric corresponds to a much later state.

Uniform `900M/24B`:

- Checkpoint root: `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_1_2b_24b/baseline_stratified-6dbf91`
- Target final step: `45775`
- Only saved checkpoint: `step-48504`
- Latest plotted BPB: `1.048645` at `step-48504`
- No eval record at `step-45775`
- Nearest evals around target: `step-45000` BPB `1.049415`, `step-46000` BPB `1.050667`

This is also not target-ready. The near-target evals are similarly poor, but there is no exact target checkpoint or target eval to use for the paper plot.

## Hypothesis 2: relaunch/resume logic reused stale overshot checkpoints

Read `.executor_info` from both checkpoint roots.

## Results

Confirmed.

The saved executor configs have correct target `num_train_steps`, but also set `initialize_from` to the already-overshot checkpoint in the same experiment prefix:

- `340M/10.4B`: `trainer.num_train_steps = 19836`, `initialize_from = .../step-39671`
- `900M/24B`: `trainer.num_train_steps = 45776`, `initialize_from = .../step-48504`

That means the relaunch path did not produce a fresh target checkpoint. It loaded a checkpoint past the target step, then exited/succeeded without creating the missing target-step artifact.

## Changes

- `paper_plots/baseline_scaling_trajectories.py` now treats non-target-ready rows with `max_checkpoint_step > target_final_checkpoint_step` as `needs_relaunch`, not as diagnostic plotted values.
- `launch_baseline_scaling_cell.py` now supports `BaselineScalingMethod.UNIFORM` so the central baseline-scaling workflow can launch replacement Uniform cells.
- `run_registry/build_run_registry.py` now tracks central Uniform replacement cells for `340M/10.4B` and `900M/24B`, and its resubmit hints include `--no-resume-latest-checkpoints`.

## Future Work

- [x] Submit replacement Uniform `340M/10.4B` with `--no-resume-latest-checkpoints`.
- [x] Submit replacement Uniform `900M/24B` with `--no-resume-latest-checkpoints` on `v5p-64` in `us-east5-a`.
- [ ] Refresh the registry and plot after completion.
- [ ] Consider making `resolve_latest_checkpoint_path` reject checkpoints whose step exceeds the requested target final step.

## Replacement submissions

Submitted on 2026-04-25:

- `/calvinxu/dm-baseline-scaling-uniform-340m-20260425-102914`
- `/calvinxu/dm-baseline-scaling-uniform-900m-20260425-102915`

Both parent jobs were running after submission. The new checkpoint roots are:

- `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling_uniform_520m_10p4b/baseline_stratified-0fab44`
- `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling_uniform_1_2b_24b/baseline_stratified-b504ac`

Their `.executor_info` files show `initialize_from = None` and `load_checkpoint_path = None`, so the stale overshot checkpoints are no longer being reused.

## Full baseline-plot checkpoint audit

After the Uniform issue, the full 25-cell baseline-scaling manifest was audited against checkpoint roots,
HF checkpoint steps, and exact target eval records.

Results as of 2026-04-25:

- `17` cells have exact target-step checkpoints/evals and are valid for downstream eval.
- `6` cells are diagnostic historical rows only: 60M rows for all methods plus Olmix `100M/6B`.
- `2` cells are invalid and require relaunch: Uniform `340M/10.4B` and Uniform `900M/24B`.

One additional manifest bug was found and fixed: UniMax `900M/24B` had a valid target BPB label but a stale
analysis-dataset checkpoint root in `us-central1`. The canonical run-registry row points to
`gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_1_2b_chinchilla/baseline_unimax-857505`,
which has exact `step-45775` checkpoint/eval. The plot builder now prefers exact target-ready registry rows
over analysis-dataset rows.

## Downstream eval contamination audit

The already-launched downstream eval states were inspected:

- `ngd3dm2_baseline_scaling_downstream_evals_us_east5_gen` deferred Uniform `340M/10.4B` and Uniform
  `900M/24B` because they were not target-ready.
- `ngd3dm2_baseline_scaling_downstream_evals_us_east5_mmlu_retry1` also deferred both invalid Uniform rows.
- The stale UniMax `900M/24B` analysis root was deferred because it had no HF checkpoint, so it was not
  launched.
- The post-refresh job `dm-baseline-scaling-evals-new-targets-20260425-025012` used only exact target-ready
  GRP no-L2 and Olmix rows and collected all 10 launched eval outputs successfully.

The downstream eval launcher now records expected and latest HF checkpoint steps and refuses to launch if
there is no exact HF checkpoint at the expected target step. This prevents future stale-root or overshot-root
rows from being evaluated by `discover_latest_checkpoint=True`.

## Historical diagnostic downstream eval follow-up

The initial downstream eval launcher only evaluated `target_ready` rows unless `--include-diagnostic` was
passed. That protected against the bad Uniform roots, but it also skipped useful historical BPB rows that
are plotted as context.

Follow-up action:

- Built `baseline_scaling_downstream_eval_historical_exact_manifest.csv` with the six diagnostic historical
  rows.
- Five 60M rows have exact `step-4576` HF checkpoints and are safe to evaluate.
- Olmix `100M/6B` remains deferred because it has no exact HF checkpoint available.
- Submitted `/calvinxu/dm-baseline-scaling-evals-historical-exact-20260425-105951` to run the missing
  historical evals:
  - GSM8K/HumanEval for all five exact 60M rows.
  - MMLU for Proportional `60M/1.2B` and UniMax `60M/1.2B`; MMLU already existed for GRP no-L2, Olmix,
    and Uniform `60M/1.2B`.

## Olmix `100M/6B` checkpoint audit

Olmix `100M/6B` was marked diagnostic because the original executor status is `FAILED`, but the final
training artifacts are present:

- Levanter checkpoint: `checkpoints/step-22887`
- Exact final eval: `eval/uncheatable_eval/bpb = 0.956061840057373` at `step-22887`
- Existing MMLU-like lm-eval artifacts at intermediate steps.

The missing piece was a complete HF export. The `hf/step-22887` directory had `model.safetensors` but was
missing `config.json` and tokenizer files, so downstream vLLM eval discovery rejected it. The model config
matches the other `100M/6B` rows exactly, so the same-scale Proportional HF metadata/tokenizer files were
copied into the Olmix HF directory.

Follow-up launches:

- `/calvinxu/dm-baseline-scaling-evals-olmix-100m-20260425-111212`: runs GSM8K/HumanEval for Olmix
  `100M/6B`; MMLU is skipped because an artifact already exists.
- `/calvinxu/dm-baseline-scaling-evals-uniform-100m-20260425-111255`: runs GSM8K/HumanEval for Uniform
  `100M/6B` from its `us-central1` checkpoint; MMLU is skipped because an artifact already exists.

## 60M diagnostic label audit

The five `60M/1.2B` baseline-scaling cells were showing as `diagnostic_only`.
This was a labeling/backfill problem, not a data-quality distinction.

Root cause:

- The 60M export CSV includes exact final perplexity metrics, including `eval/uncheatable_eval/bpb`.
- `run_registry/build_run_registry.py` intentionally loads only provenance columns from that old export.
- The registry rows therefore lack `is_perplexity_ready`, `has_target_eval`, `checkpoint_root`, and
  `target_eval_step` even though the underlying step-4576 artifacts exist.
- The plot then fell back to `packet_historical_metric` / legacy GCS metric paths and called those rows
  diagnostic.

Action:

- The central plot now treats audited 60M rows with an exact step-4576 checkpoint-backed metric as
  `target_ready`.
- GRP no-L2 `60M/1.2B` is similarly marked `target_ready` from its legacy GCS target eval path.

The broader registry still has a known legacy-export limitation: it is a provenance table for all old
60M rows, but it does not fully backfill artifact metadata for every historical 60M run because scanning
all legacy GCS roots during every registry refresh is too slow. The paper-plot path now handles the five
central baseline-scaling 60M cells explicitly.

## Uniform `340M/10.4B` completion and downstream eval launch

CC reported the Uniform `340M/10.4B` run as succeeded. Iris marks the parent job
`/calvinxu/dm-baseline-scaling-uniform-340m-20260425-102914` as failed because the parent process had one
failed step, but the actual checkpoint root is usable:

- Checkpoint root: `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling_uniform_520m_10p4b/baseline_stratified-0fab44`
- `.executor_status`: `SUCCESS`
- Exact Levanter checkpoint: `checkpoints/step-19835`
- Exact HF checkpoint: `hf/step-19835`
- Exact target eval: `eval/uncheatable_eval/bpb = 0.8795892596244812` at `step=19835`

The central scaling plot now reads this direct target artifact and marks Uniform `340M/10.4B` as
`target_ready`. Olmix `100M/6B` is handled the same way because it has an exact target checkpoint/eval
despite the original parent failure.

Submitted newly unblocked downstream evals:

- `/calvinxu/dm-baseline-scaling-evals-uniform-340m-20260425-1630`
- Includes GSM8K/HumanEval and MMLU.
- Both child evals started on `v5p-8` in `us-east5-a`.

Uniform `900M/24B` remains the only missing baseline-scaling training cell at this point.
