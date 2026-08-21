# Uniform 900M Missing Permanent Checkpoints

## Context

`baseline_scaling` Uniform at corrected `900M/24B` failed after a long
preemptible `v5p-64` run. The permanent checkpoint root only contained
`eval_metrics.jsonl`:

```text
gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/
  ngd3dm2_baseline_scaling_uniform_1_2b_24b/
  baseline_stratified-b504ac/checkpoints/eval_metrics.jsonl
```

This looked like a complete loss of work, but the training logs showed regular
time-policy checkpoints under the Marin temp bucket:

```text
gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/
  baseline_stratified-b504ac/step-41449
```

The temp checkpoint has `metadata.json`, `manifest.ocdbt`, and tensor data.

## Root Cause

The replay launchers only searched permanent checkpoint roots:

```text
gs://marin-<region>/checkpoints/<experiment>/<run>-*/checkpoints/step-*/metadata.json
```

For these runs, `steps_per_export` is the final train step, so permanent
checkpoints are intentionally only saved at the end. Preemption recovery depends
on rolling time-policy checkpoints, which are written to:

```text
gs://marin-tmp-<region>/ttl=14d/checkpoints-temp/<run>-*/step-*
```

The launcher therefore failed to discover the valid temp checkpoint and would
restart from scratch after a parent-job resubmission.

There was a second bug: if a temp checkpoint were discovered, passing it through
`mirror://` would be wrong because the mirror filesystem scans primary
`marin-*` data buckets, not `marin-tmp-*` buckets.

## Fix

Updated `qsplit240_replay` checkpoint discovery to scan both:

- permanent checkpoint roots under `gs://marin-<region>/checkpoints/...`
- temp checkpoint roots under `gs://marin-tmp-<region>/ttl=14d/checkpoints-temp/...`

Temporary checkpoints are accepted only if they have tensor data
(`manifest.ocdbt` or `d/`). Metadata-only temp directories are skipped.

Added `checkpoint_initialization_path()` so permanent checkpoints still use
`mirror://`, while temp-bucket checkpoints remain concrete `gs://` paths.

Updated the baseline-scaling and stratified launchers to use that helper.

## Current State

Stopped the accidental scratch resubmission:

```text
/calvinxu/dm-baseline-scaling-uniform-900m-20260426-054640
/calvinxu/dm-baseline-scaling-uniform-900m-20260426-054640/train_lm
```

Submitted the corrected resume parent:

```text
/calvinxu/dm-baseline-scaling-uniform-900m-resume-20260425-2309
```

Remote executor metadata confirms the child config will resume from:

```text
trainer.initialize_from =
gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/baseline_stratified-b504ac/step-41449
```

The TPU child was pending on `v5p-64` capacity in `us-east5-a` at the time of
this note.

## Validation

Local direct resolver check:

```text
resolve_latest_checkpoint_path(...) ->
gs://marin-tmp-us-east5/ttl=14d/checkpoints-temp/baseline_stratified-b504ac/step-41449
```

Compile check:

```text
uv run --with torch python -m py_compile \
  experiments/domain_phase_mix/qsplit240_replay.py \
  experiments/domain_phase_mix/launch_baseline_scaling_cell.py \
  experiments/domain_phase_mix/launch_two_phase_many_stratified_baseline.py
```

The full `tests/test_domain_phase_mix_determinism.py` targeted pytest could not
collect because the file imports stale `marin.utils.create_cache_tokenizer_step`
through `two_phase_starcoder_determinism_wsd`; this is unrelated to this fix.
