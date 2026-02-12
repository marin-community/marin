# Multi-Host Inference Debug (simpo vs multihost_inference_work)

This document is a step-by-step, novice-friendly record of how we reproduced and fixed the multi-host inference failure on the `simpo` branch, using a known-good baseline from `multihost_inference_work`.

Everything below is reproducible on the same machine and TPU used in the investigation.

## Goal

- Run multi-host sampling on a v5p-16 TPU with the same command on two branches.
- Identify why `simpo` fails while `multihost_inference_work` works.
- Apply a fix and confirm the command finishes successfully.

## Environment

- Repo root: `/Users/ahmed/code/marin3`
- Levanter working dir: `/Users/ahmed/code/marin3/lib/levanter`
- TPU: `simpo_worker_2` in `us-central1-a`
- Command uses `uv run python` per repo tooling guidelines.

## Known-Good Command

This command is the reference for success:

```bash
cd /Users/ahmed/code/marin3/lib/levanter
uv run python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand \
  -- uv run src/levanter/main/sample_lm_multihost.py \
  --config_path config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml
```

## Quick Summary of the Fix

The fix was to **restore the multi-host inference stack to the working versions** and **remove stale config fields** that `simpo` introduced but are no longer supported by the restored code:

- Restored from `multihost_inference_work`:
  - `src/levanter/inference/engine.py`
  - `src/levanter/inference/jit_scheduler.py`
  - `src/levanter/inference/page_table.py`
  - `src/levanter/layers/attention.py`
  - `src/levanter/layers/kv_cache.py`
  - `src/levanter/main/sample_lm_multihost.py`
- Removed unsupported config keys from:
  - `config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml`
    - `engine.use_logical_reset`
    - `max_prompts_per_batch`

After these changes, the command completes with:

```
Job finished with no error.
```

## Reproduction (Simpo Failure)

### 1) Confirm the branch

```bash
cd /Users/ahmed/code/marin3

git branch --show-current
```

### 2) Run the command on `simpo`

```bash
cd /Users/ahmed/code/marin3/lib/levanter

uv run python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand \
  -- uv run src/levanter/main/sample_lm_multihost.py \
  --config_path config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml
```

### 3) Observe the first failure

On `simpo`, the run fails early with a config parsing error:

```
draccus.utils.DecodingError: `engine`: The fields `use_logical_reset` are not valid for InferenceEngineConfig
```

This indicates a config mismatch: `use_logical_reset` is present in the YAML, but the current `InferenceEngineConfig` does not accept it.

### 4) Remove the invalid field

Edit the config file and remove:

```yaml
engine:
  use_logical_reset: true
```

### 5) Second failure (after fixing `use_logical_reset`)

If you restore `sample_lm_multihost.py` to the working version (see below), you may hit a second config error:

```
draccus.utils.DecodingError: The fields `max_prompts_per_batch` are not valid for SampleLmMultihostConfig
```

This is because the working `sample_lm_multihost.py` does not define `max_prompts_per_batch`, but the `simpo` YAML still includes it.

### 6) Remove the second invalid field

Edit the config file and remove:

```yaml
max_prompts_per_batch: 1
```

At this point, the config is compatible with the restored script.

## Fix Steps (Exact Commands)

### 1) Restore working versions of inference files

```bash
cd /Users/ahmed/code/marin3/lib/levanter

git restore --source multihost_inference_work -- \
  src/levanter/inference/engine.py \
  src/levanter/inference/jit_scheduler.py \
  src/levanter/inference/page_table.py \
  src/levanter/layers/attention.py \
  src/levanter/layers/kv_cache.py \
  src/levanter/main/sample_lm_multihost.py
```

### 2) Update the config file

Remove the unsupported fields:

```yaml
engine:
  use_logical_reset: true

max_prompts_per_batch: 1
```

### 3) Clean up any stale TPU containers (optional but recommended)

```bash
cd /Users/ahmed/code/marin3

gcloud alpha compute tpus tpu-vm ssh simpo_worker_2 --quiet --worker=all \
  --zone=us-central1-a --command="docker rm -f levanter || true"
```

### 4) Re-run the launcher command

```bash
cd /Users/ahmed/code/marin3/lib/levanter

uv run python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand \
  -- uv run src/levanter/main/sample_lm_multihost.py \
  --config_path config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml
```

## Expected Successful Output

You should see:

- Model loading logs (safetensors reading)
- WandB run initialization
- Prefill + decode logs
- WandB run summary
- Final line:

```
Job finished with no error.
```

## Offending Change Summary

The `simpo` branch introduced several inference changes and configuration knobs that do not work together in this setup. The working version of multi-host inference is the one in `multihost_inference_work`.

The concrete differences that broke `simpo` for this command were:

1) **Inference stack changes**
   - `InferenceEngine` and `jit_scheduler` behavior diverged from known-good logic.
   - These were restored from `multihost_inference_work` to fix the run.

2) **Config mismatches**
   - `engine.use_logical_reset` exists in the YAML but is not supported by the current `InferenceEngineConfig`.
   - `max_prompts_per_batch` exists in the YAML but is not supported by the restored `SampleLmMultihostConfig`.

Once the inference files were restored and the config was cleaned, the run completed successfully.

## Troubleshooting Notes

### Stop a stuck launcher

```bash
pgrep -fl "infra/launch.py"
kill <pid>
```

### Check container logs directly

```bash
gcloud alpha compute tpus tpu-vm ssh simpo_worker_2 --quiet --worker=all \
  --zone=us-central1-a --command="docker logs --tail 200 levanter"
```

### Ignore benign warnings

You may see these messages; they are not fatal:

- `Failed to add ssh key. This may lead to problems.`
- JAX `FutureWarning` about scatter dtype promotion

They did not prevent the successful run.

## Final Validation

The fixed code was validated on 2026-02-04 with the command above, and completed with `Job finished with no error.`
