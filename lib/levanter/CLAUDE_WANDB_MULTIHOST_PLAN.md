# Plan: WandB Logging for Host-Data-Parallel Multi-Host Inference

Date: 2026-02-07

## Problem

The `host_data_parallel` inference mode (M10) currently uses `tracker.type: noop` and `skip_samples_table: true` in all configs. This means:
- No WandB run is created
- No samples are logged to WandB
- Generated text only exists in per-host JSONL files and a leader-merged JSONL file on disk

We want WandB runs to start properly and generated samples to be logged as WandB artifacts (tables) for both v5p-16 and v5p-32.

## Current State

### What Works Today
- Global-mesh path has full WandB support: metrics, samples table, deferred logging
- Host-DP path writes per-host JSONL files + leader-merged JSONL via `_gather_host_rows_to_leader()` (M10.6)
- The tracker is initialized via `levanter.initialize(config)` on all hosts at startup
- `levanter.current_tracker().finish()` is called at the end of both paths

### What's Missing
- Host-DP path in `_run_host_data_parallel_inference()` does not call `levanter.tracker.log()` at all
- Configs use `trainer.tracker.type: noop` — no WandB run is created
- `skip_samples_table: true` is set in all host-DP configs

### Key Constraints (from M5/M5.2 lessons)
1. **All tracker logging must happen on leader only** after generation completes (deferred pattern)
2. **No in-loop tracker emission** between rounds — causes TPU launch-group failures
3. **No leader-only `jax.device_get`** on globally sharded arrays — causes crashes/hangs
4. Host-DP mode is constrained to `n_rounds=1`, `n_generations=1`, `defer_tracker_logs_until_end=true`

Since host-DP is `n_rounds=1` only, the M5 round-boundary crash class does not apply. The constraint is simpler: tracker logging must happen **after** generation completes, and only on the leader.

## Plan

### Step 1: Config Changes (No Code Changes)

Create two new config files based on the M10.7 validated host-DP configs. The only differences from the existing M10.7 configs:

**`config/sampler/sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_wandb_v5p16.yaml`**:
```yaml
# Changes from m107_v5p16_hostdp.yaml:
trainer:
  tracker:
    type: wandb                             # was: noop
    project: levanter-multihost-inference   # WandB project name
    tags: ["m10", "host-dp", "v5p-16", "128x2048", "wandb"]
skip_samples_table: false                   # was: true
host_data_parallel_output_dir: /tmp/m10_host_outputs_wandb_v5p16
```

**`config/sampler/sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_wandb_v5p32.yaml`**:
```yaml
# Changes from m107_v5p32_hostdp.yaml:
trainer:
  tracker:
    type: wandb
    project: levanter-multihost-inference
    tags: ["m10", "host-dp", "v5p-32", "128x2048", "wandb"]
skip_samples_table: false
host_data_parallel_output_dir: /tmp/m10_host_outputs_wandb_v5p32
```

All other settings (model, engine, prompts, inference_mode, enable_host_rows_gather, etc.) remain identical to the validated M10.7 configs.

### Step 2: Code Changes in `sample_lm_multihost.py`

The host-DP path (`_run_host_data_parallel_inference()`) currently ends after writing JSONL outputs and gathering to leader. We need to add WandB logging **after** the gather step, **leader-only**, using the merged rows.

#### 2a. Add WandB samples table logging after leader gather

Location: `_run_host_data_parallel_inference()`, after the `_gather_host_rows_to_leader()` / merged JSONL write block.

```python
# After merged JSONL is written on leader...
if is_leader and not config.skip_samples_table:
    # Build samples table from merged rows (already gathered and ordered)
    samples_rows = []
    for row in merged_rows:
        samples_rows.append((
            row.get("round_index", 0),
            row["global_prompt_index"],
            row.get("generation_index", 0),
            row.get("prompt_text", ""),
            row.get("generated_text", ""),
            row.get("generated_token_count", 0),
        ))

    try:
        import wandb
        samples_table = wandb.Table(
            columns=["round", "prompt_id", "generation_id", "prompt", "generated_text", "num_tokens"]
        )
        for r in samples_rows:
            samples_table.add_data(*r)
        levanter.tracker.log({"sample/samples": samples_table}, step=0)
    except ImportError:
        levanter.tracker.log({"sample/samples_rows": samples_rows}, step=0)
```

Key design points:
- This runs **after** generation and gather are complete, so no risk of round-boundary interference
- Uses `merged_rows` from the M10.6 gather step — already ordered by `(round, prompt_id, generation_id)`
- Only runs on leader (`is_leader`)
- Falls back gracefully if wandb is not importable
- Uses `step=0` since host-DP is `n_rounds=1`

#### 2b. Add metrics logging after generation

Also on leader, log aggregate metrics (total tokens, wall time, per-host round times):

```python
if is_leader:
    metrics = {
        "sample/total_prompts": len(merged_rows),
        "sample/total_generated_tokens": sum(r.get("generated_token_count", 0) for r in merged_rows),
        "sample/num_hosts": jax.process_count(),
        "sample/inference_mode": "host_data_parallel",
    }
    levanter.tracker.log(metrics, step=0)
```

#### 2c. Ensure merged_rows is available for wandb logging

Currently, the merged JSONL write path reads from `_merge_gathered_host_rows(...)`. We need to ensure the merged rows list is retained in memory for the wandb logging step. This is already the case — `merged_rows` is a Python list returned by the merge helper and used for JSONL writing.

The key code flow change is:

```
current:  generate → build rows → gather → write merged JSONL → finish
proposed: generate → build rows → gather → write merged JSONL → log to wandb → finish
```

#### 2d. Handle the case where gather is disabled

If `enable_host_rows_gather: false`, there are no merged rows on leader. In that case:
- Skip WandB samples table logging (only per-host JSONL is written)
- Log a warning that WandB samples require `enable_host_rows_gather: true`
- Still log aggregate metrics if the leader's local rows are available

For the configs we create, we always set `enable_host_rows_gather: true`.

### Step 3: Ensure `prompt_text` is in gathered rows

Looking at the current host output row format, rows include `generated_text` and `generated_token_count` but need `prompt_text` for the samples table. Check that `_build_host_output_rows()` includes the original prompt text. If not, add it.

From the M10.3 implementation, rows already include prompt information via the request metadata. Verify this field name matches what the samples table builder expects.

### Step 4: Validation

#### 4a. Unit tests
- Existing tests in `test_sample_lm_multihost_host_outputs.py` should still pass
- Add a test that verifies the samples table building logic produces the expected columns from merged rows

#### 4b. TPU validation (v5p-16)

```bash
cd /Users/ahmed/code/marin3/lib/levanter
python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name simpo_worker_2 --tpu_type v5p-16 --capacity_type on-demand \
  -- uv run src/levanter/main/sample_lm_multihost.py \
  --config_path config/sampler/sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_wandb_v5p16.yaml \
  2>&1 | tee /tmp/m10_wandb_v5p16.log
```

Expected:
- WandB run initializes on leader
- Generation completes (~254s wall clock based on M10.7 v5p-16 host-DP result)
- WandB samples table with 128 rows logged
- WandB run finishes with summary metrics
- Per-host JSONL + merged JSONL still written as before

#### 4c. TPU validation (v5p-32)

```bash
cd /Users/ahmed/code/marin3/lib/levanter
python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name simpo_worker_3 --tpu_type v5p-32 --capacity_type on-demand \
  -- uv run src/levanter/main/sample_lm_multihost.py \
  --config_path config/sampler/sample_llama8b_multihost_real_128prompts_2048_m10_hostdp_wandb_v5p32.yaml \
  2>&1 | tee /tmp/m10_wandb_v5p32.log
```

Expected:
- WandB run initializes on leader
- Generation completes (~222s wall clock based on M10.7 v5p-32 host-DP result)
- WandB samples table with 128 rows logged
- WandB run finishes

### Step 5: Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| WandB init on non-leader hosts causes issues | Low | `levanter.initialize()` already handles multi-host WandB init correctly (only leader creates run) |
| WandB table upload adds significant wall time | Low | Table has 128 rows; upload happens after generation |
| Tracker logging triggers TPU issues | Very Low | No in-loop logging; single `tracker.log()` call after all generation + gather complete; host-DP is `n_rounds=1` only |
| `prompt_text` field missing from gathered rows | Low | Verify field name in `_build_host_output_rows()`; add if missing |

### Summary of Changes

| File | Change |
|------|--------|
| `config/sampler/...wandb_v5p16.yaml` | New config: host-DP + `tracker.type: wandb` + `skip_samples_table: false` |
| `config/sampler/...wandb_v5p32.yaml` | New config: host-DP + `tracker.type: wandb` + `skip_samples_table: false` |
| `src/levanter/main/sample_lm_multihost.py` | Add post-gather wandb logging in `_run_host_data_parallel_inference()` |
| `tests/inference/test_sample_lm_multihost_host_outputs.py` | Add test for samples table building from merged rows |

### Execution Order

1. Verify `prompt_text` is available in gathered rows (read code)
2. Add wandb logging code to `_run_host_data_parallel_inference()`
3. Create both config files
4. Run unit tests
5. Run v5p-16 TPU validation
6. Run v5p-32 TPU validation
7. Confirm WandB dashboard shows samples table
