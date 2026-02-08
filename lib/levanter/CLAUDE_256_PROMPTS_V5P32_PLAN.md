# Plan: 256-Prompt Host-Data-Parallel Inference on v5p-32

Status: PLAN (2026-02-07)

## Goal

Run 256 prompts with `max_new_tokens=2048` on `v5p-32` (16 chips, 4 hosts) using
`inference_mode: host_data_parallel` with wandb logging.

## Key Insight: Per-Host Workload is Already Validated

With host-DP on v5p-32 (4 hosts): **256 / 4 = 64 prompts per host**.

This is the **exact same per-host workload** that was successfully validated today on v5p-16
(128 prompts / 2 hosts = 64 per host). The engine config does not need to change; only the
prompt list needs to expand from 128 to 256.

## Constraints Review (from CODEX Notes M5-M10)

### 1. max_seqs Boundary (M9)

- **M9 global-mesh boundary**: `max_seqs=56` is the hard ceiling with `page_size=64` in
  global mesh mode. At 57+, reproducible first-decode `Anomalies` crash.
- **Host-DP mode**: Uses `page_size=128` and `max_seqs=128`. Each host only receives its
  shard (64 prompts). The M9 boundary at 56 was discovered in a different engine
  configuration (page_size=64, global mesh) and does not apply here.
- **Validation**: v5p-16 wandb run successfully processed 64 prompts per host with
  `max_seqs=128` today. v5p-32 wandb run processed 32 per host. Both passed.

### 2. max_pages Ceiling (M9)

- **M9 length-fixed boundary**: `max_pages=4096` is the hard ceiling with `page_size=64`.
  At `max_pages=4096`, stable up to 124 prompts. At 125+, fast-rejected by validation.
- **Host-DP mode with page_size=128**: Current config uses `max_pages=2304`.
- **Page usage observed (64 per host)**: `pages_in_use=1088 free=1216` at after_decode.
  This is well within the 2304 budget.
- **Page capacity math**: `ceil(2560/128) = 20 pages/seq`. For 64 seqs: `64 * 20 = 1280`
  minimum. `2304 - 1280 = 1024` free pages. Ample headroom.

### 3. Scheduler Budget

- Current: `max_tokens_per_round=128, max_rounds=64, max_queued_tokens=128`.
- For 64 active seqs generating 2048 tokens each = 131,072 total tokens per host.
- Each decode iteration processes `64 * 64 = 4096` tokens (max_tokens_per_round * active).
  Wait, more precisely: with `max_tokens_per_round=128` and 64 active sequences, each iter
  generates min(128, 64) * 64 = 4096 new tokens if all are active. Observed in wandb v5p-16
  run: exactly 32 decode iterations producing 4096 tokens each = 131,072 per host.
- Budget check: `32 decode iters <= max_rounds=64`. Passes with headroom.

### 4. Asymmetric Tracker Logging (M5)

- **Critical lesson**: In-loop tracker emission (wandb tables mid-generation) causes
  asymmetric collective operations across TPU hosts, leading to slice crashes.
- **Mitigation**: `defer_tracker_logs_until_end: true` (already set in all host-DP configs).
- **Additional safety**: Our wandb logging function `_log_host_dp_samples_to_tracker()` only
  runs on leader after all generation is complete and after cross-host gather.

### 5. Leader-Only jax.device_get (M5 debug)

- Not applicable in host-DP mode. Each host runs its own local engine; there is no
  cross-host collective during generation.

### 6. VMEM OOM in Ragged Prefill (M10.3/M10.4)

- **Issue**: `page_size=128` with `q_block_size=32` caused VMEM OOM during ragged prefill.
- **Fix (M10.4)**: Reduced kernel block sizes to `q_block_size=16, kv_block_pages=8`.
  This is already set in all current host-DP configs.

### 7. Cross-Host Gather Overhead (M10.6)

- Gather uses JAX distributed KV store with JSON+zlib+base64 encoding.
- Measured overhead: `~0.47%` decode-time, `~6.84%` end-to-end wall time for 128 prompts.
- For 256 prompts: payload sizes double, but still well within KV store limits. Expect
  similar percentage overhead.

## What Needs to Change

### Config Changes Only (No Code Changes Needed)

1. **Expand prompt list**: 128 -> 256 prompts
2. **Update output directory**: New path for 256-prompt outputs
3. **Update wandb tags**: Reflect 256-prompt workload

### Engine Parameters: NO CHANGES

All engine parameters stay the same because per-host workload (64 prompts) is identical:

| Parameter | Value | Rationale |
|---|---|---|
| `max_seq_len` | 2560 | Unchanged (prompt + generation headroom) |
| `max_pages` | 2304 | Unchanged (1280 needed for 64 seqs, 1024 free) |
| `page_size` | 128 | Unchanged (M10 host-DP standard) |
| `max_seqs` | 128 | Unchanged (>= 64 local prompts) |
| `max_seqs_in_prefill` | 128 | Unchanged (>= 64 local prompts) |
| `max_prefill_size` | 2048 | Unchanged |
| `max_queued_tokens` | 128 | Unchanged |
| `max_tokens_per_round` | 128 | Unchanged |
| `max_rounds` | 64 | Unchanged (32 iters needed for 64 prompts) |
| `reset_mode` | physical | Unchanged |
| `cleanup_mode` | none | Unchanged |

## Expected Performance

Based on validated v5p-16 results (64 per host) and v5p-32 results (32 per host):

| Metric | v5p-16 128p (64/host) | v5p-32 128p (32/host) | v5p-32 256p (64/host) predicted |
|---|---|---|---|
| Hosts | 2 | 4 | 4 |
| Prompts per host | 64 | 32 | 64 |
| Per-host round_total_s | ~107s | ~86s | ~107s (same as v5p-16) |
| Decode iterations | 32 | 32 | 32 |
| Per-host decode tok/s | ~1,768 | ~1,148 | ~1,768 (same as v5p-16) |
| Total tokens | 262,144 | 262,144 | 524,288 |
| End-to-end wall clock | ~200s | ~167s | ~250s (model load + 107s gen) |

The per-host round time should be ~107s (matching v5p-16 with 64/host), since each host
has the identical chip count (4) and workload (64 prompts). v5p-32 256p should produce
**2x the total tokens** (524,288 vs 262,144) in roughly the same wall time as v5p-16 128p.

## Implementation Steps

### Step 1: Create Config File

Create `config/sampler/sample_llama8b_multihost_real_256prompts_2048_m10_hostdp_wandb_v5p32.yaml`:
- Copy from the working 128-prompt v5p-32 wandb config
- Expand prompts list from 128 to 256
- Update tags: `["m10", "host-dp", "v5p-32", "256x2048", "wandb"]`
- Update output dir: `/tmp/m10_host_outputs_256p_wandb_v5p32`
- Keep all engine params identical

### Step 2: Run on v5p-32 (simpo_worker_3)

```bash
uv run python infra/launch.py --foreground \
  --zone us-central1-a \
  --tpu_name simpo_worker_3 \
  --tpu_type v5p-32 \
  --capacity_type on-demand \
  -- uv run src/levanter/main/sample_lm_multihost.py \
     --config_path config/sampler/sample_llama8b_multihost_real_256prompts_2048_m10_hostdp_wandb_v5p32.yaml \
  2>&1 | tee /tmp/m10_256p_wandb_v5p32.log
```

### Step 3: Validate Results

1. Check job completes with no errors
2. Verify WandB run shows:
   - `sample/total_prompts: 256`
   - `sample/total_generated_tokens: 524288` (256 * 2048)
   - `sample/num_hosts: 4`
   - Samples table with 256 rows
3. Verify per-host outputs:
   - 4 shard files, 64 rows each
   - Merged leader file: 256 rows
4. Compare per-host round time to v5p-16 64/host baseline (~107s)

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Per-host VMEM exhaustion at 64 prompts | **Very Low** | Already validated on v5p-16 today |
| max_seqs=128 boundary crash | **Very Low** | Only 64 prompts per host, well below config limit |
| Page capacity exhaustion | **Very Low** | 1088/2304 pages used at 64 seqs (observed) |
| Gather payload too large for 256 rows | **Low** | JSON+zlib; 256 rows is ~2x 128 rows, well within limits |
| WandB table upload timeout at 256 rows | **Low** | 128 rows uploaded fine; 256 is incremental |
| Scheduler budget insufficient | **None** | 32 iters needed, 64 max_rounds available |

## Potential Risks NOT Expected

- The `max_seqs=56` M9 boundary does NOT apply here (different page_size, different mode)
- VMEM OOM in ragged prefill does NOT apply here (already using q_block=16, kv_pages=8)
- Asymmetric tracker crash does NOT apply here (deferred + leader-only logging)
