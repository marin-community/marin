# Claude Subagent: OOM Fix for Multi-Host Sampling

## Task Overview

A subagent was spawned to fix OOM (Out of Memory) errors in the multi-host TPU sampling pipeline for Llama 3.1 8B on v5p-64 (8 hosts, 64 TPU devices).

**Constraints:**
- Must keep 100 prompts
- Must keep 4096 max_new_tokens (max response length)
- Everything else (memory/inference parameters) is fair game

## What Happened

### Run 1: Initial Execution

The subagent ran the launch command:
```bash
python infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker --tpu_type v5p-64 --capacity_type on-demand -- \
    uv run src/levanter/main/sample_lm_multihost.py \
    --config_path config/sampler/sample_llama8b_multihost_real.yaml
```

**Result:** RESOURCE_EXHAUSTED error in vmem during `ragged_paged_attention` prefill phase.

Error details:
```
RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem while allocating on stack for
%ragged_paged_attention.32 = bf16[1024,32,128]{2,1,0:T(8,128)(2,1)} custom-call(...)
```

The attention kernel was trying to allocate 37.54M but only had 16M VMEM limit.

### Analysis

The subagent identified the root cause:
- The prefill size and number of sequences processed together was too large for the TPU's vector memory (VMEM)
- VMEM is separate from HBM and has tighter constraints for kernel operations

Original config parameters:
- `max_prompts_per_batch: 8`
- `max_prefill_size: 1024`
- `max_seqs: 8`
- `max_seqs_in_prefill: 8`
- `max_pages: 288`

### Parameter Changes Made

The subagent reduced memory-intensive parameters:

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `max_prompts_per_batch` | 8 | 4 | Process fewer prompts per batch |
| `max_prefill_size` | 1024 | 512 | Reduce prefill buffer size |
| `max_seqs` | 8 | 4 | Reduce concurrent sequences |
| `max_seqs_in_prefill` | 8 | 4 | Reduce sequences in prefill |

### Run 2: With Reduced Parameters

The command was re-launched with the updated config. Model loading completed successfully across all 8 hosts.

**Status when stopped:** Model was loaded and inference was about to begin. The subagent was interrupted before the second run could complete to determine if the OOM was fixed.

## Current Config State

The config file `config/sampler/sample_llama8b_multihost_real.yaml` now contains:
```yaml
max_new_tokens: 4096
max_prompts_per_batch: 4
engine:
  max_seq_len: 4608
  max_pages: 288
  page_size: 128
  max_seqs: 4
  max_seqs_in_prefill: 4
  max_prefill_size: 512
  max_queued_tokens: 64
  max_rounds: 64
prompts:
  # 100 prompts (Prompt 001 through Prompt 100)
```

## Next Steps If OOM Persists

If OOM still occurs with current parameters, try:
1. Reduce `max_prompts_per_batch` to 2
2. Reduce `max_prefill_size` to 256
3. Reduce `max_pages` (affects KV cache size)
4. Consider `hbm_utilization` parameter if available

## Key Learnings

1. **VMEM vs HBM:** TPU has separate memory spaces. The `ragged_paged_attention` kernel runs in VMEM which has stricter limits than HBM.

2. **Prefill is memory-intensive:** The prefill phase processes prompt tokens and requires significant VMEM for the attention kernel.

3. **Multi-host adds complexity:** With 8 hosts, each processing a portion of the globally-sharded model, memory constraints apply per-chip but coordination happens globally.

## Command to Resume Testing

```bash
python infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker --tpu_type v5p-64 --capacity_type on-demand -- \
    uv run src/levanter/main/sample_lm_multihost.py \
    --config_path config/sampler/sample_llama8b_multihost_real.yaml
```
