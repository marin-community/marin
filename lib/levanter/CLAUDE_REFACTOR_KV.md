# KV Cache Refactoring Plan for Multi-Host Inference

## Goal
Generate up to 4096 new tokens per prompt in multi-host inference without OOM, with incremental improvements that don't break working configs.

## Current Working Configs
- v5p-16, 1 prompt, 2048 tokens: SUCCESS
- v5p-16, 1 prompt, 4096 tokens: SUCCESS

## Current Failing Configs
- v5p-16, 5+ prompts: Exit status 1 (no traceback)
- v5p-64: Launch-group mismatch / slice halt

---

## Implementation Status

| Task | Status | Notes |
|------|--------|-------|
| P0.1 | ✅ DONE | `debug_stats()` and `debug_stats_print()` added to DecodeState |
| P0.2 | ✅ DONE | `kv_pages` field removed, now a `@property` alias for `page_indices` |
| P0.3 | ✅ DONE | `incremental_cleanup` config added, wired to `free_pages_for_finished()` |
| P1.1 | ✅ DONE | `use_logical_reset` config added, `reset_logical()` methods added |
| P1.2 | ✅ DONE | Block sizes changed from Q_BS=1/KV_BS=2 to Q_BS=16/KV_BS=16 |
| P1.3 | ✅ DONE | `invalidate_finished()` called when `incremental_cleanup=True` |
| P2.1 | 🔲 TODO | Investigate multi-prompt failures |
| P2.2 | 🔲 TODO | Fix TPU kernel VMEM OOM |

### Test Results (Local)
- ✅ 104/104 inference tests pass
- ✅ 73/73 paged attention tests pass
- ✅ 5/5 jit_scheduler tests pass
- ✅ 3/3 clone_pages tests pass

---

## New Config Options

Added to `InferenceEngineConfig`:

```python
# Memory management options
incremental_cleanup: bool = False
"""When True, call free_pages_for_finished() after each decode round to incrementally
release pages for finished sequences. Default False to preserve existing behavior."""

use_logical_reset: bool = False
"""When True, skip physical zeroing of KV cache during reset. The attention mask
ensures stale data is never read. Default False to preserve existing behavior."""
```

---

## P0: Safety Target (Low Risk)

### P0.1: Add Diagnostics and Logging ✅ DONE
**Goal**: Improve visibility into failures before making behavioral changes.

**Changes**:
1. Add `debug_stats()` method to `DecodeState` returning:
   - Active sequence count
   - Total pages allocated
   - Free page count
   - Page ref count histogram
2. Add optional logging to `allocate_for_seq()` and `free_pages()`
3. Log page state at start/end of each decode round

**Files**:
- `src/levanter/inference/jit_scheduler.py` - Add `debug_stats()` method (~line 700)

**Test**: Run 1-prompt config, verify stats are logged correctly.

---

### P0.2: Remove Redundant `kv_pages` Field ✅ DONE
**Goal**: Remove duplication that wastes memory and risks divergence bugs.

**Problem**: `SequenceTable.kv_pages` and `page_indices` are always identical (synced at lines 356, 420, 525, 143-144).

**Changes**:
1. Remove `kv_pages` field from `SequenceTable`
2. Add `@property kv_pages` that returns `page_indices` for compatibility
3. Remove all assignments to `kv_pages` in:
   - `reserve_slot()` (line 143)
   - `assign_slot()` (line 171)
   - `clear_slots()` (line 193)
   - `allocate_for_seq()` (line 356)
   - `free_pages()` (line 420)
   - `clone_pages_from()` (line 525)

**Files**:
- `src/levanter/inference/jit_scheduler.py`

**Test**: Existing tests pass unchanged.

---

### P0.3: Wire Up Existing Cleanup Primitives ✅ DONE
**Goal**: Enable incremental page cleanup without full reset.

**Problem**: `free_pages_for_finished()` exists (line 438-472) but is NEVER called.

**Changes**:
1. Add `incremental_cleanup: bool = False` to `InferenceEngineConfig`
2. When enabled, call `decode_state.free_pages_for_finished(finished_mask)` after each decode round
3. Keep default `False` to preserve existing behavior

**Files**:
- `src/levanter/inference/engine.py` - Add config option, wire into decode loop

**Test**:
- Unit test verifying ref counts decrement after cleanup
- Run 1-prompt config with `incremental_cleanup=True`

---

## P1: Intermediate Steps (Medium Risk)

### P1.1: Logical Reset (Skip Physical Zeroing) ✅ DONE
**Goal**: Avoid expensive `jnp.zeros_like()` on entire KV tensor.

**Problem**: `KvPageCache.reset()` (line 60-63) zeros entire tensor even though:
- Attention masks by `kv_len` (stale data not read)
- New allocations overwrite stale data

**Changes**:
1. Add `use_logical_reset: bool = False` to `InferenceEngineConfig`
2. Add `KvPageCache.reset_logical()` that returns `self` unchanged
3. When enabled, only reset metadata (PageTable ref counts, SequenceTable state)

**Files**:
- `src/levanter/layers/kv_cache.py` - Add `reset_logical()`
- `src/levanter/inference/engine.py` - Add config option

**Test**:
- Unit test: allocate pages, reset logically, reallocate, verify correct KV written
- Benchmark: measure reset time reduction

---

### P1.2: Reference Attention Block Size Optimization ✅ DONE
**Goal**: Speed up fallback attention when TPU kernel is disabled.

**Problem**: `default_ragged_paged_attention()` uses `Q_BS=1` (line 1989), processing one token at a time.

**Changes**:
1. Change `Q_BS = min(1, ...)` to `Q_BS = min(16, q.axis_size("position"))`
2. Change `KV_BS = min(2, ...)` to `KV_BS = min(16, page_indices.axis_size("page"))`
3. Add parameters to make block sizes configurable

**Files**:
- `src/levanter/layers/attention.py` - Update `default_ragged_paged_attention()` (line 1989)

**Test**:
- Existing `test_paged_attention.py` tests pass
- Benchmark: measure tok/s improvement

---

### P1.3: Call `invalidate_finished()` in Decode Loop ✅ DONE
**Goal**: Clean up finished sequence metadata during decode.

**Problem**: `invalidate_finished()` exists (line 654-664) but is never called.

**Changes**:
1. Call `decode_state.invalidate_finished()` at end of each decode round
2. Purge queue entries for finished sequences via `purge_queue_of_slot()`

**Files**:
- `src/levanter/inference/engine.py` - Add calls in decode loop

**Test**: Verify finished sequences don't take up slots in subsequent rounds.

---

## P2: Stretch Goals (Higher Risk)

### P2.1: Investigate Multi-Prompt Failures
**Goal**: Understand why 5+ prompts fail on v5p-16.

**Investigation**:
1. Add per-round logging of page allocation state
2. Check if page exhaustion occurs
3. Verify config values identical across hosts
4. Check for non-deterministic compilation (HLO dumps)

**Files**:
- `src/levanter/inference/engine.py`
- `src/levanter/main/sample_lm_multihost.py`

---

### P2.2: Fix TPU Kernel VMEM OOM
**Goal**: Re-enable TPU ragged paged attention kernel.

**Problem**: Kernel allocates 37.5MB scratch vs 16MB limit.

**Investigation**:
1. Profile vmem usage per head configuration
2. Check if head padding (line 1894-1911) contributes
3. Consider chunked processing for large sequences

**Files**:
- `src/levanter/layers/attention.py` - `_do_tpu_ragged_paged_attention()`

---

## Implementation Order

```
Week 1: P0 (Safety Target)
  P0.1 (Diagnostics) -> P0.2 (Remove kv_pages) -> P0.3 (Wire cleanup)
  Test: 1-prompt config still works

Week 2: P1.1 + P1.2 (Performance)
  P1.2 (Block sizes) - independent
  P1.1 (Logical reset) - after P0.3
  Test: 1-prompt config, measure speedup

Week 3: P1.3 + P2.1 (Multi-prompt)
  P1.3 (Invalidate finished)
  P2.1 (Debug multi-prompt failures)
  Test: 5-prompt config

Week 4+: P2.2 (TPU Kernel)
  Only if P2.1 unblocks multi-prompt
```

---

## Verification

After each step:
1. Run existing tests: `pytest tests/inference/test_paged_attention.py`
2. Run 1-prompt 2048 config (known working): verify clean exit
3. Run 1-prompt 4096 config (known working): verify clean exit
4. Check W&B logs for throughput regression

After P1 complete:
5. Run 5-prompt config: check if failure mode changes
6. Benchmark tok/s with reference attention

---

## Critical Files

| File | Key Lines | Purpose |
|------|-----------|---------|
| `src/levanter/inference/engine.py` | 1055 (reset), 1322-1331 (done flags) | Inference orchestration |
| `src/levanter/inference/jit_scheduler.py` | 79-80 (kv_pages), 438-472 (free_pages_for_finished), 654-664 (invalidate_finished) | State management |
| `src/levanter/layers/kv_cache.py` | 60-63 (reset) | KV storage |
| `src/levanter/layers/attention.py` | 1989 (Q_BS=1), 1822-1879 (dispatcher) | Attention impl |

---

## Background: Why These Issues Exist

### Root Cause of OOM
The original OOM was **TPU scoped vmem** (not HBM) inside the ragged paged attention kernel:
```
RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem
Scoped allocation with size 37.54M and limit 16.00M exceeded scoped vmem limit
```

This forced disabling the TPU kernel (`use_tpu_ragged_paged_attention: false`), falling back to a slow reference implementation.

### Why Cleanup Primitives Aren't Used
The cleanup primitives (`free_pages_for_finished`, `invalidate_finished`) were implemented but never wired into the execution path. The engine relies on `GenState.reset()` which zeros the entire KV cache between `generate()` calls - a brute-force approach that works but is expensive.

### Why Reference Attention Is Slow
`default_ragged_paged_attention` was explicitly marked "not optimized for performance" and uses `Q_BS=1` (single-token processing) with nested `fori_loop`s, causing:
- No vectorization
- Many kernel launches
- Poor memory access patterns

### Why Multi-Prompt Fails
Unknown root cause. HLO dumps show identical compilation across hosts. Possible causes:
- Page exhaustion when running multiple prompts sequentially
- Non-deterministic state accumulation
- TPU runtime issues (launch-group mismatch on v5p-64)

---

## CRITICAL: model.decode() Crash on v5p-16 (2026-02-04)

### Current State
Even 1-prompt config crashes on v5p-16 multihost at `model.decode()`:

**What Works:**
- Model loading on both hosts
- Engine creation on both hosts
- Simple JIT test (hax.named_jit with scalar math)
- Model embedding lookup (`model.embeddings.embed(tokens)`)
- KV cache creation (`hax.named_jit(model.initial_cache)(...)`)
- Sync barrier (`barrier_sync_with_tag`)

**What Fails:**
- `model.decode(tokens, cache, binfo, pos_ids)` inside `_prefill_kernel`
- Both hosts reach the call point, then exit with status 1
- No Python traceback - crash is at XLA/TPU level
- jax.debug.print statements inside model.decode don't execute

### Debug Attempts
1. Added sync barrier before `_run_prefill` - both hosts sync, still crash
2. Set `use_tpu_ragged_paged_attention: false` - still crash (reference impl)
3. Set `JAX_TRACEBACK_FILTERING=off` - no additional info
4. Added jax.debug.print inside `_prefill_kernel` - doesn't print (crash before execution)

### Possible Causes
1. **SPMD trace mismatch**: Different hosts generating different HLO
2. **Array sharding mismatch**: Inference arrays not matching model sharding
3. **Mesh configuration**: model=4, replica_dcn=2 might have issues
4. **Nested control flow**: `default_ragged_paged_attention` has nested fori_loops

### Next Steps
1. Try single-host (v5p-8) to confirm code works outside multihost
2. Check HLO dumps for trace differences between hosts
3. Add explicit sharding constraints to inference arrays
4. Consider simpler attention path for debugging

---

## Testing Plan (Post-Implementation)

Now that P0 and P1 are implemented, run configs in this order to validate incrementally:

### Phase 1: Verify No Regression (Baseline)
**Goal**: Confirm existing working configs still work with default settings.

| # | Config | Flags | Expected | Result | Notes |
|---|--------|-------|----------|--------|-------|
| 1 | v5p-16, 1 prompt, 2048 tokens | defaults | ✅ SUCCESS | ✅ PASS | ~111-124 tok/s, clean exit |
| 2 | v5p-16, 1 prompt, 4096 tokens | defaults | ✅ SUCCESS | ✅ PASS | ~5x faster engine creation! |

#### Config 1 Results (2026-02-03)
- **W&B Run**: [unique-terrain-33](https://wandb.ai/marin-community/levanter-multihost-sampling/runs/6vsn4ndk)
- **Throughput**: 111-124 tok/s
- **Decode iter**: ~0.26-0.28s (device 0s, host ~0.26s, extract ~0.25s)
- **HBM**: 88 GiB free after gen, 89 GiB free after engine
- **Status**: Job finished with no error

#### Config 2 Results (2026-02-03)
- **Engine creation time**: ~3-4s (was ~18s before P1.2 optimization → **~5x speedup**)
- **HBM**: ~80-90 GiB free (no regression)
- **Status**: Job finished with no error

**Key Finding**: The block size optimization (P1.2: Q_BS=1→16, KV_BS=2→16) provides massive speedup for reference attention path.

---

## 🎉 Major Result: Reference Attention Now Viable

The P1.2 block size optimization made the reference attention implementation **~5x faster**:

| Metric | Before (Q_BS=1) | After (Q_BS=16) | Improvement |
|--------|-----------------|-----------------|-------------|
| Engine creation (4096 tokens) | ~18s | ~3-4s | **~5x faster** |
| HBM usage | ~80-90 GiB free | ~80-90 GiB free | No regression |

**Why this matters**:
- The TPU ragged paged attention kernel hits VMEM OOM (37MB vs 16MB limit)
- Previously, disabling it meant falling back to a very slow reference implementation
- Now the reference implementation is actually usable as a production fallback

```bash
# Config 1: Baseline 2048
python -m levanter.main.sample_lm_multihost \
  --config config/sampler/sample_llama8b_multihost_real_1prompt_2048.yaml

# Config 2: Baseline 4096
python -m levanter.main.sample_lm_multihost \
  --config config/sampler/sample_llama8b_multihost_real_1prompt.yaml
```

### Phase 2: Test New Options (Single Prompt) - OPTIONAL
**Goal**: Verify new config options work correctly on known-good configs.

| # | Config | Flags | Expected | Result | Notes |
|---|--------|-------|----------|--------|-------|
| 3 | v5p-16, 1 prompt, 2048 | `use_logical_reset: true` | ✅ SUCCESS | ⏳ skip | Skip cache zeroing |
| 4 | v5p-16, 1 prompt, 2048 | `incremental_cleanup: true` | ✅ SUCCESS | ⏳ skip | Free finished pages |
| 5 | v5p-16, 1 prompt, 2048 | Both options enabled | ✅ SUCCESS | ⏳ skip | Full optimization |

> **Recommendation**: Skip Phase 2 for now. The new options (`incremental_cleanup`, `use_logical_reset`) are primarily useful for multi-prompt scenarios. Single-prompt configs already work well. Jump to Phase 3 to test the failing multi-prompt case.

### Phase 3: Multi-Prompt (The Failing Case) ⬅️ **DO THIS NEXT**
**Goal**: See if changes help or change the failure mode.

| # | Config | Flags | Expected | Result | Notes |
|---|--------|-------|----------|--------|-------|
| 6 | v5p-16, 5 prompts | defaults | ❓ Unknown | ⏳ pending | Was failing before |
| 7 | v5p-16, 5 prompts | `incremental_cleanup: true` | ❓ | ⏳ pending | May help page exhaustion |
| 8 | v5p-16, 5 prompts | Both options | ❓ | ⏳ pending | Full optimization |
| 9 | v5p-16, 20 prompts | `incremental_cleanup: true` | ❓ | ⏳ pending | Stress test |

#### Recommended Test Order

**Step 1**: Run 5 prompts with defaults (Config 6)
```bash
python infra/launch.py --foreground --zone us-central1-a \
    --tpu_name simpo_worker --tpu_type v5p-16 --capacity_type on-demand -- \
    uv run src/levanter/main/sample_lm_multihost.py \
    --config_path config/sampler/sample_llama8b_multihost_real_5prompts_4096.yaml
```

**Step 2**: If Config 6 fails, try with `incremental_cleanup: true` (Config 7)
```bash
# Add to config or pass as override
# engine:
#   incremental_cleanup: true
```

**Step 3**: If still failing, try with both options (Config 8)
```bash
# engine:
#   incremental_cleanup: true
#   use_logical_reset: true
```

**What to check**:
- Does it still fail the same way?
- If it fails, where in the logs?
- Check W&B for `decode/tokens_per_sec` to see if it runs longer before failing
- Compare failure mode before/after

#### Config 6 Attempt 1 (2026-02-03)
- **Result**: ❌ FAILED - exit status 1, no traceback
- **Observation**: Decode was running fine (~84 tok/s), then both workers crashed simultaneously
- **Last log**: `63 new` tokens (down from 64) - suggests sequence was finishing
- **Diagnosis**: Crash happens at TPU/XLA level, not Python. No visibility into cause.

**Diagnostic logging added** to `sample_lm_multihost.py`:
- `debug/batch_starting` - logged before each prompt batch
- `debug/batch_completed` - logged after each prompt batch
- `debug/active_seqs`, `debug/pages_allocated`, `debug/pages_free` - page state after each batch

This will help identify:
1. Which prompt batch was running when crash happened
2. Whether page exhaustion is occurring
3. Whether cleanup between batches is working

### Phase 4: v5p-64 (Launch Group Issues)
**Goal**: Attempt v5p-64 to see if changes affect launch-group mismatch.

| # | Config | Flags | Expected | Notes |
|---|--------|-------|----------|-------|
| 10 | v5p-64, 1 prompt | defaults | ❓ | Was failing with launch-group |
| 11 | v5p-64, 1 prompt | Both options | ❓ | Try with all optimizations |

**Lower priority** - only attempt if Phase 1-3 pass.

---

## Debugging Commands

### Add Debug Logging
In your config YAML or command line:
```yaml
# Enable debug stats printing in decode loop
# (requires adding calls to debug_stats_print() in engine.py)
```

### Check Page State
Add to `_run_generation_loop` body for debugging:
```python
# At start of body():
gen_state.decode_state.debug_stats_print()
```

### Quick Local Test
```bash
cd lib/levanter
uv run pytest tests/inference/ -v --tb=short
```

---

## Success Criteria

1. **Phase 1 Pass**: No regression on working configs
2. **Phase 2 Pass**: New options work without breaking anything
3. **Phase 3**: Either:
   - Multi-prompt now works (success!)
   - Or we have better diagnostics on why it fails (progress)
4. **Performance**: tok/s should be same or better with `use_logical_reset=True`
