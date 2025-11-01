# Multi-Host Inference Plan

## Goal
Enable `levanter.inference.engine.InferenceEngine` to execute inference across multiple hosts with both tensor parallelism (TP) and data parallelism (DP). The engine should coordinate prefill/decode work, token queues, and page-table state such that data shards run in lock-step, while preserving the current single-host behavior when `jax.process_count() == 1`.

## Current Architecture (Single Host)
- **GenState / DecodeState**: Own the KV `PageTable`, `TokenQueue`, sequence metadata, and per-sequence decoding params. Assumes all sequences live in a single host.
- **Prefill / Decode loops**: Controlled by `InferenceEngine.generate` which polls request queues, performs prefill batches, and runs decode rounds until sequences finish or buffers fill.
- **Token Queue**: `TokenQueue` exposes named axes (`seq`, `token`) sized for a single process. Entries are produced on the leader, consumed locally.
- **Page Table**: Tracks ownership of KV pages, seq lengths, etc. Allocations are per-host.
- **Control Flow**: `generation_loop` (approx. lines 860-1030) assumes single-host; no explicit `jax.lax.psum` or collective synchronization.

## Design Constraints for Multi-Host DP + TP
1. Introduce a **Batch/Data axis** that is sharded across hosts for all mutable decode state (token queues, seq buffers, page table entries, status arrays).
2. **Leader (process 0)** orchestrates request packing, broadcasts work to other hosts, and aggregates completion signals.
3. Synchronize generation loop across hosts so each round runs collectively; workers only exit once leader signals no more work.
4. Maintain compatibility with TP axis usage in existing jit-compiled functions.

## Open Questions / Investigation Items
- Where to add `Batch` axis definitions? (Likely in `TokenQueue` / `PageTable` constructors.)
- What collective primitives are already wrapped by `levanter`? (Check for helpers around `hax.axis_mapping`, `with_sharding_constraint`, etc.)
- How does `TokenQueue` represent empties? Need to understand interplay with `INVALID` sentinel when queue is sharded.
- Determine how requests are currently distributed; is there existing support for host-local request routing?

## Implementation Plan (Checklist)

### 1. Audit and Document Current State Layout ✅
- [ ] Read `TokenQueue`, `DecodeState`, `PageTable`, and `generation_loop` implementations to map all tensors that need a new Batch axis.
- [ ] Produce diagrams / notes on how seq slots map to queue entries in single-host setup.

### 2. Introduce Batch Axis Types
- [ ] Define `Batch = Axis("batch", data_parallel_world_size)` (or similar) in `engine.py` or a central module.
- [ ] Update constructors in `TokenQueue` / `PageTable` to accept optional `batch_axis` and shard lengths accordingly.
- [ ] Ensure dtype/shape init uses `hax.ones` / `zeros` with both `Batch` and existing axes.

### 3. Leader-Based Request Packing
- [ ] Modify request ingestion so `process_index == 0` gathers all pending prompts, batches them (respecting DP world size), and broadcasts packed tensors to other processes using `jax.lax.broadcast` or `with_sharding` utilities.
- [ ] Non-leader hosts should receive the replicated packed batches and skip local host-side sampling.

### 4. Sharded Token Queue Handling
- [ ] Expand `TokenQueue` buffers to include `Batch` axis. Each host should enqueue/dequeue only its shard.
- [ ] Implement collective enqueue/dequeue coordination so that leader populates queue slices and other hosts read matching slices.
- [ ] Audit queue invariants (head/tail pointers) to ensure they are tracked per-batch element and sharded safely.

### 5. Page Table & Decode State Sharding
- [ ] Add Batch axis to `PageTable` fields (`page_indices`, `seq_lens`, ownership flags).
- [ ] Update allocation helpers (`assign_seq_id_to_seq`, `clone_pages_from`, etc.) to operate per-batch shard, likely by indexing with `(Batch, Seq)`.
- [ ] Ensure KV cache (`KvPageCache`) integrates Batch axis or is already sharded via TP.

### 6. Generation Loop Synchronization
- [ ] Wrap `generation_loop` to include a barrier at each iteration boundary (e.g., `jax.lax.psum(done_flags)`), ensuring all hosts agree when to exit.
- [ ] Introduce leader signals for `should_run_decode` / `should_exit`, broadcast to workers.
- [ ] Handle cases where some hosts run out of local sequences while others still decode (need collective OR).

### 7. Prefill Coordination
- [ ] Leader prefill step should prepare batched inputs (with Batch axis) and broadcast to workers before `engine.prefill` executes.
- [ ] Ensure attention cache writes happen only on local shard but remain synchronized.

### 8. Sampling & Output Aggregation
- [ ] Adapt sampler outputs to gather finished sequences back on leader for final response assembly.
- [ ] Define communication path to send generated tokens / logprobs from workers to leader, minimizing host-to-host traffic.

### 9. Testing & Validation Strategy
- [ ] Add unit tests for single-host behavior regression.
- [ ] Introduce multi-host simulation tests using `jax.experimental.multihost_utils` or `pytest` multi-process harness if available.
- [ ] Create integration test plan for DP+TP (maybe using `pjit` with `Mesh` of size >1).

### 10. Documentation & Follow-up
- [ ] Update docs (e.g., `docs/inference.md`) describing new multi-host mode and configuration knobs.
- [ ] Provide migration notes for existing users (env vars, CLI flags).

## Risks & Mitigations
- **Complex synchronization bugs** → Add extensive logging guarded by `if process_index == 0`, and consider tracing utilities for collective calls.
- **Performance regressions** → Measure latency before/after enabling DP; cache-shard layout should avoid unnecessary broadcasts.
- **State divergence** → Use assertive checksums (`jax.debug.callback`) to compare key tensors across hosts during development.

## Dependencies / Resources
- Review `levanter.inference.jit_scheduler` for existing collective patterns.
- Check if `levanter.distributed` utilities already manage DP axes.
- Confirm expected APIs from `kv_cache` and `Sampler` when additional axes are introduced.
