# GCS TensorStore Loader Revamp

## Background

The current GCS safetensors loader asks TensorStore for one slice per FSDP shard. Each slice triggers a tiny random read across the safetensors file, so multi-host checkpoints make millions of independent byte-range requests. Even with batching, the async path ends up waiting on one slice at a time, yielding very slow staging relative to copying the full files locally.

We want to keep the “no local copy” property for TPU jobs (local disk is scarce) while avoiding the pathological access pattern. Safetensors provides the metadata we need: each tensor lives in a flat file with known dtype, shape, and byte offsets. We can leverage that to schedule a small number of large, contiguous reads and shuffle the bytes to the devices we actually need.

## Constraints & Targets

- Do not require staging the entire checkpoint to local disk; operate directly on GCS via `fsspec`.
- Preserve existing HF checkpoint interfaces (`create_async_array_from_callback` consumers should keep working).
- Support multi-host setups using the existing device mesh; downstream code assumes tensors arrive with shardings from `best_effort_sharding`.
- Read complete tensors (never split a single key across multiple chunks) to keep bookkeeping manageable.
- Default chunk size: 2 GB, but expose an override (env or config) so we can tune for memory pressure.
- Keep the legacy small-slice loader behind a feature flag for rollback while the new path hardens.

## Proposed Algorithm

1. **Metadata pass (all hosts).**
   - Use the safetensors header to build `TensorMeta(key, file_path, dtype, byte_start, byte_end, shape)`.
   - Cache file-level metadata so repeated loads do not re-download headers.

2. **Chunk construction (all hosts, deterministic).**
   - For each file, walk tensors in storage order and pack them into contiguous `ChunkSpec`s.
   - Each `ChunkSpec` keeps: `file_path`, `byte_start`, `byte_end`, ordered list of `(key, dtype, shape, offset_within_chunk)`, and the `PartitionSpec` for each key (via `best_effort_sharding`).
   - Never split a tensor across chunks; if one tensor exceeds the chunk size limit, let the chunk size grow to accommodate it.

3. **Chunk assignment (all hosts, deterministic).**
   - Enumerate chunks globally (e.g., sorted by `(file_path, byte_start)`).
   - Assign each chunk to a host `owner_rank = chunk_index % world_size` (or similar simple rule) so every process arrives at the same mapping without communication.

4. **Chunk materialization (owner host only).**
   - Owner performs a single `fsspec` range read for `[byte_start, byte_end)` and converts the result to a NumPy buffer.
   - Produce views for each tensor using the metadata offsets; apply dtype conversion if requested.

5. **Hand-off via callbacks (future refinement).**
   - Initial goal was to wrap each chunk in `jax.make_array_from_callback`, returning real data on the owning host and lightweight placeholders elsewhere.
   - Current implementation keeps the chunk owner model logically, but for now every host issues its own range read for the chunk (still coalesced) and materialises tensors locally. This avoids the TPU round-trip penalty we saw with `broadcast_one_to_all`, at the cost of duplicate network I/O. The next iteration should restore single-owner reads while keeping the data on TPU during redistribution (see “Next Steps”).

6. **Per-key extraction.**
   - After reshaping, cut the chunk back into individual tensors using the recorded offsets.
   - Supply each tensor to the existing `create_async_array_from_callback` path, preserving the computed sharding.

7. **Cleanup / Feature flag.**
   - Provide a configuration flag (env var `LEVANTER_USE_CHUNKED_GCS_LOADER`, default off initially) to pick between loaders.
   - Ensure we can fall back if issues arise during rollout.

### Pseudocode Sketch

```python
def build_chunk_specs(tensor_meta, chunk_limit):
    specs = []
    current = new_chunk()
    for meta in sorted(tensor_meta, key=lambda m: (m.file_path, m.byte_start)):
        if meta.file_path != current.file_path or current.size + meta.size > chunk_limit:
            specs.append(current)
            current = new_chunk(meta.file_path)
        current.add_tensor(meta)
    if current.tensors:
        specs.append(current)
    return specs

def chunk_owner(chunk_index, world_size):
    return chunk_index % world_size
```

## Current Status

- Implemented a chunked async loader (`read_safetensors_fsspec`) that parses metadata once, builds deterministic chunk specs, and reads contiguous byte ranges via `fsspec`.
- Each host now coalesces its reads (large sequential ranges instead of per-tensor slices) and reconstructs tensors locally; sharded outputs are assembled via `best_effort_sharding`.
- HF checkpoint loading routes through the new loader.
- Concurrency and chunk sizing are externally tunable via env vars (`LEVANTER_FSSPEC_CHUNK_BYTES`, `LEVANTER_FSSPEC_MAX_CONCURRENT_CHUNKS`, `LEVANTER_FSSPEC_MAX_WORKERS`).
- Current limitation: every host still downloads every chunk; parallel TPU redistribution is not yet implemented.

## Implementation Tasks

- [x] Parse safetensors metadata on every worker and build `{key: TensorMeta}` without redundant I/O.
- [x] Add chunk builder producing deterministic `ChunkSpec`s that never split tensors and respect the configured size limit (default 2 GB).
- [ ] Implement deterministic host assignment (`chunk_index % world_size`) and expose helper to query chunk ownership. (Removed from the current host-local implementation; will return when we do owner-based reads.)
- [x] Fetch contiguous byte ranges via `fsspec`, reconstruct tensors, apply dtype conversions, and pass them to existing consumers with their target shardings.
- [ ] Reintroduce a callback-based hand-off (or equivalent) that keeps non-owners from allocating each chunk, while still integrating with `best_effort_sharding`.
- [ ] Add per-process chunk prefetch so the next chunk begins materialising while the current tensors are extracted.
- [x] Add configuration toggles (env vars for chunk size, read concurrency, and worker pool sizing) and document defaults.
- [ ] Add broader tests (multi-host mock or integration) to verify redistribution behaviour, dtype handling, and the fallback switch.

## Open Questions / Follow-Ups

- Validate memory pressure on TPU hosts for 2GB reads; adjust defaults if necessary.
- Confirm any future placeholder/callback approach doesn’t confuse downstream shape inference; document expected array shapes.
- Decide where to log chunk assignments / timings for profiling.
- Consider caching chunk specs / metadata between runs to avoid recomputation when repeatedly loading the same checkpoint revision.
- Investigate alternatives to full per-host reads: ideal plan is “owner host performs read → chunk stays on TPU”:
  1. Deterministically assign each chunk to an owner host (`chunk_index % world_size`).
  2. Owner reads the chunk and uploads it directly to TPU memory; non-owners upload zeros.
  3. Within a jit context, perform an `all_gather` that keeps data on TPU (no CPU round-trip).
  4. Slice and apply `best_effort_sharding` inside that jit so tensors materialize directly on device.
  5. Only fall back to host reconstruction if the TPU path fails.

### JIT Sketch for TPU-Resident Redistribution

```python
import jax


@functools.partial(jax.jit, out_shardings=None)
def assemble_chunks(chunks, chunk_specs):
    """
    Args:

    Returns:
        dict mapping tensor key -> sharded jax.Array.
    """

    # Step 1: collective gather so every device sees every host's chunk.
    # this has to be in a shard_map i think
    gathered = chunks =

    tensors = {}
    cursor = 0
    for host_idx, chunk_meta in enumerate(chunk_specs):
        chunk_bytes = gathered[host_idx]
        for tensor_meta in chunk_meta.tensors:
            start = tensor_meta.byte_offset // tensor_meta.dtype.itemsize
            end = start + np.prod(tensor_meta.shape, dtype=int)
            raw = chunk_bytes[start:end].reshape(tensor_meta.shape).astype(tensor_meta.dtype)

            sharding = best_effort_sharding(tensor_meta.shape)
            if sharding is None:
                tensors[tensor_meta.key] = raw
            else:
                tensors[tensor_meta.key] = jax.experimental.pjit.with_sharding_constraint(raw, sharding)

    return tensors


# Host-side flow
chunk_specs = chunk_specs + [None] * (num_hosts - (len(chunk_specs) % num_hosts))

for k in range(len(chunk_specs) // num_hosts):
    chunk_specs = chunks[k * num_hosts: (k + 1) * num_hosts]
    max_chunk_size = max(spec.size for spec in chunk_specs)

    local_chunk = np.zeros(max_chunk_size, dtype=np.uint8)
    owned_chunk = maybe_fetch_chunk_for_host(...)
    if owned_chunk is not None:
        local_chunk[: len(owned_chunk)] = owned_chunk


    # stack with a process mesh?
    chunks = jax.make_array_from_single_device_arrays()
    assembled = assemble_chunks(device_chunk, chunk_specs)
```

- Padding ensures `all_gather` operates on equal-length buffers (TPU requirement).
- `chunk_metadata` is a lightweight pytree describing per-tensor offsets and shapes.
- The jit body never touches host memory; `best_effort_sharding` (or the equivalent logic) must be expressed in a jit-friendly way so shard placement happens inside the device context.
