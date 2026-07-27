# UB-X JAX FFI

This package is a bounded raw transport layer for NVIDIA NCCL UB-X at commit
`db0c814185a0415cc2e23dca387fecb9282de551`.

The runtime supports one process with exactly eight visible local GPUs. It owns
one source-built NCCL communicator, symmetric allocation, collective symmetric
window, and NCCL device communicator per GPU. The pool contains:

1. UB-X REG0 synchronization storage.
2. Two `max_tokens_per_rank` BF16 dispatch staging buffers.
3. Two push3 destination buffers of shape
   `[max_local_tokens, top_k, hidden_size]`.

Dispatch accepts precomputed `[token, top_k]` global expert IDs and arbitrary
compact destination slots. It returns `[max_tokens_per_rank, hidden_size]` and
zeros rows where `dispatch_valid` is false. Push3 combine accepts
`inverse_map[max_tokens_per_rank, 4]`, `topk_idx`, and dense FP32 gate weights.

Both FFI calls are effectful. They are intended for a rank-local `shard_map`
body, not as globally partitioned collectives. There is no custom VJP.
Each JaxPP stage process must expose exactly eight process-local CUDA hardware
ordinals `0..7`. Global JAX device IDs may differ between processes and are not
used: the FFI handler selects runtime state with `cudaGetDevice()`.

The build compiles the pinned `ubx.cu` directly into the FFI library and links
the pinned source-built `libnccl.so.2.30.7`. Initialization rejects any
dynamically resolved NCCL runtime other than that exact library and version.

## Current Boundaries

- The pinned checkout must already contain `build/include` and
  `build/lib/libnccl.so.2.30.7` from a source NCCL build.
- Dispatch stages through the symmetric pool and copies into the XLA-owned
  result. Removing that copy requires an XLA-compatible symmetric allocator.
- UB-X's compiled push3 timeout prints and exits its poll loop but does not
  return a failure status. A timeout can therefore yield incomplete data.
- Initialization uses one host thread per local CUDA ordinal for collective
  window registration and device-communicator creation. This path still needs
  an H100 build and runtime gate.
