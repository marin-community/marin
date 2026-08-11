# MoK-like XLA collective-memory probe

Status: completed and superseded by the production integration and research record in
[`jax_ffi_peer_visible_buffers/research.md`](jax_ffi_peer_visible_buffers/research.md).

## Objective

Determine whether JAX 0.11 can give a typed GPU FFI peer-visible operand and
destination-passing results with an explicit XLA lifetime, without changing the
supported staged `mok_like` backend.

## Contract

The first experiment is deliberately smaller than MoK:

```text
four local u32 shards
    -> XLA custom call with operand/result memory space 1
    -> rank r remotely reads rank r+1's operand
    -> rank r remotely writes rank r+1's result
    -> synchronize every supplied FFI stream
    -> four-rank completion rendezvous
    -> return both destination-passing results
```

The native coordinator is keyed by `ffi::RunId`. It rejects duplicate or
missing ranks, waits at most five minutes, and does not let any handler return
until every remote access has completed. Exact output vectors prove both the
remote read and remote write. A concurrent two-execution gate follows only
after the single invocation passes.

The JAX lowering uses
`jax.ffi.ffi_lowering(..., extra_attributes={"mhlo.frontend_attributes": ...})`
to attach `operands_memory_spaces` and `results_memory_spaces` with color 1.
StableHLO attributes are necessary but not sufficient: the run must retain HLO
or buffer-assignment evidence that the allocations received collective color.
Invalid color 99 is the compile-failure plumbing control; default-space peer
access is not a negative control because this OpenXLA pin peer-maps the ordinary
CUDA VMM allocator too.

## Gates

1. CPU/lowering tests validate shapes, dtypes, memory-space attributes, and
   import/build laziness.
2. Four GB200s: one RunId, four distinct ranks, exact ring read/write outputs,
   zero timeout or coordinator error.
3. Two concurrent invocations of the same compiled executable use distinct
   RunIds and retain exact outputs without pointer mixing.
4. Inspect optimized HLO/buffer assignment and identify any color-boundary
   copies. Do not claim zero-copy from StableHLO text alone.
5. Only after those pass, add custom-VJP/rematerialization and repeated buffer
   reuse probes. MoK integration and staging-copy removal are explicitly out of
   scope until the lifetime and allocation evidence is complete.

## Falsifiers

- JAX 0.11 drops the frontend attributes.
- XLA rejects color 1 on the FFI operand/results.
- Buffer assignment does not use collective memory or inserts unavoidable
  copies around the operation.
- Peer access is not valid for the colored buffers.
- A handler can return while a remote stream still accesses its buffer.
- Concurrent RunIds mix pointers or deadlock.
