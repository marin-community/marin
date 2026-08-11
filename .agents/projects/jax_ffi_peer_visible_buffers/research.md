# JAX FFI peer-visible buffers: research

## TL;DR

- OpenXLA collective memory makes custom-call buffers peer-visible, but it added
  three boundary copies in the isolated probe.
- The pinned JAX 0.11 default CUDA allocator also peer-maps ordinary buffers. A
  memory-space-zero probe performed exact remote reads/writes with no optimized
  HLO copy, including two concurrent executions.
- `mok_like` now has explicit experimental direct-storage modes with staged
  fallback. Direct forward `x` passed the full VJP/concurrency matrix and
  improved four-GB200 steady throughput by 0.51%. Direct backward inputs did
  not improve end-to-end throughput; direct router-gradient output regressed it
  by about 2.4%.
- Equal rank-relative virtual addresses are unnecessary. Each owner pointer
  must be dereferenceable unchanged by its peers, and the invocation lease
  carries one pointer per owner rank.
- Default-space peer mapping remains an implementation property of the pinned
  runtime. The proposed JAX/PJRT capability is still needed for a portable,
  supported contract.

## Framing

The barrier-free staged `mok_like` backend copies about 108.09 GiB per GPU per
training step between XLA buffers and runtime-owned peer-visible workspaces.
The experimental direct modes replace selected staging copies with XLA-owned
operands and results. Their current feasibility depends on the pinned CUDA
allocator peer-mapping ordinary memory-space-zero buffers.

## In-repository findings

- Levanter's GPU extra pins `jax[cuda13]==0.11.0` in
  `lib/levanter/pyproject.toml:82-89`.
- The staged forward currently passes ordinary JAX arrays to `jax.ffi.ffi_call`
  and copies remote-access inputs into its two-slot native runtime
  (`lib/levanter/src/levanter/kernels/mixture_of_kittens/ffi.py:84-144`).
- The barrier-free native path already has the ordering protocol a zero-copy
  call needs: generation-tagged source readiness, destination completion, a
  local stream dependency before returning results, RunId/call-site isolation,
  and saved-context validation. Zero-copy should replace storage ownership, not
  replace this protocol.
- The isolated probe is implemented in
  `lib/levanter/src/levanter/kernels/mixture_of_kittens/collective_memory_probe.py`
  and `csrc/collective_memory_probe.cu`. The harness is
  `experiments/grug/moe_hero_ep/mok_like_collective_memory_probe.py`.

## Upstream findings

- JAX 0.11.0 pins OpenXLA commit
  `131bf41acb4650e4391a640c3f1859c1c86ad74b`. That revision includes OpenXLA
  PR [#39834](https://github.com/openxla/xla/pull/39834), which maps custom-call
  frontend attributes `operands_memory_spaces` and `results_memory_spaces` to
  GPU buffer colors. Color 1 is collective memory and color 2 is temporary
  memory.
- JAX 0.11's public `jax.ffi.ffi_lowering` forwards arbitrary MLIR custom-call
  attributes, although `ffi_call` has no first-class memory-space parameter.
  The working bridge is a custom primitive whose lowering constructs the
  context-local `mhlo.frontend_attributes` dictionary.
- At this OpenXLA pin, color 1 selects collective allocation. CUDA VMM grants
  peer read/write access to capable devices. The ordinary CUDA allocator also
  receives peer grants, so successful remote access to default-space buffers is
  not evidence of a supported peer-visibility contract.
- XLA owns only local buffer liveness. It cannot infer an access performed by a
  different GPU. A multi-device FFI must enqueue a local-stream dependency that
  closes every remote read and write before XLA may reuse each local operand or
  expose each result.

Primary upstream references:

- [JAX 0.11 XLA pin](https://github.com/jax-ml/jax/blob/jax-v0.11.0/third_party/xla/revision.bzl)
- [JAX 0.11 FFI lowering](https://github.com/jax-ml/jax/blob/jax-v0.11.0/jax/_src/ffi.py#L286-L336)
- [JAX 0.11 MLIR custom call](https://github.com/jax-ml/jax/blob/jax-v0.11.0/jax/_src/interpreters/mlir.py#L3329-L3415)
- [OpenXLA GPU memory-space colors](https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/service/gpu/gpu_memory_space_assignment.h#L31-L76)
- [OpenXLA collective-memory FFI test](https://github.com/openxla/xla/blob/c73d5b8af35c381dbefb1fd47adcce63fce4cb5e/xla/tests/collective_ops_ffi_test.cc#L550-L608)

## Four-GB200 probe results

`/dlwh/mok-zc-001-single-20260810-1947` compiled and loaded the isolated CUDA
FFI, retained the exact memory-space attributes, and assigned its operand and
two results to `S(1)`. All four ranks remotely read the next rank's XLA operand
and remotely wrote the next rank's destination-passing result exactly. The
optimized HLO inserted three boundary copies:

```text
default input -> S(1) operand
S(1) result 0 -> default result 0
S(1) result 1 -> default result 1
```

`/dlwh/mok-zc-002-concurrent-20260810-1949` invoked the same compiled
executable twice from two host threads with distinct inputs. Both remote read
and write outputs were exact, proving RunId isolation without pointer mixing or
deadlock. `/dlwh/mok-zc-003-invalid-color-20260810-1949` failed promptly at
compile time with `Invalid memory space 99`, confirming that XLA interpreted
the attributes instead of retaining inert metadata.

The isolated probe synchronizes each CUDA stream on the host before pointer
exchange and again before the completion rendezvous. That proves a safe
operation-scoped lifetime is possible, but it does not prove that a production
handler can return after enqueueing nonblocking GPU readiness/completion waits.

`/dlwh/mok-zc-004-default-space-20260810-2255` repeated the ring in memory
space zero. Single and concurrent executions produced exact remote reads and
writes. Optimized HLO made the custom call the root and contained zero copy,
copy-start, or copy-done instructions. This proves direct access for the pinned
allocator/runtime combination; it does not establish a public JAX guarantee.

## Production integration results

The production FFI exchanges pointers per invocation, keyed by `RunId`, static
collective ID, phase, and ordinal. It publishes owner readiness on each XLA
stream, waits before peer access, publishes completion after remote work, and
closes remote lifetimes before XLA may reuse operands or consume results. XLA
pointers remain in the invocation lease and are never cached in the persistent
runtime.

Four-GB200 correctness covered forward, `dx`, combine/router gradients, every
routed/shared weight gradient, offloaded saved context, two dependent calls,
two concurrent executions, zero-token experts, 3:1 skew, all-to-one routing,
and one, two, eight, and 32 real macrobuffers. Handler counts and slot telemetry
showed no replay, pointer mixing, generation mismatch, or premature reuse.

The matched 48-layer results use steps 60-79 before profiling:

| Storage mode | Tokens/s | Step time | MFU | Delta vs staged |
|---|---:|---:|---:|---:|
| staged forward/backward | 20,966.88 | 12.5030 s | 24.5217% | baseline |
| direct forward `x`, staged backward | 21,074.51 | 12.4394 s | 24.6476% | +0.51% |
| direct forward and backward inputs, staged router-gradient output | 20,999.04 | 12.4839 s | 24.5593% | +0.15% |
| all direct | 20,463.28 | 12.8108 s | 23.9327% | -2.40% |

The all-direct regression came from destination-readiness overhead for a one-MiB
router-gradient output. Keeping that output staged recovered the regression.
Moving backward readiness before local-only clears removed 2.51 trillion
recorded wait cycles but did not change step time, showing that those waits were
overlapped rather than critical-path work. Direct forward `x` remains the best
measured storage choice.

## Staging-traffic derivation

The production local shape is 65,536 tokens, hidden width 6,144, top-k 4, and
BF16 activations. Each activation-sized copy is
`65,536 * 6,144 * 2 = 805,306,368` bytes, or 0.75 GiB. The five explicit copies
per layer are forward `x`; backward `d_y`, saved `x`, router/combine input; and
router/combine gradient output. The two router tensors are each
`65,536 * 4 * 4 = 1,048,576` bytes. Total traffic is therefore 2.251953 GiB per
layer per GPU, 108.09375 GiB across 48 layers, and 432.375 GiB across the
four-GPU process per step. These are code-derived logical bytes, not Nsight
copy-engine measurements.

## Conclusions

Color one does not solve the MoK boundary-copy problem. It moved each copy to a
compiler-inserted memory-space transition. Memory-space-zero buffers avoid
those transitions on the pinned runtime and are sufficient for a measured
forward-path win.

The supported contract still needs two upstream pieces. A PJRT memory kind must
let external parameters and donated arrays arrive in peer-visible storage. An
HLO allocation capability must propagate through complete alias sets for
compiler-owned temporaries. The capability stays orthogonal to HLO memory space
so ordinary producers and consumers do not create color-boundary copies.

`mok_like` requires peer-addressable UVA/VMM and per-owner pointer exchange. It
does not require equal rank-relative addresses or NCCL window registration.
The direct modes remain explicitly experimental and pin-gated; staged storage
remains the fallback.
