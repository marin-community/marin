# Generated Grug Contract/Map chain

## Result

Shuttle now generates the recovered low-rank two-Contract training family as
two bounded one-CTA CUDA handlers. The forward handler owns:

~~~
Contract -> scalar Map -> Contract -> scalar Map
~~~

It returns the output and three BF16 save values used by JAX's VJP. The reverse
handler consumes JAX-recovered scalar VJP ASTs and owns:

~~~
scalar Map -> Contract -> scalar Map -> Contract -> scalar Map
             + two weight-adjoint Contracts
~~~

The generator interface contains Contract shapes, scalar ASTs, and bindings to
generic chain values. It has no model or workload dispatch key.

## Numerical boundaries

Both physical handlers use fixed left-to-right FP32 multiply/add loops for every
Contract. Each Contract result is rounded to BF16 round-to-nearest-even before a
Map or subsequent Contract consumes it. The imported scalar AST preserves every
BF16 operation and carries `source_ordered`.

The reverse emits BF16 dX, dW0, and dW1. The dW outputs therefore preserve the
recovered Contract boundary before the surrounding optimizer converts them to
FP32. Placement collectives remain outside this physical family.

## Mutation and parity

The SiLU hidden Map and all three reverse Maps come from the natural Grug HLO.
An independent mutation compiles `tanh` and its JAX VJP to HLO, imports both
scalar programs, and passes them through the same physical generator. The
kernel count, shapes, shared-memory allocation, and handler interface do not
change.

The CPU reference executes the same ordered BF16/FP32 boundaries independently
of CUDA. On random 8x32 inputs with rank 128, the forward is bitwise equal to the
natural JAX function. The recovered reverse has maximum errors of 0.0009765625,
0.00012207031, and 0.00012207031 for dX, dW0, and dW1. The corresponding mean
errors are 0.000079870224, 0.000014353722, and 0.000010687894. The tanh mutation
also matches its natural JAX forward and VJP within BF16 tolerance.

## Physical boundary audit

The accepted generated source depends only on CUDA BF16/runtime primitives and
XLA typed FFI. It contains two CUDA kernels, no atomics, no cuBLAS call, and no
opaque model kernel. The runtime wrapper imports JAX and NumPy, with no Torch
dependency.

For the recovered 8x32x128 family:

| Handler | CTA count | Threads | Static shared memory |
| --- | ---: | ---: | ---: |
| Forward | 1 | 256 | 4,096 bytes |
| Reverse | 1 | 256 | 2,560 bytes |

The implementation is a correctness-first physical family. The scalar loops do
not yet use tensor cores, and no GPU latency claim is attached to this commit.
This host has no NVCC, so CUDA compile/load validation remains the next required
preflight before an H100 run.

## H100 component harness

`h100_generated_contract_map_chain_training.py` is the bounded component gate.
It reconstructs the generic program from the frozen natural Grug HLO, generates
the two CUDA handlers, compiles and loads them through JAX typed FFI, and compares
one generated forward/reverse step with the same ordinary JAX forward and
`jax.vjp`. JAX, rather than Shuttle, defines the reverse scalar programs.

The full run records the raw counterbalanced timing distributions, deterministic
output hashes, source and semantic digests, generated-source audit, StableHLO,
optimized matched-JAX HLO, and H100 telemetry. Host-side instrumentation requires
exactly one forward and one reverse handler invocation per generated step. It is
non-atomic because this bounded harness performs one sequential dispatch stream;
it is evidence instrumentation, not part of the physical computation.

The required environment is Linux x86-64 with one visible H100, CUDA 13 NVCC and
runtime libraries, and matching versions of `jax`, `jaxlib`,
`jax-cuda13-plugin`, and `jax-cuda13-pjrt`, all pinned to 0.11.0. The frozen HLO
fixture is:

~~~
lib/tile_lifetime/benchmarks/artifacts/
  xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/
  transformed-gpu-pre-scheduler-hlo.txt.gz
~~~

From a checkout at the revision under test, the one-run command is:

~~~bash
PYTHONPATH="$PWD/lib/tile_lifetime/src:$PWD" \
/app/.venv/bin/python \
  lib/tile_lifetime/benchmarks/h100_generated_contract_map_chain_training.py \
  --hlo-fixture \
    lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/transformed-gpu-pre-scheduler-hlo.txt.gz \
  --nvcc /app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
  --architecture sm_90a \
  --artifact-directory /tmp/shuttle-contract-map-chain-h100/generated \
  --json-output /tmp/shuttle-contract-map-chain-h100/result.json \
  --shuttle-revision "$(git rev-parse HEAD)" \
  --require-jax-version 0.11.0 \
  --threads 256 \
  --seed 20260809 \
  --warmups 10 \
  --repeats 30 \
  --iterations 1000
~~~

Appending `--preflight-only` performs source generation, NVCC compile/link,
library load, symbol resolution, and zero-count verification without allocating
or executing a GPU buffer. This local macOS host has no NVCC, so neither preflight
nor H100 execution has been claimed yet.

## Capture-safe physical candidate

The generator exposes two physical host-dispatch candidates for the same
Contract/Map semantic program. `launch_checked` performs an immediate CUDA
launch-status query and remains useful for standalone debugging.
`command_buffer_capture_safe` omits that host query and marks both typed-FFI
handlers with `kCmdBufferCompatible`. The capture-safe source contains direct
launches on the FFI-provided stream and no allocation, library handles,
autotuning, synchronization, or CUDA status queries.

The choice is represented as a physical-candidate enum rather than a semantic
option. Consequently, both candidates have the same semantic digest but distinct
source digests. The generic command-buffer audit owns the forbidden-operation
policy and validates both handlers before traits are emitted. SiLU and tanh
hidden Maps use the same capture-safe wrapper; only their generated scalar ASTs
change.

The capture gate records forward and reverse callback counts before and after
every variant in every counterbalanced sample. Each checkpoint includes the
sample index, complete execution order, variant, count delta, and number of
logical handler calls. This distinguishes callbacks caused by the generated
variant from callbacks observed while the matched JAX variant runs.

The bounded acceptance policy requires at least one pre-timing callback for
each handler. It permits at most one further callback per handler on only the
first sample of each distinct counterbalanced order. Later recapture, more than
one recapture per order, a callback attributed to a variant with no declared
handler calls, or a callback count at least as large as the logical call count
rejects the run. The latter is classified separately as per-logical-call
fallback. The result exposes the policy, every checkpoint, per-order deltas,
and the final classification.

The harness writes a pending result containing the full raw timing distributions
before it evaluates this policy. It then rewrites the result with the accepted
or rejected classification and raises only after the rejected result is on
disk. This preserves performance evidence even when capture accounting fails,
as it did for TLTC-XLA-063. No GPU result is attached to the revised gate yet.

## Next bounded candidate: predicated BF16 MMA

The next candidate should replace scalar Contract loops with one-CTA predicated
BF16 MMA while retaining the existing generated preparation and finalization
ASTs. This is a design target, not an implementation in the current checkpoint.

The initial family remains bounded to the recovered rows=8, hidden=32, rank=128
shape class. It pads the physical M dimension to an MMA-compatible tile, predicates
all loads and stores back to the logical shape, and selects fixed N/K tiles from
a small generic set. Forward and reverse continue to round every Contract result
to BF16 round-to-nearest-even before a Map or subsequent Contract consumes it.
Weight adjoints retain their recovered physical layouts rather than assuming
logical row-major storage. The same scalar Map AST attachment accepts SiLU, tanh,
and future mutations without changing the MMA mainloop.

MMA changes reduction association relative to the current fixed left-to-right
FP32 loops. It is therefore legal only under an explicit bounded-reassociation
numerical contract. A source-ordered request must retain the scalar candidate;
the planner must not silently upgrade it to MMA. Before GPU measurement, CPU and
JAX tests should cover logical padding predicates, both weight-adjoint layouts,
BF16 boundaries, scalar-Map mutation, rejection under source-order policy, and
numerical error under the declared contract.

The first implementation should remain one CTA per handler. Multi-CTA reduction,
cluster scheduling, and EventTensor readiness are later physical candidates, not
requirements for this bounded MMA experiment.
