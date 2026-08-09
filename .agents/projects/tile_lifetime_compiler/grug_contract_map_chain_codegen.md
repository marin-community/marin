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
