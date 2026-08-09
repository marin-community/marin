# Routed Grug weight-gradient GB200 checkpoint

This artifact records two generated group-batched weight-gradient Contracts
inside a natural one-layer Grug training step. JAX owns differentiation.
Shuttle recovers the Contracts from the GPU `PRE_SCHEDULER` HLO and replaces
each exact dot boundary with an independent typed FFI call:

```text
lhs[E,K,M] x rhs[E,K,N] -> gradient[E,M,N]
```

The two recovered shapes are `[4,512,32] x [4,512,64] -> [4,32,64]` and
`[4,512,32] x [4,512,32] -> [4,32,32]`. The generated physical primitive is a
generic strided-batched cuBLAS Contract. It has no MoE routing, activation,
transport, or reduction logic. Each output has one writer and no atomics.

## Result

The measurement used one NVIDIA GB200 at compute capability 10.0 with driver
595.71.05. The workload used JAX/JAXlib 0.11.0 and CUDA 13.2.78 compiler
components. The batch reservation requested one GPU, four CPUs, and 64 GiB of
host memory. `telemetry.csv` contains 291 samples at approximately 100 ms
intervals. Graphics clocks ranged from 120 to 1950 MHz, memory clocks remained
at 3996 MHz, and sampled power ranged from 149.93 to 212.01 W under a 1200 W
limit.

Thirty paired whole-step samples alternate launch order, with 15 pairs starting
from each implementation:

| Path | Median whole-step latency |
| --- | ---: |
| Stock XLA | 0.725728 ms |
| Shuttle weight Contracts | 0.774129 ms |
| Shuttle / XLA | 1.066692× |

This telemetry-instrumented replay is the primary frozen result. An immediately
preceding independent 30-pair capture on the same GB200, code revision, and
software environment measured 0.681649 ms for stock XLA and 0.744593 ms for
Shuttle, or 1.092341×. `independent-confirmation-summary.json` preserves its
complete raw sample and output-hash distribution.

Both generated handlers executed 35 times. The transformed HLO contains exactly
two custom calls. Maximum absolute error is `3.7252903e-9`, mean absolute error
is `1.1881582e-12`, and 49 of 53 result leaves are bitwise equal in the direct
comparison.

The whole training step produces several output hashes in both paths because
XLA still owns other reductions. The generated Contract boundaries are
deterministic by construction: one Contract owns each output element, uses BF16
operands with FP32 accumulation, and rounds once to BF16 before the unchanged
placement collective. The declared policy is `allow_rounding_reorder`; the
recovery path rejects `bitwise_exact` because the source HLO does not specify a
bitwise dot-reduction tree.

## Boundary audit

The generated calls replace only `dot.6` and `dot.7`. `psum.52` and `psum.53`
still consume those results directly, so all placement collectives remain in
XLA. The transformation adds no copies or transposes. All four FFI operands have
runtime parameter ancestry, both outputs are fresh allocations, and the
generated CUDA contains no `atomicAdd`.

This is a whole-step comparison against stock XLA, not a comparison with an
opaque MoE kernel. The routed forward path, input adjoint, relation index plane,
and collectives remain separate Shuttle/XLA boundaries.

`summary.json` preserves every primary timing sample, launch order, output hash,
correctness statistic, operand binding, and numerical contract.
`independent-confirmation-summary.json` preserves the independent confirmation.
The original and transformed HLO snapshots preserve the compiler boundary. The
two top-level generated CUDA files and their compile-helper copies preserve the
generic physical body. `environment.json`, `hardware.txt`, and `telemetry.csv`
record the reproducibility context. `SHA256SUMS` covers every file other than
itself.
