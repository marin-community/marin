# Combined routed Grug training GB200 checkpoint

This artifact records one natural one-layer Grug training step transformed once
at XLA's GPU `PRE_SCHEDULER` stage. JAX owns differentiation. Shuttle recovers
and independently replaces four generic regions:

- routed forward `Contract -> Map -> Contract -> source Fold`;
- routed input adjoint `Contract -> reverse Map -> Contract -> source Fold`;
- two group-batched weight-gradient Contracts.

The generated calls reuse the existing generic physical generators. They are
not combined into a workload-specific megakernel. XLA still owns the relation
index plane, the rematerialized routed chain around these regions, and all
placement collectives.

## Result

The primary measurement used one NVIDIA GB200 at compute capability 10.0 with
driver 595.71.05. The batch reservation requested one GPU, four CPU cores,
64 GB host memory, and 100 GB ephemeral disk. The GPU had a 1200 W power limit.
During the 402 telemetry samples, SM clocks ranged from 120
to 1950 MHz, memory clocks remained at 3996 MHz, and sampled power ranged from
175.61 to 245.37 W. The toolchain used JAX/JAXlib 0.11.0 and CUDA 13.2.78.

Thirty paired whole-step measurements alternate launch order, with 15 pairs
starting from each implementation:

| Path | Median whole-step latency |
| --- | ---: |
| Stock XLA | 0.554336 ms |
| Shuttle combined replacement | 0.654897 ms |
| Shuttle / XLA | 1.181407x |

An immediately preceding two-pair smoke at the same revision measured 1.190x.
It was a validation run only and is not pooled into the primary distribution.
`summary.json` and `stdout.json` contain only the 30-pair primary capture.

Maximum absolute error is `3.7252903e-9`, mean absolute error is
`1.1989784e-12`, and 49 of 53 result leaves are bitwise equal in the direct
comparison. The generated result hash is identical across all 30 executions.
Stock XLA's full-step hashes vary because it still owns other reduction trees.

## Boundary audit

The post-roundtrip HLO contains each generated target exactly once. Every
handler executed 35 times: one correctness execution, four warmups, and thirty
measurements. The generated input-adjoint auxiliary `%select.7` remains a
direct operand of the first weight Contract. Each generated weight Contract has
exactly one direct external all-reduce consumer (`%psum.58` and `%psum.59` in
this roundtrip), so psums remain outside Shuttle. The transformation adds no
copies and removes one transpose (`51 -> 50`).

Forward and input-adjoint regions declare `source_ordered`. Weight Contracts
declare `allow_rounding_reorder`: BF16 operands, FP32 pedantic accumulation,
and one BF16 round-to-nearest-even output conversion before the unchanged
collective. All dynamic operands have natural parameter ancestry; the only
static operand is the input Fold identity. Weight outputs are fresh. Generated
sources contain no `atomicAdd` semantic accumulation.

`summary.json` preserves every timing sample, launch order, output hash,
correctness statistic, operand ancestor, numerical policy, target occurrence,
handler call count, and post-roundtrip wiring audit. Original and transformed
HLO snapshots preserve the compiler boundary. Generated CUDA files preserve
the four physical bodies. `allocation.txt`, `hardware.csv`, `nvidia-smi-q.txt`,
`nvcc-version.txt`, `python-environment.txt`, and `telemetry.csv` preserve the
execution context. `SHA256SUMS` covers every file other than itself.
