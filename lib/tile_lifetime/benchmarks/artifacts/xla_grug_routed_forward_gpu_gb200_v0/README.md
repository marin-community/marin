# Generated routed Grug forward on GB200

This artifact replaces one routed forward region in an ordinary one-layer Grug
train step at XLA's `PRE_SCHEDULER` boundary. JAX owns differentiation. Shuttle
recovers and generates:

```text
Contract -> Map -> Contract -> source-keyed Fold
```

The generated path uses generic cuBLAS Contracts, a scalar Map AST, and a
single-writer Fold. It contains no model-name dispatch, opaque MoE kernel, or
atomic accumulation. Every generated operand has ancestry in the natural
train-step parameters or batch.

## Result

The run used four warmups and 30 counterbalanced repeated samples:

| Path | Minimum | Median | P90 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| XLA baseline | 0.8030 ms | 0.8517 ms | 0.8848 ms | 0.9159 ms |
| Generated region | 0.9275 ms | 0.9911 ms | 1.0432 ms | 1.1351 ms |

The generated-to-baseline median ratio is `1.1637x`. Maximum absolute error is
`2.33e-10`, mean absolute error is `4.18e-15`, and 52 of 53 result leaves are
bitwise equal in the direct correctness comparison.

Whole-train-step hashes vary on both paths because other XLA-owned reductions
remain nondeterministic. The generated routed Fold itself has one writer per
source-feature and a fixed destination-major edge traversal. This artifact
therefore does not claim whole-step bitwise determinism.

## Provenance

- Compiler tree: remote branch parent `40a479bf67fd8835e37942057b76f6d69425d02b`
  plus the source and artifact changes in this checkpoint.
- Iris control client: main revision `eafa4d49f7`.
- Iris job: `/dlwh/dev-gpu-routed-grug-ffi-aug09`.
- Hardware: one NVIDIA GB200, compute capability 10.0.
- Driver: `595.71.05`.
- Power limit: 1200 W, P0 when sampled after the run. The sampled graphics
  clock was idle and is not reported as a benchmark clock.
- JAX/JAXlib/CUDA plugin: 0.11.0.
- CUDA compiler, CRT, and NVVM: 13.0.88.
- CUDA CCCL: 13.0.85.
- cuBLAS: 13.4.1.1.
- Architecture passed to NVCC: `sm_100a`.

`summary.json` contains every repeated latency and output hash. The compressed
HLO files preserve the original and transformed modules. The CUDA files are
identical copies of the exact generated translation unit retained under both
the generator-specific and compile-helper filenames.
