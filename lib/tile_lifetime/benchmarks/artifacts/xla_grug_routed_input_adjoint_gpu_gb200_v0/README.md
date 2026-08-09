# Routed Grug input-adjoint GB200 checkpoint

This artifact records a natural one-layer Grug training step whose JAX-owned
backward graph is intercepted at XLA's `PRE_SCHEDULER` stage. Shuttle recovers
and replaces one routed input-adjoint region:

```text
Contract
→ generated reverse scalar Map
→ Contract
→ generated deterministic source Fold
```

The custom call also returns the segmented reverse-Map buffer consumed by the
unchanged grouped weight-adjoint Contract. The generated path uses generic
cuBLAS Contracts, generated source-ordered BF16 scalar ASTs, and a single-writer
Fold. It contains no atomic accumulation or workload-specific semantic kernel.

## Result

The measurement used one NVIDIA GB200 at compute capability 10.0 with driver
595.71.05. The workload container used JAX/JAXlib 0.11.0 and CUDA 13.0.88
compiler components. `telemetry.csv` contains 263 samples; graphics and SM
clocks reached 1950 MHz, and power ranged from 152.07 W to 221.02 W under the
cluster's 1200 W limit.

Thirty paired steady-state samples alternate launch order, with 15 pairs
starting from each implementation:

| Path | Median whole-step latency |
| --- | ---: |
| Stock XLA | 0.904177 ms |
| Shuttle input adjoint | 0.837056 ms |
| Shuttle / XLA | 0.925766× |

The generated execution has one custom-call occurrence and 35 observed handler
executions. The complete 53-leaf result has maximum absolute error
`2.3283064e-10` and mean absolute error `1.3082361e-14`; 51 leaves are bitwise
equal in the direct comparison. All dynamic operands have runtime parameter
ancestry. The Fold initial value is the only static operand.

The stock path produced nine whole-step output hashes across repeated runs,
while the generated path produced one. These hashes cover the complete training
step, so they should not be interpreted as an isolated-kernel determinism test.
The generated Fold itself is deterministic by construction: one thread owns
each source-feature result and visits compact edges in destination-major order.

## Scope

This is a whole-step replacement comparison against stock XLA, not a standalone
expert-kernel oracle. The routed forward/recompute chains, grouped weight
adjoints, relation index plane, and collectives remain under XLA.

`summary.json` preserves every latency sample, launch order, output hash,
correctness statistic, operand role, and parameter-ancestry audit.
`original-gpu-pre-scheduler-hlo.txt.gz` and
`transformed-gpu-pre-scheduler-hlo.txt.gz` preserve the exact compiler boundary.
`generated_routed_input_adjoint_ffi.cu` is the generated implementation;
`generated_pair_map_handler.cu` is the identical compile-helper snapshot.
`hardware.txt`, `environment.json`, `telemetry.csv`, and `reproduction.txt`
record the hardware, software, resource request, and in-container command.
