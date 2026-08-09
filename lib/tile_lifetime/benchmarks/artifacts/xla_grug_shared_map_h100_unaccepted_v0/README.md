# Shared-map Grug H100 replay: unaccepted

This artifact records an H100 replay of the seven-call natural Grug training
replacement at Shuttle revision `9798ebd794e9a20a5a3f0f4b493400bb77e748ae`.
The run is not an accepted result because the generated output was not bitwise
deterministic.

## Configuration

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0
- Driver: 595.71.05
- Power limit: 700 W
- JAX and JAXlib: 0.11.0
- Triton: 3.6.0
- CUDA compiler: 13.2.78
- Composition: `shared_map_xla_remainder`
- Warmups: 4
- Timed repeats: 30, counterbalanced by execution order

## Result

Each of the seven selected custom-call targets occurs once in the transformed
HLO. Each handler executed 35 times: one correctness call, four warmups, and 30
timed calls.

The generated result passed the ordered-floating-point comparison against XLA:

- result leaves: 53
- bitwise-equal leaves: 38
- maximum absolute error: `9.760260581970215e-7`
- mean absolute error: `7.976652347623191e-11`

The XLA output produced one hash in all 30 runs. The generated output produced
one hash in 27 runs and a second hash in 3 runs. The accepted-path determinism
guard rejected the run.

Median latency was `0.528433 ms` for XLA and `0.603667 ms` for the generated
seven-call path, a ratio of `1.142374x`. These timings are diagnostic only because
the run failed determinism.

## Contents

- `unaccepted-execution-result.json`: raw paired samples, hashes, handler counts,
  target counts, and numerical comparison
- `latency-summary.json`: distribution summary
- original and transformed pre-scheduler HLO
- source attention VJP StableHLO
- generated CUDA for all seven calls
- `SHA256SUMS`: content hashes

The failed run did not retain per-leaf hashes. The benchmark now records a hash
for every output leaf so the next bounded replay can identify the varying state
or metric before any component-specific rerun.
