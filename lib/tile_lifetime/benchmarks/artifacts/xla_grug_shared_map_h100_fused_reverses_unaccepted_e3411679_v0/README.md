# Unaccepted thirteen-call H100 Grug replay

This artifact records the only physical replay of the natural one-layer Grug
train step with the `shared_map_fused_reverses` composition. The measured
Shuttle revision is `e34116793d63f8d1a2c61c21c02b1ff37d44f0d6`.

The generated path owns thirteen generic regions:

- the routed forward Contract/Map/Contract/Fold region;
- two input-adjoint Contracts, one generated multi-Map, and the source Fold;
- two group-batched weight-gradient Contracts;
- the fused weighted RelationProgram reverse;
- compact normalized-exp Contract/Map/Fold forward and reverse regions;
- the streaming attention reverse; and
- two row-axis Folds.

The transformed module contains exactly one occurrence of every selected
custom-call target. Every handler executed exactly 35 times: one correctness
execution, four warmups, and 30 measurements. The normalized-exp reverse reads
the generated forward saved-state output, and the audit records the old
forward and reverse arithmetic as dead. Placement all-reduces remain outside
the Shuttle regions.

All semantic and execution guards pass. The 53-leaf train-step output has
maximum absolute error `9.760260582e-7`, mean absolute error
`7.977541458e-11`, and 38 bitwise-equal leaves. The generated and stock paths
each produce one stable whole-tree hash across all samples. Generated source
and linked-library audits find no Torch or Triton runtime dependency, and the
generated reductions use no atomic accumulation.

The result is not performance-accepted. Thirty counterbalanced samples measure:

| Path | Median | Min | Max | Standard deviation |
| --- | ---: | ---: | ---: | ---: |
| Shuttle | 0.731302 ms | 0.693048 ms | 0.765498 ms | 0.011501 ms |
| stock JAX/XLA | 0.591416 ms | 0.583925 ms | 0.600955 ms | 0.003970 ms |

The ratio is `1.236525x`, above the `1.20x` proof threshold. The absolute gap
is `0.139885 ms`. Dividing that gap by thirteen calls gives `10.760 us` per
call, but this is scale context only: it is not a causal attribution because
the transformed executable also changes the kernels around those boundaries.

The run used one NVIDIA H100 80GB HBM3, compute capability 9.0, driver
595.71.05, a 700 W power limit, one requested CPU, 32 GB host memory, 50 GB
ephemeral disk, and batch priority. Clocks were not pinned. JAX, JAXLIB, the
CUDA 13 plugin, and PJRT were all 0.11.0. The holder used current Iris revision
`eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb`; the measured checkout remained
the exact Shuttle revision above.

The benchmark command in `invocation.txt` was issued once with no schedule
tuning or benchmark retry. Earlier holder-submission attempts failed locally
before Iris created a job because of bundle size and stale client protocol;
they consumed no GPU time and did not invoke the benchmark. The accepted
holder was released immediately after artifact copy. `allocation-release.txt`
records the inactive job and absent pod.

`summary.json.gz` and `execution-evidence.json.gz` preserve all 30 timing
pairs, alternating order, whole-tree and per-leaf hashes, correctness metrics,
target occurrences, handler counts, numerical policies, and liveness audits.
The original/transformed HLO and all thirteen generated semantic sources are
also retained. Shared objects, cubins, build caches, and duplicate generated
source copies are excluded.

