# GB200 clean Map/Fold MoE replay

This artifact replays the natural StableHLO distributed BF16 MoE boundary after
replacing the handwritten SwiGLU and semantic merge arithmetic with CUDA scalar
bodies generated from Shuttle `Map` and `Fold` expressions.

The measured Shuttle path uses DeepEP only for forward payload permutation,
generic grouped GEMMs for the segmented contracts, and
`all_to_all_single` for reverse payload movement. The activation, route-weight
contribution, ordered Fold update, and final shared-output Map come from the
recovered plan's scalar ASTs. The complete MoK forward is oracle-only.

## Result

The workload is four GB200 GPUs, 2,048 local tokens, hidden size 7,168,
intermediate size 3,072, 384 global experts, and top-6 routing. The selected
candidate uses 56 DeepEP communication SMs and concatenated `[E, 2I, K]` W13.

| Capture order | Shuttle median | MoK median | Ratio |
| --- | ---: | ---: | ---: |
| Shuttle first | 4.147536 ms | 3.630992 ms | 1.142260× |
| MoK first | 4.144560 ms | 3.711088 ms | 1.116805× |
| Pooled 60 rank-maximum samples | 4.147536 ms | 3.647136 ms | 1.137204× |

The pooled means are 4.145514 ms for Shuttle and 3.659444 ms for MoK, a
1.132826× ratio. The clean generated path remains below the 1.20× completion
threshold.

Both captures use 10 warmups and 30 measured iterations. Each reported sample
is the maximum latency across the four ranks. Launch order is reversed between
captures.

## Correctness and determinism

- RelationPlan counts, rows, weights, and payload mappings match the independent
  reference on every rank.
- Relation overflow is zero.
- Repeated Shuttle outputs are bitwise identical on every rank and in both
  captures.
- Shuttle versus MoK maximum absolute error is `0.0001220703125`.
- The largest recorded mean absolute error is approximately `2.67e-6`.
- The generated scalar-program digest is
  `3048c6b922de317e556ff4e1a6fe9c81a22bfc9ba4d6582d0245fbf275f81fba`.
  The recovered plan, generated include, and loaded extension agree on this
  digest.

## Source-lineage audit

| Dependency | Classification | Accepted-path use |
| --- | --- | --- |
| CUDA/WGMMA/barriers | hardware/runtime primitive | matrix work and synchronization |
| MoK-derived grouped GEMM core | generic compute primitive | segmented W13 and W2 only |
| DeepEP dispatch | generic communication primitive | forward payload permutation only |
| `all_to_all_single` | generic communication primitive | reverse payload permutation only |
| generated pair Map | generated Shuttle kernel | activation over separate or concatenated pairs |
| generated indexed ordered Fold | generated Shuttle kernel | route-weight contribution and source-order merge |
| generated partitioned Fold/base Map | generated Shuttle kernel | deterministic cross-rank merge and shared add |
| complete MoK forward | expert/oracle-only | comparison path only |
| DeepEP semantic combine | legacy control only | excluded from accepted Shuttle path |

The generated-code audit in each JSON records an empty
`accepted_path_external_semantic_kernels` list. The generic loop and indexing
skeletons call generated scalar device functions; changing the Map or Fold AST
changes the generated include and digest without editing CUDA loop bodies.

The natural runtime path still uses a small generic Torch executor for the
recovered router Contract, top-k selection, and normalized weights. Shapes and
Relation semantics come from the StableHLO plan, but this adapter is not a
generated router kernel. This is a remaining frontend/runtime-lineage cleanup,
not an opaque MoE kernel or an excluded benchmark boundary.

## Revisions and environment

```text
Shuttle:            31f600f22837ec4a3c4c3eaf07c2e4a9a5ddc268
MoK:                3e1cf43ab93ad040afed52a45ab03cb490ffe4be
ThunderKittens:     1c3920d993404dd49a6d4c7267ea11d583bd5c68
DeepEP:             7febc6e25660af0f54d95dd781ecdcd62265ecca
CUDA / NVCC:        13.0.88
NVIDIA driver:      595.71.05
PyTorch:            2.10.0+cu130
GPU:                4× NVIDIA GB200
clock/power policy: cluster default, unpinned / 1200 W limit
```

One batch-priority pod was preempted before the smoke reached Shuttle kernel
execution. The complete environment and extensions were rebuilt on the
replacement pod; both final captures ran there without another preemption.

## Contents

- `raw/`: both counterbalanced JSON distributions and stdout, the smoke,
  toolchain and telemetry records, the MoK build log, and build provenance;
- `fixtures/`: per-rank semantic correctness fixtures for the smoke and both
  final captures;
- `source/`: exact generated and generic source files used by the replay;
- `SHA256SUMS`: content hashes for every artifact file other than itself.

The platform-specific compiled extension is intentionally not checked in. Its
SHA-256 is
`1e08e83fbc51ab681e589af6bf173f02823eae9add710779c30a59b1c3d699f4`,
and `raw/probe-build.json` records the build.
