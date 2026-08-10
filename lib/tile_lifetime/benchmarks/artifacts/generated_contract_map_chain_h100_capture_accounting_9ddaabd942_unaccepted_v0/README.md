# Contract/Map capture-accounting H100 replay

## Result

TLTC-XLA-068 is an unaccepted command-buffer replay from canonical Shuttle
revision `9ddaabd9420d96eb06cb11f5d2efb7085677bb5e`. The one authorized measured
process used one batch-priority H100, four warmups, 30 counterbalanced samples,
and 1,000 iterations per variant and sample. It was not retried or tuned.

The raw medians were:

| Variant | Median (ms) |
| --- | ---: |
| Generated two-FFI-call Contract/Map chain | 0.0591248865 |
| Matched natural-JAX forward plus VJP | 0.0570275875 |

The generated-to-JAX ratio was `1.03677692x`. This is recorded as descriptive
timing evidence only. The result is rejected because it failed the capture
acceptance gate, so it is not a performance acceptance result.

Correctness passed against both the ordered CPU reference and natural JAX. The
largest observed maximum absolute error was `0.0009765625`; the largest mean
absolute error was `0.0000694394`. Three repeated output tuples had identical
hashes.

## Capture assessment

Handler counts after warmup were `(forward=4, reverse=4)` and final counts were
`(forward=6, reverse=6)`. The generated-first order added `+2/+2` during its
first timed sample. Every subsequent checkpoint, including all samples of the
natural-first order, added zero.

The policy committed before the run requires:

1. Every declared handler has a positive pre-measurement capture count.
2. Each full counterbalanced order may add at most one callback per handler.
3. Any allowed recapture occurs only in that order's first sample.
4. A callback from a variant declaring zero logical handler calls is rejected.
5. A timed callback count at least as large as a phase's logical call count is
   rejected as per-logical-call fallback.
6. Negative, discontinuous, or inconsistent checkpoints are rejected.

The first order's `+2/+2` exceeds item 2. The committed classifier therefore
reports `unbounded_recapture`, writes the full result, and exits with status 1.
The gate was not weakened or reinterpreted after observing the data.

`result.json` contains all 30 samples for both variants, every per-variant
callback checkpoint, both order summaries, numerical errors, deterministic
hashes, environment versions, and the final rejection. Before evaluating the
gate, the harness wrote the same raw result with
`capture_acceptance.status = "pending"`; assessment then replaced that field
without removing any timing or checkpoint data. This ordering is tested in the
canonical source.

## Reproducibility

The holder bootstrap used a detached Marin control worktree with an explicit
empty `[dependency-groups] dev = []` and passed frozen dependency import and
toolchain preflight before allocation/device access. The measured environment
was Torch-free and pinned:

- JAX, JAXLIB, `jax-cuda13-plugin`, and `jax-cuda13-pjrt`: `0.11.0`
- NVCC: `13.3.73`
- GPU: NVIDIA H100 80 GB HBM3
- driver: `595.71.05`
- target: `sm_90a`

The source archive SHA-256 was
`0fe90f48ae1cb5a7f3f39606f6c21ac69128835e2cd876ead88688de712bad9e`.
The source preflight produced semantic digest
`7bf1bafa3966278fdc46de0e149a2a9fff050105f7f3474fa56319c44a31380a`
and source digest
`87e470d00945d490d8acb3d28999de23eb9df216e7ca67daae132a6ec5581fe8`.
It found command-buffer traits on both generic handlers, two generated kernels,
explicit BF16 boundaries, no atomics, no opaque semantic dependency, and no
launch-status query.

## Release

The copied result SHA-256 matched the remote result before release. The holder
was then explicitly terminated. The controller reports `killed` with reason
`Terminated by user`; the exact pod is absent; a namespace pod-name search is
empty; the local session cache is absent; and no holder process remains. See
`release-proof.txt`.
