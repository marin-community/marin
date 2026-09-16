# CC runtime-provenance review

Reviewer: `claude-opus-5[1m]` at maximum reasoning, read-only, subscription account `plambdafour@proton.me`.

## Verdict

Correct the runtime; do not relax the parent-reproduction tolerance.

The v3 canary did its job by showing that exact losses and batches do not establish interchangeable gradient geometry after a JAX/Levanter runtime change. Because completion rows are interleaved into the old v10 trajectories, even a small systematic shift can create artificial temporal structure. The eight v3 rows must remain non-consumable and must never be rendered.

## Required correction

- Anchor recovery to the v10 execution stack that produced the existing plotted rows. The v6 and v10 numerical stacks are identical, but v10 is the direct provenance anchor.
- Pin and record the source commit, JAX/JAXLIB/libtpu versions, Levanter trainer and gradient-accumulation sources, Python patch version, task-image digest, exact TPU topology, XLA/libtpu flags, and JAX matmul precision.
- Retain the frozen `5e-6` parent-statistic reproduction tolerance.
- Run a small A/B canary before all 288 groups.
- Expand the canary beyond v3: it must cover every missing checkpoint-label class, both target-bearing shapes, the H5 source-only shape, both H5 policies, and the final zero-learning-rate path.
- Obtain another independent review of the corrected manifest and runtime before launch.

The review also requested signed drift summaries rather than only maxima. The v3 diagnostic records signed and absolute deviations; mean signed cosine drift was -0.00000185, while the largest absolute cosine drift was 0.000620. Exact losses alongside shifted gradient geometry support numerical-runtime drift rather than checkpoint or batch mismatch.
