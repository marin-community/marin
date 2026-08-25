PASS_AFTER_BLOCKERS_RESOLVED

# CC review: saved-checkpoint gradient plot completion v4

Review date: 2026-08-22. Reviewer: `claude-opus-5[1m]` at maximum effort, read-only, subscription account `plambdafour@proton.me`. No files were edited.

## Verdict

No launch blocker remains. The four prior blockers are genuinely fixed, and the split host/worker package verification behaves as specified.

## Resolved blockers

1. **Source-only final-state reproduction is non-vacuous and discriminating.** `_assert_source_only_parent_statistics` fires exactly when target IDs are empty. It compares both sources, raw and projected geometry, and all 11 components against pinned v6 parent source norms at the immutable `5e-6` tolerance. The left/right orientation is correct because each parent pair stores the requested source as its left operand, and the source norm is independent of the paired target. Stage 1 therefore makes 88 discriminating comparisons across its two final rows; this is the statistic that exposed v3 runtime drift while losses remained exact.
2. **Runtime provenance is stated honestly.** The release records a requested task-image digest and explicitly says it is not observed. It calls the pinned commit the historical library-source commit rather than claiming the hybrid tree is one commit. Each worker records the complete installed distribution inventory and its canonical hash.
3. **Every worker verifies the hybrid source tree.** It verifies the 881-file historical root/lib manifest, eight recovery/v10 implementation files, both v6 parent implementation files, and the combined implementation-manifest hash before gradient work. The frozen full train-configuration hash closes the remaining pod-config construction path.
4. **The execution-tree language is accurate.** Root lockfiles and `lib/*` come from the v10-era commit; the runner and probe kernels are a separately hash-pinned overlay. Numerical equivalence rests on source hashes, package inventory, and parent-statistic gates rather than self-reported container identity.

## Host and worker runtime split

- Host readiness, audit, authorization, and materialization do not require `libtpu`. `_configure_mechanism_runtime` defaults to static source verification only, and audit uses worker-recorded environment payloads rather than the local interpreter.
- Every TPU worker calls `_configure_mechanism_runtime(..., verify_worker_runtime=True)` before `mechanism.run_mechanism_group`. It fails before gradient calculation unless Python is `3.12.13` and JAX/JAXLIB/libtpu are exactly `0.10.1`/`0.10.1`/`0.0.41`. Missing `libtpu` resolves to `None` and fails.

## Other verified properties

- The release, coverage report, v10 release, superseded v3 release and failure marker, execution-reference manifest, implementation files, parent release/design, CC review, and locality contract are hash-checked.
- Stage 1 freezes a create-only environment baseline only after output, per-row runtime, source-only, and environment-uniformity gates pass. Later stages and final materialization must match it. Deleting it cannot bypass the gate because it can only be recreated from immutable Stage-1 result documents.
- Launch enforces stage-specific concurrency ceilings and audits every prior stage. Stage 1 spans all six missing temporal labels, all three supports, both H5 policies, both target-bearing workload shapes, and the source-only shape.
- Materialization performs an exact 9,856-row overlap A/B for newly computed common-policy source geometry against v10 at `5e-6`, including cosine-defined flags, before publishing visualization-only tables.
- Materialization requires an operator-supplied release hash and audits all 288 outputs against the Stage-1 environment baseline.
- Monkeypatching is idempotent and non-recursive; one process cannot verify two releases. The release hash and implementation-manifest hash have no circular definition.
- Completion target tables have `_visualization_only.csv` names, completion points use distinct open diamonds, final target-update cosine remains structurally undefined at zero learning rate, and the superseded v3 outputs remain non-consumable.

## Nonblocking recommendations

- Smoke-test `_steps()` in the historical worktree before launch.
- The worker could additionally recompute the release canonical hash and compare imported JAX module versions to distribution metadata; host verification and immutable parent-statistic gates already close these paths for this release.
- An explicit positive comparison-count assertion could make the final source-only gate even more regression-resistant; the frozen Stage-1 inventory already guarantees 88 comparisons.
- Keep the full installed-package inventory fail-closed across workers. Diagnose any mismatch rather than relaxing it.

The reviewed implementation is approved for freeze and staged launch. Iris launch commands must explicitly include this review file and explicitly exclude visualization-only inputs.
