# StarCoder WSD80 gradient plot completion v4 follow-up

The first v4 review returned `BLOCK` with four blockers. The corrected implementation now:

1. Compares raw and projected source-gradient norms for every target-less final row against both pinned v6 parent source documents at the immutable `5e-6` tolerance. This gives the two final Stage-1 rows a discriminating numerical-runtime gate.
2. Renames container and commit metadata so requested inputs are not presented as observations. Every worker records the complete installed distribution inventory and its canonical hash; required JAX/JAXLIB/libtpu versions are checked before gradient work. Stage 1 freezes a create-only environment baseline, and every later stage and final materialization must match it.
3. Verifies both `implementation_files` and `parent_implementation_files` on each worker, in addition to the 881-file historical root/lib manifest. The release records the canonical combined implementation-manifest hash.
4. Describes the execution tree accurately as historical root lockfiles and lib sources plus a separately hash-pinned recovery/probe overlay. No claim remains that all files come from one commit.

Additional review recommendations adopted:

- Common-policy completion source geometry is compared against the overlapping v10 rows during materialization, with the comparison inventory and maximum differences recorded.
- Materialization requires an operator-supplied release hash.
- Target-side merged files use `_visualization_only.csv` names.
- Completion source points use a distinct diamond marker.
- The superseded v3 failure marker is hash-pinned and states that v3 cannot be regenerated from the in-place v4 scripts.
- Device-kind validation is non-vacuous and exact for the observed JAX label `TPU v5`.
- Package resolution is validated before expensive computation.
- Static source and release verification runs in host readiness, authorization, audit, and materialization. Exact
  Python/JAX/JAXLIB/libtpu checks run only inside `run_completion_group`, because macOS orchestration correctly has
  no `libtpu` installation. Every worker still fails before gradient calculation if the TPU numerical stack drifts;
  the worker records the complete package inventory for the staged audit.

The v10 mechanism code remains hash-pinned and unmodified. Stages 2-4 remain blocked by exact prior-stage audits.
