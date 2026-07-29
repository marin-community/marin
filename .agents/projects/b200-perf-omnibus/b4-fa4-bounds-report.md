# B4 FA4 segment-bound hoist

## TL;DR

B4 now sits on the A0 scan-layer prerequisite at `5f0efe967`. The homogeneous
layer scan computes the full-causal and sliding-window FA4 bounds once, carries
the two bounds arrays and their validity mask as loop-invariant operands, and
selects the bounds for each layer without a conditional around the rematerialized
block.

The functional diff is +159/-30 against A0, compared with the +156/-30 estimate.
Focused numerical, JAXPR, and StableHLO checks pass. This worktree has no CUDA
JAX runtime, so the real CuTe kernel and the 8-rack device-placement claim were
not tested locally.

## Implementation

`AttentionMask` can carry precomputed FA4 lower bounds and token validity.
`fa4_cute_segment_bounds` exposes the existing packed or simple causal bound
calculation, and the FA4 frontend uses attached bounds instead of recomputing
them. Metadata follows the query batch sharding at the kernel boundary.

The scanned model computes the long- and short-window metadata before
`jax.lax.scan`. The scan body receives three `[B, S]` loop-invariant operands,
uses `jnp.where` to select the current layer's lower bounds, and passes the result
through `AttentionMask.with_fa4_bounds`. Long-layer RoPE disabling uses the same
traced layer selector, so it does not reintroduce the removed `jax.lax.cond`.

The source patch unconditionally applied `reshard(..., P(None, None))` in the
bounds helpers. JAX rejects a `PartitionSpec` when no mesh exists, breaking four
existing CPU tests. `_replicate_metadata` now applies replicated sharding only
under a mesh and otherwise leaves the array unchanged.

## Diff size

The four functional files are +159/-30 against `5f0efe967`:

- `experiments/grug/moe/model.py`: +52/-16
- `attention/__init__.py`: +1/-0
- `attention/_core.py`: +16/-0
- `attention/_fa4_cute.py`: +90/-14

The estimate was +156/-30. The three additional insertions are the net result of
the no-mesh sharding fix and compressing stale source comments. Tests add
+81/-1. The report is excluded from both counts.

## Verification

The B4-focused selection passes three tests:

- scanned and unscanned five-layer model outputs and router metrics match at
  `rtol=atol=1e-5` with long-layer RoPE disabled;
- direct and precomputed packed-window FA4 metadata produce matching attention
  outputs at `rtol=atol=1e-6` through a CPU reference at the CuTe boundary;
- the layer-scan JAXPR has loop-invariant long bounds, short bounds, and validity
  operands, selects the bounds before the rematerialized block, and has no
  conditional at that boundary. Its lowered StableHLO has one layer
  `stablehlo.while` and no `stablehlo.case`.

The full touched test files produced 21 passes and 6 expected GPU skips before
failing one unrelated base-Grug test. The default repository selection completed
with 1,256 passes, 17 skips, 47 deselections, 5 expected failures, and the same
single failure:
`test_grug_base_run_emits_expected_metrics_with_json_tracker`. The failure is an
explicit-sharding mismatch in `experiments/grug/base/model.py:227`, which B4 does
not change. The test fails identically on the A0 worktree at `5f0efe967`.

The default environment omits three dependencies needed by its own configured
test run. The completed run used `uv run --with pytest-timeout --with sympy
--with pylatexenc pytest` without overriding the repository marker expression.
`uv run --with 'pyrefly>=1.0.0,<1.1.0' pyrefly check --baseline
.pyrefly-baseline.json` reports `0 errors`.
`./infra/pre-commit.py --all-files --fix` passes.

No GPU kernel compile, SPMD warning census, device-placement inspection, or
multi-rack run was performed. The structural test lowers a CPU graph with the
unavailable CuTe boundary replaced by a metadata-consuming function. It verifies
the hoist in the model graph but does not establish the source commit's 8-rack
operational claim.

## Excluded work

The CUTLASS 4.6 `make_fragment` migration is not on
`origin/main@6ce4a7e68`: `_fa4_cute_segmented_bwd.py` still has five top-level
calls. It is needed separately and is owned by
`agent/deri-fa4-slot@19463a5c3`, which replaces those five calls with
`make_rmem_tensor`. B4 does not touch that file or duplicate the migration.

The `flash-attn-4` requirement and lock remain at 4.0.0b16. No wheel bump,
`subtile_factor` rename, dependency change, smoke script, vendored file, debug
print, or cluster job was added.

## Plan mismatches and uncertainty

The earlier blocker was real: `a33e16ced` depended on the scan path from
`97b53fe0e`, which was absent from the original sequence. A0 at `5f0efe967`
resolves that mismatch, and all four B4 source hunks apply to their intended
symbols on top of it.

The source patch and commit message describe the FA4 boundary as a
`jax.pure_callback`, but neither the source parent nor current main uses one.
The backend calls the CuTe launcher through the CUTLASS JAX custom-call path.
The implementation comments now describe conditional kernel metadata rather
than a callback.

The source did not account for bounds-helper calls outside a mesh. The
`_replicate_metadata` adjustment is the only functional deviation from the four
source hunks. The device-0 placement mechanism and rack-scale wedge are inherited
from the evidence record; they were not reproduced in this worktree.
