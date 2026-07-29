# A1 drop-metric extraction report

## Result

`SCALE_REPORT_DROPS=1` now travels from `launch_cw_scale.py` into
`GrugModelConfig.report_capacity_overflow` and controls whether `MoEMLP` requests
the backend's dropped-assignment count. The disabled path returns the existing
`_zero_dropped_assignments()` scalar, which keeps the router-metric structure
stable without duplicating the helper.

The chunked `sonic_cute` fix is a separate commit on
`agent/impl-b12-sonic`, after B1 and B2. It adds `_chunk_capacity_drops` to the
shared MoE helpers and changes only
`_moe_mlp_local_sonic_cute_chunked` to return that count. The unchunked and
intermediate-dimension chunked paths remain structural zero because they have
no fixed assignment capacity.

B2 must not land without the follow-up fix. B2 introduces an expert-dimension
chunker that can drop assignments while returning zero. A1 makes drop reporting
the basis for fidelity comparisons, so leaving B2 at `74bcdcf40` would put a
silent false-clean result behind a flag that derisking runs rely on.

The child-job priority change from `fc5532108` is extracted as a separate
launcher commit in this worktree. Its `docs/debug-log-d1-child-priority.md`
artifact is excluded because `AGENTS.md` forbids `docs/debug-log-*`.

## Diff size

The A1 code change is +11/-2 lines, plus +37/-1 lines of regression coverage.
The B2 follow-up is +12/-2 functional lines and +40/-1 test lines. Together the
split drop-metric implementation is +23/-4 functional lines, 0.77 times the
sequence estimate of approximately 30 added lines.

The source estimate assumed that more of the metric path was missing. At
`origin/main` `6ce4a7e68`, the per-layer counter, router summaries, tracker
keys, ring/ragged accounting, and `_zero_dropped_assignments` already exist.
Only the launcher/config gate and B2's chunk-capacity count remained.

The priority extraction is +18 lines in `experiments/grug/dispatch.py`. It is
not included in A1's size because it is an independent launcher commit.

## Backend accounting

- `ring`: returns a real clipped-assignment count. Both the public MoE test and
  the launcher-to-model regression propagated a positive count on eight virtual
  CPU devices. The latter produced 12 drops for an overloaded eight-expert
  case.
- `ragged_all_to_all`: the implementation computes a real receiver-clipping
  count. The positive-count test reaches `jax.lax.ragged_all_to_all`, then
  XLA:CPU fails with `UNIMPLEMENTED: HLO opcode ragged-all-to-all`. Its returned
  count was not exercised.
- `deepep`: returns a structural zero. The receiver capacity is sized for all
  assignments and this implementation has no clipping counter. DeepEP cannot be
  exercised in the CPU environment.
- `scatter`: returns a structural zero. Its grouped local dispatch processes all
  assignments, so zero is correct.
- `sonic`: returns a structural zero. Its grouped local dispatch has no capacity,
  so zero is correct. The Triton path cannot be exercised in the CPU
  environment.
- `sonic_cute` unchunked: returns a structural zero and has no assignment
  capacity. It requires the SM100 QuACK stack and was not executed.
- `sonic_cute` intermediate-dimension chunked: returns a structural zero and
  partitions compute without capping assignments. It requires the SM100 QuACK
  stack and was not executed.
- `sonic_cute` expert-dimension chunked: now returns a real static-capacity
  overflow count. Four CPU cases matched an independent oracle that walks the
  sorted rows each chunk processes, including three overloaded layouts. The
  QuACK backend itself was not executed.

The accelerator-side follow-up list is therefore
`ragged_all_to_all`, `deepep`, raw `sonic`, and all `sonic_cute` variants. The
load-bearing checks are a positive overloaded count for ragged all-to-all and
the expert-dimension `sonic_cute` path. The other entries should confirm that
their structural zero still corresponds to a dropless schedule.

## Verification

The launcher-to-model regression was run with eight virtual CPU devices. Before
the A1 code change it failed because both reporting modes returned 12; after the
change, reporting disabled returned zero and `SCALE_REPORT_DROPS=1` returned a
positive count.

The existing ring overload test passed on the same eight-device CPU mesh. The
equivalent ragged-all-to-all test failed only at the unsupported XLA:CPU HLO,
as described above. The B2 helper tests passed all four balanced and unbalanced
layouts.

The priority helper returned Iris's unspecified band when unset, mapped
`production`, `interactive`, and `batch` to their Iris priority bands, and
rejected `urgent`. No Fray or Iris job was submitted.

`./infra/pre-commit.py --all-files --fix` passed. The repository's pinned
Pyrefly invocation reported 0 errors. The literal `uv run pyrefly` command
cannot start because this workspace does not install a root `pyrefly`
executable; `infra/pre-commit.py` runs
`uvx --from 'pyrefly>=1.0.0,<1.1.0' pyrefly check` instead.

The default pytest marker selection completed with 1,252 passed, 18 skipped,
47 deselected, 5 xfailed, and one unrelated failure:
`test_grug_base_run_emits_expected_metrics_with_json_tracker`. The failure is
an explicit-sharding mismatch in `experiments/grug/base/model.py`; that model
and its training code are unchanged from `origin/main`, and the same failure
was already reproduced by the B1/B2 extraction. The root environment also
lacks its declared math and timeout test dependencies, so the complete run used
the locked `marin-core` test and math groups.

No GPU, cluster job, QuACK kernel, raw Sonic kernel, DeepEP transport, or
ragged-all-to-all collective was executed.

## Extraction decisions and uncertainties

The older `model.py` and `train.py` plumbing from `4fbc89152` and `2d4a87395`
was dropped because `origin/main` already returns per-layer counts and logs the
router overflow rates. Copying it would replace the current metric API with an
older parallel path.

The one-line chunked return from `cefc6d47b` was applied only after B2 created
`sonic_cute.py`. B1/B2 were not imported into A1. The source's longer helper
docstring was shortened, while preserving the same count:
`sum(max(chunk_load - chunk_capacity, 0))`.

The first mechanical patch to `sonic_cute.py` matched the duplicated unchunked
return instead of the expert-dimension chunker. Reading the enclosing symbols
caught it immediately; the unchunked return was restored before testing or
committing.

The plan's approximately 30-line estimate is accurate only across the resolved
two-branch split. A1 alone is smaller because the current `origin/main` has
more complete reporting infrastructure than the plan's earlier base
description implied.
