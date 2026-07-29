# C1, C2, and D4 Newton–Schulz sharding report

## TL;DR

The branch contains three ordered implementation units for scanned expert
Newton–Schulz: C1 routes 4D expert stacks to MuonH, C2 keeps the expert dimension
sharded without an `(L, E) -> LE` merge while preserving the SYRK branch, and D4
returns padded non-expert stacks directly to the parameter sharding before
slicing. Four-device CPU tests match the replicated Newton–Schulz reference
exactly for both the 4D and padded paths.

The known sharding hazards are covered by jaxpr assertions. They reject an
expert-stack merge, a direct multi-axis inbound reshard, and a fully replicated
padded outbound reshard. CPU XLA did not emit the
`spmd_partitioner.cc:668` warning for the pre-C2 negative control, so a CPU
warning-count test cannot distinguish the broken and fixed layouts. The remaining
gate is a full-rack compile on the exact assembled series at
`replica_dcn=1,data=1,expert=64,model=1`, with complete-log warning counts of zero
for both `SCALE_MUON_SYRK=0` and `1`.

## Implementation

C1 is commit `2ca4bc914`. It routes scanned 4D expert matrices through MuonH,
adds the distributed 4D Newton–Schulz path and local matrix kernel, and updates
the Grug MoE optimizer mask so scanned normalization gains remain on Adam. The
`SCALE_OPTIMIZER` launcher hunk from `b0c7a1b56` was dropped as required.

C2 is commit `efb1a410a`. It ports the no-merge EP path from `75c517148`: the
expert dimension remains on `"expert"`, only the layer dimension can move to
non-expert axes, and the result returns to `orig_4d_spec` for both gate/up and
down orientations. The source lineage had no `SCALE_MUON_SYRK` branch. This
branch preserves it using the reconciliation qualified in `888fff904`: each
shard-local `(L, E_local)` tile is flattened inside `shard_map` before calling
the batched QuACK symmetric kernel. The isolated branch does not contain
`quack_symmetric_cute.py`; Phase B2 supplies that predecessor in the assembled
series.

D4 threads `_target_sharding(param)` into
`_newtonschulz_padded_stack_sharded`. The padded result is resharded directly to
that parameter layout and sliced only afterward. This is the
`target_sharding=` change from `497423bc6`; the pre-existing padding flag is not
counted as D4.

## Diff size

The functional sizes exclude tests and this report.

| Item | Estimate | Actual functional diff | Test diff |
|---|---:|---:|---:|
| C1 | +222 / -20 | +155 / -18 | +34 / -5 |
| C2 | +54 / -11 | +103 / -13 | +125 / -0 |
| D4 | +18 / -1 | +15 / -1 | +97 / -8 |

C1 is 0.70 times the estimated added lines because the extracted source comments
and probe descriptions were compressed. C2 is 1.91 times the estimated added
lines because the standalone base lacked the SYRK helper and call path that the
porting hazard requires preserving. D4's functional diff matches the estimate;
its test diff is larger because it includes an executable four-device CPU parity
test and factors the subprocess setup shared with C2.

## Verification

`uv run pytest lib/levanter/tests/test_grugmuon.py
experiments/grug/moe/test_optimizer.py` passed all 12 tests. The numerical tests
force four CPU devices in a child process. Random FP32 inputs produced
bit-identical outputs against the replicated Newton–Schulz path for the 4D
expert case and for the zero-padded non-expert case. The padded result also
returned with the requested parameter sharding.

Abstract-mesh jaxpr tests assert:

- neither gate/up nor down expert stacks create an `(L, E) -> LE` reshape;
- both orientations restore their original 4D sharding;
- a multi-axis padded inbound move uses a single-axis reshard followed by local
  extension to the axis tuple;
- the padded output has no `P(None, None, None)` reshard before slicing and
  returns in the parameter sharding.

`./infra/pre-commit.py --all-files --fix` passed. `uv run pyrefly check` reported
zero errors.

The default `uv run pytest` selection completed with 1 failure, 1,253 passes, 17
skips, 47 deselections, and 5 expected failures. The failure is
`test_grug_base_run_emits_expected_metrics_with_json_tracker`, which raises a
`ShardingTypeError` at `experiments/grug/base/model.py:227` while concatenating
two differently sharded label slices. The failing test, base model, base trainer,
dispatch code, and local Fray backend are unchanged from `origin/main`. A focused
rerun failed identically. No out-of-scope fix was made.

No GPU or cluster job was submitted. SYRK execution was not tested locally
because the kernel is Blackwell-only and its B2 module is not in this isolated
worktree. The sibling D2 qualification compiled the same local-tile SYRK
reconciliation on four GB200s and restored the expected expert shardings, but
that result does not replace the assembled full-rack gate.

## Warning gate and remaining uncertainty

A four-device CPU compile emitted no involuntary-rematerialization warning for
the pre-C2 merged negative control. Absence of the warning on CPU is therefore
not a useful assertion. The checked-in tests instead assert the sharding
structures that caused the warning on GPU.

The outstanding acceptance test needs the exact assembled C1+C2+D4 code plus B2
on a 64-device explicit mesh with
`replica_dcn=1,data=1,expert=64,model=1`. Both SYRK settings must compile and
execute finite realistic expert and padded non-expert cases, restore
`P(None,"expert","data","model")`,
`P(None,"expert","model","data")`, and `P(None,"data","model")` as appropriate,
avoid any expert merge or replicated padded outbound reshard, and contain zero
`spmd_partitioner.cc:668` warnings in the complete compiler logs.

The sibling four-GPU `data=1,expert=4` zero-warning result is mesh-scoped. At
`data=model=1`, `P(None,"data","model")` is physically replicated across expert
devices, so it does not establish the memory mechanism behind D4's measured
+1.78 percentage-point result. This branch does not claim an independent
reproduction of that gain.

## Plan discrepancies and dropped work

The plan header cites an older `origin/main`; this worktree started from the
user-specified `6ce4a7e68`. The relevant source commits and hunks were present,
and no assigned mechanism was already on that base.

The C1 estimate does not exactly match `b0c7a1b56` after the launcher is removed:
the source's optimizer and `grugmuon.py` hunks total +215 / -18. This remained
within the estimate and did not change the extraction.

No re-run variant was taken for C2. No `SCALE_OPTIMIZER` launcher wiring, smoke
script, vendored kernel, host-offload code, shape grouping, or other branch-only
feature was included.
