# A4 dispatcher environment forwarding

## Summary

The Grug dispatcher now forwards every `XLA_*` variable to accelerator tasks.
This includes `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`; the prior `XLA_FLAGS`
prefix did not match it. The existing `JAX_PLATFORMS` exclusion is unchanged.

## Scope and diff size

The functional diff is +1/−1 in `experiments/grug/dispatch.py`. The regression
test adds 14 lines in `tests/test_grug_dispatch.py`. `sequence.md` does not give
A4 a numeric estimate in the Phase A table, but its prescribed implementation is
the same one-line `XLA_FLAGS` to `XLA_` widening. The functional diff matches
that scope.

I did not take the research branches' `CE_` or `SCALE_` additions. No `CE_*`
environment reader exists on this main snapshot. The current `SCALE_*` variables
are read by `launch_cw_scale.py` while it builds the serialized launch
configuration, before `dispatch_grug_training_run` creates the accelerator task.
Worker-side `SCALE_*` readers belong to the unlanded research changes, so
forwarding that namespace in A4 would create a broader contract than the current
code needs.

I also left `experiments/june_tpu_67b_a2b/dispatch.py` unchanged. A4 is scoped to
the Grug dispatcher, and the allocator defect concerns its accelerator tasks.

## Repository audit

Before adding the test, a repository-wide search found no
`XLA_PYTHON_CLIENT_ALLOCATOR` occurrence in code, YAML, TOML, shell scripts, or
documentation. The Grug dispatcher used
`("XLA_FLAGS", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_")`, confirming that the
allocator was neither set elsewhere nor matched for forwarding.

The existing repository uses `XLA_FLAGS` and documents
`XLA_PYTHON_CLIENT_MEM_FRACTION`. Both are accelerator runtime controls, and the
broader prefix did not reveal an `XLA_*` value that would be harmful to forward
from the CPU dispatcher. `JAX_PLATFORMS` remains excluded explicitly and is
absent from the forwarded result in the regression test.

Both `agent/deri-d67` and `agent/deri-d2-build` carry the broader
`("XLA_", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_", "CE_", "SCALE_")` tuple.
Commit `75c517148` on the `moe-standalone-ep` line still carries the narrow
tuple. Its 32-way rematerialization measurements may therefore have used the
default allocator even when the submit command requested `cuda_async`. The code
record cannot establish the allocator used by those completed jobs, so those
numbers should not serve as controls until their runtime environments are
verified.

## Verification

`tests/test_grug_dispatch.py` calls `_forwarded_env_vars` with
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` and dispatcher-only
`JAX_PLATFORMS=cpu`. Before the implementation change, the test failed with a
missing allocator key. After the change, it passed and confirmed that
`JAX_PLATFORMS` was absent.

`./infra/pre-commit.py --all-files --fix` passed, including its configured
Pyrefly check. The documented standalone `uv run pyrefly` command exits 2 after
printing the current Pyrefly CLI help because this snapshot requires a
subcommand; `uv run pyrefly check` passed with 0 errors.

The default `uv run pytest` selection completed 1,274 selected tests with 1,253
passed, 17 skipped, 5 xfailed, and one failure. The failure is
`test_grug_base_run_emits_expected_metrics_with_json_tracker`, at the untouched
`experiments/grug/base/model.py:227`: JAX rejects concatenating explicit
`P(("replica_dcn", "data"), None)` and `P(None, None)` shardings. The same test
fails when run alone. No forwarded-prefix environment variables are set in this
process, and `_forwarded_env_vars()` returns an empty dictionary, so A4 does not
change that test's dispatched environment. The targeted A4 regression test
passes.

No GPU or cluster job was submitted. This forwarding behavior is observable in
the dispatcher process and does not require accelerator execution; allocator
behavior at 64×GB200 was not remeasured in this slice.

The plan cites an older `origin/main` snapshot than this worktree
(`1c631c4c0` in `sequence.md` versus `6ce4a7e68` here). The A4 premise and source
shape still match at `6ce4a7e68`. The only unresolved evidence question is the
allocator used by historical `moe-standalone-ep` jobs. The repository guidance's
bare Pyrefly command is also stale for the installed CLI, and the default test
selection is not green at this main snapshot because of the explicit-sharding
failure above.
