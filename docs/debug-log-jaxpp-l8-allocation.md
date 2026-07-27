# Debugging log for JaxPP L8 first-execution allocation

Identify the named JaxPP task or receive-buffer category that requests
6,039,815,424 bytes on stage 1, 6,040,339,456 bytes on stage 2, and
6,543,655,424 bytes on stage 3 for the matched L8 explicit-MPMD group-size-two
run.

## Initial status

r7 and r8 compile 68 tasks, then fail during first execution. Disabling
`JAXPP_REUSE_RECV_BUFFERS` leaves the requested byte counts unchanged.
Whole-state donation fails earlier because a reused `float32[2,64]` QB state
leaf is deleted.

## Hypothesis 1

The requests belong either to a JaxPP receive destination or to the XLA
temporary allocation for the final stage optimizer task. The r7 logs alone do
not distinguish those categories because the fatal stack is reached while
DIME materializes a transfer buffer.

## Changes to make

Add an opt-in `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN` diagnostic before
`eval_local`. It statically reports the per-device receive-pool and
per-transfer destination bytes, including adjacent producer and consumer task
names. It also precompiles each local `task_p` without executing it and reports
JAX `CompiledMemoryStats` under the task name.

## Results

- r7 rank 0 compiled `grug_1f1b_stage0_update_grouped_components` at
  `04:18:36`, completed it, and compiled `grug_1f1b_keep_step` at `04:18:43`.
  Ranks 1-3 each compiled
  `grug_1f1b_stage{1,2,3}_update_grouped_components` at `04:18:36`; those are
  the last task executables on those ranks before their allocator failures at
  `04:18:54-04:18:55`.
- The stage-3 request exceeds stage 2 by `503,315,968` bytes. This is
  `24 * (8192 * 2560) - 512`, tying the delta to the stage-3 vocabulary-head
  parameter count inside the grouped optimizer update. A pipeline activation
  receive does not contain that parameter-shaped payload.
- The fatal DIME stack is secondary evidence only: task execution is
  asynchronous, and the main thread reaches DLPack materialization while the
  preceding device executable reports its allocator failure.
- The current evidence points first to
  `grug_1f1b_stage{1,2,3}_update_grouped_components`, but compile order alone
  does not identify the failing executable. It also does not prove whether the
  single BFC request is an XLA temporary allocation or a concurrently
  materialized receive pool. The new diagnostic reports both categories before
  `eval_local` and will settle that distinction by exact byte equality.
- `uv run pytest tests/test_grug_moe_jaxpp_task_validation.py
  tests/test_grug_moe_explicit_stage_task_grouping.py` passes `48/48`.
- No cluster run was launched.

## Future work

- [ ] Run one compile-only or smallest L8 gate with the diagnostic enabled.
- [ ] Match the failed byte count against exactly one receive or task-temp
      record before changing memory behavior.
