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
- r10 parent `/dlwh/iris-run-job-20260727-052358` ran the unchanged exact L8
  graph with memory-plan logging from `e96ac89d21`. Parent, child, and all four
  tasks succeeded; one training step completed with loss `9.04`.
- The prior failed allocations match `joined_expert_backward` temporary
  allocations after 256-byte alignment:
  - `6,039,815,424 = 6,039,815,272 + 152` bytes on stage 1/2 block 1.
  - `6,040,339,456 = 6,040,339,304 + 152` bytes on stage 0 block 1.
  - `6,543,655,424 = 6,543,655,288 + 136` bytes on stage 1/2 block 0 and
    stage 3 blocks 0/1.
- Receive pools were only `167,772,164`, `838,860,800`, `671,088,640`, and
  `335,544,320` bytes on stages 0-3. They do not match the failed requests.
- `update_grouped_components` temporary allocations were `52,506,648`,
  `138,488,344`, `138,488,344`, and `52,507,168` bytes on stages 0-3. The
  initial optimizer-update attribution was wrong.
- The memory-plan path precompiles and caches every local task before
  `eval_local`. Separating compilation from execution is sufficient for r10 to
  avoid the prior first-execution OOM despite retaining the same large backward
  executable temporaries.
- r11 parent `/dlwh/iris-run-job-20260727-053238` extended the same exact graph
  to 20 steps. Parent, child, and all four ranks succeeded after compiling all
  `68` tasks; no allocation failure, retry, pod, or workload remained.
- r11 steps 2-19 measured MFU mean/p50/p90
  `10.683126/11.088952/11.121371` and duration mean/p50/p90
  `2.971911/2.845301/3.467565s`. Mean MFU is `33.71%` below the valid
  group-size-one L8 control at `16.116235`.
- Precompile/cache therefore resolves the first-execution allocation but does
  not make this grouped explicit-routing graph a viable performance path.
  Keep the memory-plan mode opt-in and do not promote the grouped graph to L24.
- `uv run pytest tests/test_grug_moe_jaxpp_task_validation.py
  tests/test_grug_moe_explicit_stage_task_grouping.py` passes `48/48`.

## Future work

- [x] Run a 20-step L8 gate with task precompile/cache enabled and measure
      steady-state MFU.
- [x] Retain precompile with verbose memory-plan logging as an opt-in debugging
      path; the throughput gate does not justify a production mode.
