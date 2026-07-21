# B200 NCCL fast-restart reproduction

This probe has not reproduced the GPU deadlock on this branch yet. Local CPU
runs validate the repeated-process and compilation-cache mechanics only. Do not
file or update an upstream issue from those runs.

The source observation used GB200 GPUs, JAX and jaxlib 0.10.1, CUDA 13.2,
driver 595.71.05, and NCCL with GIN. A process containing ten fusions and one
`psum` passed on a cold start, then a fresh process launched on the same
allocation hung in `ncclCommInitRank` at first execution. The source campaign
observed 22/22 fast-restart hangs and 10/10 cold-start passes. All ranks reached
`Init START`; leader threads then blocked in `ib_uverbs_event_read`.
[The source comment](https://github.com/marin-community/marin/issues/7279#issuecomment-5009950227)
and [evidence commit](https://github.com/marin-community/marin/commit/3705f4acb09df84dcc6ccb7e6e770fae97c65ecf)
contain the full prior result.

`b200_nccl_fast_restart.py` keeps the smallest source workload. The Iris task
launcher runs it as three fresh process groups in one bounded allocation,
shares the JAX compilation cache across repeats, records software and hardware
versions, and limits each repeat to 180 seconds. Natural process exit is
intentional; an explicit `jax.distributed.shutdown()` would change the teardown
path under test. The runner stops after the first failed repeat so distributed
tasks cannot mix process generations.

Run `run_b200_nccl_fast_restart.sh` inside a bounded Iris B200 job. Start with
`MARIN_REPRO_PROCESSES_PER_TASK=2`; submission commands and run identifiers
belong in private operator notes, not this repository.

Scale one dimension at a time only when every repeat passes:

| Order | Process topology | Purpose |
|---:|---|---|
| 1 | 2 local processes | Smallest local NCCL clique |
| 2 | 4 local processes | Larger local clique |
| 3 | 8 distributed processes | Smallest distributed arm |

Set `MARIN_REPRO_PROCESSES_PER_TASK` to the local process count for each arm.
Stop scaling when repeat two or three times out. A candidate reproduction
requires a successful earlier repeat, then a later repeat that reaches NCCL
`Init START` without `Init COMPLETE` or `REPRO_OK`. A compilation failure,
allocation timeout, or first-run failure is not this bug.

The expected result is `REPRO_OK` for every rank and repeat. Iris preserves the
combined task output; the runner also stores per-repeat logs under a temporary
task-local directory and prints its path in `RESULT`. Record the process
topology, GPU name, driver, CUDA, JAX, jaxlib, NCCL, per-repeat exit status, and
pass/hang count before changing the matrix. If the two-to-eight-process matrix
stays green, the remaining boundary is the larger source topology; do not infer
that the original defect is fixed.
