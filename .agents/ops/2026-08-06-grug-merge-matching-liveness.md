# Grug merge matching: long cost-matrix phase

Determine whether the d512 expert-matching retry was computing, blocked, or
deadlocked after its uninstrumented functional-cost phase ran for several
hours.

## Initial status

The retry controller
`/dlwh/grug-xem-merge-spectral-prefit-r2-20260806` and matching child remained
`RUNNING`. Matching attempt 0 was preempted after a worker reconcile failure;
attempt 1 resumed on a `us-central1` v5p-8 worker. All 256 spectral probe sets
completed by 2026-08-06 13:28:22 PT. The original `functional_cost_matrix`
implementation did not log row-level progress.

## Hypothesis 1

The task is computing the 256 cost rows. The matching implementation
accumulates every row in memory and writes the probe, cost, assignment, and
manifest objects only after all rows finish, so lack of artifact growth is
expected during this phase.

## Results

Read-only checks at approximately 18:20 PT found:

- `iris.task` heartbeats every approximately 6.6 seconds, with CPU use varying
  from 490 to 1,786 millicores, RSS approximately 16.3 GiB, peak RSS 19.6 GiB,
  and disk use 3.367 GiB.
- TPU utilization columns were unavailable for this worker.
- The container's Python PID 1 was in `R (running)` state with 1,639 threads and
  approximately 16.8 GiB RSS.
- The exact artifact prefix
  `gs://marin-us-central1/grug/expert_merge/d512/matching-layers-2-3/2026.08.06`
  contained only executor metadata. Its `.executor_status.lock` timestamp
  advanced from 18:18:18 to 18:18:49 PT, confirming a fresh executor heartbeat.
- No traceback, OOM, bad-node signal, new preemption, matching manifest, or
  prefit child was present.
- An on-demand py-spy capture could not attach to the container process, so the
  diagnosis relies on independent task telemetry, procfs state, and the
  executor heartbeat.

The task was computing rather than deadlocked. Restarting it while these
signals remained fresh would have discarded useful work.

## Hypothesis 2

The long phase is repeatedly compiling shape-varying XLA programs rather than
evaluating the cost matrix.

## Results

A second read-only audit at approximately 00:47 PT found no `XLA`, compilation,
cache-miss, executable, or slow-operation messages in the complete retained
worker log. The last application log remained the completion of probe set 255
at 20:28:22 UTC. Finelog telemetry over the two latest five-minute windows
remained fresh: CPU averaged 853 millicores in the first window and 732
millicores in the second, RSS stayed between 16.37 and 16.55 GiB, and TPU
utilization remained unavailable.

The implementation made slow cost evaluation more plausible than repeated
compilation. `functional_cost_matrix` evaluated 256 source rows without an
enclosing `jax.jit`. Each row performed source and candidate evaluations for
ordinary and spectral inputs; the candidate bank was evaluated in 16 expert
chunks. This produced approximately 26,000 major einsum dispatches across the
matrix, followed by a device-to-host transfer for each row. The tensor shapes
were stable across rows, so compilation-cache reuse should have dominated after
the first row. The observed runtime corresponded to approximately 1.5 seconds
per major dispatch, which is plausible for sequential TPU launch and execution
overhead.

## Outcome

Attempt 1 remained live for nearly 23 hours before its worker was preempted at
2026-08-07 12:51 PT. It had not committed a matching payload. Iris started
building attempt 2, but the controller was stopped at 12:53 PT to avoid
repeating the obsolete matcher.

Commit `b02efb4698` batches and JIT-compiles the expert cost calculation. Retry3
submitted its matching child at 12:56:17 PT. The first row-progress line arrived
at 12:59:01 PT, 2 minutes 43 seconds after submission, and row 256 logged one
second later. The task succeeded in 3 minutes 36.57 seconds with zero failures
and zero preemptions. No compile, HBM, OOM, or traceback signal appeared.

The committed artifact contains 263 objects totaling 206.05 MiB, including
`.artifact.json`, `cost_matrix.npz`, `assignments.json`,
`matching_manifest.json`, `matching_metrics.json`, and 256 per-expert probe
sets. The spectral prefit child launched at 13:00:10 PT.

The original liveness diagnosis was correct: the worker was active, and the
cost-matrix implementation was dominated by sequential dispatch overhead.
Batching reduced matching from more than 22 hours without completion to less
than four minutes including startup and artifact commit.

## Future work

- [x] Add per-row progress logs inside `functional_cost_matrix`.
- [ ] Record TPU utilization for GCE/TPU task telemetry when available.
