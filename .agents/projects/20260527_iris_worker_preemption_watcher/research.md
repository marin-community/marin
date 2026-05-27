# Research: Iris worker preemption watcher

Notes gathered for `design.md`. Issue: https://github.com/marin-community/marin/issues/5872. Predecessor issues: #5753 (race), #5747 (RCA), #5754 (whack-a-mole pattern-list workaround that motivated this principled fix).

## Why the current behavior is wrong

When a GCP TPU VM is preempted, JAX distributed loses one peer and the rest of the slice exits with a mix of SIGABRT / RPC-peer-loss / non-zero codes. The sibling workers classify these in `lib/iris/src/iris/cluster/worker/task_attempt.py:892-931`:

- `L909-925` promotes recognized TPU bad-node signatures to `TASK_STATE_WORKER_FAILED`.
- `L927-931` default-routes everything else to `TASK_STATE_FAILED`.

`TASK_STATE_FAILED` charges `max_retries_failure`, which is 0 for training jobs (PR #4615), so one preempt burns the entire job. Stderr-pattern-matching the JAX failure modes (proposed in #5754) was rejected as fragile.

## Worker architecture relevant to the fix

- **Entry:** `lib/iris/src/iris/cluster/worker/main.py:48-82` → `Worker.start()` → spawns the uvicorn server then the lifecycle thread.
- **Threads spawned at `worker.py:308-311`:** `worker-lifecycle`, `profile-loop`. New `preempt-watcher` thread fits here.
- **Reconcile handler:** `worker.py:856` (`handle_reconcile`). This is the only worker→controller payload path that carries task state. Observation set is built inside this method; omitting attempts here is how the worker "lies by silence" while draining.
- **Task lifecycle:** `task_attempt.py:419-446` (`kill()` does SIGTERM then SIGKILL with `term_timeout_ms` grace). Worker already owns task termination — we just need to invoke it on preempt.

The worker is RPC-driven; there's no existing per-worker tick loop other than `profile-loop`. The preempt watcher is the second non-RPC-driven background thread.

## Controller architecture relevant to the fix

- **Apply reconcile:** `lib/iris/src/iris/cluster/controller/transitions.py:1986-2054` (`apply_reconcile_result`). Reads `result.observations`, filters to the plan, converts to `TaskUpdate`s via `_observations_to_updates` (`L2096-2168`), applies via `_apply_task_transitions`.
- **PREEMPTED handling:** dedicated handler around `transitions.py:2440-2548`. Already has the requeue-coscheduled-siblings cascade at `L2527-2537` — but that fires *after* the per-task transition machinery has already run, which means sibling FAILEDs from racing exits land first. The atomic-slice rule needs to run *before* `_observations_to_updates` translates sibling exits.
- **Coscheduled helpers:** `_find_coscheduled_siblings` (`transitions.py:733`), `_terminate_coscheduled_siblings` (`L750`, moves siblings to `TASK_STATE_COSCHED_FAILED`), `_requeue_coscheduled_siblings` (`L792`, bounces siblings to PENDING). The requeue helper is exactly what we want post-preempt.
- **Worker health:** `lib/iris/src/iris/cluster/controller/worker_health.py:50-130` (`WorkerHealthTracker`). Has `register`, `heartbeat`, `ping`, `mark_unhealthy`, `forget`. Needs a new `mark_draining(worker_id)` (or equivalent) so the scheduler reaper can fast-path removal.

## Proto surface

- **TaskState enum:** `lib/iris/src/iris/rpc/job.proto:195-218`. Already has `TASK_STATE_PREEMPTED=10`, `TASK_STATE_WORKER_FAILED=7`, `TASK_STATE_COSCHED_FAILED=11`. No new TaskState needed — the worker should report PREEMPTED, not invent a new state.
- **WorkerHealth message:** `lib/iris/src/iris/rpc/worker.proto:146-150`. Add `bool draining = 4;`.
- **StopReason enum:** `worker.proto:136-144` already has `STOP_REASON_WORKER_DRAIN=6` (controller→worker stop intent for graceful drain). This is the inverse direction of what we need but confirms the "drain" name is precedented.

## GCP metadata access (already exists)

- `lib/iris/src/iris/cluster/worker/env_probe.py:29-62`:
  - `_GCP_METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1/instance"`
  - `_is_gcp_vm()` (`L34-47`) checks DMI strings.
  - `_get_gcp_metadata(path)` (`L50-62`) issues authenticated GET with 2s timeout.
- The preempt endpoint per GCP docs: `${_GCP_METADATA_ROOT}/preempted` — returns `"TRUE"` once preemption fires. We can either poll at ~1s cadence or long-poll with `?wait_for_change=true`. Polling is simpler and good enough for the ~10s budget.
- **No existing preemption-detection code anywhere in `lib/iris/`.** Grep for `preempted`, `metadata.google.internal`, `ACPI` confirms this is greenfield.

## ACPI soft-off

Issue mentions hooking ACPI (G2/S5) per the GCE docs. In practice on a GCE Linux VM, the kernel will receive `SIGTERM`-as-systemd-shutdown and then SIGKILL ~30s later. We don't need to register an ACPI handler ourselves — the metadata server flips `preempted=TRUE` before the ACPI sequence starts, and our metadata-watcher is the cleaner signal. ACPI hookup is a fallback only worth adding if the metadata watcher proves unreliable in production.

## Out-of-scope reminders (from the issue)

- Substring-pattern fix in #5754 — discarded once this lands.
- CoreWeave / on-prem preempt sources — controller-side atomic rule covers them via abrupt-loss heuristics; per-platform watchers are future work.
- Revisiting `max_retries_failure=0` (PR #4615) — the budget is correct, the classification was the bug.
- Checkpoint-on-preempt — Levanter checkpoint writes don't fit in 10s, and 30s isn't reliable either.

## Sequencing constraint

@rjpower's note in the issue: land *after* the reconcile-PR series settles. The reconcile machinery this design hooks into (`apply_reconcile_result`, `_observations_to_updates`) was just stabilized in commits `946f5f953c`, `9b5afc103b`, `1f19d486ab`. Branch should rebase on `main` immediately before the spiral starts.
