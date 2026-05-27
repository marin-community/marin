# Worker GCP Preemption Watcher + Controller Atomic Slice-Preempt

_Why are we doing this? What's the benefit?_

Today a GCP preempt of one TPU VM in a coscheduled slice racing-terminates the JAX distributed RPC mesh on its siblings. The siblings exit with assorted non-zero codes that the worker classifies as `TASK_STATE_FAILED` (`task_attempt.py:927-931`) — charged against `max_retries_failure`, which for training jobs is 0 (PR #4615). One preempt thus burns the whole job. The current workaround in #5754 was substring-matching JAX peer-loss strings out of stderr; @rjpower vetoed it as whack-a-mole and asked for this principled fix instead.

Goal: when a worker knows its host is going away, every subsequent exit on that worker is preempt-induced by definition; the controller should see preemption, not a swarm of FAILEDs.

Issue: https://github.com/marin-community/marin/issues/5872. Sibling/dup: #5753, #5754, #5747.

## Challenges

- **Atomicity under teardown race.** The preempt-watcher fires, the worker tries to SIGTERM tasks and finalize its observation set, but the host may vanish mid-flight. We can't rely on the worker successfully reporting "I'm draining" — the controller-side rule has to make the slice safe even if the worker dies abruptly.
- **The 30s/10s GCP window.** GCP spec says ~30s notice; field reality (per @rjpower's note in the issue) is closer to ~10s. Anything we do on the watcher path has to be cheap. Checkpoint-on-preempt is a non-goal here.
- **Coscheduled-slice ground truth.** Today preemption cascades requeue siblings only after the *originally-preempted* task transitions out of PREEMPTED back to PENDING (`transitions.py:2527`). For atomicity we need a rule that fires the instant *any* sibling reports PREEMPTED — before the per-task transition machinery turns sibling exits into FAILEDs.
- **Non-GCP environments.** CoreWeave (k8s) and on-prem don't expose the GCE metadata `preempted` endpoint. The watcher abstraction must accommodate them but the first cut is GCP-only — the controller-side rule (part 2) covers the abrupt-loss path on every platform.

## Costs / Risks

- New background thread per worker, polling `metadata.google.internal` every ~1s. Negligible CPU; small chance of metadata-server flakiness causing spurious draining latches if we don't bound retries carefully.
- A controller bug in the atomic-slice-preempt path could mass-cancel a healthy slice if `draining=True` is ever set spuriously. Mitigation: the bit is set on the worker only by a confirmed `preempted=TRUE` reading from the metadata server, never by metadata-server errors or timeouts.

## Design

Two cooperating pieces.

### 1. Worker-side preempt watcher + drain latch

Plumb a `PreemptWatcher` started alongside the profile loop (`worker.py:311`). Implementation in a new `lib/iris/src/iris/cluster/worker/preempt_watcher.py` (~60 lines):

```python
_PREEMPT_URL = f"{_GCP_METADATA_ROOT}/preempted"  # reuse env_probe.py:29

class PreemptWatcher:
    def __init__(self, on_preempt: Callable[[], None], poll: Duration = Duration.from_seconds(1.0)):
        ...
    def run(self) -> None:
        if not _is_gcp_vm():  # env_probe.py:34
            return  # no-op on CoreWeave / on-prem; controller-side rule covers them
        while not self._stop.is_set():
            if _get_gcp_metadata("preempted") == "TRUE":
                self._on_preempt()
                return
            self._stop.wait(self._poll.seconds())
```

The watcher only latches on a confirmed `"TRUE"` body. Metadata-server errors, timeouts, and non-`TRUE` responses keep polling; they never set the bit. Once latched, the watcher exits — preempt is monotonic and there's no clearing path.

`Worker._on_preempt()`:
1. Sets `self._draining = True` (a single bool guarded by the existing worker lock).
2. Iterates `self._attempts` and calls `attempt.kill(reason=PREEMPT)` (`task_attempt.py:419-446`) — SIGTERM with the existing `term_timeout_ms` grace period, then SIGKILL.
3. Causes the next `handle_reconcile` reply to stamp `WorkerHealth.draining = True`.

`Worker.handle_reconcile` (`worker.py:856`) keeps emitting `AttemptObservation`s exactly as today — the controller decides what to do with them. The only behavior change is `ReconcileResponse.health.draining = True` while latched. Observations emitted *before* the latch flipped are legitimate; observations emitted *after* are ignored by the controller per (2), so the worker doesn't need to filter.

`WorkerHealth.draining` is the single load-bearing signal. The controller treats it as ground truth and overrides per-task classifications below; the worker is no longer the source of truth for task state once the bit flips.

Proto change in `lib/iris/src/iris/rpc/worker.proto:146`:

```proto
message WorkerHealth {
  bool healthy = 1;
  string health_error = 2;
  iris.job.WorkerResourceSnapshot resources = 3;
  bool draining = 4;  // NEW: worker is preempting; do not score subsequent attempt exits
}
```

Exit-classification site `task_attempt.py:927-931` does not need a check: the latch lives at the worker level, and `handle_reconcile` is where the suppression happens. Keep the per-attempt classifier honest; only the reporting layer lies.

### 2. Controller-side atomic slice-preempt + expedited removal

In `Transitions.apply_reconcile_result` (`transitions.py:1986-2054`), between the filter at L2035 and the update conversion at L2039:

- If `result.health.draining`:
  - **Discard `result.observations` entirely** — the draining worker is no longer authoritative on per-task state.
  - Look up every active task currently assigned to `worker_id`.
  - For each such task, mark it PREEMPTED in this transaction using `_mark_task_producing_transition(..., attempt_state=TASK_STATE_PREEMPTED, ...)` (`transitions.py:2511-2521`). For coscheduled tasks, also fan out via `_find_coscheduled_siblings()` (`transitions.py:733`) and mark all siblings PREEMPTED in the same tx.
  - Call `WorkerHealthTracker.mark_draining(worker_id)` (new method on `worker_health.py:50`). The scheduler's reaper checks this: a worker with `draining=True` AND zero RUNNING-state assignments is eligible for immediate removal, bypassing the standard `PING_FAILURE_THRESHOLD=10` ping window.

Discarding observations on the draining path is what makes "just the bit" work: the worker can keep emitting whatever observations naturally fall out without the controller having to disambiguate which exits were the preempt and which were real. The atomic rule also means that even if the worker dies after sending one drain bit but before subsequent observations land, the whole coscheduled slice is already PREEMPTED. The retry path is the existing one — `TASK_STATE_PREEMPTED` charges `preemption_count` not `failure_count`, and `_requeue_coscheduled_siblings` (`transitions.py:792`) re-coschedules.

Spiral order: (a) proto field + worker latch + force `draining=True` from a test fixture → (b) controller atomic-slice rule reading the flag → (c) GCP metadata poll wired in. Each stage is independently testable.

## Testing

- **Unit (`tests/test_transitions.py`):** Construct a coscheduled job with N tasks across 1 worker. Feed an `apply_reconcile_result` with `health.draining=True` and observations that *contradict* the desired draining behavior (e.g. attempts in `FAILED`). Assert: observations are discarded; all N siblings transition to PREEMPTED in the same tx; `preemption_count` bumps once per task; `failure_count` unchanged.
- **Unit (`tests/test_worker.py`):** Inject a fake `PreemptWatcher.on_preempt()` call. Assert: `_draining` flips; subsequent `handle_reconcile` stamps `ReconcileResponse.health.draining=True`. Separately, verify the watcher does *not* latch when the metadata stub returns errors / timeouts / non-`TRUE` bodies.
- **Integration (`tests/integration/`):** Stub GCP metadata server returning `TRUE`. Bring up a worker + controller in-process; assert end-to-end that a coscheduled 2-task job whose worker preempts ends up with both tasks in PREEMPTED and the job re-coscheduled.
- **Live (Iris dev cluster):** Simulate preempt via `gcloud compute instances simulate-maintenance-event` against a v4 slice running a multi-host JAX job. Observe in the dashboard: slice transitions to PREEMPTED atomically, no FAILED tombstones, retry lands on a fresh slice.

## Open Questions

- **Naming:** issue suggests `WORKER_SHUTTING_DOWN`; proto already has `STOP_REASON_WORKER_DRAIN` and `WorkerHealth` is the natural carrier. Proposing `WorkerHealth.draining` (bool) + observability via `WorkerLiveness.draining=True`. Pushback welcome.
- **Non-coscheduled tasks on a draining worker:** mark PREEMPTED (current proposal) or let normal `_apply_task_transitions` route their exits? Marking PREEMPTED is cleaner — the worker can no longer be trusted to classify, and PREEMPTED retries on a fresh worker anyway. Calling out because it slightly broadens the blast radius vs. the issue's explicit scope (coscheduled only).
- **Should the watcher poll or long-poll?** GCE supports `?wait_for_change=true` on metadata. Long-poll cuts latency from ~1s to ~ms but ties up a urllib connection. Polling at 1s is fine for ~10s budget; proposing poll.
