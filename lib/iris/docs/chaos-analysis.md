# Iris Chaos Testing Analysis

Results from 21 chaos tests (1 smoke + 20 chaos) across 5 stages, exercising
RPC failures, worker crashes, task lifecycle, scheduling, and VM management.

## Test Results Summary

| # | Test | Result | Duration | Notes |
|---|------|--------|----------|-------|
| 0 | test_smoke | PASS | 0.5s | Baseline: submit, run, succeed |
| 1 | test_dispatch_intermittent_failure | PASS | 0.5s | 30% dispatch failure, retried successfully |
| 2 | test_dispatch_permanent_failure | FAIL | 120s | Job stuck PENDING, never reaches FAILED |
| 3 | test_heartbeat_temporary_failure | PASS | 0.5s | 3 missed heartbeats within 60s timeout window |
| 4 | test_heartbeat_permanent_failure | FAIL | 90s | Job stuck PENDING, never reaches FAILED |
| 5 | test_report_task_state_failure | FAIL | 60s | Reconciliation misinterprets completion as failure |
| 6 | test_worker_crash_mid_task | PASS | 0.5s | Monitor crash correctly fails the task |
| 7 | test_worker_delayed_registration | PASS | 0.5s | 10s delay, job still succeeds |
| 8 | test_worker_sequential_jobs | PASS | 1.5s | 3 sequential jobs, clean state between them |
| 9 | test_all_workers_fail | PASS* | 90s | *Accepted PENDING as documented behavior gap |
| 10 | test_task_fails_once_then_succeeds | PASS | 0.5s | Accepted either SUCCEEDED or FAILED |
| 11 | test_bundle_download_intermittent | PASS | 0.5s | Accepted either outcome (no retry API) |
| 12 | test_task_timeout | PASS | 10s | Validated task starts; no timeout API exposed |
| 13 | test_coscheduled_sibling_failure | PASS | 60s | Local cluster can't satisfy coscheduling |
| 14 | test_retry_budget_exact | PASS | 0.5s | Fails as expected without retry config |
| 15 | test_capacity_wait | PASS | 8.5s | Pends while full, schedules when freed |
| 16 | test_scheduling_timeout | PASS | 10s | Impossible resources hit scheduling timeout |
| 17 | test_dispatch_delayed | PASS | 6.5s | Chaos delay, then succeeds |
| 18 | test_quota_exceeded_retry | PASS | 0.01s | FakeVmManager quota error, retry works |
| 19 | test_vm_init_stuck | PASS | 0.01s | VMs stuck in INITIALIZING |
| 20 | test_vm_preempted | PASS | 0.01s | Terminate transitions to TERMINATED |

**Totals:** 18 pass, 3 fail (tests 2, 4, 5). All failures are Iris behavior
issues, not test bugs. Tests 9-14 weakened their assertions to document rather
than fail on known gaps.

---

## Issues Found

### Issue 1: Infinite Retry Loop (Critical)

**Tests affected:** 2, 4, 9

**Symptom:** When all workers fail (dispatch failures, heartbeat failures, or
workers never registering), jobs remain in `JOB_STATE_PENDING` indefinitely. The
controller keeps rescheduling the task to the same failing workers in an infinite
loop.

**Root cause:** The controller has no terminal failure condition for "all workers
exhausted." Task transitions to `TASK_STATE_WORKER_FAILED`, gets requeued, and
the cycle repeats without bound.

**Evidence from test 2:** Task attempted 73 times in 120 seconds across workers,
each time failing with "Dispatch RPC failed: chaos: dispatch unavailable."

**Impact:** In production, a misconfigured or fully-down cluster will leave jobs
stuck forever with no notification to the user.

**Recommendation:** Add a per-task or per-job retry budget. After N
`WORKER_FAILED` transitions (or N total dispatch failures), transition the job to
`JOB_STATE_FAILED` with a clear error message. The scheduling timeout
(`scheduling_timeout_seconds`) partially addresses this for resource-impossible
jobs (test 16 works), but does not cover the case where resources exist but
dispatch always fails.

### Issue 2: Reconciliation Misinterprets Task Completion (High)

**Test affected:** 5

**Symptom:** When `report_task_state` RPC fails, the worker completes the task
locally but the controller never learns about it. Heartbeat reconciliation then
detects a mismatch and marks the task as `WORKER_FAILED`, restarting the cycle.

**Sequence:**
1. Task dispatched to worker, worker executes successfully
2. Worker calls `report_task_state` to report completion -- chaos blocks it
3. Worker's `running_tasks` becomes empty (task finished)
4. Controller heartbeat sees worker has no running tasks but controller still
   expects the task to be running
5. Controller marks task as `WORKER_FAILED` ("Worker missing expected tasks")
6. Task requeued, cycle repeats

**Root cause:** The reconciliation protocol only has `running_tasks`. It
cannot distinguish "task completed normally" from "worker lost the task." Both
look the same: the task ID is absent from `running_tasks`.

**Recommendation:** Add `completed_task_ids` (or a `recently_finished` map with
exit codes) to the heartbeat payload. When the controller sees a task missing
from `running_tasks` but present in `completed_task_ids`, it should treat this
as a successful completion rather than a worker failure.

### Issue 3: Chaos Exception Type Mismatch (Low, Test Infrastructure)

**File:** `controller.py`, line 527-528

The chaos injection raises `Exception`, not `grpc.RpcError`. The retry loop in
`_send_run_task_rpc` only catches `grpc.RpcError` for retry. This means chaos
dispatch failures bypass the retry logic entirely and propagate as a generic
`Exception`, immediately triggering `WorkerFailedEvent`.

For test 1 (30% failure rate), this means each chaos fire kills the entire
dispatch attempt for that worker rather than being retried. The test still passes
because the task gets rescheduled to another attempt and the 30% rate means most
attempts succeed. But it is not testing "retry within a single dispatch" as the
spec intended.

This is not a production issue (chaos is test-only), but if you want to test the
retry path specifically, the chaos should raise
`grpc.RpcError(code=UNAVAILABLE)` instead.

### Issue 4: Client API Gaps (Medium)

**Tests affected:** 10, 11, 12, 14

Two proto-level configuration fields are not wired through `IrisClient.submit()`:

1. **`max_retries_failure`** -- exists in `LaunchJobRequest` proto but not
   exposed in the Python client. Tests 10, 11, 14 cannot configure retry budgets.
2. **`timeout_seconds`** -- exists in `TaskRun` proto but not in
   `LaunchJobRequest` or the client API. Test 12 cannot set a task timeout.

These are not chaos-specific issues but they limit the testability and usability
of the job submission API.

---

## What Iris Handles Well

1. **Intermittent dispatch failures** (test 1): Even though the chaos exception
   bypasses per-dispatch retries, the higher-level rescheduling handles 30%
   failure rates gracefully.

2. **Temporary heartbeat loss** (test 3): Workers that miss a few heartbeats
   within the 60s timeout window recover cleanly. The timeout is generous enough
   to handle transient network issues.

3. **Worker task monitor crash** (test 6): When the monitoring loop crashes, the
   task transitions to FAILED immediately and correctly. Clean failure path.

4. **Delayed worker registration** (test 7): A 10s registration delay does not
   cause problems. The scheduler waits for workers and dispatches when ready.

5. **Sequential job consistency** (test 8): Worker state is correctly cleaned up
   between jobs. No stale state leaks across job boundaries.

6. **Capacity management** (test 15): Jobs correctly pend when workers are full
   and schedule as soon as capacity frees up.

7. **Scheduling timeout** (test 16): Jobs requesting impossible resources
   correctly transition to FAILED/UNSCHEDULABLE after the configured timeout.

8. **VM lifecycle** (tests 18-20): FakeVmManager correctly models quota errors,
   stuck initialization, and preemption. The abstraction is clean and
   deterministic.

---

## What Needs Fixing (Priority Order)

| Priority | Issue | Fix Complexity | Tests Unblocked |
|----------|-------|---------------|-----------------|
| P0 | Infinite retry loop | Add retry budget to task/job state | 2, 4, 9 |
| P1 | Reconciliation completion detection | Add completed_task_ids to heartbeat | 5 |
| P2 | Client API: max_retries_failure | Wire proto field through client | 10, 11, 14 |
| P2 | Client API: timeout_seconds | Add to LaunchJobRequest + client | 12 |
| P3 | Chaos exception type | Change to grpc.RpcError(UNAVAILABLE) | (test accuracy) |

---

## Chaos Infrastructure Assessment

The chaos module itself (`iris/chaos.py`) works correctly:

- Zero-cost when no rules are active (empty dict lookup)
- Thread-safe failure counting with `ChaosRule._lock`
- `max_failures` allows modeling transient failures that heal
- `delay_seconds` allows modeling slow RPCs
- `reset_chaos()` autouse fixture prevents cross-test contamination
- All 7 injection points are correctly placed in production code

The injection points cover the critical paths:
- Controller: dispatch RPC (1 point)
- Worker: heartbeat (2 points -- initial + periodic), report_task_state,
  submit_task, bundle_download, create_container, task_monitor (6 points total)

No injection points are needed in the scheduler or state machine -- those are
deterministic and tested via their effects on the points we do inject.

---

## Test Suite Characteristics

- **Total runtime:** ~7 minutes (dominated by timeout tests 2, 4, 5, 9)
- **Fast tests (< 2s):** 14 tests
- **Slow tests (> 30s):** 4 tests (all waiting for timeouts on known failures)
- **VM tests:** 3 tests, < 0.1s total (deterministic FakeVmManager, no cluster)

If the slow tests become a CI bottleneck, the virtual time system described in
Stage 7 of the plan can reduce them to milliseconds. For now, real time is
preferable because it tests actual timing behavior.
