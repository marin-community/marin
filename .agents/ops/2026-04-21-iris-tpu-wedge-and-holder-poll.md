---
date: 2026-04-21
system: iris
severity: degraded
resolution: fixed
pr: none
issue: none
---

# Iris TPU worker wedge + reservation-holder poll storm

## TL;DR

- User-visible symptom: jobs on TPU v5p-preemptible-8 workers looping with `TPU init failure ("Couldn't open iommu group")` / `RuntimeError: No accelerator found`; `docker ps` on a suspect VM showed 3 `iris-managed=true` containers while the controller listed only 2 tasks for that worker.
- Three independent bugs stacked:
  1. `ManagedThread._safe_target` never fired `on_stop` on the natural-return path (`lib/iris/src/iris/managed_thread.py:85-115`), so task-container `docker kill`+`docker rm` were skipped. Containers that wedged on the vfio driver stayed `Up`, still owning the iommu group, and every subsequent task on the VM failed identically.
  2. `_apply_task_transitions` treated `ASSIGNED -> WORKER_FAILED` as a free retry (`lib/iris/src/iris/cluster/controller/transitions.py:1878`): no `preemption_count` bump and no health-tracker signal. A host that kept losing the iommu race stayed in rotation indefinitely while child tasks looped at full scheduler velocity.
  3. `get_running_tasks_for_poll` (`lib/iris/src/iris/cluster/controller/transitions.py:3116`) did not filter `is_reservation_holder = 1` rows. Every 60s poll cycle, reservation-holder tasks were shipped to the claimed worker in `PollTasksRequest.expected_tasks`, the worker's `_reconcile_expected_tasks` missed them (holders are virtual), and the controller applied `WORKER_FAILED("Task not found on worker")`. Five active holders were each accumulating ~51 attempts/hour against the same claimed worker.
- Fixes: fire `on_stop` unconditionally in the ManagedThread finally; add a `build_failed` health-tracker bump on `ASSIGNED -> WORKER_FAILED`; add `JOIN jobs j ON j.job_id = t.job_id ... AND j.is_reservation_holder = 0` to the poll SQL.
- Mitigation applied live: operator SSH'd the poisoned VM and `sudo docker kill`'d the wedged iris-managed containers. Worker recovered immediately, no hardware intervention.
- Not rolled out yet: existing orphans on other workers will bleed off naturally via preemption. No automated sweep added.

## Original problem report

> {"worker_id": "marin-tpu-v5p-preemptible-8-us-east5-a-20260421-0641-2a0d895e-worker-0",
>  "updates": [{"task_id": "/larry/iris-run-job-20260422-004848/grug-train-partial-rope-every_layer-d1280-2.83e-19/0",
>               "attempt_id": 61,
>               "state": "TASK_STATE_WORKER_FAILED",
>               "exit_code": 1,
>               "error": "TPU init failure (\"Couldn't open iommu group\"): Exit code: 1. stderr: RuntimeError: No accelerator found. Please run on a TPU or GPU."}]}
>
> this means they've been scheduled to an invalid worker. the fix we applied seems invalid. research and report

Follow-up observation from the user after SSH'ing the VM:

> it looks like this worker has "stale tasks": [3 iris-task docker containers, ages 46m / 47m / 11h] -- but the worker status only shows 2 tasks -- /larry/.../grug-train-partial-rope-... Running, /tonyhlee/.../ Succeeded -- can you check the worker docker logs, is it out of sync with the controller, or with docker?

## Investigation path

1. Confirmed the failing task was NOT a reservation-holder (no `:reservation:` segment in task_id) — ruled out that the 2026-04-21 holder-reset fix (`a0fcf0fd2` era) was the regression. The fix only touched `_remove_failed_worker`'s holder branch at `transitions.py:2160-2183`.
2. Read `_apply_task_transitions` at `transitions.py:1851-1892`. Found that `ASSIGNED -> WORKER_FAILED` unconditionally retries to PENDING without incrementing `preemption_count` or `failure_count` (line 1878-1880). Only `EXECUTING -> WORKER_FAILED` bumps the budget (1873, 1886-1891). TPU init failure dies during container startup → task never reaches RUNNING → retry budget never drains.
3. Checked `WorkerHealthTracker` (`lib/iris/src/iris/cluster/controller/worker_health.py`): only hooks are `ping` (liveness-loop failure) and `build_failed` (BUILDING→FAILED). No hook on ASSIGNED→WORKER_FAILED and no hook on generic launch failure. Worker's pings kept succeeding (workerd alive, only TPU broken), so `workers_over_threshold()` returned `[]` forever.
4. User clarified this was not a bad machine and to check for co-schedule — "check that worker, i suspect another job is scheduled to the same machine." Walked the reservation taint logic at `controller.py:718-759 _inject_taint_constraints`. Observed that descendants of reservation jobs get *no* taint constraint (line 753, `elif job_id in has_reservation: modified[job_id] = req`) while non-reservation jobs get `NOT_EXISTS`. Hypothesised cross-reservation co-scheduling. User then reported live `docker ps` showing 3 iris containers vs controller's 2 tasks — confirming a container-vs-controller desync rather than a scheduler bug.
5. Read `worker.py:794-826 _reconcile_expected_tasks`. Confirmed it iterates `self._tasks` only, never `docker ps` — so any container present in Docker but missing from the in-memory task dict is invisible to reconcile and never killed. `_TERMINAL_STATES` check at line 824 also skips tasks whose status is already terminal, so a failed attempt that didn't get its container removed is never revisited.
6. Fetched `sudo docker logs iris-worker` from the poisoned VM (`marin-tpu-v5p-preemptible-8-us-east5-a-20260421-0641-2a0d895e`, zone `us-east5-a`). Single iris-worker container, 15h uptime, no restart. 548 KB / 2661 lines. Ran `logscan` (gemini) and then line-level grep for `on_stop`, iommu, and specific task_ids.
7. Pattern: every task-completion moment — including natural success and the 58× `/tim/iris-run-job-20260421-181136/train_lm/0` iommu failures — ends with `W iris.managed_thread on_stop callback for task-<id> did not complete`. This fires before the next attempt even submits. No `adopt_running_containers`, no `_reset_worker_state`, no `Container not found`, no `failed to remove` anywhere in the log.
8. Read `lib/iris/src/iris/managed_thread.py:85-115`. Identified the bug: the watcher thread blocks on `self._stop_event.wait()` inside `_watch_stop`. `_stop_event` is only set by explicit `ManagedThread.stop()` (line 137). Natural return from the target never sets it → watcher parks forever → `finally` joins with 1s timeout → warning logged → `on_stop` never invoked → `attempt.stop(force=True)` (which does `container.kill() + container.remove()`) never runs → container lingers. If the container's PID 1 happens to be wedged in a kernel-level vfio teardown, the container stays `Up`, still holding the iommu group, and every subsequent task on the host fails identically.
9. Operator `sudo docker kill`'d the wedged iris-managed containers on the VM. Worker immediately started accepting new tasks successfully. Confirmed the wedge was a plain orphan (SIGKILL reachable), not kernel D-state — meaning the only missing step all along was our `docker kill` invocation that `on_stop` should have issued.
10. Queried the controller checkpoint at `/tmp/iris-debug/controller.sqlite3` (decompressed from `gs://marin-us-central2/iris/marin/state/controller-state/1776818488274/controller.sqlite3.zst`, ~2h old) for fleet-wide scope: `SELECT task_id, COUNT(*) FROM task_attempts GROUP BY task_id HAVING attempts > 50`. 51 hits. Restricting to "active in the last 60 minutes" returned exclusively reservation-holder tasks (`:reservation:/0`) at ~51 attempts/hour each — a different failure mode than the iommu loop.
11. Pulled `SELECT attempt_id, state, worker_id, error, started_at_ms, finished_at_ms FROM task_attempts WHERE task_id = '/larry/iris-run-job-20260420-220049/:reservation:/0' ORDER BY attempt_id DESC LIMIT 15`. Every row: `state=7 (WORKER_FAILED)`, `error='Task not found on worker'`, `started_at_ms IS NULL`, `finished_at_ms` ~60s apart, same worker repeatedly. User confirmed reservations are virtual and should never appear on a worker — "somehow this path has been lost."
12. `grep -n 'running_tasks\b' cluster/controller/*.py`. Inventoried every producer: `drain_dispatch` (`transitions.py:2630`) filters holders, `drain_dispatch_all` (`transitions.py:2686`) filters holders, `_building_counts` (`controller.py:613`) filters holders, `drain_for_direct_provider` (`transitions.py:3384`) filters holders. `get_running_tasks_for_poll` at `transitions.py:3116` did NOT filter — only path feeding `PollTasksRequest.expected_tasks` via `controller.py:2321 _poll_all_workers` → `worker_provider.py:332 poll_workers` → `PollTasksRequest`.

## User course corrections

- **"this is NOT a bad machine"** — cut off the hardware-failure branch and the instinct to fail the worker. Correct: the machine was healthy, only a wedged container held the TPU.
- **"i suspect another job is scheduled to the same machine. the reserved job logic is complicated. make a diagram prove you understand this"** — forced a reread of reservation taint injection and preference pass before writing any code. Model was drifting toward a generic "retry budget" patch; the user's framing pushed toward state-desync investigation.
- **"check the worker docker logs, is it out of sync with the controller, or with docker?"** — distinguished the two orthogonal desync axes. Correct answer was worker↔docker, not worker↔controller. Framing the question as a binary kept the investigation from conflating the cases.
- **"after i manually docker killed the running tasks, the worker recovered"** — confirmed the wedge was plain-orphan (SIGKILL reachable), not kernel D-state. Collapsed the remaining solution space: fix the code path that was supposed to issue `docker kill` and didn't.
- **"yes, reservations are _never_ found on a worker because they are virtual. somehow this path has been lost"** — accelerated the diagnosis once it narrowed to the poll loop. Told the model exactly which invariant had been violated, so the model could skip hypotheses about holder lifecycle and go directly to finding the unfiltered SQL.
- **"what's this docker run --init, would that help"** — tested whether defensive hardening would substitute for the real fix. Answer: no (signal was never sent; tini can't route a signal you didn't issue, and kernel D-state is beyond tini anyway). Kept the scope tight.

## Root cause

Three separate defects in different files, each independently sufficient to produce the observed failure modes:

### A. ManagedThread drops on_stop on the natural-return path

`lib/iris/src/iris/managed_thread.py:85-115`. `_safe_target` spawns a watcher that runs `on_stop` when `_stop_event` fires. `_stop_event` is only set by `ManagedThread.stop()`; when the target returns normally, the watcher stays blocked on `.wait()`, the 1s `watcher.join` times out, a warning is logged, and `on_stop` is silently skipped. For task threads `on_stop` is `attempt.stop(force=True)` → `container.kill() + container.remove()`. So every task that completes without an explicit stop — whether it succeeded, failed, or WORKER_FAILED — leaks its container. Most become harmless `Exited` rows in `docker ps -a`, but any container whose PID 1 was wedged in the vfio teardown path of a failed TPU init stays `Up`, holds the iommu group, and poisons the VM.

Class: **missed default case in a signal/event-driven cleanup handshake** — callback only runs on the "external stop" branch, not the "work completed" branch.

### B. ASSIGNED → WORKER_FAILED has no budget cost and no health signal

`lib/iris/src/iris/cluster/controller/transitions.py:1878-1880`. Only `EXECUTING_TASK_STATES -> WORKER_FAILED` bumps `preemption_count` (line 1873-1877) and only `BUILDING -> FAILED` bumps `self._health.build_failed` (line 1871-1872). A task that dies during ASSIGNED (TPU init, image pre-flight, any setup before BUILDING completes) retries to PENDING with zero cost to the retry budget and zero signal to `WorkerHealthTracker`. Combined with (A), the scheduler loop could send the same task to the same wedged worker forever.

### C. Poll loop sends virtual reservation holders to real workers

`lib/iris/src/iris/cluster/controller/transitions.py:3116-3154`. `get_running_tasks_for_poll` feeds `PollTasksRequest.expected_tasks` through `_poll_all_workers` (`controller.py:2321`, 60s cadence). Its SELECT lacked the `is_reservation_holder = 0` filter that every sibling path uses. Holder tasks have `current_worker_id` set (scheduler anchor) and `state=RUNNING` (from `_assign_task`), so they match the filter. The worker's `_reconcile_expected_tasks` looks the holder up in `self._tasks`, fails, and returns `WORKER_FAILED("Task not found on worker")`. Controller applies the update, holder retries to PENDING, scheduler re-assigns the same claimed worker. One `WORKER_FAILED` per holder per 60s — 51 active holders were each taking ~51 attempts/hour. Holders avoided going terminal only because `_remove_failed_worker`'s holder path resets `preemption_count=0`; per-task-update `WORKER_FAILED` does not reset, so slow drain toward the retry ceiling was also happening.

Class: **invariant drift in a replicated SQL filter** — five places apply the same "exclude holders from worker-facing snapshots" rule; one grew apart.

## Fix

### A. ManagedThread — always run on_stop in finally

`lib/iris/src/iris/managed_thread.py`, single edit inside `_safe_target.finally`:

```python
finally:
    if watcher:
        # Wake the watcher regardless of how the target exited so on_stop runs
        # on the natural-completion path too. Otherwise cleanup (e.g. docker
        # kill+rm for task containers) is silently skipped whenever the target
        # returns without an explicit stop() — leaving wedged containers that
        # keep holding TPU vfio/iommu groups and break subsequent tasks.
        self._stop_event.set()
        watcher.join(timeout=5.0)
        if watcher.is_alive():
            logger.warning("on_stop callback for %s did not complete", name)
```

Join timeout widened from 1s → 5s because `docker kill` + `docker rm` legitimately need more than 1s. New test file: `lib/iris/tests/test_managed_thread.py` — four cases: stop()-called, natural return, target raises, no-double-fire.

### B. ASSIGNED → WORKER_FAILED bumps health tracker

`lib/iris/src/iris/cluster/controller/transitions.py:1878-1886`. Added:

```python
if update.new_state == job_pb2.TASK_STATE_WORKER_FAILED and prior_state == job_pb2.TASK_STATE_ASSIGNED:
    task_state = job_pb2.TASK_STATE_PENDING
    terminal_ms = None
    # ASSIGNED -> WORKER_FAILED means the worker accepted the task but
    # couldn't bring it up (e.g. TPU iommu/vfio already held by another
    # process on the VM). Attribute the failure to the worker so a host
    # that keeps failing launches gets reaped; otherwise the task loops
    # forever without draining preemption budget.
    if worker_id is not None:
        self._health.build_failed(WorkerId(str(worker_id)))
```

Reuses the existing `build_failures` counter and `BUILD_FAILURE_THRESHOLD` — no new field, no new threshold. Task-side retry policy unchanged (still free retry, by design: launch failures shouldn't cost task budget; the cost goes to the worker). Regression test: `lib/iris/tests/cluster/controller/test_transitions.py::test_worker_failed_from_assigned_bumps_health_tracker`.

**Caveat worth future work**: bug C made holders emit `WORKER_FAILED` per poll cycle. Under fix B in isolation, that would have reaped every reservation-claimed worker within ~3 minutes. Fix C must ship with or before B. Post-C this is academic, but if the poll path ever reintroduces virtual-task polling, B will amplify it.

### C. Poll SQL filters reservation holders

`lib/iris/src/iris/cluster/controller/transitions.py:3137-3150`. Added JOIN + filter:

```sql
SELECT t.task_id, t.current_attempt_id, t.current_worker_id
FROM tasks t JOIN jobs j ON j.job_id = t.job_id
WHERE t.current_worker_id IN (...) AND t.state IN (?, ?, ?)
AND j.is_reservation_holder = 0
ORDER BY t.task_id ASC
```

Mirrors the filter in `drain_dispatch` (2652-2662), `drain_dispatch_all` (2715), `_building_counts` (`controller.py:624`), `drain_for_direct_provider` (`transitions.py:3411`). Regression test: `lib/iris/tests/cluster/controller/test_reservation.py::test_get_running_tasks_for_poll_excludes_reservation_holders`.

### Data repair

None staged in code. Fleet-wide orphan cleanup not attempted — user opted to let existing orphan containers bleed off naturally as preemptibles recycle. A one-shot `docker kill` / `docker rm --label iris.managed=true` sweep across active workers would clean them sooner; not worth the blast-radius for most cases.

## How OPS.md could have shortened this

Generic patterns only; none specific to this bug.

1. **`lib/iris/OPS.md` → new subsection "Worker-VM inspection recipes"** under the existing worker operations area. Current OPS.md has no canonical command for `gcloud compute tpus tpu-vm ssh ... --impersonate-service-account=iris-controller@hai-gcp-models.iam.gserviceaccount.com` nor for `sudo docker logs iris-worker 2>&1 | gzip`. Those two invocations recurred through the session. Add:

   ```bash
   # SSH to a TPU worker VM (strip "-worker-N" suffix for the VM name)
   gcloud compute tpus tpu-vm ssh <vm-name> --zone=<zone> \
     --impersonate-service-account=iris-controller@hai-gcp-models.iam.gserviceaccount.com \
     --command='sudo <command>'

   # Pull the workerd container log, gzipped, straight to local disk
   gcloud compute tpus tpu-vm ssh <vm-name> --zone=<zone> \
     --impersonate-service-account=iris-controller@hai-gcp-models.iam.gserviceaccount.com \
     --command='sudo docker logs iris-worker 2>&1 | gzip' > /tmp/iris-debug/iris-worker.log.gz

   # Enumerate iris-managed task containers on a VM
   gcloud compute tpus tpu-vm ssh <vm-name> --zone=<zone> \
     --impersonate-service-account=iris-controller@hai-gcp-models.iam.gserviceaccount.com \
     --command='sudo docker ps --filter label=iris.managed=true --format "{{.ID}}\t{{.CreatedAt}}\t{{.Status}}\t{{.Label \"iris.task_id\"}}"'
   ```

   These are needed any time a worker misbehaves. Note `sudo` — docker.sock is root-owned on these VMs. Document this explicitly; otherwise the next engineer burns several minutes on "permission denied while trying to connect to the Docker daemon socket."

2. **`lib/iris/OPS.md` → new subsection "Worker-vs-Docker desync: diagnostic pattern"** under troubleshooting. Recurring smell, generic across bugs: `docker ps` on a worker shows more iris-labeled containers than the controller lists for that worker. Diagnostic steps (not a fix for any specific bug):

   - Compare `docker ps --filter label=iris.managed=true` vs `iris rpc controller get-worker-state <worker-id>` (or the equivalent RPC that returns the worker's running-tasks list).
   - If container count > controller count: `docker inspect <container-id> --format '{{.Config.Labels.iris.task_id}} / attempt={{.Config.Labels.iris.attempt_id}} / worker={{.Config.Labels.iris.worker_id}}'`. If `iris.worker_id` matches the current worker, the orphan was born on this process (cleanup path skipped). If it differs, a previous worker process left it behind and `adopt_running_containers` didn't reclaim it.
   - Workerd's own memory: `sudo docker logs iris-worker 2>&1 | grep -E 'on_stop callback|Cleanup|adopt_running_containers|Resetting worker state'`. The presence of many `on_stop callback ... did not complete` warnings is a generic signal that container cleanup is being skipped — *class of bug*, not this specific one.

3. **`lib/iris/OPS.md` → add a row to "Troubleshooting" table: "same failure reason text on many pending attempts → cache-update or snapshot-source is unfiltered."** This is a generic smell that recurs. Whenever a pending-reason field or error string is identical across many unrelated tasks, the snapshot-building query that populates it has probably lost a filter. Same shape as the previous ops-log's "Pending scheduler feedback" pattern. Name the pattern so the next engineer recognizes it without re-reading this log.

4. **`lib/iris/OPS.md` → "Replicated SQL filter" invariant under code-review checklist.** Five places in the controller enforce "exclude reservation holders from worker-facing snapshots" (`drain_dispatch`, `drain_dispatch_all`, `_building_counts`, `drain_for_direct_provider`, and — now — `get_running_tasks_for_poll`). When adding a sixth path that sends tasks to a worker, *grep for `is_reservation_holder = 0` and copy the filter*. This is a recurring category of drift; worth naming in OPS.md's "when adding new worker-RPC paths" section, if one exists, or as a new short note.

5. **`lib/iris/OPS.md` → "Retry budgets" explainer.** The ASSIGNED→WORKER_FAILED free-retry rule is non-obvious on first read. One paragraph explaining that `preemption_count` bumps only on `EXECUTING -> WORKER_FAILED` (line 1873) and `build_failures` bumps only on `BUILDING -> FAILED` (line 1871), with ASSIGNED being a no-signal state, would save the next engineer from rediscovering it. This is generic to "why is my task looping?" triage.

## Artifacts

- Worker log (548 KB, 2661 lines) fetched during investigation: `/tmp/iris-debug/iris-worker.log` — local, already discarded by now; re-fetch with the command in "How OPS.md could have shortened this" §1.
- Controller checkpoint used for fleet-wide queries: `gs://marin-us-central2/iris/marin/state/controller-state/1776818488274/controller.sqlite3.zst` (taken 2026-04-22T00:41:28 UTC). Queries in the "Investigation path" §10-11 reproduce.
- Affected worker (confirmed wedged, cleared): `marin-tpu-v5p-preemptible-8-us-east5-a-20260421-0641-2a0d895e-worker-0` (zone `us-east5-a`).
- Other workers showing the high-attempt signature at investigation time: `marin-tpu-v5p-preemptible-8-us-central1-20260421-0625-801865e5-worker-0` (in-sync, not wedged), `marin-tpu-v5p-preemptible-8-us-central1-20260421-0137-dad717ca-worker-0` (zero iris containers live; controller snapshot stale), `marin-tpu-v5p-preemptible-8-us-east5-a-20260421-1628-2b4231e7-worker-0` (possibly out-of-sync; snapshot was 2h old at check time).
- Code changes:
  - `lib/iris/src/iris/managed_thread.py:111-120`
  - `lib/iris/src/iris/cluster/controller/transitions.py:1878-1886`
  - `lib/iris/src/iris/cluster/controller/transitions.py:3137-3150`
- Tests:
  - `lib/iris/tests/test_managed_thread.py` (new file)
  - `lib/iris/tests/cluster/controller/test_transitions.py::test_worker_failed_from_assigned_bumps_health_tracker`
  - `lib/iris/tests/cluster/controller/test_reservation.py::test_get_running_tasks_for_poll_excludes_reservation_holders`
- Prior ops log that lightly touches this cluster: `.agents/ops/2026-04-21-iris-scheduler-freeze.md` — different bug (scheduler-loop crash via task_attempts IntegrityError); the `_remove_failed_worker` holder fix landed there was confirmed not responsible for today's symptoms.
