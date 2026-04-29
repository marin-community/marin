# Research: Iris worker cordon

Background notes that fed `design.md` and `spec.md`. References pinned to `main@9bec9c06b`.

## Problem framing

From [#5270](https://github.com/marin-community/marin/issues/5270): `RestartWorker` is the only manual recycle primitive and it hard-kills in-flight tasks. Surfaced during #5268 triage when 7 wedged CPU on-demand workers were running coordinator/parent jobs. The original DoD asked for a `DrainWorker` (cordon + wait + recycle, with timeout fallback). After investigation we narrowed scope to **cordon + uncordon only**; the wait-and-recycle "drain" semantics are deferred. See `design.md` § Open Questions and "Out of scope" for the why.

## In-repo findings

### `RestartWorker` today

- Proto: [`controller.proto:473-480`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/rpc/controller.proto#L473-L480) — `RestartWorkerRequest{worker_id}`, `RestartWorkerResponse{accepted, error}`.
- Service handler: [`service.py:2404-2430`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/service.py#L2404-L2430) — calls `autoscaler.restart_worker(worker_id)`.
- Autoscaler op: [`operations.py:39-78`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/autoscaler/operations.py#L39-L78) — looks up `slice_id`, calls `slice_handle.restart_worker(handle)`. Worker process restarts in-place; in-flight tasks are dropped at the process level (no scheduler-side coordination).

### Worker state representation

- `WorkerRow`: [`schema.py:1481-1502`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/schema.py#L1481-L1502) — fields include `worker_id`, `address`, `healthy: bool`, `active: bool`, `consecutive_failures`, `last_heartbeat`, resource committed/total, `attributes`.
- Worker table: [`schema.py:839-925`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/schema.py#L839-L925) — `healthy INTEGER CHECK (0 or 1)`, `active INTEGER CHECK (0 or 1)`. Binary flags pattern, no enum. New `cordoned INTEGER CHECK (0 or 1) DEFAULT 0` column slots in cleanly.

### Scheduler eligibility

- Worker query: `db.healthy_active_workers_with_attributes()` returns rows where both flags are 1. Adding `AND cordoned = 0` is the minimal-surface change (filtered at DB layer).
- Scheduler: [`scheduler.py`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/scheduler.py) — `try_schedule_task()` (line ~458) iterates candidate workers from constraint matching, then capacity. Filtering at the DB layer keeps the scheduler unchanged.
- Scheduling cycle: `controller.create_scheduling_context(workers)` (controller.py:~2047) builds the snapshot from filtered worker rows.

### In-flight task tracking

- `db.running_tasks_by_worker(db, worker_ids)` ([`db.py:822`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/db.py#L822)) returns `dict[WorkerId, set[JobName]]`. Already exists; used at controller.py:1681, 1842, 2531.

### Autoscaler

- Scale-down: `ScalingGroup.scale_down_if_idle()` ([`scaling_group.py:835-890`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/autoscaler/scaling_group.py#L835-L890)) terminates idle slices when `ready + pending <= target_capacity` and the slice has been idle. Idle is computed from `worker_status_map` updated by the scheduler's `last_active`.
- Scale-up cooldown: `can_scale_up()` (line ~938-958) enforces 1-minute cooldown.
- Routing: `routing.py:189-210` computes `max_vms` as sum of REQUESTING+BOOTING+INITIALIZING+READY slices. Counts slices, not individual workers.
- Slice ownership: `ScalingGroup._slices` dict guarded by `_slices_lock` (line 308). `detach_slice()` (line ~635) is the single termination path.
- **Hazard A (load-bearing).** A cordoned worker that finishes its last task becomes idle. The autoscaler's existing `scale_down_if_idle` will only terminate it if `ready + pending <= target_capacity` — i.e., demand has dropped. While there is still demand for the slice's shape, the autoscaler will keep it alive even when idle, so termination won't happen on its own. **Implication:** the cordon path itself must drive termination of cordoned-and-idle workers, not rely on autoscaler scale-down.
- **Hazard B.** If demand still requires N workers and we cordon one, autoscaler will scale up a replacement (after the 1-min cooldown). This is the *desired* behavior per #5270 framing — replace the wedged worker without losing in-flight tasks. We just need cordoned workers to not count toward `target_capacity`.
- **Hazard C.** Race between cordon-reaper terminating an idle cordoned slice and autoscaler scale-down terminating the same slice for unrelated reasons (demand fell). `detach_slice()` is guarded by `_slices_lock`; the second caller gets `None` from the dict. Cloud termination should be idempotent. Low risk but worth a sentinel — see `design.md` § "Termination ownership".

### CLI

- Pattern in [`cli/cluster.py:1050`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cli/cluster.py#L1050): `client.restart_worker(controller_pb2.Controller.RestartWorkerRequest(worker_id=wid))`. New verbs follow this exactly.

### K8s provider taint code

- Lives at `lib/iris/src/iris/cluster/providers/k8s/tasks.py` and `.../k8s/fake.py`. Concrete to GPU-pod admission (NVIDIA_GPU_TOLERATION etc.). Not reusable as a cordon abstraction; cordon here is a SQLite flag and scheduler filter, not a K8s taint.

### OPS.md

- `lib/iris/OPS.md` covers cluster lifecycle, job/task management, scheduler/autoscaler queries. No drain or cordon documentation today. Insertion point: new section "Recovering a wedged worker" after "Task Operations".

### Adjacent designs in `.agents/projects/`

- `20260303_iris_autoscaler_design.md` — autoscaler demand/routing/cooldown. Adjacent, no overlap; cordon adds a per-worker eligibility gate orthogonal to scaling decisions.
- `20260129_iris_chaos_design.md` — chaos injection. Unrelated.
- No prior cordon/drain design in tree.

## Prior art (cluster-manager cordon/drain)

Short comparison; full pass under 200 words.

- **Kubernetes.** Splits cordon and drain. Cordon is `Node.Spec.Unschedulable: bool` (sticky, declarative) plus a mirrored `node.kubernetes.io/unschedulable:NoSchedule` taint. The scheduler has a dedicated `NodeUnschedulable` filter plugin. `kubectl drain` is a *client-side loop*: cordon, then iterate pods and POST to the eviction API (which goes through PodDisruptionBudget admission). No server-side drain coordinator. ([safely-drain-node](https://kubernetes.io/docs/tasks/administer-cluster/safely-drain-node/), [api-eviction](https://kubernetes.io/docs/concepts/scheduling-eviction/api-eviction/)).
- **Nomad.** Splits eligibility (`node eligibility -disable` = cordon) from drain (`node drain -enable` = cordon + drain with deadline). `-keep-ineligible` preserves cordon when drain is cancelled. Best practice: cordon many before draining any, to avoid task ping-pong.
- **Slurm.** Single state machine `DRAIN -> DRAINING -> DRAINED`; no separate cordon. Older design.

### Lessons applied to this design

1. **Cordon as a sticky declarative flag is the right primitive.** Both K8s and Nomad split it from active operations. Crash-safe: if the controller restarts mid-cordon, the flag persists and new tasks are still blocked.
2. **Server-side ownership beats client-side polling for Iris.** K8s went client-side because clusters are multi-tenant and disruption policy is per-workload (PDBs). Iris is single-operator and owns the scheduler — a server-driven cordon-and-reap loop avoids reinventing a polling CLI and survives operator disconnects.
3. **No PDB analog needed.** Iris has no per-job disruption budget; skip the eviction-vs-delete split until that concept exists.
4. **Drain naming is loaded.** Since we're not implementing the wait-and-recycle behavior in this round, calling the RPC `CordonWorker`/`UncordonWorker` (not `DrainWorker`) signals scope honestly and leaves room for a later `DrainWorker` if we add explicit timeouts/forced-recycle.

## Q&A summary that shaped the design

User answers from interrogation phase:

1. *Split or single?* → Cordon + Uncordon only, no Drain.
2. *Recycle on completion?* → Yes — when the cordoned worker completes all in-flight tasks, terminate it. Let the autoscaler scale up a replacement if demand requires.
3. *Timeout fallback?* → N/A; no drain.
4. *Multi-VM slices?* → Cordon at slice granularity. Cordoning a worker on a multi-VM slice cordons the entire slice.
5. *Open questions?* → Simplified per scope.
6. *Out of scope?* → No drain in this round; explicit in `design.md`.
