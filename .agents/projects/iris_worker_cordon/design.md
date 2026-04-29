# Iris Worker Cordon

_Why are we doing this? What's the benefit?_

Today the only manual worker-recycling primitive is `RestartWorker`, which hard-kills in-flight tasks ([`controller.proto:473-480`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/rpc/controller.proto#L473-L480)). Operators have no way to say "stop assigning new tasks here; let the running ones finish; then take this worker out." Surfaced concretely during #5268 triage: 7 wedged CPU on-demand workers were each running coordinator/parent jobs, and the only options were to wait or to hard-restart and lose those coordinators.

This design adds **`CordonWorker` / `UncordonWorker`** — a sticky eligibility flag on the worker row, mirrored to all sibling workers in the same slice. A small reaper terminates cordoned workers once they finish their last in-flight task. The autoscaler is left to scale up replacements on its own, driven by unmet demand. A separate "drain with timeout / forced recycle" primitive is **explicitly out of scope** for this round (see Open Questions); we want the simplest thing that addresses the wedged-worker case before adding more verbs.

Background notes — including prior-art comparison with K8s/Nomad/Slurm and the autoscaler hazard analysis — are in [`research.md`](./research.md).

## Challenges

The proposal is small in surface area but interacts with two pieces of state machinery that are easy to get wrong:

**Autoscaler interactions.** The autoscaler's `ScalingGroup.scale_down_if_idle()` ([`scaling_group.py:835-890`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/autoscaler/scaling_group.py#L835-L890)) only terminates an idle slice when demand has fallen (`ready + pending <= target_capacity`). A cordoned worker that finishes its last task while demand is still high will sit idle indefinitely if we rely on autoscaler scale-down. Conversely, we want the autoscaler to *replace* the cordoned worker with a fresh one — that path works automatically, but only if the cordoned worker stops counting toward `target_capacity`. So cordon needs two effects on the autoscaler: hide cordoned workers from capacity totals (to provoke scale-up) and own termination of cordoned-and-idle workers (since scale-down won't fire).

**Multi-VM slices.** Iris slices can have multiple VMs (e.g. v5e-64 = 16 VMs). Per-worker cordon on a multi-VM slice is meaningless: terminating one VM means terminating the whole slice. The user-facing semantic is therefore "cordon at slice granularity" — cordoning a worker propagates the flag to all siblings in its slice. This keeps the operator mental model simple and makes recycle straightforward. Cordon is only ever applied to slices in `READY` state (workers don't exist as DB rows until the slice is ready); cordoning a `worker_id` for an INITIALIZING/BOOTING slice returns `WORKER_NOT_FOUND`.

## Costs / Risks

- New RPC verb pair, new SQLite column, new CLI subcommands. Modest churn.
- Cordon is sticky — an operator who cordons a worker and forgets about it leaves capacity unreachable until uncordoned. Mitigation: surface cordoned workers prominently in `iris cluster status` and stamp a `cordoned_at` timestamp so stale cordons are visible. We considered but rejected (for this round) a TTL on the flag, a `--reason` field, and a per-cordon audit row; revisit if cordon-and-forget incidents recur.
- The reaper introduces a second termination path alongside autoscaler scale-down. Race is benign (both go through `detach_slice()` under `_slices_lock`, the loser observes `None` and no-ops, and cloud termination is idempotent), but worth a regression test.
- No timeout / forced-recycle. If a cordoned worker has a wedged task that never terminates, the worker stays alive forever. Operators can still fall back to `RestartWorker` (hard kill); a follow-up `DrainWorker(worker_id, timeout)` is the natural extension if we hit this in practice.

### Alternatives considered

- **`RestartWorker(graceful=true)`** instead of a new verb pair. Rejected: cordon is naturally sticky and idempotent, restart is naturally one-shot. Conflating them via a flag complicates `RestartWorker`'s contract and loses the "cordon many before recycling any" pattern that K8s and Nomad both standardize on.
- **Reaper as a standalone loop** (not embedded in the autoscaler cycle). Rejected for this round: the autoscaler already runs a per-cycle reconciliation pass, already holds the slice/group state we need, and already handles termination. Coupling the reaper to its tick keeps the new code minimal. Cost: cordon recycle latency is bounded by autoscaler tick (currently fast); if we ever pause or slow the autoscaler for unrelated reasons, recycle slows too.

## Design

### State

Add a `cordoned INTEGER CHECK (cordoned IN (0, 1)) DEFAULT 0` column to the `workers` table ([`schema.py:839-925`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/schema.py#L839-L925)) and a matching `cordoned: bool` field on `WorkerRow` ([`schema.py:1481-1502`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/schema.py#L1481-L1502)). Binary flag, mirrors `healthy` and `active`. No new enum.

### RPCs

- `CordonWorker(worker_id) -> {accepted, error, cordoned_worker_ids}` — sets `cordoned=1` for the named worker and **all sibling workers in its slice** (multi-VM cordon-the-slice rule). Idempotent: cordoning an already-cordoned worker is a no-op success. Returns the full list of worker_ids touched so the operator sees the slice expansion.
- `UncordonWorker(worker_id) -> {accepted, error, uncordoned_worker_ids}` — symmetric reverse. Only meaningful before the reaper has terminated the slice; after termination the worker is gone and uncordon returns `not_found`. Also idempotent.

Both RPCs follow the existing `RestartWorker` shape (`accepted`, `error` fields). CLI surface follows the existing pattern in [`cli/cluster.py:1050`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cli/cluster.py#L1050):

```
iris rpc controller cordon-worker --json '{"worker_id":"..."}'
iris rpc controller uncordon-worker --json '{"worker_id":"..."}'
```

### Scheduler

One-line change to `db.healthy_active_workers_with_attributes()`: append `AND cordoned = 0` to the eligibility predicate. The scheduler ([`scheduler.py`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/scheduler.py)) is unchanged — it never sees cordoned workers in its candidate pool, so no logic in `try_schedule_task` needs to know about cordon.

### Autoscaler

Two narrow changes, both in `lib/iris/src/iris/cluster/controller/autoscaler/`:

1. **Capacity accounting.** When the routing layer ([`routing.py:189-210`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/autoscaler/routing.py#L189-L210)) computes ready capacity, slices where every worker is cordoned are excluded from `max_vms`. This makes the autoscaler see the cordoned slice as "missing" from the fleet and provision a replacement after the standard 1-minute scale-up cooldown. (We exclude at slice granularity rather than worker granularity because cordon is slice-scoped and a partially-cordoned slice is impossible by construction.)

2. **Cordon reaper.** Add a small reconciliation step to the autoscaler's per-cycle loop in `runtime.py`, called immediately after `scale_down_if_idle`. For each fully-cordoned slice with zero in-flight tasks (checked via `db.running_tasks_by_worker`), call `terminate_slices_for_workers([worker_ids])` ([`operations.py`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/autoscaler/operations.py)). Runs alongside `scale_down_if_idle` but never on the same slices: cordoned slices are excluded from scale-down (they're already excluded from `target_capacity`, so the predicate would not fire anyway, but we add an explicit guard to make the contract clear).

The combination yields the desired flow: cordon a worker → slice marked ineligible → scheduler stops sending new tasks → autoscaler provisions a replacement → in-flight tasks drain naturally → reaper terminates the original slice. **Ownership boundary:** the reaper terminates cordoned-and-idle slices; autoscaler scale-down terminates uncordoned-and-surplus slices.

### OPS.md

New section "Recovering a wedged worker" added to `lib/iris/OPS.md` covering the cordon/uncordon recipe and how to verify the slice is recycled. Full text in `spec.md`.

## Testing

The smallest test that catches a regression is an integration test against the dev cluster (or `provider/k8s/fake.py`) that:

1. Submits N+1 tasks to a cluster of N workers; confirms all run.
2. Cordons one worker mid-flight; confirms (a) the cordoned worker's existing task continues to completion, (b) no new task is assigned to it, (c) the autoscaler provisions a replacement and the queue drains.
3. After the cordoned worker's task completes, confirms the slice is terminated within one autoscaler cycle.
4. Repeats the cordon, then uncordons before completion, and confirms the worker rejoins eligibility without recycle.

A unit test on `db.healthy_active_workers_with_attributes` covers the SQL filter directly. The reaper logic gets a unit test against an in-memory DB with a fake `terminate_slices_for_workers`. Multi-VM-slice behavior gets one parametrized integration test.

## Open Questions

- **Naming: `CordonWorker` vs `CordonSlice`?** The user-facing semantic is slice-scoped, but the issue and operator mental model talk about workers. We've kept "Worker" in the RPC name to match the existing `RestartWorker`, returning the slice's full worker list in the response. Alternative: introduce `CordonSlice(slice_id)` as the primary verb with `CordonWorker` as a thin wrapper. Reviewers' call.
- **Forced-recycle / drain follow-up.** This design intentionally omits a timeout fallback. If we hit cases in production where cordoned workers stay alive indefinitely because of a wedged task, the natural follow-up is a `DrainWorker(worker_id, timeout)` RPC that calls `RestartWorker` once the deadline passes. Worth filing as a tracking issue if we ship cordon and observe the gap.

### Out of scope (for this design)

- **Drain semantics.** No timeout, no forced-recycle, no `DrainWorker` RPC. The reaper waits forever for in-flight tasks to finish naturally.
- **Cluster-wide / batch cordon.** No "cordon all workers matching attribute X" verb.
- **In-task signaling.** We do not send a SIGTERM-style "wrap up" signal to running tasks on a cordoned worker.
- **Per-job disruption budgets.** No PDB analog, no eviction-vs-delete split.
