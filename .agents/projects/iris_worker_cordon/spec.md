# Spec: Iris Worker Cordon

Concrete contracts implied by [`design.md`](./design.md). This file is the surface reviewers should read to check "would I build this exact API?" — not an implementation plan.

## Schema delta

`workers` table ([`schema.py:839-925`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/schema.py#L839-L925)) gains two columns in a new schema migration (next available migration number, follows existing `# Migration NNNN` comment convention):

```sql
ALTER TABLE workers
ADD COLUMN cordoned INTEGER NOT NULL CHECK (cordoned IN (0, 1)) DEFAULT 0;

ALTER TABLE workers
ADD COLUMN cordoned_at INTEGER NULL;  -- unix epoch ms, set when cordoned flips 0→1, NULL otherwise

CREATE INDEX idx_workers_cordoned ON workers(cordoned) WHERE cordoned = 1;
```

The partial index keeps the common-case eligibility query (cordoned = 0) fast and gives the reaper an O(cordoned-count) scan rather than a fleet-wide table scan. `cordoned_at` is set when the flag flips 0→1 (so `iris cluster status` can render "CORDONED 12m ago") and reset to NULL when the flag flips 1→0.

`WorkerRow` dataclass ([`schema.py:1481-1502`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/schema.py#L1481-L1502)) gains matching fields:

```python
@dataclass(frozen=True)
class WorkerRow:
    worker_id: WorkerId
    address: str
    healthy: bool
    active: bool
    cordoned: bool                  # NEW: ineligible for new task assignment; reaper-owned termination once idle
    cordoned_at: datetime | None    # NEW: when cordoned flipped to True; None when not cordoned
    consecutive_failures: int
    last_heartbeat: datetime | None
    # ... existing fields unchanged
```

## Proto

Added to `lib/iris/src/iris/rpc/controller.proto`, immediately after `RestartWorker` (line 477):

```proto
// Mark a worker (and all sibling workers in its slice) as ineligible for new
// task assignments. Existing in-flight tasks continue until completion. Once
// every worker in the cordoned slice has zero in-flight tasks, the controller
// terminates the slice. The autoscaler may scale up a replacement slice
// independently if demand requires it. Cordon is only valid for workers on
// slices in READY state.
//
// Idempotent: cordoning an already-cordoned worker returns accepted=true and
// the same slice_worker_ids (slice membership is unchanged by re-cordon).
//
// Error conditions (returned via `accepted=false` and `error` string; CLIs
// match on the prefix before any colon):
//   - "worker_not_found": worker_id does not match a registered worker on a
//     READY slice. This includes pre-READY (BOOTING/INITIALIZING) workers and
//     workers whose slice has already been terminated by the reaper.
rpc CordonWorker(CordonWorkerRequest) returns (CordonWorkerResponse);

message CordonWorkerRequest {
  string worker_id = 1;
}

message CordonWorkerResponse {
  bool accepted = 1;
  string error = 2;
  // All worker_ids on the slice the request targeted. Always returns the full
  // slice membership regardless of whether this call modified any flags
  // (idempotent: re-cordon returns the same list). For single-VM slices this
  // is a 1-element list. Empty when accepted=false.
  repeated string slice_worker_ids = 3;
}

// Reverse a cordon. Clears the `cordoned` flag and `cordoned_at` timestamp on
// the named worker and all sibling workers in its slice. Only meaningful
// before the reaper has terminated the slice; after termination the worker
// no longer exists and the call returns "worker_not_found: already_terminated".
//
// Idempotent: uncordoning a non-cordoned worker returns accepted=true with
// the slice's worker_ids (no flags were changed).
//
// Error conditions (same string-prefix matching contract as CordonWorker):
//   - "worker_not_found": worker_id does not match a registered worker on a
//     READY slice.
//   - "worker_not_found: already_terminated": the worker existed but its
//     slice has been reaped since cordon. Distinct error string so the CLI
//     can render "you raced the reaper" rather than "typo in worker id".
rpc UncordonWorker(UncordonWorkerRequest) returns (UncordonWorkerResponse);

message UncordonWorkerRequest {
  string worker_id = 1;
}

message UncordonWorkerResponse {
  bool accepted = 1;
  string error = 2;
  // All worker_ids on the slice the request targeted. Always returns the full
  // slice membership regardless of whether this call modified any flags.
  // Empty when accepted=false.
  repeated string slice_worker_ids = 3;
}
```

## Service handler signatures

In `lib/iris/src/iris/cluster/controller/service.py`, paralleling `restart_worker` at line 2404. Both handlers use the existing `ScalingGroup.find_slice_for_worker(worker_id)` and `ScalingGroup.get_slice_worker_ids(slice_id)` helpers (`scaling_group.py:1142, 1151`) for slice membership lookup — no new SQL membership query is introduced for the RPC path.

```python
def cordon_worker(
    self,
    request: controller_pb2.Controller.CordonWorkerRequest,
    context: grpc.ServicerContext,
) -> controller_pb2.Controller.CordonWorkerResponse:
    """Set cordoned=1 and cordoned_at=now for the named worker and all
    siblings on the same slice.

    Atomic: a single UPDATE statement of the form

        UPDATE workers
        SET cordoned = 1,
            cordoned_at = COALESCE(cordoned_at, :now_ms)
        WHERE slice_id = (SELECT slice_id FROM workers WHERE worker_id = :wid)
          AND slice_id IS NOT NULL

    sets the flag on every member of the slice in one statement, so the
    scheduler never observes a partially cordoned slice. `cordoned_at` is
    preserved on re-cordon (`COALESCE`). Idempotent.
    """

def uncordon_worker(
    self,
    request: controller_pb2.Controller.UncordonWorkerRequest,
    context: grpc.ServicerContext,
) -> controller_pb2.Controller.UncordonWorkerResponse:
    """Reverse cordon_worker. Clears cordoned=0 and cordoned_at=NULL for
    every worker on the slice in a single UPDATE.

    Returns "worker_not_found: already_terminated" if the slice has been
    reaped since cordon (worker_id no longer in workers table). Idempotent
    on already-uncordoned workers.
    """
```

## Autoscaler operations

New module-level helper in `lib/iris/src/iris/cluster/controller/autoscaler/operations.py`. The cordon/uncordon SQL lives in `db.py` (next section) since it is pure DB state; only the reaper needs an `operations.py` helper because it ties DB state to slice termination.

```python
def reap_cordoned_idle_slices(
    db: ControllerDb,
    groups: dict[GroupKey, ScalingGroup],
) -> list[SliceId]:
    """Per-cycle reconciliation step.

    Calls `db.cordoned_idle_slices(db)` to get the list of slice_ids whose
    every worker is cordoned AND has zero in-flight tasks (single SQL JOIN,
    not N+1). For each, calls `terminate_slices_for_workers` to release it.
    Returns the list of slice_ids actually terminated (possibly empty).

    Idempotent across cycles: a slice already terminated does not reappear
    in `cordoned_idle_slices`. Concurrent termination races are safe: the
    autoscaler's `scale_down_if_idle` skips cordoned slices via the explicit
    guard, and if some other path races us into `detach_slice`, the loser
    observes `_slices.pop` returning `None` and no-ops. Cloud termination
    via the platform layer is idempotent.
    """
```

## DB query changes

`db.healthy_active_workers_with_attributes` ([`db.py`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/db.py)) gains one predicate:

```python
# BEFORE
WHERE healthy = 1 AND active = 1

# AFTER
WHERE healthy = 1 AND active = 1 AND cordoned = 0
```

New query function added next to it:

```python
def cordoned_idle_slices(db: ControllerDb) -> list[SliceId]:
    """Return slice_ids where every worker is cordoned AND no worker has
    any in-flight task. Single query (joins `workers` with `task_attempts`
    filtered on RUNNING/BUILDING/ASSIGNED states), not N+1.

    By design, cordon propagates to all siblings, so "every worker cordoned"
    and "any worker cordoned" coincide. Used exclusively by the reaper.
    """
```

## Autoscaler routing change

`routing.py` ([`routing.py:189-210`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/autoscaler/routing.py#L189-L210)) excludes cordoned READY slices from the ready-capacity total. Routing today computes `max_vms` from slice-state counts (REQUESTING + BOOTING + INITIALIZING + READY + headroom), not by enumerating workers. The change subtracts cordoned READY slices from the READY contribution:

```python
# Conceptual change in compute_max_vms:
# READY contribution = (READY slices) - (READY slices fully cordoned)
# REQUESTING, BOOTING, INITIALIZING contributions: unchanged. Cordon is only
# valid for READY slices (workers don't exist as DB rows pre-READY), so other
# states cannot be cordoned and need no exclusion logic.
# Fully-cordoned READY slices are treated as "missing from the fleet" so the
# autoscaler provisions a replacement when demand requires.
```

Partial cordon of a slice is structurally impossible (CordonWorker propagates to all siblings), so "fully-cordoned" and "any-cordoned" coincide.

## Autoscaler scale-down guard

`ScalingGroup.scale_down_if_idle` ([`scaling_group.py:835-890`](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cluster/controller/autoscaler/scaling_group.py#L835-L890)) gets an explicit cordoned-slice skip near the top of its candidate-selection loop. Belt-and-suspenders: the existing predicate would not fire on cordoned slices anyway (they are excluded from `target_capacity`), but the guard makes the ownership boundary unambiguous and survives future refactors of the capacity-counting logic.

## CLI

In `lib/iris/src/iris/cli/cluster.py` (paralleling `restart-worker` at [line 1050](https://github.com/marin-community/marin/blob/9bec9c06b/lib/iris/src/iris/cli/cluster.py#L1050)):

```
iris rpc controller cordon-worker --json '{"worker_id": "<id>"}'
iris rpc controller uncordon-worker --json '{"worker_id": "<id>"}'
```

Both follow the existing `client.<rpc_name>(proto_request)` pattern; no new CLI dispatch machinery needed. Output is the JSON-rendered response (including `cordoned_worker_ids` / `uncordoned_worker_ids` so the operator sees the slice expansion).

`iris cluster status` is extended to show a `CORDONED` badge for cordoned workers in its tabular output. Implementation: one new column read from `WorkerRow.cordoned`.

## Errors

Errors are returned via the existing `accepted=false; error="..."` pattern (matching `RestartWorker`). The error string follows a known shape: a snake_case prefix optionally followed by `: <detail>`. CLIs match on the prefix.

| Error string | Trigger |
|--------------|---------|
| `worker_not_found` | `worker_id` does not match a registered worker on a READY slice (typo, pre-READY worker, etc.). Raised by both RPCs. |
| `worker_not_found: already_terminated` | Distinct shape returned by `UncordonWorker` only, when the worker existed but its slice has been reaper-terminated since cordon was set. Lets the CLI render "you raced the reaper" rather than "typo". |

No new proto error enum. Idempotent success cases (already cordoned / already uncordoned) return `accepted=true` with the slice's full worker-id list.

## OPS.md addition

A new section in `lib/iris/OPS.md`, inserted after the existing "Task Operations" section. Exact text:

````markdown
## Recovering a wedged worker

If a worker is reachable but should stop receiving new tasks (e.g. broken-but-running, pre-deploy maintenance, reserved-capacity in a known-bad state), use cordon. Existing in-flight tasks finish naturally; the slice is recycled and a replacement provisioned by the autoscaler.

```
# Identify the wedged worker
iris cluster status                              # CORDONED column shows current cordon state

# Cordon it (also cordons siblings on the same slice)
iris rpc controller cordon-worker --json '{"worker_id":"<id>"}'

# Confirm it stopped receiving new tasks; watch in-flight ones drain
iris cluster tasks --worker-id <id>

# To cancel before recycle:
iris rpc controller uncordon-worker --json '{"worker_id":"<id>"}'
```

The slice is recycled automatically when every cordoned worker on it has zero in-flight tasks. Cordon is sticky across controller restarts. There is no timeout — a wedged task will keep its worker alive indefinitely; in that case fall back to `restart-worker` for hard recycle.
````

## File summary

| Concern | File | Change |
|---------|------|--------|
| Proto | `lib/iris/src/iris/rpc/controller.proto` | Add `CordonWorker` + `UncordonWorker` RPCs and 4 messages |
| Schema | `lib/iris/src/iris/cluster/controller/schema.py` | New `cordoned` and `cordoned_at` columns + index; matching `WorkerRow` fields; new migration |
| DB queries | `lib/iris/src/iris/cluster/controller/db.py` | Filter cordoned in `healthy_active_workers_with_attributes`; add `cordoned_idle_slices` |
| Service | `lib/iris/src/iris/cluster/controller/service.py` | Add `cordon_worker` and `uncordon_worker` handlers; both use existing `find_slice_for_worker` / `get_slice_worker_ids` for membership |
| Autoscaler ops | `lib/iris/src/iris/cluster/controller/autoscaler/operations.py` | Add `reap_cordoned_idle_slices` |
| Autoscaler routing | `lib/iris/src/iris/cluster/controller/autoscaler/routing.py` | Exclude cordoned slices from `max_vms` |
| Autoscaler scale-down | `lib/iris/src/iris/cluster/controller/autoscaler/scaling_group.py` | Guard skipping cordoned slices in `scale_down_if_idle` |
| Autoscaler runtime | `lib/iris/src/iris/cluster/controller/autoscaler/runtime.py` | Call `reap_cordoned_idle_slices` once per cycle |
| CLI | `lib/iris/src/iris/cli/cluster.py` | Add `cordon-worker` / `uncordon-worker` subcommands; CORDONED badge in status |
| Docs | `lib/iris/OPS.md` | Add "Recovering a wedged worker" section |

## Out of scope

- `DrainWorker` RPC (with timeout / forced-recycle fallback) — deferred; if cordoned-with-wedged-task becomes a recurring issue, file follow-up.
- `CordonSlice(slice_id)` as primary verb — see Open Questions in `design.md`.
- Cluster-wide / batch cordon (e.g. "cordon all workers in zone X").
- In-task SIGTERM-style "wrap up" signaling.
- Per-job disruption budgets (PDB analog).
- Migrating `RestartWorker` to use cordon under the hood.
