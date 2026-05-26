# Reconcile RPC: worker control plane

The Reconcile RPC is the sole controller-to-worker channel for task lifecycle.
One unary call per worker per tick carries the controller's **complete desired
attempt set** for that worker; the response carries the **complete observed
set** plus a `WorkerHealth` block. There is no separate start/stop/poll wire
and no worker-initiated push channel for state changes: reconcile is the only
path.

## Wire

Proto: [`lib/iris/src/iris/rpc/worker.proto`](../src/iris/rpc/worker.proto)
defines `ReconcileRequest` (`worker_id`, repeated `DesiredAttempt`) and
`ReconcileResponse` (`worker_id`, repeated `AttemptObservation`,
`WorkerHealth`). The service method is `WorkerService.Reconcile`.

`DesiredAttempt.intent` is a `oneof { AttemptSpec run; StopReason stop }`
keyed by a 16-hex-char `attempt_uid`. `AttemptSpec.request` is populated only
on the dispatch tick for an `ASSIGNED` attempt; subsequent ticks omit it and
the worker pulls from its local cache (see
[`reconcile.py`](../src/iris/cluster/controller/reconcile.py),
`reconcile_workers`). `AttemptObservation` echoes the same `attempt_uid`, so
routing is purely UID-based — there is no (task_id, attempt_id) fallback.

## Control flow

```mermaid
sequenceDiagram
    participant Loop as Controller<br/>_reconcile_worker_batch
    participant Pure as reconcile.py<br/>reconcile_workers()
    participant Prov as WorkerProvider<br/>reconcile_workers()
    participant W as Worker
    participant Apply as transitions<br/>apply_reconcile_result

    Loop->>Pure: ReconcileInputs (rows + job_specs)
    Pure-->>Loop: list[WorkerReconcilePlan]
    Loop->>Prov: plans, addresses
    Prov->>W: Reconcile(ReconcileRequest)
    W-->>Prov: ReconcileResponse(observed, health)
    Prov-->>Loop: list[ReconcileResult]
    Loop->>Apply: per-worker (plan, result)
```

## Pure compute vs. transport

`controller._reconcile_worker_batch`
([`controller.py`](../src/iris/cluster/controller/controller.py)) snapshots DB
rows and calls `reconcile_workers(inputs)`
([`reconcile.py`](../src/iris/cluster/controller/reconcile.py),
`reconcile_workers`) to produce one `WorkerReconcilePlan` per worker (the
`ReconcileRequest` proto is built once inside the plan). The plans flow
through `WorkerProvider.reconcile_workers`
([`worker_provider.py`](../src/iris/cluster/controller/worker_provider.py))
which fans them out under a single `asyncio.gather` capped by
`self.parallelism` and returns a `ReconcileResult` per worker. The apply layer
([`transitions.apply_reconcile_result`](../src/iris/cluster/controller/transitions.py))
consumes those results.

## Worker side

`WorkerLifecycle.handle_reconcile`
([`worker.py`](../src/iris/cluster/worker/worker.py)) processes each
`DesiredAttempt` (run or stop intent), kills any local attempt not in the
desired set ("zombie"), synthesizes `TASK_STATE_MISSING` observations for run
intents that resolved to nothing locally, and attaches a fresh `WorkerHealth`.
The RPC entry point is `WorkerService.reconcile`
([`service.py`](../src/iris/cluster/worker/service.py)).

The observation set is bounded: the worker emits an `AttemptObservation` only
for attempts the controller asked about (`DesiredAttempt` resolves to a local
attempt) or for zombies it is killing this tick (so the controller can confirm
the implicit kill). Terminal local history outside the desired set is
suppressed — otherwise a worker could emit hundreds of stale terminal
observations per tick, each driving a DB write. The controller mirrors this
in `_filter_observations_to_plan`
([`transitions.py`](../src/iris/cluster/controller/transitions.py)):
observations whose attempt is not in the per-worker `WorkerReconcilePlan` are
dropped (DEBUG-logged) before any work is done.
