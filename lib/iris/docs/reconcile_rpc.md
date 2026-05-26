# Reconcile RPC: worker control plane

The Reconcile RPC collapses the legacy `StartTasks` + `PollTasks` pair (and the
implicit `StopTasks` dispatch) into a single unary call per worker per tick.
One request carries the controller's **complete desired attempt set** for that
worker; one response carries the **complete observed set** plus a
`WorkerHealth` block. No more split-brain windows between "we told the worker
to start X" and "we polled and the worker hadn't seen X yet"; no third RPC for
stops. Both the legacy and new wires remain in tree behind a rollout flag
through Phase D.

## Wire

Proto: [`lib/iris/src/iris/rpc/worker.proto:106`](../src/iris/rpc/worker.proto)
defines `ReconcileRequest` (`worker_id`, repeated `DesiredAttempt`) and
`ReconcileResponse` (`worker_id`, repeated `AttemptObservation`,
`WorkerHealth`). Service method is
[`worker.proto:194`](../src/iris/rpc/worker.proto). `DesiredAttempt.intent`
is a `oneof { AttemptSpec run; StopReason stop }`. `AttemptSpec.request` is
populated only on the dispatch tick for an `ASSIGNED` attempt; subsequent ticks
omit it and the worker pulls from cache (see
[`reconcile.py:71`](../src/iris/cluster/controller/reconcile.py)).

## Control flow

```mermaid
sequenceDiagram
    participant Loop as Controller<br/>_reconcile_worker_batch
    participant Pure as reconcile.py<br/>reconcile_workers()
    participant Prov as WorkerProvider<br/>reconcile_workers()
    participant W1 as Worker A<br/>(RPC path)
    participant W2 as Worker B<br/>(legacy path)
    participant Apply as transitions<br/>apply_reconcile_result

    Loop->>Pure: ReconcileInputs (rows + job_specs)
    Pure-->>Loop: list[WorkerReconcilePlan]
    Loop->>Prov: plans, addresses, rpc_worker_ids
    par RPC fanout
        Prov->>W1: Reconcile(ReconcileRequest)
        W1-->>Prov: ReconcileResponse(observed, health)
    and Legacy fanout
        Prov->>W2: StartTasks(...)
        Prov->>W2: PollTasks(...)
        W2-->>Prov: acks + task statuses
    end
    Prov-->>Loop: list[ReconcileResult]
    Loop->>Apply: per-worker (plan, result)
```

## Pure compute vs. transport

`controller._reconcile_worker_batch`
([`controller.py:2391`](../src/iris/cluster/controller/controller.py)) snapshots
DB rows, calls `reconcile_workers(inputs)`
([`reconcile.py:66`](../src/iris/cluster/controller/reconcile.py)) to produce
one `WorkerReconcilePlan` per worker (proto built once, reused on both wires),
then routes through `WorkerProvider.reconcile_workers`
([`worker_provider.py:271`](../src/iris/cluster/controller/worker_provider.py)).
The provider partitions plans by `rpc_worker_ids` and runs both fanouts under a
single `asyncio.gather` capped by `self.parallelism`. The legacy path
([`_reconcile_one_legacy`, `worker_provider.py:202`](../src/iris/cluster/controller/worker_provider.py))
synthesizes a `ReconcileResult` from its `StartTasks` acks + `PollTasks`
response via
[`_legacy_results_to_reconcile_results`, `worker_provider.py:362`](../src/iris/cluster/controller/worker_provider.py)
so the apply layer
([`transitions.apply_reconcile_result`, `transitions.py:1988`](../src/iris/cluster/controller/transitions.py))
sees one shape regardless of wire.

## Worker side

`WorkerLifecycle.handle_reconcile`
([`worker.py:1050`](../src/iris/cluster/worker/worker.py)) processes each
`DesiredAttempt` (run or stop intent), kills any local attempt not in the
desired set ("zombie"), synthesizes `TASK_STATE_MISSING` observations for run
intents that resolved to nothing locally, and attaches a fresh `WorkerHealth`.
The RPC entry point is
[`WorkerService.reconcile`, `service.py:171`](../src/iris/cluster/worker/service.py).

The observation set is bounded: the worker emits an `AttemptObservation` only
for attempts the controller asked about (`DesiredAttempt` resolves to a local
attempt) or for zombies it is killing this tick (so the controller can confirm
the kill it implicitly requested). Terminal local history outside the desired
set is suppressed — previously a worker could emit hundreds of stale terminal
observations per tick, each driving a DB write on the apply side. The
controller mirrors this in
[`transitions._filter_observations_to_plan`, `transitions.py`](../src/iris/cluster/controller/transitions.py):
observations whose attempt is not in the per-worker `WorkerReconcilePlan` are
dropped (DEBUG-logged) before any work is done.

## Rollout

Resolved once at controller startup from `IRIS_RECONCILE_PREFIX`
([`main.py:218`](../src/iris/cluster/controller/main.py)) into
`ControllerConfig.reconcile_rpc_prefix`
([`controller.py:1300`](../src/iris/cluster/controller/controller.py)):

| `IRIS_RECONCILE_PREFIX` | Behavior |
|---|---|
| unset | legacy wire for all workers |
| `*` | Reconcile RPC for all workers |
| `<prefix>` | Reconcile RPC for `worker_id.startswith(prefix)`, legacy otherwise |

Per-tick routing lives at
[`controller.py:2402`](../src/iris/cluster/controller/controller.py). The
prefix is frozen at startup — widen it by restarting the controller (see
the `restart-iris` skill). Tests covering each mode:
[`test_reconcile.py:1199`](../tests/cluster/controller/test_reconcile.py) and
the `_observation_for_all_run` cases around
[`test_reconcile.py:1078`](../tests/cluster/controller/test_reconcile.py).

## Monitoring

Every worker handled via the RPC wire emits one structured INFO line per tick
to the `iris.cluster.controller.reconcile_rollout` logger
([`worker_provider.py:441`](../src/iris/cluster/controller/worker_provider.py),
emitter at
[`worker_provider.py:446`](../src/iris/cluster/controller/worker_provider.py)):
`desired_run`, `desired_stop`, observation counts by state, run/stop ids, and
`error`. Dial verbosity on that logger independently of the controller root.
The logger and `_log_rpc_rollout` are explicitly tagged for deletion alongside
the legacy wire.
