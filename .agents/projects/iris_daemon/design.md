# IrisDaemon

_Why are we doing this? What's the benefit?_

Iris can schedule discrete jobs, but it cannot say "run exactly one instance of X on every node
of category C, for that node's lifetime" — the DaemonSet pattern. Several workloads want this:
node-local log shippers and metric exporters, cache/proxy sidecars, and — the motivating case —
a standing fleet of warm query workers for smallquery (`.agents/projects/smallquery/`) that must
cover every preemptible VM and automatically track autoscaler churn. Today each of these is
hand-rolled as a fixed-size coscheduled gang (`WorkerPool`) that doesn't follow nodes joining or
leaving. `IrisDaemon` makes "one per node of a category" a first-class, backend-agnostic
primitive.

## Challenges

The honest difficulty is that Iris has **no spread/anti-affinity scheduling primitive** —
constraints are positive-match only and coscheduling `group_by` is the *opposite* (gang
affinity) (`lib/iris/src/iris/scheduling/scheduler.py:539-548`,
`lib/iris/src/iris/cluster/constraints.py:129`). So we can't express "one per distinct worker"
through the scheduler as-is. The design has to either add such an operator or sidestep it. It
also has to compose cleanly with the autoscaler, which already owns the authoritative
worker-join/leave signal (`autoscaler/recovery.py:124-184`), without racing it. And it must
settle a policy question that touches every other job on the node: does a daemon's resource
request **reserve** capacity (shrinking what co-located jobs see as free) or run as best-effort
overflow?

## Costs / Risks

- **Controller complexity**: a new reconciliation loop maintaining a per-worker invariant, with
  all the edge cases of node churn, restarts, and config changes.
- **Resource-accounting policy is load-bearing and global** — get it wrong and daemons either
  starve real jobs or get starved themselves.
- **Backend skew**: the GCP/TPU path and the k8s/CoreWeave path have different composition models
  (k8s already does pod sidecars, `backends/k8s/tasks.py:460-470`; GCP bootstrap launches a
  single fixed container, `worker/worker_bootstrap.py:259-268`). One primitive, two lowerings.
- Yet another way to run a workload — must not become a confusing overlap with jobs/actors.

## Design

**A `DaemonSpec` declared in the cluster config, reconciled by the controller into one task per
matching worker.** The operator declares daemons in the cluster YAML (alongside scale groups);
the controller runs a **`DaemonReconciler`** that maintains the invariant:

> for every live worker `W` matching `DaemonSpec.selector`, there is exactly one running daemon
> task of that spec **pinned to `W`**.

**Route A — reconciler over existing task placement (recommended).** The reconciler reuses
machinery that already exists rather than adding a scheduler mode:

- **Placement by pinning, not spreading.** Each daemon instance is an ordinary task carrying a
  HARD constraint on the target worker's unique id (`tpu-worker-id` / worker id already exist as
  attributes, `cluster/types.py:58-72`). This sidesteps the missing anti-affinity operator — we
  never ask the scheduler to spread; we place one task per known worker id.
- **Join/leave via the autoscaler signal.** The reconciler subscribes to the same
  worker-health/membership signal the autoscaler folds from Reconcile-RPC liveness
  (`autoscaler/recovery.py:124-184`). New worker → create its daemon task; worker gone → its
  daemon task is already `WORKER_FAILED` by normal task↔worker binding (`task-states.md`), so the
  reconciler just stops tracking it. The reconciler only ever *creates*; node loss tears down for
  free.
- **Discovery via EndpointService.** Each daemon `RegisterEndpoint`s under a spec-derived prefix
  (e.g. `smallquery/worker/<worker_id>`); consumers list by prefix and get the live set, with
  dead instances auto-expiring (`controller/endpoint_service.py:83-132`).
- **Supervision for free.** A daemon is a task, so it inherits restart, health, and
  preemption accounting — no separate health wiring.

**Route B — bootstrap sidecar (k8s-first, optional).** For workloads that must escape the task
cgroup and grab leftover host cycles, lower a `DaemonSpec` to a co-resident container: on k8s a
pod sidecar (already supported, `backends/k8s/tasks.py:460-470`); on GCP a new
`WorkerConfig.sidecar_containers` field consumed by the bootstrap template (today a single
hardcoded `docker run`, `worker_bootstrap.py:259-268`). Route B trades Iris lifecycle management
for direct cycle-stealing; we start with Route A and add B only if a consumer needs it.

**Resource semantics.** A `DaemonSpec` requests resources like any task. The recommended default
for opportunistic daemons (smallquery): **BATCH priority band**, **tiny/zero `cpu_millicores`**
(bypasses CPU fit, `scheduler.py:271-276`), and a **real hard `memory_bytes`** reservation
(hard cgroup cap, `runtime/docker.py:684-686`) — reserve RAM, yield CPU. Whether that memory
reservation counts against co-located jobs' visible free capacity is the policy decision in Open
Questions; the spec should make the choice explicit per daemon, not implicit.

**Selector.** v1 selector is a **scale group** (the natural operator-facing unit,
`ScaleGroupConfig`, `config.py:322-340`), optionally narrowed by an attribute predicate reusing
the existing constraint vocabulary.

## Testing

- **Reconciler unit tests** against a fake cluster: assert the invariant holds across worker
  add/remove/restart and `DaemonSpec` add/change/delete — exactly one instance per matching
  worker, none on non-matching workers.
- **Churn integration test** on an Iris dev cluster: autoscale a scale group up and down (and
  `iris job kick --state preempted` daemon instances, `cli/job.py:1045-1080`) and assert the
  fleet re-converges and endpoints reflect the live set.
- **Resource-accounting test**: a daemon with a memory reservation + zero CPU co-located with a
  normal job; assert the normal job's scheduling sees the policy-intended free capacity and
  neither OOMs the other.
- **Both backends**: GCP/TPU (Route A task) and k8s/CoreWeave (Route A task; Route B sidecar if
  built).

## Open Questions

- **Resource accounting**: does a daemon's request subtract from the capacity co-located jobs see
  as available (reserve) or is it best-effort overflow that yields on contention? This is the
  central policy call and affects every other workload on the node.
- **Reconciler placement**: a sibling controller component vs folded into the autoscaler loop
  (which already owns membership). Coupling vs separation of concerns.
- **Route B scope**: do we build the GCP bootstrap-sidecar lowering at all, or is Route A (task
  pinned per worker) sufficient for every near-term consumer including smallquery?
- **Restart/backoff policy** for a daemon whose container keeps crashing on one node — cap
  retries per worker (à la zephyr's infra-failure ceiling) before marking that node's daemon
  degraded?
