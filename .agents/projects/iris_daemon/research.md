# IrisDaemon — research

What exists in Iris today, what's missing, and where a per-node-daemon primitive would hook.
All refs `lib/iris/...`. Motivated by smallquery (`.agents/projects/smallquery/`), which needs
"exactly one warm worker on every node of a scale group," but the primitive is general (log
shippers, metric exporters, node-local caches/proxies all want the same).

## Verdict: no DaemonSet-equivalent today
All workloads flow through the scheduler as job tasks. There is no way to declare "run one
instance of X on every worker of category C."

## What's missing (the gaps a daemon must fill)
- **Worker bootstrap launches exactly one container — the agent — from a hardcoded template**
  with no injection point for extra containers (`worker/worker_bootstrap.py:259-268`; template
  `:109-294`; only `cache_dir`/`docker_image`/`worker_port`/`worker_config_json` substituted at
  `:297-325`). So a VM cannot bring up a co-resident sidecar by config.
- **No scale-group sidecar field.** `WorkerSettings` carries only `attributes`+`cache_dir`
  (`config.py:315-319`); `ScaleGroupConfig` has resources/buffers/priority/quota
  (`config.py:322-340`); `WorkerConfig` (wire config to the VM) has a single `docker_image`
  (`config.py:365-393`). Nowhere to declare a co-resident container.
- **No anti-affinity / one-per-VM scheduling operator.** Constraints are positive-match only
  (EQ/IN/EXISTS, `constraints.py:129,186,609`); coscheduling `group_by` is gang *affinity* — the
  opposite of spread (`scheduler.py:539-548,799-862`). One-per-VM is only weakly approximable via
  unique per-VM attributes + N constrained jobs.

## What exists to build on (the hooks)
- **Autoscaler already tracks worker join/leave** via Reconcile-RPC liveness (no ping loop):
  `WorkerHealthTracker` folds REACHED/UNREACHABLE; dead workers reaped and reprovisioned
  (`autoscaler/recovery.py:77+,124-184`; provisioning health incl. `PREEMPTED`,
  `autoscaler/provisioning.py:27-36`). A daemon reconciler hooks the same join/leave signal.
- **Worker identity attributes exist** for pinning: `tpu-name`, `tpu-worker-id`, plus the worker
  id; well-known attrs at `cluster/types.py:58-72`, `WorkerMetadata` proto `job.proto:537-576`.
  A daemon task pins to a worker via a HARD constraint on its unique id — no new scheduler mode
  needed.
- **EndpointService** gives free discovery: each daemon `RegisterEndpoint`s its address; the
  consumer lists by prefix; endpoints auto-expire when the task dies
  (`controller/endpoint_service.py:83-132`; client renews at 1/3 lease,
  `client/endpoint_client.py:83-154`).
- **Task↔worker binding already dies correctly**: a task whose worker vanishes goes
  `WORKER_FAILED` (`task-states.md`), so a daemon instance naturally dies with its node — the
  reconciler only needs to (re)create, never tear down on node loss.
- **Priority bands** give the "yield to everyone" semantics: BATCH never preempts, is preempted
  by higher bands (`scheduling/policy.py:567-585`; `docs/priority-bands.md`).
- **Spare-capacity packing**: a 0-CPU CPU-only task bypasses CPU fit (`scheduler.py:271-276`);
  memory is a hard cgroup cap (`runtime/docker.py:684-686`). So "reserve RAM, take leftover CPU"
  is expressible per-task today.
- **k8s backend already composes a per-pod sidecar** (log-shipper, `backends/k8s/tasks.py:460-470`)
  — proof Iris composes multi-container pods on that backend; a native DaemonSet/sidecar is
  cheap to add on the k8s/CoreWeave path specifically.

## Closest existing patterns (reference designs, not the primitive)
- **`WorkerPool`** (`client/worker_pool.py:4-9,135-151`) — a coscheduled gang of standing worker
  tasks receiving dispatched callables. Fixed-size; not per-VM; doesn't track autoscaler churn.
- **Actor framework + `ActorPool`** (`actor/server.py:4-9`, `actor/pool.py:58`) — long-lived RPC
  servers registered as endpoints, with load-balance/broadcast. Exactly the
  "consumer pushes RPCs to standing workers" shape — but scheduler-placed, not one-per-node.

## Two implementation routes
- **Route A — controller-side reconciler (recommended).** Maintain the invariant "one running
  daemon task of spec D on every live worker matching selector S." Reuses task placement (pin to
  worker id) + autoscaler join/leave + EndpointService discovery. Daemon = a task → gets Iris
  supervision/health/preemption accounting for free. Sidesteps the missing spread operator.
- **Route B — worker-bootstrap sidecar.** Add `sidecar_containers` to the wire `WorkerConfig` +
  a loop in the bootstrap template so every VM launches D at boot. Closest to "true daemon on
  every VM," can escape the cgroup to grab leftover cycles, but Iris wouldn't
  supervise/health-check/preemption-account it unless lifecycle is also wired. On k8s, this is a
  pod sidecar and nearly free (`tasks.py:460-470`).

## Unclear / to confirm with Iris owners
- **Resource accounting policy**: does a daemon's request count against the worker's committed
  capacity (other tasks see less free), or is it best-effort overflow? smallquery wants RAM
  reserved + ~zero CPU; the policy is theirs to set.
- Whether long-lived tasks are kept alive on exit or treated as finished (`service_mode.py` is
  only `dry_run/local/cloud`, unrelated) — affects restart semantics for a crashed daemon.
- Whether the reconciler should live in the autoscaler loop or be a sibling controller component.
