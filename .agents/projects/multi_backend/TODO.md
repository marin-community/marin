# Multi-backend: remaining work

The in-process multi-backend foundation and the partitioned-authority contract
are in #6730: backends author their own task status / worker liveness /
placement, the controller only routes a task to a backend and stores what it
reports, `ReconcileResult` is effects-only (no `WorkerId` crosses the boundary),
and teardown + liveness live in the backend. At a single backend it is
byte-identical to today. What is left, as discrete PRs.

## In #6730 before merge

- **Read-only dashboard cannot list backends.** `ListBackends` is missing from
  `DASHBOARD_READABLE_RPCS` (`lib/iris/src/iris/rpc/auth.py`). The always-shown
  Capacity tab and the Backends tab both call it, so the read-only
  `DASHBOARD_ROLE` is denied. One-line allowlist add, alongside `ListWorkers`.
- Narrative-docstring cleanup over the branch diff.

## PR: generic BackendStatus + always-on Backends tab

Replace the k8s-special-cased `GetKubernetesClusterStatus` /
`backend.get_cluster_status()` — which is not on the `TaskBackend` protocol and
leaks through `# type: ignore[union-attr]` in `service.py` — with a uniform
per-backend status every backend populates.

- Extend `BackendSummary` with `oneof detail { GetKubernetesClusterStatusResponse
  kubernetes; WorkerFleetDetail worker; }`. Reuse the existing k8s message; add a
  thin `WorkerFleetDetail` = `AutoscalerStatus` (already stamped per backend) +
  worker counts.
- Populate in `list_backends`: k8s via `get_cluster_status()`, worker via
  `_merge_autoscaler_status(only_backend_id=...)` + the existing per-backend
  worker counts. No new data sources.
- Decide: put a `status()` (or `get_cluster_status()`) on the `TaskBackend`
  protocol — kills the `type: ignore` — versus keep it capability-gated.
- Dashboard: drop the `requiresMultiBackend` gate so the tab is always on
  (one line in `App.vue`); add per-backend detail panels reusing
  `KubernetesClusterTab` and `FleetOverview`; decide whether the standalone
  Cluster tab folds into the Backends tab.
- Independent of the tracker refactor below — per-backend status is already
  derived from the `scale_group -> backend_id` read-partition, not the tracker —
  so this can land before the remote stack.

## PR: remote backend transport (reopen #6731)

The levels-of-controllers payoff: a backend that *is* a remote Iris controller.

- Wire/session transport over connect/RPC to a remote cluster (C-1b).
- `RemoteAgent` RPC + the Iris execution stack running in "backend mode" (C-2).
- Auth-stack extension: identity binding for remote backends — who may drive
  which backend.
- Cloud-smoke for a two-controller topology (C-4).

## PR: per-backend tracker construction (deferred internal refactor)

Move `WorkerHealthTracker` construction into each backend (its own store),
replacing the single shared tracker the controller injects via `attach_health`.

- Not required for BackendStatus or Fleet aggregation — both already partition by
  the `scale_group -> backend_id` map at read time, not by splitting the tracker.
- Pull-in trigger: the moment a second worker-bearing backend is testable
  in-process. Until then partitioning has zero observable effect at N=1 and, done
  without the matching re-aggregation, would regress the currently-working global
  view.
- Blockers to resolve first: `unreachable_grace` lives in `ControllerConfig` and
  the composer builds backends without it; k8s-only clusters have no worker
  tracker (aggregation must tolerate zero); the `TaskBackend` protocol has no
  `health`.

## PR: remove the CLUSTER_VIEW dispatch-drain input path

The k8s backend still has its pending promoted by the controller draining
dispatch updates into its `reconcile`. Removing that needs the design decision
for how a workerless backend promotes its own pending (PENDING -> ASSIGNED)
without writing the global DB directly.
