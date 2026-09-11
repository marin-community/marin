# Research: multi-backend dashboard + RPC surface

Findings from a three-agent sweep of the dashboard frontend, the controller RPC
surface, and the controller's per-backend exposure. File:line refs are against
the `iris-mb-2-multi-backend` branch.

## Frontend (lib/iris/dashboard/, Vue 3 + rsbuild + Tailwind)

- Served by the controller: HTML shell at `/`, static assets from `dashboard/dist/static/`
  (`dashboard.py`), hash routing, Connect RPCs proxied at `/iris.cluster.ControllerService/*`.
- Tabs (gated by capability strings from `/auth/config` `backend.capabilities`): Jobs (`/`),
  Capacity (`/capacity`), Fleet (`/fleet`, requires `workers`), Cluster (`/cluster`, requires
  `cluster`), Endpoints, Account, Status; detail pages Job/Task/Worker.
- **Capability gating is singular today**: `App.vue` reads `config.backend?.capabilities` (the
  representative backend only) and shows Fleet if `workers`, Cluster if `cluster`. A controller
  with a worker-daemon **and** a k8s backend shows only the representative's tabs — a bug.
- Each tab already scopes through a URL query param and spreads `...route.query` on write
  (`?user=`, `?contains=`, `?sort/dir/page`). Adding `?backend=` is the native idiom.
- `TabNav.vue` exposes a `<slot/>` left of Refresh — a free home for a scope control.
- Region/zone are parsed from scale-group **names** (`FleetOverview.vue`); backend is a distinct
  real dimension. `backend_id` is on every job/task/worker DB row (migration 0032) but absent
  from `rpc.ts` and the UI.
- Reusable: `DataTable` (sortable/paginated/column slots), `InfoCard`/`InfoRow`, `StatusBadge`,
  `MetricCard`, `ConstraintChip` (ideal for advertised attrs), `FleetOverview`, `useControllerRpc`,
  `useAutoRefresh`.

## Controller RPC surface (service.py, controller.proto / job.proto / vm.proto)

Row shapes consumed by the dashboard and their `backend_id` status:

| RPC | Row message | backend_id? | scale_group? |
|---|---|---|---|
| ListJobs / GetJobStatus | JobStatus (job.proto) | ❌ (in DB, not serialized) | n/a |
| ListTasks / GetTaskStatus | TaskStatus (job.proto) | ❌ (in DB, not serialized) | n/a |
| ListWorkers / GetWorkerStatus | WorkerHealthStatus (controller.proto) | ❌ | ❌ (resp field exists, never set) |
| GetAutoscalerStatus | ScaleGroupStatus/SliceInfo/VmInfo (vm.proto) | ❌ | ✓ on slice/vm |
| GetSchedulerState | Pending/RunningTaskBucket | ❌ | n/a |

- `backend_id` columns exist on `tasks`/`jobs`/`task_attempts` (schema.py) with an
  `idx_tasks_backend_state` index. `reads.list_active_tasks` already accepts a `backend_id`
  filter; `PENDING_TASK_COLS` already selects it and `PendingTask` carries it.
- `workers` has **no** `backend_id` — only `scale_group`; worker→backend is the in-memory
  `controller._scale_group_to_backend.get(sg, DEFAULT_BACKEND_ID)`.
- A job routes to exactly one backend (meta-scheduler pins `(job_id, backend_id)`), so all a
  job's tasks share one backend — no `backend_id` filter needed on `ListTasks`.

## Single-backend assumptions baked into the service/dashboard

1. **/auth/config** builds `backend` from `backend_descriptor(self._service.provider)` — the
   representative backend only. `controller.capabilities` is already the union; it just isn't served.
2. **GetAutoscalerStatus** reads `self._controller.autoscaler` = the first backend with an
   autoscaler. Multiple worker-daemon backends (GCP + N CoreWeave) each own an autoscaler →
   only one's capacity shows.
3. **GetKubernetesClusterStatus** calls `self._controller.provider.get_cluster_status()`, assuming
   the representative backend is the k8s one → breaks if k8s isn't representative.

## Available per-backend, already in memory

`controller.backends` (`{id: TaskBackend}`), `controller.capabilities` (union), `_backend_ids`,
`_backend_routing`, `_scale_group_to_backend`. Each backend: `.name`, `.capabilities`,
`.autoscaler`, `.advertised_attributes()`, `.admits(user)`. Backend allow policy persisted in
`backends_table.allow_policy_json`. `ControllerProtocol` (service.py) currently exposes
`autoscaler`/`provider`/`capabilities` but **not** `backends` or a scale-group→backend resolver —
must be widened.

## New failure mode multi-backend introduces

A job can match **no backend** ("no backend matches the job's constraints" / "no backend permits
this user") or be pinned via `--backend X`. Unschedulable routing decisions are already finalized
to the DB with a reason, so they surface today via task `pending_reason`/`error`; what's missing is
per-backend attribution and a distinct "unroutable" (vs "no capacity") signal.
