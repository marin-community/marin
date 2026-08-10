# Spec: multi-backend dashboard + RPC surface

Concrete contracts. Field numbers are proposals against the current `.proto`
files (verify next-free at implementation time). All additions are optional;
empty default == "all backends" / single-backend identity.

## Proto changes

### job.proto — row messages
```proto
message TaskStatus {
  // ...existing through field 23 (18/20/21/22 reserved)...
  // Literal owning backend id: DEFAULT_BACKEND_ID ("default") on single-backend
  // clusters, the routed id once pinned, empty ONLY while a job is genuinely
  // unrouted. Never an overloaded sentinel; the UI hides the column via multiBackend.
  string backend_id = 24;
}

message JobStatus {
  // ...existing through field 33 (8/13 reserved)...
  string backend_id = 34;   // literal id (see TaskStatus.backend_id)
}
```

### controller.proto — WorkerHealthStatus + request filters + scheduler buckets
```proto
message WorkerHealthStatus {
  // ...existing through field 8...
  string backend_id = 9;     // resolved from scale_group via the controller map
  string scale_group = 10;   // was absent from the ListWorkers row
}

// All request filters are FEATURE PR. Core merges/locates unconditionally and does
// not read a request backend_id (see handler notes).
message WorkerQuery { /* ...*/ string backend_id = 6; }   // empty = all backends
message JobQuery    { /* ...*/ string backend_id = 10; }
message Controller.GetAutoscalerStatusRequest        { string backend_id = 1; }  // drill-down; empty = merge all
message Controller.GetKubernetesClusterStatusRequest { string backend_id = 1; }  // required when >1 CLUSTER_VIEW backend

message PendingTaskBucket { /* band,user,job,count... */ string backend_id = 5; }
message RunningTaskBucket { /* band,user,worker,job,count... */ string backend_id = 6; }
```
No filter on `ListTasksRequest` (job→backend is 1:1; the `TaskStatus.backend_id`
row field suffices).

### vm.proto — autoscaler view (CORE PR: this is the one core-PR wire change)
```proto
message ScaleGroupStatus {
  string name = 1;
  string backend_id = 2;   // owning backend; disambiguates the merged groups list
  // ...existing fields shift unaffected (additive)...
}
```
Do **not** tag `SliceInfo`/`VmInfo` — they nest under a `ScaleGroupStatus` that now
carries `backend_id`, and both already carry `scale_group`. One normalized join point.

### controller.proto — new ListBackends RPC (FEATURE PR)
```proto
message StringList { repeated string values = 1; }   // map values cannot be repeated

message Controller.ListBackendsRequest {}

message Controller.BackendSummary {
  string backend_id = 1;
  string name = 2;
  string kind = 3;                                    // BackendConfig.kind ("worker_daemon"/"k8s")
  repeated string capabilities = 4;                   // sorted BackendCapability.value
  map<string, StringList> advertised_attributes = 5;  // backend.advertised_attributes()
  bool restricted = 6;                                // allow policy != "*"
  int32 allowed_user_count = 7;                       // for "restricted (N)" without shipping the list
  repeated string scale_groups = 8;                   // owned scale groups
  int32 worker_count = 9;
  int32 pending_task_count = 10;
  int32 running_task_count = 11;
  bool has_autoscaler = 12;
  map<string, int32> capacity_health = 13;            // availability_status -> group count
  // No stored health field: the card's health dot is derived frontend-side from
  // capacity_health + counts. A backend-liveness field is a natural addition when
  // remote backends land (PR3).
}

message Controller.UnroutableJob {
  string job_id = 1;
  string reason = 2;   // "no backend matches the job's constraints" / "no backend permits this user"
}

message Controller.ListBackendsResponse {
  repeated BackendSummary backends = 1;
  int32 unroutable_job_count = 2;          // structured signal for the Unroutable card/banner
  repeated UnroutableJob unroutable_sample = 3;   // small sample; do not parse pending_reason strings
}

service ControllerService {
  // ...
  rpc ListBackends(Controller.ListBackendsRequest) returns (Controller.ListBackendsResponse);
}
```

## Python: ControllerProtocol widening (service.py)
```python
class ControllerProtocol(Protocol):
    # ...existing autoscaler / provider / capabilities...
    @property
    def backends(self) -> dict[str, TaskBackend]: ...
    def backend_id_for_scale_group(self, scale_group: str) -> str:
        """Owning backend id for a scale group; DEFAULT_BACKEND_ID when unmapped."""
        ...
```
`Controller` already has `backends`; add `backend_id_for_scale_group` wrapping
`_scale_group_to_backend`. `ControllerServiceImpl` also exposes a `backends`
accessor for `dashboard.py`.

## Python: handler changes

### CORE PR — correctness fixes
```python
# dashboard.py _auth_config: serve union + per-backend list, keep legacy key
backends = {bid: backend_descriptor(b) for bid, b in self._service.backends.items()}
union = sorted({c for d in backends.values() for c in d.capabilities})
rep = backend_descriptor(self._service.provider)
return JSONResponse({ ...,
    "capabilities": union,
    "backends": [{"id": bid, "name": d.name, "capabilities": d.capabilities} for bid, d in backends.items()],
    "backend": {"name": rep.name, "capabilities": rep.capabilities},  # deprecated
})

# service.get_autoscaler_status: UNCONDITIONALLY merge across all autoscalers, tag groups.
# Core reads no request.backend_id (drill-down is feature-PR). Invariant: scale-group names
# are globally unique across backends (single _scale_group_to_backend key space), so the
# per-group current_demand/recent_actions need no further disambiguation.
merged = vm_pb2.AutoscalerStatus()
for backend_id, backend in self._controller.backends.items():
    asc = backend.autoscaler
    if asc is None: continue
    sub = asc.get_status()
    for g in sub.groups: g.backend_id = backend_id
    merged.groups.extend(sub.groups)
    for k, v in sub.current_demand.items(): merged.current_demand[k] = v
    merged.recent_actions.extend(sub.recent_actions)
merged.recent_actions.sort(key=lambda a: a.timestamp, reverse=True)
del merged.recent_actions[RECENT_ACTIONS_CAP:]
merged.last_evaluation = max_last_evaluation
# last_routing_decision is a single per-autoscaler snapshot: left unset in the merged view
# (a feature-PR per-backend drill-down can surface it). Then the existing
# _overlay_worker_usability(merged, ...) is unchanged (it scans all groups/VMs).
return ...Response(status=merged)

# service.get_kubernetes_cluster_status: find CLUSTER_VIEW backend by capability.
# Core reads no request.backend_id; returns the first CLUSTER_VIEW backend by sorted id
# (today there is at most one). The feature PR adds the backend_id filter and the
# "require it when >1 CLUSTER_VIEW backend" rule.
if BackendCapability.CLUSTER_VIEW not in self._controller.capabilities:
    return ...Response()
candidates = [b for bid, b in sorted(self._controller.backends.items())
              if BackendCapability.CLUSTER_VIEW in b.capabilities]
return candidates[0].get_cluster_status() if candidates else ...Response()
```

### FEATURE PR — row tagging, filters, ListBackends
- `TASK_DETAIL_COLS` (reads.py): add `tasks_table.c.backend_id`; add `backend_id: str`
  to `TaskWithAttempts` (+ `from_row`); set `proto.backend_id` in `task_to_proto`.
- `WORKER_DETAIL_COLS` (reads.py): add `workers_table.c.scale_group`; handler stamps
  `backend_id = controller.backend_id_for_scale_group(row.scale_group)` and sets
  `WorkerHealthStatus.backend_id/scale_group` (and finally fills the long-dead
  `GetWorkerStatusResponse.scale_group`).
- `_JOB_ROW_COLUMNS` / JobRow (reads.py): add `jobs_table.c.backend_id`; set
  `JobStatus.backend_id` in `_jobs_to_protos`.
- `get_scheduler_state`: add `backend_id` to bucket keys/protos (running query must also
  select `tasks_table.c.backend_id`; `PENDING_TASK_COLS` already has it).
- `JobQuery.backend_id`: `AND jobs.backend_id == query.backend_id` in `_apply_job_filters`.
- `WorkerQuery.backend_id`: filter in `list_workers` using the in-memory scale-group→backend
  map (workers have no `backend_id` column; the roster is cached in-process).
- `ListBackends` handler: source `name/capabilities/advertised_attributes/has_autoscaler`
  from `controller.backends`; `scale_groups` by inverting `_scale_group_to_backend`;
  `restricted/allowed_users` from `backends_table.allow_policy_json`; `worker_count` from a
  grouped count over `workers.scale_group` mapped to backend; `pending/running_task_count`
  from a grouped count over `tasks(backend_id, state)` (uses `idx_tasks_backend_state`);
  `capacity_health` by rolling up `backend.autoscaler.get_status().groups[*].availability_status`.

## Frontend (FEATURE PR)

New:
- `components/controller/BackendsTab.vue` — `InfoCard` grid overview.
- `components/shared/BackendScope.vue` — `?backend=` `<select>` in the `TabNav` slot.
- `composables/useBackends.ts` — one `/auth/config` fetch; exposes `backends`,
  `multiBackend`, union `capabilities`, `currentBackend`.
- `/backends` route; `ListBackends` client call + `BackendInfo`/`BackendSummary` types in `rpc.ts`.

Extend:
- `App.vue` — gate tabs on union `capabilities`; add Backends tab + `BackendScope` slot when
  `multiBackend`; source from `useBackends`.
- `types/rpc.ts` — `backendId` on `JobStatus`/`TaskStatus`/`WorkerHealthStatus`; `backendId?`
  on `JobQuery`/`WorkerQuery`; `backendId` tag on `ScaleGroupStatus`.
- `JobsTab.vue`/`FleetTab.vue` — read `route.query.backend`, pass into the RPC query, add a
  Backend column shown only in All mode.
- `CapacityTab.vue` — per-backend `FleetOverview` sections; Pools grouped by backend; `Unroutable`
  `MetricCard`; Backend column on Pending/Unmet tables.
- `FleetOverview.vue` — optional `backendLabel` prop so it renders once per backend.
- `JobDetail.vue`/`TaskDetail.vue`/`WorkerDetail.vue` — `Backend` `InfoRow` (+ pin chip on
  JobDetail), `v-if="multiBackend"`.

## Tests
- Service: unit tests for the three correctness fixes (union capabilities; autoscaler merge
  across two fake backends with distinct scale groups → groups tagged + both present; k8s
  status finds the CLUSTER_VIEW backend when it is not representative). FEATURE PR: `ListBackends`
  rollup; `backend_id` present on row protos; `JobQuery`/`WorkerQuery` filter.
- Frontend: the existing dashboard e2e screenshot smoke (controller serves `dist/`), extended
  with a 2-backend mock to capture the overview tab + scoped Jobs view. Single-backend screenshots
  must be unchanged (the gating contract).

## Out of scope
- `SliceInfo`/`VmInfo` `backend_id` (resolved via the parent `ScaleGroupStatus`).
- `workers.backend_id` column (resolved in-memory; see design open question 3).
- Residual-demand timeseries; per-backend auth boundaries.
