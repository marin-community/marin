# Iris JobQuery Job Tree

## Problem

The dashboard was building job trees from a flat `ListJobs` response and
inferring hierarchy from slash-delimited names in
`lib/iris/dashboard/src/components/controller/JobsTab.vue` and
`lib/iris/dashboard/src/components/controller/JobDetail.vue`.

That conflicted with the wider `ListJobs` contract used by
`lib/iris/src/iris/client/client.py`, which expects broad job listing behavior
rather than dashboard-specific tree semantics. The result was fragile nested
expansion logic and no explicit server contract for direct child queries.

## Approach

Keep `ListJobs` as the single RPC, but add a typed `JobQuery` envelope in
`lib/iris/src/iris/rpc/controller.proto`:

- `scope=ALL` preserves the current client behavior
- `scope=ROOTS` returns top-level jobs for the dashboard jobs table
- `scope=CHILDREN` returns direct children for a parent job

Also add `has_children` to `iris.job.JobStatus` in
`lib/iris/src/iris/rpc/job.proto` so the UI does not guess whether an expand
button should exist.

## Key Code

```python
if scope == controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS:
    jobs, total_count = _jobs_paginated(...)
elif scope == controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN:
    if not parent_job_id:
        raise ConnectError(Code.INVALID_ARGUMENT, ...)
    jobs, total_count = _child_jobs_paginated(...)
else:
    jobs, total_count = _jobs_all_filtered(...)

has_children = _parent_ids_with_children(self._db, [j.job_id for j in jobs])
all_jobs = self._jobs_to_protos(
    jobs, task_summaries, autoscaler_pending_hints, has_children=has_children
)
```

On the frontend, `JobsTab.vue` and `JobDetail.vue` now cache loaded child rows
by parent job id and track expanded state by job id.

## Tests

- `tests/cluster/controller/test_service.py -k list_jobs`
  covers legacy all-jobs behavior and new `ROOTS` / `CHILDREN` scopes.
- `tests/cluster/controller/test_dashboard.py -k "ListJobs or list_jobs"`
  covers the dashboard JSON/RPC surface.
- `lib/iris/tests/e2e/test_smoke.py::test_dashboard_job_expand_nested`
  still validates nested expansion behavior through the UI path.
