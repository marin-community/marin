# Patch Nested Jobs In Iris Job Detail

## Problem

The Iris dashboard job detail page does not show nested RL child jobs such as
`train` and `rollout` for executor-wrapped runs, even though those jobs exist
and are visible via CLI.

Concrete repro from the successful `E3b` run:

- Root job:
  - `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934`
- Executor child:
  - `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3`
- RL coordinator:
  - `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3/rl-exec-gcs-small-20260324-212413`
- Train child:
  - `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3/rl-exec-gcs-small-20260324-212413/rl-exec-gcs-small-20260324-212413-train`
- Rollout child:
  - `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934/rl_testing-exec-gcs-small-20260324-212413_db03625d-642dd6e3/rl-exec-gcs-small-20260324-212413/rl-exec-gcs-small-20260324-212413-rollout-0`

CLI sees the full tree correctly:

```bash
uv run iris --config=lib/iris/examples/marin.yaml job list --json \
  --prefix /ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934
```

The dashboard does not flatten or recursively render that subtree on the job
detail page. Users therefore land on a page that says the executor step
`SUCCEEDED`, but still cannot see the actual nested trainer and rollout jobs
without manually drilling into deeper job IDs.

The current frontend behavior is caused by
[JobDetail.vue:81-94](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/dashboard/src/components/controller/JobDetail.vue:81),
which fetches `ListJobs(nameFilter=jobName)` and then filters to direct
children only:

```ts
const prefix = jobName + '/'
childJobs.value = (childResp.jobs ?? []).filter(j => {
  if (!j.name.startsWith(prefix)) return false
  const suffix = j.name.slice(prefix.length)
  return suffix.length > 0 && !suffix.includes('/')
})
```

That `!suffix.includes('/')` line is the functional source of the issue.

This is mismatched with the controller contract. The backend already returns
descendants so the dashboard can build a tree:

- [service.py:1021-1025](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/src/iris/cluster/controller/service.py:1021)
- [test_service.py:870-892](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/tests/cluster/controller/test_service.py:870)

The jobs list page already understands how to build a nested job tree from the
same `ListJobs` response:

- [JobsTab.vue:107-150](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/dashboard/src/components/controller/JobsTab.vue:107)

So the data path already exists. The detail page is just discarding it.

## Goals

- Show nested descendant jobs on the Iris job detail page.
- Make executor-wrapped RL runs understandable from the UI.
- Reuse existing controller behavior; avoid backend RPC changes if possible.
- Preserve current detail-page task/log behavior for the selected job.

Non-goals:

- Changing job submission topology.
- Changing the backend `ListJobs` API.
- Reworking the entire jobs dashboard.
- Hiding intermediate executor/coordinator jobs; those should remain visible.

## Proposed Solution

Replace the job detail page's direct-child-only filtering with the same
tree-building approach already used by `JobsTab.vue`.

Recommended UI behavior:

- Keep the existing `CHILD JOBS` section, but render it as a nested expandable
  tree of descendants under the current job.
- Show direct child rows first.
- Allow expansion into grandchildren and deeper descendants.
- For linear chains like `root -> executor -> coordinator -> train/rollout`,
  the user should be able to reach train/rollout from the same detail page
  without manual router navigation.

No backend change is required. `ListJobs(nameFilter=jobName)` already returns
the descendant jobs necessary to build the tree.

Core idea:

```ts
const descendantJobs = ref<JobStatus[]>([])

const flattenedChildJobs = computed(() => {
  const jobList = descendantJobs.value
  const jobByName = new Map(jobList.map(j => [j.name, j]))
  const childrenMap = new Map<string, JobStatus[]>()
  const roots: JobStatus[] = []

  for (const child of jobList) {
    const parent = getParentName(child.name)
    if (parent && jobByName.has(parent)) childrenMap.set(parent, [...(childrenMap.get(parent) ?? []), child])
    else roots.push(child)
  }

  return flattenExpandedTree(roots, childrenMap, expandedChildJobs.value)
})
```

The important change is conceptual, not syntactic:

- stop throwing away descendants in `JobDetail.vue`
- treat the detail page's child-jobs section as a subtree view

## Implementation Outline

1. In [JobDetail.vue](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/dashboard/src/components/controller/JobDetail.vue), replace `childJobs` direct-child filtering with storage of the full descendant set returned by `ListJobs`.
2. Reuse or copy the tree helpers from [JobsTab.vue](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/lib/iris/dashboard/src/components/controller/JobsTab.vue): `getParentName`, `getLeafName`, `childrenMap`, flattened tree, and expansion state.
3. Render descendant rows with indentation and expand/collapse affordances instead of a flat direct-child table.
4. Ensure links still navigate to the selected descendant job's detail page.
5. Add a frontend test that proves a root job detail page can show a grandchild or deeper descendant in the rendered child-jobs tree.
6. Verify the existing task table and log viewer remain scoped to the selected job, not the expanded descendants.

## Notes

- The current behavior is not a cluster or controller bug. The jobs existed and
  succeeded. Only the detail-page presentation was incomplete.
- The jobs list page already solves the same structural problem; the detail
  page should align with that behavior rather than inventing another model.
- A smaller fallback fix would be to rename the section to `DIRECT CHILD JOBS`,
  but that does not solve the actual usability problem for nested RL runs.
- For the `E3b` case, the desired visual tree on the root page is:

```text
rl_testing-exec-gcs-small-...
  rl-exec-gcs-small-...
    rl-exec-gcs-small-...-train
    rl-exec-gcs-small-...-rollout-0
```

- There may also be refresh/staleness issues while jobs are live, but the
  first bug to fix is the deterministic one-level filter in `JobDetail.vue`.

## Acceptance Criteria

- On the root detail page for `/ahmed/iris-rl-e3b-exec-gcs-small-20260324-141934`,
  the UI can reveal the nested `train` and `rollout` jobs from the `CHILD JOBS`
  section without requiring manual typing of deeper job URLs.
- The root detail page still shows only the root task table/logs for the root job.
- The executor child detail page can reveal the coordinator and its descendants.
- No backend RPC change is required for the patch to work.
- Existing `JobsTab` tree behavior remains unchanged.

## Future Work

- Factor the job-tree code into a shared composable/component so `JobsTab` and
  `JobDetail` do not maintain parallel implementations.
- Consider auto-expanding single-child chains to reduce click depth for deeply
  nested executor jobs.
- If users still find the detail page confusing, add a toggle for:
  - `Direct children only`
  - `All descendants`
