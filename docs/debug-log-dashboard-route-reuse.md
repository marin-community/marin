# Debugging log for dashboard route reuse

Route-reuse changes in the Iris dashboard left a few stale-state paths open. The goal here is to close the remaining races without changing the intended loading and refresh behavior.

## Initial status

The dashboard already cleared stale task and worker data on route changes, and it fenced `useRpc()` / `JobDetail.vue` with generation counters. Two gaps remained:

1. `LogViewer` only read `currentAttemptId` once, so same-task retries could leave it pinned to the previous attempt.
2. `useRpc()` still assigned `data.value` immediately after `await resp.json()`, so a superseded request could still commit after body parsing.

## Hypothesis 1

`LogViewer` needs to react to `currentAttemptId` changes only when it is implicitly following the current attempt or showing all attempts. It should not override a deliberate manual selection of an older attempt.

## Changes to make

Update `lib/iris/dashboard/src/components/shared/LogViewer.vue` to watch `taskId` and `currentAttemptId` together:

- Reset to `-1` on task switches.
- Refresh when the viewer is showing all attempts.
- Advance to the new current attempt only when the local selection still matches the previous current attempt.

## Future Work

- [ ] Add dashboard component tests for task retry transitions.
- [ ] Add dashboard component tests for superseded refreshes during route reuse.

## Results

`LogViewer` now preserves manual selection but still follows retries when it was already tracking the active attempt. Task switches still reset to `All attempts`.

## Hypothesis 2

The generation check in `useRpc()` must run after every `await`, not just after `fetch()`, because `fetch()` resolves before the body is fully consumed.

## Changes to make

Update `lib/iris/dashboard/src/composables/useRpc.ts` to store the parsed payload in a local variable, re-check the generation counter after `await resp.json()`, and only then assign `data.value`.

## Results

Superseded `useRpc()` calls are now discarded both after `fetch()` and after body parsing, which closes the remaining stale-response window for the reusable dashboard views that rely on `useRpc()`.

## Verification

- `./infra/pre-commit.py --fix lib/iris/dashboard/src/composables/useRpc.ts lib/iris/dashboard/src/components/shared/LogViewer.vue lib/iris/dashboard/src/components/controller/JobDetail.vue lib/iris/dashboard/src/components/controller/TaskDetail.vue lib/iris/dashboard/src/components/controller/WorkerDetail.vue docs/debug-log-dashboard-route-reuse.md`
- `npm run build:check` in `lib/iris/dashboard/` still fails on existing unrelated errors in `src/components/controller/AutoscalerTab.vue` (`VmInfo.runningTaskCount`), so there is no clean project-wide TypeScript/build signal yet.
