# Iris User-Aware Job Names

## Problem Statement

Iris currently treats the first path segment of a job name as the job root. We
need to make the submitting user explicit so future scheduler and accounting
features can aggregate by user without introducing a parallel identifier.

This changes the canonical wire format from `/<job>/...` to `/<user>/<job>/...`.

## Canonical Format

- Root job: `/<user>/<job>`
- Child job: `/<user>/<job>/<child>`
- Task: `/<user>/<job>/<child>/<task-index>`

Examples:

- `/alice/train`
- `/alice/train/eval`
- `/alice/train/0`
- `/alice/train/eval/3`

## `JobName` Semantics

`JobName` is still the single canonical identifier for jobs and tasks, but the
first component is now metadata (`user`) rather than part of the job tree.

- `JobName.root("alice", "train")` creates `/alice/train`
- `job.user` returns `alice`
- `job.root_job` returns the root job in the hierarchy (`/alice/train`)
- `job.parent` returns `None` for `/alice/train`, `/alice/train` for `/alice/train/0`,
  and `/alice/train` for `/alice/train/eval`
- `job.depth` excludes the user segment and task index
- `job.namespace` returns `alice/train`

Actor namespace intentionally remains scoped to the root job, not just the
user, so unrelated jobs from the same user do not share actor discovery scope.

## User Inference

User identity is derived before creating a top-level `JobName`.

Resolution order:

1. Explicit override (`IrisClient.submit(..., user=...)` or `iris job run --user`)
2. Current Iris context (`get_job_info().user`) for nested submissions
3. Local OS user via `getpass.getuser()`

No separate `IRIS_USER` environment variable is needed because the user is
already encoded in `IRIS_JOB_ID`.

## Client And CLI Changes

- `IrisClient.submit(..., user: str | None = None)` accepts an explicit override
- Top-level submissions construct `JobName.root(resolved_user, name)`
- Nested submissions keep inheriting the parent job's user
- `iris job run` accepts `--user`
- Auto-generated leaf job names no longer embed the OS user, since the user is
  already present in the `JobName` prefix

## Dashboard And RPC Surface

The controller exposes live user aggregates over RPC:

- `ListUsers` returns one `UserSummary` per user
- `GetClusterSummary` includes `total_users`

The controller dashboard now has a `/users` entrypoint and a `users` tab that
renders:

- user name
- active jobs
- running jobs
- pending jobs
- total tasks
- running tasks
- completed tasks

## Future Work

`ControllerState.list_user_stats()` is the intended aggregation seam for future
per-user resource tracking, scheduler guidance, and quota/accounting features.
That work should build on the existing user-aware `JobName` model rather than
introducing another source of truth.
