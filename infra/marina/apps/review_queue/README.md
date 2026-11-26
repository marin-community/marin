# Review queue

The queue shows pending review requests for open Marin pull requests. Star a
GitHub handle to pin it in this browser, even when that person has no requests.

Marina serves the last complete open PR snapshot from Postgres. While the page
is open, it calls `POST ./api/sync` in the background. The endpoint refreshes
the open PR list and the 100 most recently updated PRs at most once every 15
minutes. It fills in detail and timeline data for up to four missing or changed
PRs per call. Each call has a 20 second budget for upstream work. One Postgres
transaction advisory lock coordinates callers across instances. Other callers
keep reading the cache and can retry. Completed work is saved after a GitHub
error or timeout, and GitHub rate limit headers set a shared retry time.

Pending lines are additions plus deletions on open PRs with known diff sizes.
Historical metrics use the available reviews from the last 14 days. The average
response pairs an individual review request with a submitted review. Lines per
week uses the latest known diff size of PRs a person reviewed, counted once per PR
and divided by two. This is a volume estimate based on the synced sample.

Validate or preview the package with:

```bash
uv run marina validate infra/marina/apps/review_queue
uv run marina release infra/marina/apps/review_queue --local --json
```

Release revisions using the [Marina app guide](../../../../docs/references/marina.md).
