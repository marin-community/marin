# Review queue

The queue groups open Marin pull requests by currently requested reviewers and
reviewers who have submitted a review. A new request takes precedence over a
previous review. Star a GitHub handle to pin it in this browser.

Marina serves the last complete open PR snapshot from Postgres. While the page
is open, it calls `POST ./api/sync` in the background. The endpoint refreshes
the open PR list and the 100 most recently updated PRs at most once every 15
minutes. It fills in detail and timeline data for up to four missing or changed
PRs per call. Pending and recently updated PRs have priority over older open
PRs. Each call has a 20 second budget for upstream work. One Postgres
transaction advisory lock coordinates callers across instances. Other callers
keep reading the cache and can retry. Completed work is saved after a GitHub
error or timeout, and GitHub rate limit headers set a shared retry time.

Pending lines are additions plus deletions on open PRs with known diff sizes.
The Reviewer load view compares pending lines and weekly reviewed lines, with
sortable columns and the underlying PRs in each row. Historical metrics use
the available reviews from the last 14 days. The average
response pairs an individual review request with a submitted review. Lines per
week uses the latest known diff size of PRs a person reviewed, counted once per PR
and divided by two. This is a volume estimate based on the synced sample.

Validate or preview the package with:

```bash
uv run marina validate infra/marina/apps/review_queue
uv run marina release infra/marina/apps/review_queue --local --json
```

Release revisions using the [Marina app guide](../../../../docs/references/marina.md).
