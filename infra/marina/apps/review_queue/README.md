# Review queue

The queue groups open Marin pull requests by currently requested reviewers and
reviewers who have submitted a review. A new request takes precedence over a
previous review. Draft PRs are excluded. Open a reviewer to see PRs that need a
first review and PRs that already have feedback or approval from someone else.
Star a GitHub handle to pin it in this browser.

Marina serves the last complete open PR snapshot from Postgres. While the page
is open, it calls `POST ./api/sync` in the background. The endpoint refreshes
the open PR list and discovers every PR updated within the review metrics
window, currently seven days. After the initial scan, it pages through changes
since the last scan and keeps known PRs still in the window. Discovery runs at
most once every 15 minutes. Each call fetches detail and timeline data for up to
four missing or changed PRs, with a 20 second budget for upstream work. One Postgres
transaction advisory lock coordinates callers across instances. Other callers
keep reading the cache and can retry. Completed work is saved after a GitHub
error or timeout, and GitHub rate limit headers set a shared retry time.

Pending lines are additions plus deletions on open PRs with known diff sizes.
The Reviewer load column sums current diff sizes across the PRs a person
reviewed in the window, counting each PR once. The average response pairs an
individual review request with a submitted review in the same window. Expand a
reviewer to see their open PRs and the recent reviews behind the metrics.
The sync widget shows progress as PR details become available.

Validate or preview the package with:

```bash
uv run marina validate infra/marina/apps/review_queue
uv run marina release infra/marina/apps/review_queue --local --json
```

Release revisions using the [Marina app guide](../../../../docs/references/marina.md).
