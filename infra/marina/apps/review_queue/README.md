# Review queue

This Marina app shows open `marin-community/marin` pull requests grouped by
GitHub's pending individual review requests. It also shows team requests and PRs
without a requested reviewer. Counts are requests, not required approvals or
reviewer availability. PR age uses the PR creation time, not the time a review
was requested.

The browser reads GitHub's public pull request API on page load and on manual
refresh. No GitHub token or backend is required. The request is subject to
GitHub's unauthenticated API rate limit, and the page reports API failures.

Validate the package with:

```bash
uv run marina validate infra/marina/apps/review_queue
uv run marina release infra/marina/apps/review_queue --dry-run --json
```

Release revisions with `marina release` following the Marina app guide in
`docs/references/marina.md`.
