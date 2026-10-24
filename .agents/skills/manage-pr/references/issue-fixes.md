# Fix an assigned GitHub issue

Use this procedure only for an explicitly identified issue. Read the issue,
root and scoped instructions, and the relevant source. Follow
`docs/dev-guide/writing-style/issues.md` for issue prose.

Before implementation, check open PRs for the issue or subsystem and open issues
for the same root cause. If work overlaps, hand off to the existing PR or scope
a distinct follow-up. Do not create a parallel implementation.

Post one concise issue comment with the confirmed cause, annotated code links,
and the proposed fix. Use `Research` and `Proposed Fix` sections when helpful.
Keep working evidence in the issue or durable session channel.

Implement on `agent/{YYYYMMDD}-fix-{issue-id}`. Follow root `TESTING.md` and
the scoped testing guides; prefer a regression that fails before the fix.
The [manage-pr workflow](../SKILL.md) owns validation, commit and PR text,
publication, and monitoring. Add `Fixes #NNNN` when the PR resolves the issue.

Link the published PR from the issue. At closure, post the landed PR or the
terminal state. If blocked, report the last completed state and blocker.
