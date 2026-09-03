---
name: fix-issue
description: Implement and land a fix for an explicitly identified marin-community/marin GitHub issue.
---

# Fix a GitHub issue

Read `AGENTS.md` and the issue guidance in `writing-style`. Keep issue comments
terse: omit filler and repetition, use annotated links, and keep each prose
section to three or four sentences. Complete the workflow in order. If blocked,
comment on the issue with the last completed state and the blocker.

## Research

Use `gh` to fetch the issue. Read the codebase for all relevant source files.
Post one concise comment with `Research` and `Proposed Fix` sections.

Before writing code, run a duplicate-work preflight:

- Check open PRs for the same issue or subsystem (`gh pr list --state open --search "<issue-id or keyword>"`).
- Check open issues for the same root cause (`gh issue list --state open --search "<keyword>"`).
- If overlapping work exists, do not open a parallel implementation PR. Add a
  short issue comment summarizing what you found, and either hand off to the
  existing PR or scope your change to non-overlapping follow-up work.

In `Research`, state the cause in two or three sentences and link at most five
relevant code locations with short annotations. In `Proposed Fix`, name the
smallest fix and include the failing call chain or a snippet only when needed.

## Implementation

Implement on `agent/{YYYYMMDD}-fix-{issue-id}`.

## Testing

Follow `write-tests` and `TESTING.md`. Prefer a regression test that fails before
the fix, extend an existing file, run the narrow test and affected safe suite,
then run `./infra/pre-commit.py --all-files --fix`. Do not override the configured
marker expression.

## Uploading

When all tests pass, upload your branch and open a PR following
`.agents/skills/commit/SKILL.md` **exactly** — use the plain-text format it
specifies (no markdown headers, bullet lists, or `## Summary` sections;
violations are rejected). Attach a comment to the Github issue summarizing the fix.

## Monitor the PR

After opening the PR, follow step 9 of `.agents/skills/commit/SKILL.md`
exactly. Its `wait_for.py` loop owns CI, review feedback, and lifecycle events.
Do not start a separate `gh pr view` or `gh pr checks` polling loop. Investigate
failures, push fixes, and re-arm the wait as that skill specifies.

Post a final issue comment with the landed PR or the last terminal state.
