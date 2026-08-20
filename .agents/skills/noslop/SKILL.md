---
name: noslop
description: Remove low-value tests, redundant prose, and unnecessary design from a branch. Use for cleanup, simplification, minimal-diff review, test-quality review, or AI-writing cleanup.
---

# No-Slop Review

Read root `AGENTS.md`, root `TESTING.md`, and
[the AI-writing checklist](../writing-style/ai-writing-donts.md). For changed
tests, read the nearest module instructions and
[the examples](references/examples.md).

## Review The Branch Diff

Review the merge-base diff, including untracked files:

```bash
UPSTREAM=$(git rev-parse --verify origin/main 2>/dev/null || git rev-parse --verify main)
BASE=$(git merge-base HEAD "$UPSTREAM")
git status --short
git diff --stat "$BASE"
git diff --check "$BASE"
git diff "$BASE"
uv run python .agents/skills/noslop/scripts/scan_diff.py "$BASE"
```

The scanner returns candidates, not verdicts. Inspect each changed test,
comment, docstring, Markdown file, commit message, and draft PR in context.

## Test Gate

For every changed test, complete: `This test fails when ___ user-visible or
system behavior is wrong.` Delete or rewrite tests whose answer is an
implementation detail, supplied fixture value, wording choice, or language
guarantee.

Reject constructor and registration checks, config round-trips, private-state
and helper-call assertions, exact human-facing prose, copied production logic,
self-generated goldens, empty smoke tests, permanent skips, and redundant
matrices. Keep coverage for real regressions, public boundaries, invariants,
state transitions, persisted effects, wire contracts, and numerical results
against an independent reference. Prefer one strong test over several weak
ones; keep wiring probes out of the repository.

## Prose Gate

Delete every sentence that adds no fact, result, decision, constraint, or
caveat. Apply the full AI-writing checklist, especially to rhetorical
contrasts, bridge phrases, importance claims, effort narration, recaps,
unsupported adjectives, and comments describing an earlier patch. Check every
retained claim against code and artifacts.

## Design Gate

For each added file, helper, type, flag, dependency, compatibility path, and
public API, identify its current consumer. Remove unused machinery, reuse a
repository concept when it fits, complete migrations, and delete stale aliases,
re-exports, shims, comments, and defensive branches. Keep one-shot migration or
debugging code out of reusable libraries.

When edits are authorized, make them. Run narrow affected tests followed by
`./infra/pre-commit.py --changed-files --fix`. Run the advisory review only at
the point required by the commit workflow.

Report what was removed or simplified and the behavioral contract behind any
retained exception. Without edit authority, report `path:line`, maintenance
cost, and the smallest fix.
