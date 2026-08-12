---
name: noslop
description: Review and clean a complete branch diff before PR publication, with strict gates for low-value tests and agentic prose. Use when asked to deslop, simplify, clean up, make a diff minimal, review test quality, tighten comments/docs/PR text, or remove AI-writing patterns such as rhetorical "X, not Y" contrasts. Apply fixes when the task authorizes code changes; otherwise report concrete findings.
---

# No-Slop Review

## Load The Policies

Read these before reviewing:

- root `AGENTS.md`
- root `TESTING.md`
- `.agents/skills/writing-style/ai-writing-donts.md`
- the nearest module `AGENTS.md` and testing guide for each changed test

Read [references/examples.md](references/examples.md) when the diff changes tests
or prose.

## Establish The Review Surface

Review the merge-base diff. Do not substitute commit history or the latest
commit.

```bash
BASE=$(git merge-base HEAD origin/main)
git status --short
git diff --stat "$BASE"
git diff --check "$BASE"
git diff "$BASE"
uv run python .agents/skills/noslop/scripts/scan_diff.py "$BASE"
```

List every changed test, comment, docstring, Markdown file, commit message, and
draft PR title/body. Read the surrounding production code and existing tests.
The scanner includes untracked files and returns candidates; inspect each one in
context.

## Gate Tests By Behavioral Value

For every changed test, finish this sentence:

> This test fails when ___ user-visible or system behavior is wrong.

Delete or rewrite the test when the blank names an implementation detail,
wording choice, supplied fixture value, or language guarantee.

Reject these patterns aggressively:

- config in, same config out: a constructed or lowered field equals the fixture
- constructor assignment, enum value, type, attribute, registration, or import exists
- exact log, help, error, command, or status prose that no machine consumes
- private state, helper call count, or `assert_called_once_with`
- production logic copied into the expected value
- `is not None`, `does not raise`, empty smoke tests, or permanent skips
- a large matrix whose rows exercise the same branch
- golden output produced by the implementation under test

Keep a test only when it protects a real boundary, regression, invariant,
round-trip, state transition, persisted effect, wire contract, or numerical
result against an independent reference.

Prefer one strong test over several weak tests. Use scratch probes for wiring
and construction checks. Do not check them in.

## Delete Prose Slop

Review every added or edited sentence in comments, docstrings, docs, reports,
commit text, and PR text.

Delete sentences that add no fact, result, decision, constraint, or caveat.
Apply the full checklist in `ai-writing-donts.md`. Search explicitly for:

- `X, not Y`, `not X, but Y`, `not just X`, `more than X`
- `rather than` used as rhetorical contrast
- `the main change`, `what this means`, `the real story`, `what matters`
- `importantly`, `notably`, `it is worth noting`, `stepping back`
- effort narration, section narration, recap paragraphs, and generic conclusions
- unsupported adjectives and claims that outrun measurements
- comments that describe an earlier version of the patch

Rewrite retained prose as the concrete claim. Split a real comparison into two
measured statements. Keep necessary negation such as `the test has not run`.

Check prose against code and artifacts. A concise false claim is still slop.

## Review The Diff As A Design

After the test and prose passes, inspect each new file, helper, type, flag,
dependency, compatibility path, and public API:

1. Name its current consumer.
2. Remove it when its consumer disappeared.
3. Reuse an existing repository concept when it fits.
4. Complete migrations; do not leave old and new paths active.
5. Keep one-shot migration and debugging machinery outside reusable libraries.
6. Remove stale comments, aliases, re-exports, shims, and defensive branches.

## Fix And Verify

When edits are authorized, make the cleanup. Do not stop at a findings table
while safe fixes remain.

Run the narrow affected tests, then:

```bash
./infra/pre-commit.py --changed-files --fix
```

Run `./infra/pre-commit.py --review` only at the point required by the commit
workflow. Do not claim a clean pass while known findings remain.

## Report

Lead with what was removed or simplified. Report retained exceptions with their
behavioral contract or evidence. If no edits were authorized, report findings
as `path:line`, the maintenance cost, and the smallest fix.
