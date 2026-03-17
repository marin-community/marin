---
name: handle-gh-scrub
description: Base execution contract for scheduled scrub runs in marin-community/marin.
---

# handle-gh-scrub

Use this skill for scheduled scrub turns.

## Workflow

1. Read the requested `scrub-*` skill and apply it for this run.
   - When inspecting GitHub issues or PRs during triage, use `gh ... --json <fields>` or other explicit narrow flags instead of plain `gh issue view` / `gh pr view`.
2. Decide whether useful progress is available now.
3. If useful work exists, implement and validate it.
4. Publish outcomes before ending the run:
   - commit/push to a branch
   - open/update a PR when code or docs changed (follow `.agents/skills/pull-request/SKILL.md`)
5. If publish is blocked (token/permissions/infra), keep the run open and schedule follow-up.
6. If no-op is correct, explicitly report inspected scope and why no change was justified.

## Completion Contract

- Do not end a run with uncommitted local changes.
- `HARNESS_SCRUB_LOOP {"needs_followup_at":null}` is only valid when either:
  - a visible artifact exists (commit/branch/PR), or
  - a justified no-op outcome is recorded.
- If work is partially done or blocked, set a future `needs_followup_at`.

## Required Footer

Your response must end with exactly one line in this format:

```text
HARNESS_SCRUB_LOOP {"needs_followup_at":null}
```

Rules:

- Keep JSON valid and on one line.
- Set `needs_followup_at = null` when the current scrub run is complete.
- Use a future RFC 3339 timestamp (with timezone/offset) when this run needs another follow-up turn.
