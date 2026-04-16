---
name: scrub-docs-code-parity
description: Scheduled scrub workflow for docs-code parity in the Marin repository.
schedule_cron: "0 0 2 * * *"
schedule_tz: America/New_York
---

# scrub-docs-code-parity

Use this skill on scheduled scrub turns for docs/code parity in `marin-community/marin`.

## Focus

- Prioritize high-confidence drift in `README.md`, `docs/`, `.agents/docs/`, and operator-facing scripts.
- Confirm command examples match current tooling conventions (`uv run`, repo scripts, and documented workflows).
- Apply concrete corrections when drift is real; avoid status-only updates.

## Decision Heuristics

- Prefer updating docs when current behavior is clear and intentional.
- If implementation is clearly wrong relative to documented intent, update code and docs together.
- Keep scope small and land one useful parity improvement per run when possible.
- If no material drift is found, choose a no-op outcome and keep cadence near daily.

## Output

- Keep the final scrub response concise and action-focused.
- Treat local-only edits as incomplete work. If you modify files, publish the result (commit/push and open or update a PR per `.agents/skills/pull-request/SKILL.md`) before finishing this scrub run.
- If publish is blocked (auth, permissions, CI infra, etc.), report the blocker and set a future `needs_followup_at` instead of ending the run.
- If no material drift is found, explicitly report inspected scope and why no change was needed.
- Always end with the required `HARNESS_SCRUB_LOOP` footer (provided by the base scrub contract).
