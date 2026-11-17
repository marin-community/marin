# scrub-docs-code-parity

Use this procedure when explicitly assigned a docs/code parity scrub in `marin-community/marin`.

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
- Treat local-only edits as incomplete work. If you modify files, publish the result (commit/push and open or update a PR per `.agents/skills/manage-pr/SKILL.md`) before finishing this scrub run.
- If publish is blocked (auth, permissions, CI infra, etc.), report the blocker and set a future `needs_followup_at` instead of ending the run.
- If no material drift is found, explicitly report inspected scope and why no change was needed.
- End the run with exactly one footer line of valid one-line JSON: `HARNESS_SCRUB_LOOP {"needs_followup_at":null}`. Set `needs_followup_at` to null when the run is complete, or a future RFC 3339 timestamp when another follow-up turn is needed.
