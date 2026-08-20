---
name: scrub-docs-code-parity
description: "Scheduled scrub: docs and code parity."
schedule_cron: "0 0 2 * * *"
schedule_tz: America/New_York
---

# Docs/code parity scrub

On scheduled turns, find high-confidence drift in README.md, docs/,
.agents/docs/, and operator-facing scripts in marin-community/marin. Check
commands against current tooling (uv run, repository scripts, and documented
workflows), and make a concrete correction when drift is real.

Prefer docs when current behavior is intentional; update code and docs together
when code is clearly wrong. Keep the scope to one useful improvement. If no
material drift exists, report the inspected scope and choose a no-op.
Code is the source of truth when documented intent does not establish that the
implementation is wrong.

Local edits are incomplete: publish via .agents/skills/commit/SKILL.md. If
publishing is blocked, report the blocker and set a future follow-up time.
Finish with exactly:

~~~text
HARNESS_SCRUB_LOOP {"needs_followup_at":null}
~~~

Use a future RFC 3339 timestamp in needs_followup_at when another turn is needed.
