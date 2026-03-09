---
name: scrub-reflection-self-improvement
description: Scheduled scrub workflow for ongoing self-improvement in the Marin repository.
schedule_cron: "0 10 2 * * *"
schedule_tz: America/New_York
---

# scrub-reflection-self-improvement

Use this skill on scheduled scrub turns to identify and land high-leverage improvements in `marin-community/marin`.

## Focus

- Look for improvements from recent issues, PR feedback, and recurring operational friction.
- Prefer one concrete implementation per run when feasible.
- If implementation is blocked, produce a concrete plan and capture follow-up work in GitHub.

## Candidate Signals

- Repeated confusion in docs, recipes, or contributor workflows.
- Recurring failures or avoidable manual steps in experiments, scripts, and infra operations.
- Capability gaps that reduce the value of agent-assisted contributions.

## Decision Heuristics

- Pick the highest-leverage change with the lowest coordination overhead.
- De-duplicate against existing issues/PRs before opening new work.
- When an improvement changes recurring workflow guidance, codify it in durable repo instructions:
  `AGENTS.md` for cross-cutting agent behavior, or `.agents/skills/` for repeatable task workflows.
- If no justified improvement exists now, choose a no-op outcome.

## Output

- Keep rationale explicit: observed gap, change made (or plan), and expected impact.
- Prefer durable artifacts over transient notes: land guidance updates in `AGENTS.md` and/or recipe docs when that is the primary improvement.
- Treat local-only edits as incomplete work. If you modify files, publish the result (commit/push and open or update a PR) before finishing this scrub run.
- If publish is blocked (auth, permissions, CI infra, etc.), report the blocker and set a future `needs_followup_at` instead of ending the run.
- If you choose no-op, include explicit inspected signals and why no justified improvement exists now.
- Always end with the required `HARNESS_SCRUB_LOOP` footer (provided by the base scrub contract).
