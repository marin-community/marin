---
name: task-docs
description: "Update documentation, runbooks, and reusable task guidance when implementation work or experiments change behavior or operational practice."
---

# Skill: Task Docs

Use this when a task changes behavior, reveals stale instructions, creates a new
operational pattern, or produces guidance that should survive outside the issue
thread.

## What to Update

- User-facing docs when behavior or configuration changes.
- Operational docs when commands, service setup, dashboards, alerts, or recovery
  procedures change.
- Skill docs when an agent workflow becomes reusable.
- Experiment reports or indexes when the work produces durable research
  findings.

Prefer editing the nearest existing doc over adding a new one. Add a new doc
only when the topic lacks a natural home or would make an existing page
unfocused.

## Content Rules

- Describe current behavior, not aspiration.
- Include exact commands and paths for operational procedures.
- Link to issues, PRs, W&B reports, logbooks, or pinned code when they are part
  of the evidence trail.
- Remove or correct stale instructions rather than adding caveats around them.
- Keep broad docs concise; put detailed run notes in logbooks or issue comments.

## Done Criteria

- Docs and code agree for the changed behavior.
- The next agent or human can find the authoritative procedure without reading
  the whole issue thread.
- Any generated guidance is generalized enough to reuse and specific enough to
  act on.
