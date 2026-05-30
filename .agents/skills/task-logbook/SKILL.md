---
name: task-logbook
description: "Maintain append-only logbooks for atomic tasks, experiments, investigations, and research threads."
---

# Skill: Task Logbook

Use this when a task needs durable context across sessions, reproducible
experiment notes, or a compact handoff trail. For long-running research,
`run-research` composes this skill with issue, W&B, docs, and snapshot skills.

## Location and Naming

- Store logbooks under `.agents/logbooks/<topic>.md`.
- Use a short, stable topic slug that can survive branch renames.
- Use the term **research logbook** for research threads and **task logbook** for
  non-research work.
- Link to the coordinating GitHub issue or PR near the top when one exists.

## Write Rules

- Append entries; do not rewrite history except to fix formatting or broken
  links.
- Each non-trivial result needs exact commands, config, key output, and the
  decision it caused.
- Record failures and negative results with enough detail to avoid rediscovery.
- Prefer terse tables for comparable numeric results.
- Link large artifacts, W&B runs, dashboards, or pinned GitHub paths instead of
  pasting dense output.

## Template

```md
# <Topic>: Task Logbook

## Scope
- Goal:
- Primary metric(s):
- Constraints:
- Coordinating issue/PR:

## Baseline
- Date:
- Code refs:
- Baseline numbers:

## Entry Log
### YYYY-MM-DD HH:MM - <short label>
- Hypothesis:
- Command:
- Config:
- Result:
- Interpretation:
- Next action:
```

For research series, add a short experiment ID prefix such as `MOE-HC` and use
IDs like `MOE-HC-001` in logbook entries, W&B run names, and issue comments.
