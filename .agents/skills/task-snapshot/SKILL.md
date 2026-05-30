---
name: task-snapshot
description: "Create stable commit or tag snapshots for task milestones, experiment results, and reproducible artifact links."
---

# Skill: Task Snapshot

Use this when a result, milestone, or handoff needs stable references that will
remain valid as a branch keeps moving.

## Snapshot Choices

- Use a commit link for ordinary task milestones.
- Use an annotated tag for meaningful experiment checkpoints, published numbers,
  or long-lived handoffs.
- Include only relevant files in the snapshot commit.

## Snapshot Comment

When posting a snapshot to an issue or PR, include:

- Commit or tag link.
- Pinned GitHub tree/file links for logbooks, benchmark outputs, configs, and
  reports.
- Exact command(s).
- Hardware/cluster and device count.
- Critical environment variables.
- Primary comparison table or a link to W&B/report artifacts.

Pinned GitHub tree links should include the commit or tag, for example:

```text
https://github.com/marin-community/marin/tree/<commit-or-tag>/.agents/logbooks/foo.md
```

## Validation

Before presenting a snapshot as evidence:

- The command is reproducible from the snapshot.
- Shapes/configs and environment are explicit.
- The logbook and issue link to the same result.
- Dense data has an artifact or W&B link.
