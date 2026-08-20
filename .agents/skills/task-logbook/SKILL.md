---
name: task-logbook
description: "Maintain append-only task/research logbooks and publish the important parts upward into coordinating GitHub issues."
---

# Task Logbook

Use when work needs durable cross-session context, reproducible experiment
notes, or a compact handoff. Without a coordinating issue, append to the
logbook; with one, update both logbook and issue. The logbook is the detailed,
append-only layer; the issue receives significant updates and durable summary.

## Location and initial shape

Store `.agents/logbooks/<topic>.md` with a short stable slug and link the issue
or PR near the top when one exists. New files use YAML frontmatter and sections
for scope, baseline, and entry log:

```md
---
topic:
issue:
description:
author:
---
# <Topic>: Task Logbook
## Scope
- Goal: | Primary metric(s): | Constraints: | Coordinating issue/PR:
## Baseline
- Date: | Code refs: | Baseline numbers:
## Entry Log
### YYYY-MM-DD HH:MM - <label>
- Hypothesis: | Commit Hash: | Command: | Config: | Result:
- Interpretation: | Next action:
```

Use the requester's identity as `author`; omit `issue` when none exists. For a
research series, use one experiment ID prefix across logbook, W&B, and issue.

## Entry rules

Append entries; edit only living indices or broken formatting/links. Each
non-trivial result records exact command, commit hash, config, key output, and
the decision it caused. Record failures/negative results enough to prevent
rediscovery, link large artifacts rather than pasting them, and keep claims
falsifiable. Lightweight WIP commits may be made before reproducibility-
relevant entries: stage only needed files; they are not production snapshots.

Maintain short, linked living indices as needed: `Current TL;DR`, `Current
baseline`, `Hypothesis queue`, `Decision log`, and `Negative results index`.
Never delete the underlying entry when a hypothesis changes. A queue item
should identify its status (active, blocked, falsified/dead end, or promoted),
evidence, and next action/blocker.

## Issue funnel

Use `file-issue`'s experiment template for a new coordinating issue. Keep its
summary current with `TL;DR`, baseline, decision log, negative-results index,
and conclusion. Promote updates as follows:

- Logbook only: raw commands/output, routine observations, local debugging,
  dead ends, and dense analysis.
- Issue comment: milestones, failures/relaunches, baseline changes, surprises,
  decisions, and a heartbeat every 6 hours of active work if no milestone.
- Issue body: durable conclusions, current status, stable baseline, decisions,
  negative-results index, and final outcome.

Agent-authored comments begin with `🤖`, preserve plain issue references, and
link commits/tags, logbook sections, and W&B artifacts. Keep comments concise.
For a non-trivial update, an isolated reader check may verify that terms,
evidence, and required links are understandable without the logbook.

## Finish

When the tracked work is complete, ensure the final logbook entry agrees with
the issue summary. The final issue update states what worked, what failed,
confidence, limitations, ordered next steps, and an explicit conclusion; then
close the issue.
