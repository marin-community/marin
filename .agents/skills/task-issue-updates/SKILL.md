---
name: task-issue-updates
description: "Create and maintain GitHub issues for experiments or atomic tasks, including progress comments, TL;DRs, decision logs, and final summaries."
---

# Skill: Task Issue Updates

Use this when work is coordinated through a GitHub issue and the issue should be
useful to a reader who did not follow the live session. This applies to
experiments, debugging tasks, operational changes, and atomic implementation
work with meaningful decisions.

## Kickoff

Create an issue early when the work needs durable coordination. For research,
title it `Experiment: <topic>` and apply the `experiment` label. Confirm timing
with the human collaborator if scope or visibility is uncertain.

Issue kickoff body:

```md
## Description

(Context someone outside the thread can understand.)

## Hypothesis or Goal

(What are you trying to learn, fix, or achieve?)

### Links

* Logbook:
* W&B Report:
* Data Browser:

## Results

(Current state; update as evidence lands.)
```

## Body Maintenance

Keep the body as the public summary layer:

- `TL;DR`
- `Scope`
- `Current baseline` when comparing behavior or metrics
- `Decision log` with decision, evidence, date, and owner
- `Negative results index` with links to comments/logbook entries
- `Conclusion` as evidence solidifies

Write for readers who know Marin/LLM systems generally but not the specific
thread. Include enough framing, exact commands, and result interpretation for
someone else to reproduce or critique the claim.

Label major claims as:

- `exploratory`: single run or weak evidence
- `replicated`: repeated and consistent
- `stable`: held across relevant shape, seed, hardware, or workflow variants

## Progress Comments

Post an update at each significant milestone. For long-running research, also
post at least every 6 hours. A significant milestone is one someone is likely to
want to find later. If no milestone occurred by the cadence deadline, post a
brief heartbeat with current status, blockers, and next ETA.

```md
Update: <short label>

- Change:
- Result delta:
- Confidence:
- Links:
  - Commit/tag:
  - Logbook section:
  - W&B:
- Next:
```

Issue comment style:

- Mostly append-only; edit only for formatting, escaping, or factual errors.
- Leave issue references like #1234 as plain text so GitHub cross-links them.
- Keep claims scoped and falsifiable.
- Link artifacts to pinned commits or tags when result reproducibility matters.

## Finish

Before closing an issue, ensure the final comment says what worked, what did
not, confidence level and limitations, ordered next steps, and an explicit
conclusion explaining the outcome.
