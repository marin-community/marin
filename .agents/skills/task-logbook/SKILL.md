---
name: task-logbook
description: Maintain a task logbook only when explicitly requested for durable cross-session notes, reproducible experiments, or a coordinating issue; entries are append-only while summary indices may change.
---

# Task logbook

## Concepts

Keep the detailed append-only record in the logbook. If a coordinating issue
exists, publish significant updates as comments and maintain its body as the
current summary.

## Location and Naming

- Store logbooks under `.agents/logbooks/<topic>.md`.
- Use a short, stable topic slug. Including the GitHub issue number is often a
  good choice.
- Link to the coordinating GitHub issue or PR near the top of the logbook when one exists.

## Logbook

New logbooks are Markdown files with YAML frontmatter.

### Logbook Template

```md
---
topic:
issue:
description:
author:
---

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
- Commit Hash:
- Command:
- Config:
- Result:
- Interpretation:
- Next action:
```

When code changes affect reproducibility, make a lightweight WIP commit and
record its hash. Stage only the files needed for the result. Apply the full
`commit` workflow when promoting the work to a production PR.

For a research series, use one short experiment ID consistently in logbook
entries, W&B runs, and issue comments.

Use the requesting user as author. Omit `issue` when none exists.

### Write Rules

- Append entries; do not rewrite history except to fix formatting or broken links or to add a coordinating issue reference.
- Each non-trivial result needs exact commands, hash, config, key output, and the decision it caused.
- Record failures and negative results with enough detail to avoid rediscovery.
- Prefer terse tables for comparable numeric results.
- Link large artifacts, W&B runs, dashboards, pinned GitHub paths, etc., instead of pasting dense output.
- Keep claims scoped and falsifiable.
- Label major claims when useful:
  - `exploratory`: single run or weak evidence.
  - `replicated`: repeated and consistent.
  - `stable`: held across relevant shape, seed, hardware, or workflow variants.

### Living Indices

The entry log is append-only. Short index sections near the top may be edited as
the current state changes.

Living indices may include `Current TL;DR`, `Current baseline`, `Hypothesis
queue`, `Decision log`, and `Negative results index`.

When updating a living index, preserve traceability by linking each item to the
logbook entry, issue comment, W&B run, commit, or tag that supports it. Do not
delete the underlying entry when a hypothesis is revised or falsified; update
the queue status and point to the evidence.

Track hypotheses as active, blocked, falsified, or promoted. Each item needs an
ID, evidence links, and either its next test, resume condition, stopping reason,
or production decision.

## Coordinating Issue

Create or use a coordinating GitHub issue when the work needs durable public
coordination. The user, supervisor agent, or outer workflow may provide one.
Confirm timing with the human collaborator if scope or visibility is uncertain.

Use the experiment body template in `.agents/skills/file-issue/SKILL.md` when
creating a new coordinating experiment issue.

Maintain the issue body for readers without thread context: current TL;DR,
baseline when relevant, decisions with evidence/date/owner, linked negative
results, and the conclusion. Do not overclaim.

### Promotion Rules

Keep commands, raw output, routine observations, dead ends, and dense analysis
in the logbook. Post milestones, failures, relaunches, baseline changes,
surprises, and decisions as issue comments. Promote durable conclusions,
baselines, decisions, negative results, and the final outcome into the issue
body.

For long-running research, post an issue update at each significant milestone
or every 6 hours of active work (either experiments in progress or agent work), whichever comes first. If no milestone occurred by the cadence
deadline, post a brief heartbeat with current status, blockers, and next ETA.

### Posting an update

Issue comment template:

```md
🤖 Update: <short label>

- Change:
- Result delta:
- Confidence:
- Links:
  - Commit/tag:
  - Logbook section:
  - W&B:
- Next:
```

- Mostly append-only; edit only for formatting, escaping, or factual errors.
- Agent-authored issue comments begin with `🤖` unless the text was explicitly approved by the user.
- Leave issue references like #1234 as plain text so GitHub cross-links them.
- Link artifacts to pinned commits or tags when result reproducibility matters.
- Keep the issue concise; move full analysis into the logbook.

When an isolated reader is available, use it for nontrivial updates to check for
undefined terms, ambiguous references, unsupported claims, missing baselines,
and missing links. Give it only the issue context and proposed update; skip this
for trivial updates.

## Finish

Before closing the issue, make the final logbook entry and issue summary agree.
The final comment records what worked, what did not, confidence and limitations,
ordered next steps, and the conclusion.
