---
name: task-logbook
description: "Maintain append-only task/research logbooks and publish the important parts upward into coordinating GitHub issues."
---

# Skill: Task Logbook

Use this when a task needs durable context across sessions, reproducible
experiment notes, or a compact handoff trail. For long-running research,
`run-research` composes this skill with a more complete experimentation lifecycle.

## Concepts

This skill has two modes:

- No coordinating issue: append progress to the logbook.
- Coordinating issue: update both the logbook and the GitHub issue.

The core model is a funnel of interestingness:

1. The logbook is maximally informative and append-only.
2. Significant or interesting updates are posted to the coordinating GitHub issue or PR.
3. The most important findings are promoted into the summary at the top of the issue.

Keep the detailed record in the logbook. Keep the issue useful to someone who
did not follow the live session.

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

In general, generate a lightweight/in-progress commit before each log if there
have been interesting code changes that affect reproducibility. Include this
commit hash in the entry.

Lightweight reproducibility commits are for research or WIP branches, not
production extraction branches. Stage only the files needed to reproduce the
entry, and leave unrelated local changes untouched. These commits are snapshots
for traceability; they do not need to satisfy the full `commit` skill or
pre-PR checklist until the work is promoted into a production PR.

For research series, add a short experiment ID prefix such as `MOE-HC` and use
IDs like `MOE-HC-001` in logbook entries, W&B run names, and issue comments.

The author should ordinarily be the user who asked you to make the logbook. If there is no GitHub issue, omit it.

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

Use living indices for:

- `Current TL;DR`
- `Current baseline`
- `Hypothesis queue`
- `Decision log`
- `Negative results index`

When updating a living index, preserve traceability by linking each item to the
logbook entry, issue comment, W&B run, commit, or tag that supports it. Do not
delete the underlying entry when a hypothesis is revised or falsified; update
the queue status and point to the evidence.

Hypothesis queue shape:

```md
## Hypothesis Queue

### Active
- `<ID>`: <hypothesis>. Evidence: <links>. Next test: <short action>.

### Blocked
- `<ID>`: <hypothesis>. Blocker: <reason>. Resume when: <condition>.

### Falsified / Dead End
- `<ID>`: <hypothesis>. Why stopped: <evidence links>.

### Promoted
- `<ID>`: <hypothesis>. Decision: <production branch/PR/report/issue link>.
```

## Coordinating Issue

Create or use a coordinating GitHub issue when the work needs durable public
coordination. The user, supervisor agent, or outer workflow may provide one.
Confirm timing with the human collaborator if scope or visibility is uncertain.

Use the experiment body template in `.agents/skills/file-issue/SKILL.md` when
creating a new coordinating experiment issue.

### Issue summary

Write the issue summary for readers who know Marin/LLM systems generally but
not the specific thread. Keep it current with the problem, goal, status,
evidence, and final outcome. Do not overclaim.

Maintain the issue body as the public summary layer:

- `TL;DR`
- `Current baseline` when comparing behavior or metrics
- `Decision log` with important decisions, evidence, date, and owner
- `Negative results index` with links to comments/logbook entries
- `Conclusion` as evidence solidifies

### Promotion Rules

Use this funnel after each meaningful logbook entry:

- **Logbook only:** detailed commands, raw outputs, routine observations,
  dead ends, local debugging notes, and dense analysis.
- **Issue comment:** significant milestones, failures, relaunches of long-running experiments, baseline
  changes, surprising observations, decisions, and periodic long-running research heartbeats.
- **Issue summary/body:** durable conclusions, current TL;DR or major status, stable baseline,
  decision log entries, negative-results index, and final outcome.

Consider all updates since the last issue comment; several small logbook entries
may add up to a useful issue update.

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

#### Issue comment style

- Mostly append-only; edit only for formatting, escaping, or factual errors.
- Agent-authored issue comments begin with `🤖` unless the text was explicitly approved by the user.
- Leave issue references like #1234 as plain text so GitHub cross-links them.
- Link artifacts to pinned commits or tags when result reproducibility matters.
- Keep the issue concise; move full analysis into the logbook.


#### Issue Reader Check

For nontrivial updates, a context-isolated subagent can check whether the update
is understandable without thread or branch context.

If available, start a subagent **without current thread context** (e.g. using
`fork_context: false`). Use this prompt:

```markdown
You are reviewing this GitHub issue update as a technically literate Marin contributor who can see only the issue body, recent issue comments, and the proposed update below.

Task: decide whether the proposed update is understandable from that visible context alone.

Flag:
- undefined terms, acronyms, run names, branch names, or experiment IDs
- references like “this”, “that”, “the previous result”, or “the baseline” whose referent is not visible
- claims without enough visible evidence, config, baseline, metric, or links to evaluate them
- links that appear necessary but are missing

You may also assume knowledge of project documentation, AGENTS.md but otherwise do not use outside knowledge beyond general knowledge of Marin and the field. Do not infer from missing context. Do not read the logbook. Suggest only edits that make the update self-contained from the visible issue context.

Issue: <issue link/id>
Update:

<...>
```

You may skip this check for trivial updates.

When updating the issue body with a new summary, use a suitable variant.

## Finish

Close the issue when what the logbook was tracking is complete.

Before closing the coordinating issue, ensure the final logbook entry and issue
summary agree. The final issue comment should say what worked, what did not,
confidence level and limitations, ordered next steps, and an explicit conclusion
explaining the outcome.
