---
name: task-logbook
description: "Maintain append-only task/research logbooks and publish the important parts upward into coordinating GitHub issues."
---

# Skill: Task Logbook

Use this when a task needs durable context across sessions, reproducible
experiment notes, or a compact handoff trail. For long-running research,
`run-research` composes this skill with a more complete experimentation lifecycle.

## Concepts

This skill can be used in one of two modes: with or without a coordinating issue. Without a coordinating issue, this skill reduces to a particular set of conventions around appending progress update to a file. With a coordinating issue, you will be updating both the file and a Github issue.

The core model is a funnel of interestingness:

1. The logbook is maximally informative and append-only.
2. Significant or interesting updates are posted to the coordinating GitHub issue or PR.
3. The most important findings are promoted into the summary at the top of the issue.

Keep the detailed record in the logbook. Keep the issue useful to someone who
did not follow the live session.

Without a coordinating issue, just do the logbook itself without the "funnel."

## Location and Naming

- Store logbooks under `.agents/logbooks/<topic>.md`.
- Use a short, stable topic slug that can survive branch renames. Including the Github issue number is often a good choice.
- Link to the coordinating GitHub issue or PR near the top of the logbook when one exists.

## Logbook

The logbook is a markdown file with optional yaml frontmatter. New logbooks should include the front matter

### Logbook Template

Not all keys below will be relevant. Baselines may not exist. Exclude them if they do not.


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

For research series, add a short experiment ID prefix such as `MOE-HC` and use IDs like `MOE-HC-001` in logbook entries, W&B run names, and issue comments.

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

Append-only does not mean every summary must be append-only. The entry log is
the immutable record; short index sections near the top may be edited as the
current state changes.

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

Ordinarily, there should be a coordinating issue associated with
each logbook. The user (or supervisor agent) may have provided it. Otherwise, they may have requested you create one, or the outer workflow may typically require it.

If so, create or use a coordinating GitHub issue when the work needs durable public
coordination. Confirm timing with the human collaborator if scope or visibility is uncertain.

Use the experiment body template in `.agents/skills/file-issue/SKILL.md` when
creating a new coordinating experiment issue.

### Issue summary

The issue summary should include a high-level summary, almost like an abstract: What is the problem trying to be solved? What is the goal? Why are we doing this? The target audience should typically be someone with limited knowledge of Marin but generally knowledgeable about the field should be able to read and understand it.

Write for readers who know Marin/LLM systems generally but not the specific
thread. Include enough framing and result interpretation for
someone else to reproduce or critique the claim. Do not overclaim.

As our understanding evolves or work progresses, the issue summary should be kept updated with the current big picture. For example if doing kernel work we might update it with "our kernel is x% of the baseline, but only y% of the target."

When finished, it should be updated with our final status and understanding.

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
- **Issue comment:** significant milestones, failures, relaunches of long running experiments, baseline
  changes, surprising observations, decisions, and periodic long-running research heartbeats.
- **Issue summary/body:** durable conclusions, current TL;DR or major status, stable baseline,
  decision log entries, negative-results index, and final outcome.

Take into account all previous updates since the last logbook entry when making this decision: a single logbook entry may not rise to the level of significance but several together might.

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

For nontrivial updates, it is useful to ask a context-isolated subagent to review to make sure the update is understandable without thread or branch context.

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
