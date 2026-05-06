---
name: agent-research
description: Long-running exploratory research workflow with logbooks, W&B, and experiment issues. Use when asked to run a research thread, benchmark study, or multi-session investigation.
---

# Skill: Agent-Directed Research

## Overview
Use this skill for long-running, exploratory research where an agent iterates on
benchmarks, experiments, and hypotheses over multiple sessions.

This workflow optimizes for:
- reproducibility,
- clear decision history,
- fast iteration loops,
- handoff quality for humans and future agents.

## Specialization Policy

`agent-research` is the base workflow for research-like work. Use domain
specialization skills for task-specific constraints while keeping lifecycle
process here.

- For Pallas kernel work, use `.agents/skills/add-pallas-kernel/SKILL.md` as
  a specialization layered on top of this skill.
- Keep branch/issue/logbook/snapshot cadence in `agent-research`; keep
  kernel-specific safety/perf rules in `add-pallas-kernel`.

## Naming
Use **research logbook** consistently in prose and file naming.

Suggested path pattern:
- `.agents/logbooks/<topic>.md`

## Core Artifacts
For each research thread, maintain all of:

1. A long-lived branch (for example `research/<topic>`).
2. A GitHub experiment issue (use the template under "Experiment Issue Template" below; label it `experiment`).
3. A local append-only research logbook in `.agents/logbooks/<topic>.md`.
4. Optional W&B runs/report for dense numeric output and charts.
5. One or more tags to seal meaningful snapshots.

## W&B Project Policy
When using W&B:
- Choose project scope by the type signature of the work.
- Use the `marin` project for pretraining runs.
- Use a new project when the work type is materially different (for example kernel development or a new RL variant).
- If explicit run-to-run comparison is required, runs must be in the same W&B project.
- You cannot reliably move/copy runs across W&B projects later, so decide project scope early.

## When to Use Marin Executor vs Dev TPU
- Use **Marin executor framework** when:
  - you need production-like pipeline behavior,
  - cluster job metadata/repro tracking matters,
  - experiment setup is non-trivial.
- Use **dev TPU** when:
  - iteration is quick,
  - you are tuning kernels or benchmarks,
  - full pipeline apparatus is unnecessary.
- Use `.agents/skills/dev-tpu/SKILL.md` for the Iris-backed workflow.

Rule of thumb:
- Start with dev TPU for fast hillclimbing.
- Confirm final claims on the execution path that matters most.

## Standard Workflow

### 1) Kickoff
1. Create/switch to a long-lived research branch.
2. Create an experiment issue using the body template under "Experiment Issue Template" below. Apply the `experiment` label.
3. Start a research logbook file in `.agents/logbooks/<topic>.md`.
4. Add links both ways immediately:
   - logbook links to the issue URL,
   - issue body links to the logbook path.
5. Add an agent-generated marker/tag for the research thread (for example in issue labels, comments, or snapshot metadata).

At kickoff, write:
- motivation,
- problem statement,
- success metrics,
- initial hypotheses,
- first experiment matrix,
- links to relevant code paths.
- key references (papers/blog posts) when applicable.
- stop criteria (what evidence is enough to stop/ship/escalate).
- a fixed baseline case for repeated comparison.

Issue timing guideline:
- Prefer creating the experiment issue sooner rather than later.
- Confirm timing with the human collaborator if there is uncertainty about scope or visibility.

Experiment ID convention:
- Assign a short experiment ID prefix for the series (for example `MOE-HC`).
- Use IDs like `MOE-HC-001`, `MOE-HC-002`, ... in:
  - logbook entries,
  - W&B run names,
  - issue comments.

### 2) Day-to-Day Research Loop
For each non-trivial experiment:
1. Run benchmark/experiment.
2. Append to the research logbook:
   - date/time,
   - exact command,
   - shape/config,
   - key outputs (tables, deltas, failures),
   - interpretation,
   - next decision.
3. Push dense scalar series and plots to W&B when tables are too large.
4. Add a GitHub issue comment with:
   - concise delta since last update,
   - important findings only,
   - links to research logbook sections and W&B runs,
   - links to artifacts in the GitHub tree pinned to the relevant commit/tag.
   - example permalink: if referencing `.agents/logbooks/foo.md`, link to `https://github.com/marin-community/marin/tree/<commit-or-tag>/.agents/logbooks/foo.md`.

Update cadence:
- Post an issue update whenever a significant milestone is reached, or every 6 hours, whichever is sooner.
- Significant milestone means someone is likely to want to find that update later.
- If no significant milestone occurred by the 6-hour mark, post a brief heartbeat update with current status, blockers, and next ETA.

Issue comment style:
- Mostly append-only.
- Do not rewrite historical comments.
- It is fine (and preferred) to edit a comment to fix formatting/escaping/errors.
- In issue bodies and comments, leave GitHub issue references like #1234 as plain text; do not wrap them in backticks, so GitHub cross-links them.
- Keep claims scoped and falsifiable.

### 3) Maintain the Issue Body
The issue body is the public summary layer:
- keep a short TL;DR up to date,
- track scope changes,
- keep links current,
- summarize takeaways that matter for non-specialists.
- maintain a short decision log (decision, evidence, date, owner).
- maintain a negative-results index with links.
- keep a short `Conclusion` section current as evidence solidifies.

Write the body for readers who:
- know Marin/LLM systems generally,
- do not know this specific thread.

Reader-context rule:
- Write issue updates/body so they stand on their own for readers who were not part of the conversation.
- Include enough framing (goal, assumptions, exact commands, and result interpretation) for someone else to reproduce or critique the claim.
- Logbook entries can be terse/context-local, but should still include exact commands and links to supporting artifacts when relevant.

Confidence labels:
- Label major claims as one of:
  - `exploratory` (single run / weak evidence),
  - `replicated` (repeated and consistent),
  - `stable` (held across shape/seed/hardware variants relevant to scope).

### 4) Snapshot and Seal
When you reach a meaningful milestone:
1. Commit only relevant files.
2. Tag the commit (annotated tag preferred).
3. Push the tag.
4. Add issue comment linking:
   - tag,
   - commit,
   - benchmark/report files (prefer GitHub tree permalinks pinned to that commit/tag).
5. Include a repro bundle for the snapshot:
   - exact command(s),
   - hardware/cluster and device count,
   - critical environment variables,
   - primary comparison table.

This creates a stable checkpoint even if the branch continues.

### 5) Finish
At completion:
1. Add a final issue summary:
   - what worked,
   - what did not,
   - confidence level and limitations.
   - explicit `Conclusion` (decision/outcome and why).
2. Add **next steps** (small, concrete, ordered).
3. Close the issue.

## Research Logbook Template
Use this structure in `.agents/logbooks/<topic>.md`:

```md
# <Topic>: Research Logbook

## Scope
- Goal:
- Primary metric(s):
- Constraints:

## Baseline
- Date:
- Code refs:
- Baseline numbers:

## Experiment Log
### YYYY-MM-DD HH:MM - <short label>
- Hypothesis:
- Command:
- Config:
- Result:
- Interpretation:
- Next action:
```

## Experiment Issue Template
Use this body when filing the GitHub experiment issue at kickoff. Title the
issue `Experiment: <topic>` and apply the `experiment` label.

```md
## Description

(Add enough context someone outside could understand what you're trying to do.
Doesn't need to be too long, but enough you could explain it to someone working
on LLMs at another lab.)

## Hypothesis or Goal

(What are you trying to learn or achieve?)

### Links

(Delete any that aren't applicable.)

* WandB Report:  (link)
* Data Browser: (link)
* (etc.)

## Results

(What did you find, including relevant evaluation metrics, etc.)
```

## Issue Update Template
Use concise updates in issue comments:

```md
Update: <short label>

- Change:
- Result delta:
- Confidence:
- Links:
  - Tag:
  - Logbook section:
  - W&B:
- Next:
```

## Issue Body Template Add-ons
Keep these sections in the issue body:
- `TL;DR`
- `Scope`
- `Decision log` (append as decisions are made)
- `Negative results index` (links to comments/logbook entries)
- `Current baseline` (shape/config + reference numbers)

## Experiment Design Rules
- Run one-axis sweeps first (change one variable at a time), then run interaction sweeps.
- Keep comparisons apples-to-apples (same shape, dtype, pass mode, and backend unless that axis is the test).
- Always compare against an explicit baseline/reference configuration.
- Only move the baseline after enough repeated evidence; note the baseline change explicitly in the logbook.

## Practical Rules
- Prefer short-lived code changes unless a persistent harness is clearly useful.
- Keep benchmark harnesses configurable and minimal.
- Record exact command lines for every headline number.
- Treat failures as first-class data (include why they failed).
- Treat negative results as first-class outcomes; they prevent repeated dead ends.
- Separate "measurement code" from "production path" whenever possible.
- Prefer persistent remote shells/scripts for long sweeps; avoid repeated sync/launch overhead when possible.
- Check accelerator contention (existing processes/locks) before attributing regressions to code.
- For long remote runs, track a monotonic progress signal (for example rows emitted, steps completed, checkpoints written) and tail recent logs for context.
- Validate machine-readable extraction before publishing results (for example expected row counts and key uniqueness) and de-duplicate when needed.

Ops hygiene checklist (before claiming regression):
- confirm no stale benchmark process is still occupying accelerator,
- confirm lockfiles/state are clean,
- confirm comparison uses same device count,
- confirm command/config/env are identical except the tested axis.

## Validation Checklist
Before posting a result:
- Command is reproducible.
- Shapes/config are explicitly listed.
- Comparison is apples-to-apples.
- Version snapshot exists (commit or tag).
- Result is written in the research logbook and linked from issue.

Before closing the issue:
- Final TL;DR is current.
- Issue body includes a clear `Conclusion`.
- Next steps are listed.
- Final snapshot tag is linked.

## See Also
- `.agents/skills/organize-experiments/`
- `.agents/skills/add-pallas-kernel/`
