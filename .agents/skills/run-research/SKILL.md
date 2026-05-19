---
name: run-research
description: Multi-session research workflow: logbooks, experiment issues, and W&B.
---

# Skill: Agent-Directed Research

## Overview
For long-running, exploratory research where an agent iterates on benchmarks,
experiments, and hypotheses over multiple sessions. Optimizes for
reproducibility, clear decision history, fast iteration, and handoff quality.

## Specialization Policy
`run-research` is the base workflow for research-like work. Layer domain
skills on top for task-specific constraints, keeping lifecycle process here.
- For Pallas kernel work, use `.agents/skills/add-pallas-kernel/SKILL.md` as a
  specialization on top of this skill.
- Keep branch/issue/logbook/snapshot cadence here; keep kernel-specific
  safety/perf rules in `add-pallas-kernel`.

## Core Artifacts
For each research thread, maintain all of:
1. A long-lived branch (e.g. `research/<topic>`).
2. A GitHub experiment issue (use the template below; label `experiment`).
3. An append-only research logbook at `.agents/logbooks/<topic>.md`. Use the
   term **research logbook** consistently in prose and file naming.
4. Optional W&B runs/report for dense numeric output and charts.
5. One or more tags to seal meaningful snapshots.

## W&B Project Policy
- Choose project scope by the type signature of the work.
- Use the `marin` project for pretraining runs.
- Use a new project for materially different work (e.g. kernel development or a
  new RL variant).
- Runs requiring explicit run-to-run comparison must share a W&B project.
- You cannot reliably move/copy runs across projects later — decide scope early.

## Marin Executor vs Dev TPU
- Use the **Marin executor framework** when you need production-like pipeline
  behavior, cluster job metadata/repro tracking, or non-trivial experiment setup.
- Use a **dev TPU** when iteration is quick, you are tuning kernels/benchmarks,
  or full pipeline apparatus is unnecessary. See
  `.agents/skills/reserve-tpu/SKILL.md` for the Iris-backed workflow.
- Rule of thumb: start with dev TPU for fast hillclimbing; confirm final claims
  on the execution path that matters most.

## Standard Workflow

### 1) Kickoff
1. Create/switch to a long-lived research branch.
2. Create an experiment issue from the template below; apply `experiment`.
3. Start a research logbook at `.agents/logbooks/<topic>.md`.
4. Link both ways immediately: logbook → issue URL, issue body → logbook path.
5. Add an agent-generated marker/tag for the thread (labels, comments, or
   snapshot metadata).

At kickoff, write: motivation, problem statement, success metrics, initial
hypotheses, first experiment matrix, links to relevant code paths, key
references (papers/blog posts), stop criteria (what evidence is enough to
stop/ship/escalate), and a fixed baseline case for repeated comparison.

Prefer creating the experiment issue sooner rather than later; confirm timing
with the human collaborator if scope or visibility is uncertain.

Experiment ID convention: assign a short prefix for the series (e.g. `MOE-HC`)
and use IDs like `MOE-HC-001` in logbook entries, W&B run names, and issue
comments.

### 2) Day-to-Day Research Loop
For each non-trivial experiment:
1. Run the benchmark/experiment.
2. Append to the research logbook: date/time, exact command, shape/config, key
   outputs (tables, deltas, failures), interpretation, next decision.
3. Push dense scalar series and plots to W&B when tables are too large.
4. Add a GitHub issue comment with: concise delta since last update, important
   findings only, links to logbook sections and W&B runs, and links to
   artifacts in the GitHub tree pinned to the relevant commit/tag (example
   permalink for `.agents/logbooks/foo.md`:
   `https://github.com/marin-community/marin/tree/<commit-or-tag>/.agents/logbooks/foo.md`).

Update cadence: post an issue update on every significant milestone, or every 6
hours, whichever is sooner. A significant milestone means someone is likely to
want to find that update later. If none occurred by the 6-hour mark, post a
brief heartbeat with current status, blockers, and next ETA.

Issue comment style:
- Mostly append-only; do not rewrite historical comments. Editing a comment to
  fix formatting/escaping/errors is fine and preferred.
- Leave issue references like #1234 as plain text (no backticks) so GitHub
  cross-links them.
- Keep claims scoped and falsifiable.

### 3) Maintain the Issue Body
The issue body is the public summary layer: keep a short TL;DR current, track
scope changes, keep links current, summarize takeaways for non-specialists,
maintain a short decision log (decision, evidence, date, owner), maintain a
negative-results index with links, and keep a `Conclusion` section current as
evidence solidifies.

Write the body for readers who know Marin/LLM systems generally but not this
specific thread. Issue updates/body must stand on their own — include enough
framing (goal, assumptions, exact commands, result interpretation) for someone
else to reproduce or critique the claim. Logbook entries can be terse and
context-local but should still include exact commands and links to supporting
artifacts.

Label major claims as one of: `exploratory` (single run / weak evidence),
`replicated` (repeated and consistent), `stable` (held across shape/seed/
hardware variants relevant to scope).

### 4) Snapshot and Seal
When you reach a meaningful milestone:
1. Commit only relevant files.
2. Tag the commit (annotated tag preferred).
3. Push the tag.
4. Add an issue comment linking the tag, commit, and benchmark/report files
   (prefer GitHub tree permalinks pinned to that commit/tag).
5. Include a repro bundle: exact command(s), hardware/cluster and device count,
   critical environment variables, primary comparison table.

This creates a stable checkpoint even if the branch continues.

### 5) Finish
1. Add a final issue summary: what worked, what did not, confidence level and
   limitations, and an explicit `Conclusion` (decision/outcome and why).
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
- Run one-axis sweeps first (one variable at a time), then interaction sweeps.
- Keep comparisons apples-to-apples (same shape, dtype, pass mode, backend
  unless that axis is the test).
- Always compare against an explicit baseline/reference configuration.
- Only move the baseline after enough repeated evidence; note the change
  explicitly in the logbook.

## Practical Rules
- Prefer short-lived code changes unless a persistent harness is clearly useful.
- Keep benchmark harnesses configurable and minimal.
- Record exact command lines for every headline number.
- Treat failures and negative results as first-class data; include why they
  failed — they prevent repeated dead ends.
- Separate measurement code from the production path whenever possible.
- Prefer persistent remote shells/scripts for long sweeps; avoid repeated
  sync/launch overhead.
- Check accelerator contention (existing processes/locks) before attributing
  regressions to code.
- For long remote runs, track a monotonic progress signal (rows emitted, steps
  completed, checkpoints written) and tail recent logs for context.
- Validate machine-readable extraction before publishing (expected row counts,
  key uniqueness) and de-duplicate when needed.

Ops hygiene checklist (before claiming a regression):
- no stale benchmark process still occupying the accelerator,
- lockfiles/state are clean,
- comparison uses the same device count,
- command/config/env identical except the tested axis.

## Validation Checklist
Before posting a result:
- Command is reproducible.
- Shapes/config are explicitly listed.
- Comparison is apples-to-apples.
- Version snapshot exists (commit or tag).
- Result is in the logbook and linked from the issue.

Before closing the issue:
- Final TL;DR is current.
- Issue body includes a clear `Conclusion`.
- Next steps are listed.
- Final snapshot tag is linked.

## See Also
- `.agents/skills/organize-experiments/`
- `.agents/skills/add-pallas-kernel/`
</content>
</invoke>
