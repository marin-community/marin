---
name: run-research
description: Run a coordinated multi-session research program only when explicitly requested with a logbook, experiment issue, and durable snapshots.
---

# Multi-session research

Long-lived work should leave a durable record: a logbook, coordinating issue
updates, and enough commands/config to reproduce results. Do not publish secrets
or private work the user asks to hold back.

Use `background-research`, `task-logbook`, `wandb-reporting`, `task-snapshot`,
and `update-docs` for their named artifacts. Add a domain skill only when the
research directly requires its workflow.

## Core Artifacts

1. A GitHub experiment issue. Agent-created experiment issues use the
   `experiment` and `agent-generated` labels.
2. A logbook at `.agents/logbooks/<topic>.md`.
3. A living hypothesis queue in the logbook, derived from append-only entries
   and updated as hypotheses are proposed, blocked, falsified, or promoted.
4. A long-lived branch, for example `research/<topic>` or
   `research/<user>/<issue>-<topic>`, with the logbook, research code, configs,
   small artifacts, and test harnesses needed to reproduce results.
5. One or more commit or tag snapshots for meaningful milestones.
6. Often a "production" branch that gets PR'd and merged.

## Standard Workflow

### 1. Prologue

1. Keep an existing user-specified branch; otherwise create or use a long-lived
   `research/<topic>` or `research/<user>/<issue>-<topic>` branch.
2. Create or use the experiment issue and link it bidirectionally with the
   logbook. Confirm scope or visibility before creating it when either is
   unclear.
3. Choose one short experiment ID prefix and use IDs such as `MOE-HC-001` in
   entries, runs, and comments; use two to four shared tags.
4. Record the goal, success and stop criteria, baseline, initial hypothesis
   queue and experiment matrix, relevant code, and references.

### 2. Research Loop

1. **Forage:** gather prior work and local context.
2. **Propose:** update the living hypothesis queue and pick the next test.
3. **Run:** implement the smallest useful experiment and collect evidence.
4. **Interpret:** compare against baseline, decide confidence, and update the
   logbook.
5. **Promote:** move only interesting, decision-relevant claims up the issue
   funnel.
6. **Seal:** snapshot durable results or extract production work.

### 2.1 Forage: Background Research

Use `background-research` at the beginning, after a significant change of
direction, or whenever you hit a wall. Assume `medium` or `low` effort unless
the decision is expensive.

Use the background-research output to update the logbook's hypothesis queue:
add new candidates, revise weak ones, mark known dead ends, and promote
well-supported ideas into the next experiment matrix. Let `background-research`
and `task-logbook` decide what belongs in the issue versus the logbook.

### 2.2 Propose / Run / Interpret: Dev Work and Experiments

For research-branch dev work, optimize for learning speed while preserving
operational security and cost controls. Ad-hoc scripts, temporary config knobs,
and copy/paste are acceptable there. Production-facing code keeps the usual
`AGENTS.md` quality bar.

For each non-trivial experiment:

1. Do the dev work needed for the experiment.
2. Run the benchmark or experiment. Use the [Iris job-monitoring workflow](../use-iris/references/monitor-job.md) for long-lived runs.
3. Append exact commands, config, key outputs, interpretation, and next decision
   to the logbook. Follow `task-logbook` for issue updates.
4. Push dense scalar series, plots, or raw artifacts to W&B or another store
   when they are too large for issue comments.

### 3. Epilogue: Seal

Seal when requested or when the defined goal is reached.

1. Update the issue body with the final TL;DR, conclusion, decision log, and negative-results index. Again, follow the `task-logbook` skill.
2. Add a final issue comment covering what worked, what did not, confidence
   level, limitations, and ordered next steps.
3. Use `update-docs` when behavior, operational practice, reusable guidance, or
   durable research findings changed.
4. Ensure the final logbook entry and snapshot links are present.
5. Close the issue when the research thread is complete.

If the research produced useful production changes, extract them into a clean
branch that links the research record without including it. Before closing,
update the TL;DR and conclusion, list next steps, and link the final snapshot
and production PR when one exists.

## Practical Rules

- Prefer short-lived code changes unless a persistent harness is clearly useful.
- Keep benchmark harnesses configurable and minimal.
- Record exact command lines for every headline number.
- Treat failures and negative results as first-class data. Record dead ends and
  excessive hyperparameter sensitivity; skip routine bugs or undertuning.
