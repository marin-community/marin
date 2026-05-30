---
name: run-research
description: "Multi-session research workflow: compose logbooks, experiment issues, W&B reporting, documentation updates, and snapshot discipline for long-running investigations."
---

# Skill: Agent-Directed Research

Use this for long-running, exploratory research where an agent iterates on
benchmarks, experiments, and hypotheses over multiple sessions. For atomic
implementation tasks, load only the narrower skills below that match the work.

## Compose These Skills

`run-research` is an orchestration layer. Load and follow these reusable pieces:

- `.agents/skills/task-logbook/SKILL.md` for append-only task or research
  logbooks.
- `.agents/skills/task-issue-updates/SKILL.md` for GitHub issue kickoff,
  body maintenance, progress comments, and final summaries.
- `.agents/skills/wandb-reporting/SKILL.md` for W&B project selection, run
  naming, report links, and artifact hygiene.
- `.agents/skills/task-docs/SKILL.md` for keeping docs and operational notes in
  sync with behavior discovered or changed by the task.
- `.agents/skills/task-snapshot/SKILL.md` for commit/tag snapshots and stable
  artifact links.

Layer domain skills on top for task-specific constraints. For example,
`.agents/skills/add-pallas-kernel/SKILL.md` adds kernel-specific safety,
correctness, and profiling rules while this skill keeps the research lifecycle.

## Core Artifacts

For each research thread, maintain all of:

1. A long-lived branch, for example `research/<topic>`.
2. A GitHub experiment issue with the `experiment` label.
3. An append-only research logbook at `.agents/logbooks/<topic>.md`.
4. Optional W&B runs/report for dense numeric output and charts.
5. One or more commit or tag snapshots for meaningful milestones.

Use the term **research logbook** consistently in prose and file naming.

## Standard Workflow

### 1. Kickoff

1. Create or switch to a long-lived research branch.
2. Create an experiment issue unless scope or visibility needs human
   confirmation.
3. Start the research logbook and link both ways: logbook to issue URL, issue
   body to logbook path.
4. Pick a short experiment ID prefix for the series, for example `MOE-HC`, and
   use IDs like `MOE-HC-001` in logbook entries, W&B run names, and issue
   comments.
5. Record motivation, problem statement, success metrics, initial hypotheses,
   first experiment matrix, relevant code paths, references, stop criteria, and
   the fixed baseline case for repeated comparison.

### 2. Research Loop

For each non-trivial experiment:

1. Run the benchmark or experiment.
2. Append exact commands, shape/config, key outputs, interpretation, and next
   decision to the logbook.
3. Push dense scalar series, plots, or raw artifacts to W&B when they are too
   large for useful issue comments.
4. Post an issue update for every significant milestone, or every 6 hours,
   whichever is sooner.
5. Keep docs current when behavior, operational procedure, or reusable guidance
   changes.

Run one-axis sweeps first, then interaction sweeps. Keep comparisons
apples-to-apples: same shape, dtype, pass mode, backend, device count, and
environment unless that axis is under test. Only move the baseline after enough
repeated evidence, and note the change explicitly.

### 3. Snapshot and Seal

When a milestone matters, create a stable snapshot using the task-snapshot
skill: commit relevant files, tag the commit when appropriate, push the tag, and
post pinned GitHub links to the issue. Include exact commands, hardware/cluster
and device count, critical environment variables, and the primary comparison
table.

### 4. Finish

1. Update the issue body with the final TL;DR, conclusion, decision log, and
   negative-results index.
2. Add a final issue comment covering what worked, what did not, confidence
   level, limitations, and ordered next steps.
3. Ensure the final logbook entry and snapshot links are present.
4. Close the issue when the research thread is complete.

## Practical Rules

- Prefer short-lived code changes unless a persistent harness is clearly useful.
- Keep benchmark harnesses configurable and minimal.
- Record exact command lines for every headline number.
- Treat failures and negative results as first-class data.
- Separate measurement code from the production path whenever possible.
- Prefer persistent remote shells/scripts for long sweeps.
- Check accelerator contention before attributing regressions to code.
- For long remote runs, track a monotonic progress signal and tail recent logs
  for context.
- Validate machine-readable extraction before publishing: expected row counts,
  key uniqueness, and de-duplication.

Ops hygiene checklist before claiming a regression:

- No stale benchmark process is still occupying the accelerator.
- Lockfiles/state are clean.
- Comparison uses the same device count.
- Command/config/env are identical except the tested axis.

## Validation Checklist

Before posting a result:

- Command is reproducible.
- Shapes/config are explicitly listed.
- Comparison is apples-to-apples.
- Version snapshot exists.
- Result is in the logbook and linked from the issue.

Before closing the issue:

- Final TL;DR is current.
- Issue body includes a clear `Conclusion`.
- Next steps are listed.
- Final snapshot is linked.

## See Also

- `.agents/skills/organize-experiments/`
- `.agents/skills/add-pallas-kernel/`
