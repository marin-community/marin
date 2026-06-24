---
name: run-research
description: "Multi-session research workflow: compose logbooks, experiment issues, documentation updates, and snapshot discipline for long-running investigations."
---

# Skill: Agent-Directed Research

Use this for long-running research where an agent iterates on benchmarks,
experiments, and hypotheses over multiple sessions. Domain skills such as
`add-pallas-kernel` may add constraints.

## General Principle: Open Development

Long-lived work should leave a durable record: a logbook, coordinating issue
updates, and enough commands/config to reproduce results. Do not publish secrets
or private work the user asks to hold back.

## Subskills

- `.agents/skills/background-research/SKILL.md` for prior-work foraging,
  source ledgers, contradiction passes, and ranked hypothesis candidates.
- `.agents/skills/task-logbook/SKILL.md` for logbooks and issue updates.
- `.agents/skills/wandb-reporting/SKILL.md` for W&B project selection, run
  naming, report links, and artifact hygiene.
- `.agents/skills/task-snapshot/SKILL.md` for commit/tag snapshots and stable
  artifact links.
- `.agents/skills/update-docs/SKILL.md` for updating durable docs, runbooks,
  reports, or skill guidance when research changes behavior or practice.

Layer domain skills on top for task-specific constraints.

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
7. A Fieldbook experiment row when the repo has an active Fieldbook ledger.

## Fieldbook First

If `.experiments/ledger.sqlite` exists, use Fieldbook as the local source of
truth for experiment state. Before checking or fixing active experiments, run:

```bash
fieldbook db where --json
fieldbook experiment list --json
fieldbook experiment status "$EXP_ID" --json
```

Use Iris, W&B, GCS, and logbooks as evidence sources, but reconcile important
changes back into Fieldbook. Record planned datapoints as `run` rows, execution
attempts as `job` rows, job fan-out with `run link-job`, and readiness checks as
`validation` rows. Before switching away from an experiment, write a checkpoint
note with `fieldbook experiment checkpoint "$EXP_ID" --json`.

## Research Logbook

Use the term **logbook** consistently. Follow `.agents/skills/task-logbook` for
formatting and issue-update rules.

## Branches

Use a research branch for the logbook, one-off scripts, harnesses, and small
artifacts. Extract a clean production branch only when the final code/docs shape
is clear.

## Standard Workflow

### 1. Prologue

1. Create or switch to a long-lived research branch. You may already be on one, or the user may have requested a specific branch name. Otherwise, pick a descriptive name like `research/<topic>` or `research/<user>/<issue>-<topic>`.
2. Create an experiment issue with `file-issue` unless scope or visibility needs
   human confirmation. If the user provides one, use it.
3. Start the logbook and link both ways: logbook to issue URL, issue body to logbook path. See the skill.
4. Pick a short experiment ID prefix for the series, for example `MOE-HC-001`, and
   use IDs like `MOE-HC-001` in logbook entries, run names, and issue comments.
5. Pick a set of tags to use for all experiments, to be used with W&B, etc.
   Typically this is the ID prefix (without the number), the issue number, and
   anything else reasonable. Try to keep to 2-4 shared tags. You may use more
   tags to distinguish runs within a project as useful.
6. Record, as applicable: motivation, problem statement, context, success
   metrics, the initial hypothesis queue, first experiment matrix, relevant
   code paths, references, stop criteria, and the fixed baseline case for
   repeated comparison.

### 2. Research Loop

1. **Forage:** gather prior work and local context.
2. **Propose:** update the living hypothesis queue and pick the next test.
3. **Run:** implement the smallest useful experiment and collect evidence.
4. **Interpret:** compare against baseline, decide confidence, and update the
   logbook.
5. **Promote:** move only interesting, decision-relevant claims up the issue
   funnel.
6. **Seal:** snapshot durable results or extract production work.

Every cycle should leave the durable record better than it found it.

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
2. Run the benchmark or experiment. Use `babysit-job` for long-lived runs.
3. Append exact commands, config, key outputs, interpretation, and next decision
   to the logbook. Follow `task-logbook` for issue updates.
4. Push dense scalar series, plots, or raw artifacts to W&B or another store
   when they are too large for issue comments.

### 3. Epilogue: Seal

Sealing should ordinarily only happen if the user requests it or the research has reached the defined goal.

1. Update the issue body with the final TL;DR, conclusion, decision log, and negative-results index. Again, follow the `task-logbook` skill.
2. Add a final issue comment covering what worked, what did not, confidence
   level, limitations, and ordered next steps.
3. Use `update-docs` when behavior, operational practice, reusable guidance, or
   durable research findings changed.
4. Ensure the final logbook entry and snapshot links are present.
5. Close the issue when the research thread is complete.

If the research produced useful production changes, extract them into a clean
branch that can link to the logbook but does not include it. Follow standard
Marin development practices on that branch.

Before closing the issue:

- Final TL;DR is current.
- Issue body includes a clear `Conclusion`.
- Next steps are listed.
- Final snapshot is linked.
- If there's a final production PR, link to it from the issue summary.

## Practical Rules

- Prefer short-lived code changes unless a persistent harness is clearly useful.
- Keep benchmark harnesses configurable and minimal.
- Record exact command lines for every headline number.
- Treat failures and negative results as first-class data. Record dead ends and
  excessive hyperparameter sensitivity; skip routine bugs or undertuning.

## See Also

- `.agents/skills/organize-experiments/`
- `.agents/skills/add-pallas-kernel/`
- `.agents/skills/task-logbook/`
- `.agents/skills/update-docs/`
