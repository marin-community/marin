---
name: run-research
description: "Multi-session research workflow: compose logbooks, experiment issues, documentation updates, and snapshot discipline for long-running investigations."
---

# Skill: Agent-Directed Research

`run-research` is an "outer loop" representing a standard research workflow in computational science. There are several pieces inside the loop (or used as loop prologue/epilogue). Not all of these pieces are relevant to all projects. Listen to the user and use judgment.

Use this for long-running, exploratory research where an agent iterates on
benchmarks, experiments, and hypotheses over multiple sessions.
There are often more specific versions of this skill for specific domains, e.g. `add-pallas-kernel`.

## General Principle: Open Development

In Marin, we strive for "open development." All work should be documented in order to make the knowledge accessible to others.

Generally, long-lived work should publish progress updates regularly throughout the development process. (By publish we typically mean updating the GitHub repo and associated issue(s). See below.)

Exceptions of course occur. Don't share secrets like API keys. The user may request that a project not be published (until they are ready). However, assume openness by default.

We instantiate this principle in the research workflow by maintaining a logbook, publishing updates to a coordinating issue, and keeping a durable record of experiments, observations, and conclusions. We also (often) produce a clean production branch with doc changes, reusable code, etc., when an experiment produces a useful result.

## Subskills

Subskills include:

- `.agents/skills/background-research/SKILL.md` for prior-work foraging,
  source ledgers, contradiction passes, and ranked hypothesis candidates.
- `.agents/skills/task-logbook/SKILL.md` for append-only task or research logbooks including publishing of summaries/updates to GitHub issues.
- `.agents/skills/wandb-reporting/SKILL.md` for W&B project selection, run
  naming, report links, and artifact hygiene.
- `.agents/skills/task-snapshot/SKILL.md` for commit/tag snapshots and stable
  artifact links.
- `.agents/skills/update-docs/SKILL.md` for updating durable docs, runbooks,
  reports, or skill guidance when research changes behavior or practice.

Layer domain skills on top for task-specific constraints. For example,
`.agents/skills/add-pallas-kernel/SKILL.md` adds kernel-specific safety,
correctness, and profiling rules while this skill keeps the research lifecycle. Domain skills may override or refine guidelines here. Users may also request only pieces of this.

## Core Artifacts

1. A GitHub experiment issue with the `experiment` label.
2. A logbook at `.agents/logbooks/<topic>.md` containing (compact) experiments, observations, and detailed analysis.
3. A living hypothesis queue in the logbook, derived from append-only entries
   and updated as hypotheses are proposed, blocked, falsified, or promoted.
4. A long-lived branch, for example `research/<topic>` or `research/<user>/<issue>-topic` that contains the latest logbook,
   research code, configs, small artifacts, test harnesses and the like. The target should be code and config sufficient to reproduce all results.
5. One or more commit or tag snapshots for meaningful milestones.
6. Often a "production" branch that gets PR'd and merged.

## Research Logbook

The logbook is a central artifact for each research thread, capturing experiments, observations, and detailed analysis. Use the term **logbook** consistently in prose.

See the `.agents/skills/task-logbook` skill for detailed instructions and formatting. The skill also describes how to keep the tracking issue up-to-date.


## Two branch workflow

Note that there are two branches in the artifacts above. One is a
"messy" work-in-progress branch that contains the logbook, one-off experiment scripts, test harnesses, (small) artifacts, and so on. The other is a "clean" production branch that only contains the polished code and docs that are intended for long-term maintenance. The former is for the research process; the latter is for the product.

Frequently you should only produce the "clean" production branch
when you are ready to create a PR and know the final shape.

## Standard Workflow

### 1. Prologue

1. Create or switch to a long-lived research branch. You may already be on one, or the user may have requested a specific branch name. Otherwise, pick a descriptive name like `research/<topic>` or `research/<user>/<issue>-<topic>`.
2. Create an experiment issue unless scope or visibility needs human confirmation. If the user provides one, use it.
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

Research is not linear. Use this loop:

1. **Forage:** gather prior work and local context.
2. **Propose:** update the living hypothesis queue and pick the next test.
3. **Run:** implement the smallest useful experiment and collect evidence.
4. **Interpret:** compare against baseline, decide confidence, and update the
   logbook.
5. **Promote:** move only interesting, decision-relevant claims up the issue
   funnel.
6. **Seal:** snapshot durable results or extract production work.

Allowed backtracks:

- From `interpret`, return to `propose`, `run`, or `forage`.
- From `promote`, return to `interpret` or `run` if the claim is unclear or
  under-supported.
- From `seal`, start a new loop only for follow-up work.

Every cycle should leave the durable record better than it found it.

### 2.1 Forage: Background Research

Use the `background-research` skill. Do this at the beginning, after a
significant change of direction, or whenever you hit a wall. Assume `medium` or
`low` effort unless the decision is expensive. If available, use a subagent for
the background pass.

Ordinarily, the goal is to find related work that will inform next steps: methods to try, libraries to use, benchmarks to run, and so on.

Use the background-research output to update the logbook's hypothesis queue:
add new candidates, revise weak ones, mark known dead ends, and promote
well-supported ideas into the next experiment matrix. Let `background-research`
and `task-logbook` decide what belongs in the issue versus the logbook.

### 2.2 Propose / Run / Interpret: Dev Work and Experiments

For dev work, you should follow the usual `AGENTS.md` guidance and your own best practices, but with assumed lower quality bar. You are exploring and should not immediately worry about maintainability, backward compatibility, and the like. Operational security and cost controls remain valid concerns.
For the research branch,  ad-hoc copy paste, environment variables, intrusive config knobs, one-off scripts should be used whenever convenient. Production-facing code
should still maintain the same quality bar.

For each non-trivial experiment:

0. Do whatever dev work you need to do.
1. Run the benchmark or experiment. Use the `babysit-job` for long-lived experiment.
2. Append to the logbook exact commands, config, key outputs, interpretation, and next decision to the logbook. Follow the `task-logbook` skill for exact procedure including updating the tracking issue.
3. Push dense scalar series, plots, or raw artifacts to W&B or other store when they are too large for useful issue comments.
4. Follow `task-logbook` for issue-update promotion and cadence.

### 3. Epilogue: Seal

Sealing should ordinarily only happen if the user requests it or the research has reached the defined goal.

1. Update the issue body with the final TL;DR, conclusion, decision log, and negative-results index. Again, follow the `task-logbook` skill.
2. Add a final issue comment covering what worked, what did not, confidence
   level, limitations, and ordered next steps.
3. Use `update-docs` when behavior, operational practice, reusable guidance, or
   durable research findings changed.
4. Ensure the final logbook entry and snapshot links are present.
5. Close the issue when the research thread is complete.

If the research resulted in useful "production" changes. Begin extracting them into a separate "clean" branch that does not have the logbook (though may reference it by direct link to GitHub).
Follow standard Marin development practices for the clean branch.

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
- Treat failures and negative results as first-class data. Interesting negative results should be recorded in the logbook and highlighted in the issue. (Bugs or undertuning are not interesting. Dead-ends or excessive hp-sensitivity are.)


## See Also

- `.agents/skills/organize-experiments/`
- `.agents/skills/add-pallas-kernel/`
- `.agents/skills/task-logbook/`
- `.agents/skills/update-docs/`
