---
name: run-research
description: "Multi-session research workflow: compose logbooks, experiment issues, documentation updates, and snapshot discipline for long-running investigations."
---

# Run Research

Use for multi-session work that iterates on experiments, benchmarks, and
hypotheses. Keep a durable record without publishing secrets or material the
user asked to keep private. Layer domain skills on top as needed.

Read the relevant companion skill at the point of use:

- `background-research`: prior work, contradiction pass, and ranked hypotheses.
- `task-logbook`: append-only entries and issue promotion rules.
- `wandb-reporting`: W&B project, naming, reports, and artifact hygiene.
- `task-snapshot`: stable commits/tags and reproducible links.
- `update-docs`: durable behavior, operational, or research guidance changes.

## Required durable shape

Normally create an experiment issue (labels `experiment` and
`agent-generated`), a logbook at `.agents/logbooks/<topic>.md`, and a
long-lived research branch containing reproducible code/configs. Maintain the
logbook's living hypothesis queue. Add commit/tag snapshots at meaningful
milestones; extract production changes to a clean branch when ready.

## Workflow

1. **Prologue:** choose/switch the research branch; create or use the issue
   (ask for human confirmation when scope or visibility is uncertain);
   link issue and logbook; choose a short experiment ID and shared tags; record
   motivation, success metrics, baseline, first matrix, code paths, references,
   and stop criteria.
2. **Loop:** forage, propose the next smallest test, run it, interpret against
   the fixed baseline, promote only decision-relevant claims, and update the
   logbook after each non-trivial experiment with exact command/config, key
   output, interpretation, and next decision. Use `babysit-job` for long runs
   and W&B for dense scalars/plots/raw artifacts.
3. **Seal:** ordinarily only on user request or when the goal is reached. Update
   the issue TL;DR, conclusion, decision log, and negative-results index; add a
   final comment with results, limitations, confidence, and next steps; update
   durable docs if behavior/practice changed; link the final logbook and
   snapshot; close the issue when complete.

Failures, negative results, and sensitivity are evidence; routine bugs and
undertuning need only enough detail to avoid rediscovery. Keep harnesses
configurable and minimal. See `organize-experiments` for report indexing.
