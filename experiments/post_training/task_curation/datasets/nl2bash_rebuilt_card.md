---
license: other
pretty_name: nl2bash rebuilt (TaskCompendium)
tags:
- agentic
- reinforcement-learning
- taskcompendium
- nl2bash
task_categories:
- text-generation
language:
- en
---

# nl2bash, converted to run in its own environment

TaskCompendium-format conversion of the TaskTrove nl2bash family, published so the RL
data Atlas points at a converted artifact rather than at a native release whose
environment the runtime cannot build.

> **Research artifact.** These tasks are training data for reinforcement learning on
> software agents. They are not a benchmark, carry no leaderboard, and their graders are
> the tasks' own verifiers rather than a reviewed specification.

> **Private pending release.** This repository is private. It is used for training and
> auditability within Marin and is not a public dataset release.

## Provenance

| Field | Value |
| --- | --- |
| Upstream source | TaskTrove `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` |
| Upstream revision | `02923004846e4e73862c20962f823a6d05100e7a` |
| Conversion | `experiments/post_training/task_curation/datasets/tasktrove/nl2bash.py` |
| Format | TaskCompendium task specs (one row per task) |

## What changed, and what did not

The native family declares one environment per task: an `ubuntu:24.04` recipe that
installs the shell tooling and creates the output directories. The conversion **lowers
that recipe onto its digest-pinned base** and replays its `RUN` steps as setup
commands, so each task runs in the environment it specifies rather than in a shared
image. Nothing else about the task changes: the instruction, the verifier data and the
graded output path are the archive's own.

Measured over all 1,498 tasks: **one** distinct environment, zero refusals, and
zero substitutions. 1,498 tasks produce 2,996 task and verifier machines on that single
identity, with no build or execution-requirement failures.

## Qualification

Setup fidelity and verifier liveness ran against these converted specs before publication:

- **Setup fidelity:** 1,498 of 1,498 specs reproduce their archive's lowered environment exactly —
  same pinned base, working directory and setup-command sequence — and every spec carries its
  verifier resources.
- **Verifier liveness:** 20 tasks sampled across the cohort's single environment. Every sampled
  grader grades an empty submission through the production grading path and returns a **negative
  verdict (reward 0.0)**, with no infrastructure errors or timeouts.

The cohort's `oracle_validation` is `verifier_executed` (the grader runs, and correctly rejects an
empty submission) and its oversight level is `sample` — liveness is sampled per environment, not
exhaustive. Graders are lock-placed and grade in the worker's local bubblewrap runtime; they add no
Daytona identity.

## License

The upstream TaskTrove tasks carry their own license and terms; this conversion
inherits them and adds no new grant. The conversion code is Apache-2.0, as with the rest
of this repository.

## Contents

`tasks.parquet` holds one row per converted task: `path` (the upstream task id) and
`lowered_task_spec` (the rollout envelope: the task spec plus its machine shapes).
