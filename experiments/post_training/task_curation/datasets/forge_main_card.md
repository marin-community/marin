---
license: openmdw-1.1
pretty_name: Forge main split, 2026-11-17 (TaskCompendium)
tags:
- agentic
- reinforcement-learning
- taskcompendium
- synthetic
- forge
task_categories:
- text-generation
language:
- en
---

# Forge main split, 2026-11-17

4,000 synthetic, verified tasks from the forge pipeline, in TaskCompendium format. Every task was
written from one skill in a 1,999-skill, 45-subject catalog, and each was accepted only after its grader
and instructions survived adversarial and transcript review.

> **Private pending release.** This repository is private. It is used for training and auditability
> within Marin and is not a public dataset release; do not cite it as one.

> **Research artifact.** These tasks are training data for reinforcement learning. They are not a
> benchmark and carry no leaderboard. Solve rates below are GLM-5.3's and describe difficulty for that
> policy only.

## Provenance

| Field | Value |
| --- | --- |
| Pipeline | forge (`lib/forge`), re-validation campaign of 2026-11-13 |
| Skill catalog | `2026.09.20-cross-domain-v3` (1,999 skills, 45 subjects) |
| Source records | `s3://marin-us-east-02a/marin/forge-viz/v1/records` (one exported record per accepted task) |
| Selection code | `experiments/post_training/task_curation/forge_release.py` |
| Split | main (tasks graded by a learned decision model are a separate split and are excluded) |
| Format | TaskCompendium task specs (one row per task) |

Each task's spec is the task its eight confirmation trials ran against (the same instructions, files,
grader and machine shape), with every machine restored to the environment its author declared; see
Qualification.

## How a task is accepted

1. A proposal for one task that exercises a catalog skill.
2. An agentic author builds it in a sandbox: instructions, files, a grader (a program, or a reference
   answer checked exactly, numerically, symbolically or as a choice), and controls, which are answers
   with a known expected grade.
3. Every version is validated: the controls are graded, GLM-5.3 attempts the task three times, a
   false-accept and a false-reject adversary attack the grader (a claim counts only if the grader
   reproduces it on a fresh machine), and a reviewer reads every attempt for flaws in the task.
4. A fixed rule reads the evidence. A failed control, a reproduced adversary claim or an established
   reviewer defect sends the task back to its author, up to four times, plus one round to adjust a task
   that is too easy or too hard.
5. Confirmation: five more GLM-5.3 trials and a final review of all eight.

## Cohort

| Outcome | Count |
| --- | --- |
| Accepted by the pipeline | 4,378 |
| Excluded: acceptance carried a reviewer note or missing evidence | 244 |
| Clean, below the coverage tier | 134 |
| **Selected** | **4,000** |

Selection is coverage first: within each skill, clean tasks are ranked by a score, every skill's best
task is taken, then every skill's second, until 4,000. The 4,000 cover 1,598 skills. The score favours an
in-band solve rate, a rate near one half, an accepted version that is not a revert, and more controls.
The ranking therefore uses GLM-5.3's solve rates, which biases the selection toward that policy's
difficulty band; most tasks are still too easy for it.

| Difficulty over eight trials | Count |
| --- | --- |
| In band (2 to 6 solved) | 955 |
| Too easy (7 or 8 solved) | 3,021 |
| Too hard (0 or 1 solved) | 24 |

| Grading | Count | Solver environment | Count |
| --- | --- | --- | --- |
| Grader program | 2,812 | Answer only (no machine) | 2,978 |
| Reference answer check | 1,188 | Linux machine (`python@sha256:a6e34c…`) | 670 |
| | | Simulated shell (ShellSim) | 352 |

Every grader program (3,293 graders, counting the reference-answer tasks that grade with a script) runs in
the same digest-pinned image, `docker.io/library/python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`.

## Qualification

Measured on all 4,000 selected tasks from their exported records:

- **Controls**: 95,706 controls (21,996 of them correct answers), at least one correct-answer control per
  task, all graded as expected on the accepted version at confirmation.
- **Adversaries**: on the accepted version, both adversaries finished and neither reproduced a grader flaw.
- **Review**: the final review of all eight trials finished on every task and found no defect.
- **Trials**: eight GLM-5.3 confirmation trials per task; 105 of the 32,000 ended at the solver's
  generation limit and count as unsolved.

This execution ran in forge's own sandboxes, and one difference matters. The campaign ran forge's local
sandbox mode: every machine that names the Python image above (the 670 Linux task machines and all 3,293
graders) ran under bubblewrap inside the forge driver's pod, with the driver's CPython and an 8 GB
address-space limit, not in the image. Those machines are recorded with no image and an empty package
spec; this release restores the image each author declared, checked against the authored spec of every
task. A task that depends on the image's contents (system packages, Python minor version, preinstalled
tools) can therefore behave differently here than in its recorded trials.

The tasks have not been re-executed through the Atlas rollout path or measured for Daytona snapshot
identity.

## Known issues

- TaskCompendium installs task resources at `/<path>` rather than under the task's working directory;
  forge observed this on 33 of 33 sampled tool tasks. Instructions that name files relative to the
  working directory can disagree with where the files land. Not fixed.

## License

First-party Marin / Open Athena data: synthetic tasks written by forge's agentic author, released
under the OpenMDW License Agreement, version 1.1 (OpenMDW-1.1); the full text is in LICENSE. Valid
for training and evaluation. The repository itself stays private until an explicit release.

## Contents

`tasks.parquet` holds one row per selected task: `path` (the forge task id, `<skill>/<slot>`) and
`lowered_task_spec` (the TaskCompendium task with its task and verifier machine shapes). Each task's `id`
and `source.row` equal its `path`; `source.dataset` is `forge-task-records`, and `source.revision` is the
SHA-256 of the task's exported forge record.

Revision history: the first revision (`b1b06874…`) kept the ids and `source` fields each author wrote,
which dropped the zero padding from slots and gave two pairs of distinct tasks the same id. The second
(`5d18615f…`) replaced only `id` and `source`; instructions, files, graders and machines are unchanged.
The third adds the OpenMDW 1.1 license and omits the optional `integrity_contract` field when it is
unset, so rows also load with TaskCompendium versions that predate it.
