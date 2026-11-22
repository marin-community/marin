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

These historical campaign measurements do not prove Atlas executability. The corrected retained
population has separate Atlas qualification below.

## Known issues

- TaskCompendium installs task resources at `/<path>` rather than under the task's working directory;
  forge observed this on 33 of 33 sampled tool tasks. Instructions that name files relative to the
  working directory can disagree with where the files land. Fixed in the fourth revision: every
  workspace resource of the 850 tasks that declare a working directory other than `/` (6,855 files) is
  re-rooted at that directory, so the runtime's root install lands each file where its author declared
  it. Verifier resources are unchanged; they install under `/tests`. A setup command that copies a
  resource from the container root now finds it already in place or no longer finds the source it
  named; a task whose recorded behavior depends on that copy is excluded by the replay with its own
  reason rather than silently kept.

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
unset, so rows also load with TaskCompendium versions that predate it. The fourth (converter version
`2026.11.21.1`) re-roots the workspace resources described under Known issues; every other field of
every row is unchanged. The fifth (`2026.11.21.2`) preserves each task's authored golden control as
private oracle resources on 1,030 tasks: `solution/solve.sh` rewrites the control's workspace files at
their authored locations, and `solution/answer.txt` carries its reply where the grader reads the reply.
The oracle role is private, so an agent never sees the answer; no other field changes.


## Atlas qualification, 2026-11-22

The corrected source is
`open-athena/forge-main-taskcompendium-20261117@2f1155cb0278fbb3dd4ee5b0685ae68800d2a143`.
It contains 1,030 original-oracle tasks and 2,970 tasks without an oracle. The private retained
population is
`open-athena/forge-main-atlas-20261122-v1@f5766a0e3da2320d30af7c5d6c245f1f1fa711d5`,
converter version `2026.11.22.1`. Its `tasks.parquet` has canonical `task_spec` and `runtime` columns.
Shape detection identifies all 3,577 rows as TaskSpec.

| Retention | Tasks |
| --- | --- |
| Qualified original oracle, with a bound non-passing NoP where required | 607 |
| No oracle, retained under the static policy; executability unmeasured | 2,970 |
| **Retained** | **3,577** |
| **Archived exclusions** | **423** |

| Primary exclusion reason | Tasks |
| --- | --- |
| Unsupported ShellSim backend contract | 352 |
| Oracle failed | 59 |
| Calibration infrastructure failure, pending | 7 |
| Calibration verifier timeout | 1 |
| Negative control pending | 3 |
| Shared executor required | 1 |

All 4,000 source rows are accounted for. Infrastructure failures remain pending. Static retention
supplies no fabricated execution pass. The retained population has a fresh actor and integrity
census, matched grading environments on all 3,577 rows, zero within-source duplicates, and zero
lexical contamination against the shared evaluation Bloom filter. Its three highest overlap
candidates were reviewed as distinct sources.

The complete source ledger and archived exclusions are under
`s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/forge-main/2026.11.21.2/static-retention-v1/`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Fatlas%2Fingest-b%2Fforge-main%2F2026.11.21.2%2Fstatic-retention-v1%2F)).
The selected-pin measurements are under the sibling `selected-qualification-v3/`,
`selected-final-evidence-v2/`, and `selected-overlap-v1/` prefixes.


The frozen seed-zero SAMPLE panel has 100 conclusive reviews, zero unavailable observations,
and zero uncertain observations under the pinned V4 reviewer. Its Result is GOOD with one recorded
finding. An independent Decimal audit of that finding counts eight events outside the window,
including two separate events at 15.50, and agrees with the task's grader. The reviewer omitted
one repeated event. The original review stays unchanged; the audit records the false positive.

The immutable SAMPLE receipt is
`s3://marin-us-east-08a/marin/rl-data/atlas/forge-main/2026.11.22.1/sample-qc-v5/receipt.json`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Frl-data%2Fatlas%2Fforge-main%2F2026.11.22.1%2Fsample-qc-v5%2Freceipt.json)).
The audit is
`s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/forge-main/2026.11.21.2/selected-qc-audit-v1/arithmetic.json`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Fatlas%2Fingest-b%2Fforge-main%2F2026.11.21.2%2Fselected-qc-audit-v1%2Farithmetic.json)).


The admission receipt binds the complete 4,000-row source ledger and the qualified 3,577-row
private release:
`s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/forge-main/2026.11.21.2/selected-admission-v1/receipt/part-000.json`
([Filer](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Fatlas%2Fingest-b%2Fforge-main%2F2026.11.21.2%2Fselected-admission-v1%2Freceipt%2Fpart-000.json)).
