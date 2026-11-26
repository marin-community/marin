---
license: other
pretty_name: CalibForge V1 rebuilt (TaskCompendium)
tags:
- agentic
- reinforcement-learning
- taskcompendium
- calibforge
task_categories:
- text-generation
language:
- en
---

# CalibForge V1, rebuilt for a bounded snapshot budget

TaskCompendium-format rebuild of the CalibForge V1 cohort, published so the RL data
Atlas points at a converted artifact rather than at the deprecated native release.

> **Private pending release.** This repository is private. It is used for training and
> auditability within Marin and is not a public dataset release; do not cite it as one.

> **Research artifact.** These tasks are training data for reinforcement learning on
> software agents. They are not a benchmark, they carry no leaderboard, and their
> graders are the original tasks' own verifiers, not a reviewed specification. Scores
> produced from them are not comparable across sources or over time.

## Provenance

| Field | Value |
| --- | --- |
| Upstream source | CalibForge V1, `smartie-continuation-20261016/calibforge` |
| Upstream revision | `6a6429ccba098e11ea9f6297a70cb43a34670c0f` |
| Decontamination | curated exclusion list from #10832 (115 tasks) |
| Rebuild code | `experiments/post_training/task_curation/calibforge_rebuild.py` |
| Format | TaskCompendium task specs (one row per task) |

The upstream cohort names its own prebuilt registry image in every task's `task.toml`,
so a run provisions 2,205 snapshots. This rebuild lowers each task onto a fixed,
digest-pinned base set and stages its fixtures through the task resources and
`environment_requirements.setup_commands` that the rollout runner executes.

Every task's own instruction, fixture bytes, file modes and verifier are preserved. The
rebuild changes the environment and the staging mechanism, not the task.

## Cohort

| Outcome | Count |
| --- | --- |
| Rebuilt and admitted | 1,357 |
| Deferred | 732 |
| Excluded by the decontamination list | 115 |
| Excluded because its own setup fails | 1 |

**Seven** distinct snapshot identities, measured through the production path
(`shellbox.snapshot_identity` with Harbor's `machine_spec`); zero snapshot-build
failures and zero tasks refused by `check_execution_requirements`.

### Deferred tasks, by reason

| Reason | Count |
| --- | --- |
| Unsupported `RUN` step | 615 |
| Multi-stage pipeline | 64 |
| Environment too large for the normalized shape | 18 |
| Machine shape larger than 2 CPU / 4 GiB / 10 GiB | 17 |
| Dynamic `COPY` source | 8 |
| Unsupported `COPY` flags | 6 |
| Heredoc | 2 |
| Unreadable instruction | 1 |
| Missing source file | 1 |

Within the unsupported `RUN` steps: python3 build scripts 220, verifier interpreters 51,
`uv` 40, `gcc` 32, `useradd` 24, `cd` 23, and others. A step the lowering cannot
express defers the task instead of approximating it, because running a task in an
environment other than the one it specifies is not admissible.

### Excluded task

| Task | Reason |
| --- | --- |
| `contrastive_solver/software-engineering_20260507_093408_007` | `setup_failure: segfault during pip install, reproducible at 1 worker` |

## Qualification

The cohort ships no oracle solutions, so execution was qualified without one:

- **Payload preservation** — every file the native recipe copies appears in the
  rebuilt setup payload with identical bytes and mode.
- **Setup execution** — each task's own setup command runs in its declared base image
  and reproduces the native recipe's files at their native destinations:
  1,357 of 1,357 pass.
- **Verifier liveness** — each sampled task's verifier runs in a real sandbox against
  the untouched workspace: 76 of 76 returned a well-formed negative verdict with
  reward 0.0 and no infrastructure error.

## License

The upstream CalibForge V1 tasks carry their own license and terms; this rebuild
inherits them and adds no new grant. Consult the upstream release at the pinned
revision before redistribution or commercial use. The rebuild code is Apache-2.0, as
with the rest of this repository.

## Contents

`tasks.parquet` holds one row per admitted task: `path` (the upstream task id) and
`lowered_task_spec` (the TaskCompendium spec the rollout consumes). The spec carries
the instruction as its context, the archived tests as verifier resources, the fixtures
as `all` resources, the pinned base image, and the setup command that stages them.
