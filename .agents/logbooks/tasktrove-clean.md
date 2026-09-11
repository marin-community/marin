---
topic: tasktrove-clean
description: TaskTrove cleanup, release validation, dashboard deployment, and merge
author: user
---

# TaskTrove Clean: Task Logbook

## Scope

- Goal: Publish a conservative normalized TaskTrove release, its audit record, and a usable Marina browser.
- Primary metrics: retained rows and sources, fail-closed verifier behavior, shipped-golden success, and release provenance.
- Constraints: deterministic normalization only; use script fallback where appropriate; no near-duplicate work; one final Parquet shard.
- Coordinating PR: [marin-community/marin#9061](https://github.com/marin-community/marin/pull/9061)

## Current TL;DR

Release `2026.09.10.8` and the Marina dashboard are live, but the release manifest and generated
Dockerfiles carry an older hard-coded verifier commit. The pipeline now derives the verifier ref
from the repository's cached `launch_provenance()` helper and is staged as `2026.09.10.9`. The remaining work is to
validate and commit that fix, rebuild and inspect `.9`, point Marina and the PR at it, then merge.

## Remaining Work

- [ ] Validate, lint, commit, and push the launch-provenance fix and Parquet layout documentation.
- [ ] Run release `2026.09.10.9` from the clean pushed commit.
- [ ] Confirm the manifest and an exported task Dockerfile use the launch commit; confirm row count and schema.
- [ ] Update the README, Marina dashboard, and PR description from `.8` to the validated `.9` release.
- [ ] Deploy Marina and verify the authenticated task table, filters, task details, and published S3 paths.
- [ ] Resolve any new review feedback, monitor required checks, and merge PR #9061.
- [ ] Remove temporary local TaskTrove audit snapshots after the release is accepted.

Near-duplicate analysis and a Hive-style `tasks/source=<name>/` publication are not planned. The
single task Parquet exposes source, family, converter, grader, environment identity, language, and
tags as ordinary selector columns.

## Entry Log

### 2026-09-11 22:40 PDT - Verifier provenance defect

- Hypothesis: The verifier ref in a release should identify the clean commit that launched the build, not a manually maintained constant.
- Commit Hash: Uncommitted follow-up on `bef70bb8584d7e0c1391d88c9eacb1605b551a90`.
- Commands: Inspected the `.8` manifest, an archived task Dockerfile, `pipeline.py`, and `rigging.provenance`; ran `uv run pytest experiments/post_training/tasktrove/tests/test_pipeline.py`.
- Config: `PIPELINE_VERSION=2026.09.10.9`; `launch_provenance().base_commit`; dirty launches rejected.
- Result: `.8` incorrectly pins `b2b68d8b0a770cdc0ab3903780172c4b3eea81b1`. Two regression tests pass after threading the captured launch commit through conversion and publication.
- Interpretation: `.8` is internally consistent but does not record the converter/verifier code that initiated its build. It must not be the final release cited by the PR.
- Next action: Run the full focused tests and repository lint, then push the fix before launching `.9`.

### 2026-09-11 22:50 PDT - Canonical Parquet slicing contract

- Hypothesis: A single output object can support source-specific consumption without publishing one physical partition per source.
- Commit Hash: Uncommitted documentation follow-up.
- Command: Inspected `publish.py` `TASK_COLUMNS` and the final release writer.
- Config: One final shard at `tasks/part-00000.parquet`; 12-column `TASKS_SCHEMA`.
- Result: Selection metadata is stored in regular columns: `source`, `family`, `template_id`, `converter`, `mode`, `dockerfile_id`, `language`, `tags`, and `has_solution`; only the task and solution archives are packed binary payloads.
- Interpretation: Documenting these columns is a first-class slicing contract while preserving the requested single-shard release.
- Next action: Add the layout to the README and PR description, then validate it against `.9`.

### 2026-09-11 23:00 PDT - Pre-launch validation

- Hypothesis: The launch-provenance change preserves existing conversion and verifier behavior.
- Commit Hash: Uncommitted follow-up on `bef70bb8584d7e0c1391d88c9eacb1605b551a90`.
- Commands: `uv run pytest experiments/post_training/tasktrove/tests lib/tasktrove-verify/tests`; `./infra/pre-commit.py --changed-files --fix`; `uv run --no-project infra/ci/run_tests.py`.
- Config: Python 3.12; eight pytest workers; repository default marker exclusions.
- Result: Focused tests passed with 426 passed and 1 skipped. Changed-file formatting, lint, and Pyrefly passed. The branch affected-test gate passed with 1,701 passed, 4 skipped, and 5 expected failures.
- Interpretation: The provenance change is ready to push. The first sandboxed attempt failed while resolving packages and cloud metadata; the same commands passed with normal repository dependency access.
- Next action: Commit and push, then launch `.9` from that clean commit.
