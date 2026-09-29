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

Release `2026.09.10.9` is published and validated with verifier launch commit `b76d03131c`. Marina
revision `marina-00055-lkm` serves that release, and its authenticated source filter and row-detail
API pass against the production Parquet. The remaining work is to update the PR's final evidence,
monitor required checks, merge, and remove temporary audit snapshots.

## Remaining Work

- [x] Validate, lint, commit, and push the launch-provenance fix and Parquet layout documentation.
- [x] Finish release `2026.09.10.9`, launched from clean pushed commit `b76d03131c`.
- [x] Confirm the manifest and an exported task Dockerfile use the launch commit; confirm row count and schema.
- [x] Update the README and Marina dashboard from `.8` to the validated `.9` release.
- [x] Deploy Marina and verify the authenticated task table, filters, task details, and published S3 paths.
- [ ] Update the PR description with `.9` release and deployment evidence.
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

### 2026-09-11 16:01 PDT - Pushed provenance fix and production launch

- Hypothesis: A release launched from clean commit `b76d03131c` will carry the same short ref in its manifest fingerprints and generated verifier Dockerfiles.
- Commit Hash: `b76d03131cd88bd9fc711dba206659027edba3a8`.
- Commands: Pushed the commit to PR head `tasktrove-conversion-pipeline`; launched `python -m experiments.post_training.tasktrove.pipeline --run --max-concurrent 8` through Iris on `cw-us-east-02a`; described the job and tailed its logs.
- Config: Iris job `/power/iris-run-job-20260911-230123`; `PIPELINE_VERSION=2026.09.10.9`; 4 CPU, 16 GiB memory, 64 GiB disk, 12-hour timeout.
- Result: The dry plan includes `tool_ref=b76d03131c` in both conversion and publication fingerprints. The live job entered `build_template_index` with no failures or preemptions.
- Interpretation: The source revision is now fetchable and fixed before the build begins, satisfying the launch-provenance contract. Artifact-level validation remains pending until publication finishes.
- Next action: Monitor to completion, then inspect the manifest, task schema, and one archived Dockerfile before changing the dashboard.

### 2026-09-11 15:59 PDT - Rejected local fallback launch

- Hypothesis: A bare pipeline invocation from the workstation would submit to the configured remote executor.
- Commit Hash: `b76d03131cd88bd9fc711dba206659027edba3a8`.
- Command: Ran the pipeline once without the Iris wrapper, stopped the exact local process tree, and removed only the newly created `/tmp/marin/raw/tasktrove/2026.09.09` and `/tmp/marin/tmp/zephyr/20260911-225930-d2ea53be` paths.
- Config: Unintended `LocalClient`; no production release output selected.
- Result: The invocation began downloading raw inputs locally, so it was stopped. The two run-owned paths used about 8.3 GiB and were permanently removed; older unrelated `/tmp/marin` contents were preserved.
- Interpretation: Production launches must use the explicit Iris wrapper. No remote release was affected.
- Next action: Keep the Iris job as the sole `.9` build and remove the remaining small audit snapshots after acceptance.

### 2026-09-11 16:25 PDT - Release `.9` validation

- Hypothesis: The finished release records the clean launch commit and preserves the `.8` data shape while changing only provenance-bearing task content.
- Commit Hash: `b76d03131cd88bd9fc711dba206659027edba3a8`.
- Commands: Waited on Iris job `/power/iris-run-job-20260911-230123`; read the remote manifest and Parquet footers; predicate-selected one task archive and inspected its Dockerfile.
- Config: Release `s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9`; expected verifier ref `b76d03131c`.
- Result: Iris succeeded. The manifest reports 1,739,326 input rows, 1,449,686 clean rows, 43 kept of 93 sources, 19 converters, 12 modes, and 39 Dockerfiles. The single task file has 1,449,686 rows, 66 row groups, and the exact 12 documented columns; the ledger has 289,640 rows. Sample `codeforces-06236` embeds `b76d03131c` and not the stale ref. The final validator exited zero.
- Interpretation: `.9` satisfies the provenance and single-shard publication contracts and can supersede `.8` in user-facing pointers.
- Next action: Build and test the `.9` dashboard pointer, deploy Marina, and verify the authenticated browser and API.

### 2026-09-11 16:39 PDT - Marina `.9` deployment

- Hypothesis: Pointing the existing TaskTrove app at `.9` will preserve the row-oriented browser behavior while exposing the corrected release paths and provenance.
- Commit Hash: `32845e24c985813813b5d7114bba0b4e8d32fd07`.
- Commands: Built the TaskTrove frontend; ran its four focused backend tests; previewed and applied Pulumi stack `marin-marina`; authenticated through the repository's cached IAP provider and exercised the landing page, manifest, exact-source filter, and row-detail endpoint.
- Config: Marina revision `marina-00055-lkm`; image `sha256:8b549ddaeba89c6548ef38698e461e87ccbd36ffa6c1d5e9aca4b198851ad488`; release `.9` task object size 3,837,032,177 bytes.
- Result: The frontend built with no npm vulnerabilities and all four backend tests passed. Pulumi updated five resources and replaced the migration command; migration execution `marina-hourly-h68qb` succeeded. The live manifest reports `b76d03131c`; source `DCAgent2__nl2bash-tasks-cleaned-oracle-v2` reports 1,497 rows; sample row 16,158 resolves to `task_5407` in script mode.
- Interpretation: The production dashboard now reads the validated `.9` release through its backend cache and row API. The local browser journey could not start because the all-app local kernel had no Echo database configured; it failed before loading TaskTrove, so authenticated production checks provide the browser/API deployment evidence.
- Next action: Publish the final PR evidence, monitor checks and review, then merge and clean up temporary audit files.
