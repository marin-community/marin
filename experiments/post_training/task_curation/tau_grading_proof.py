# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade real repaired tau tasks end to end through the production grader path.

The tau repair restores the previously missing grader lock
(``images/env-278169f50750da36/2026.10.08/requirements.lock``) and moves the tau
pivot judge temperature to the standing 0.7 policy. This module proves the repair
on a sample of published tasks from both repaired pins:

- the grader environment installs from the pinned packages lock in a fresh local
  grading machine (``grade_empty_in_sandbox`` through ``prepare_machine_spec``),
  and an empty submission scores 0.0 under the judge's declared ``empty_output``
  policy without touching the judge endpoint;
- the judge grades a real candidate reply through the production judge path
  (``grade_task`` with a live ``JudgeConnection``), with the model and
  temperature taken from the published ``JudgeSpec``.

Run it in a CPU Iris job co-located with the artifacts::

    uv run iris --config lib/iris/config/marin.yaml job run \\
      --target-cluster cw-us-east-02a --cpu 8 --memory 64G --disk 200G \\
      --enable-extra-resources --sync-package marin-core --extra cpu \\
      --priority production --timeout 259200 --no-wait -- \\
      python -m experiments.post_training.task_curation.tau_grading_proof \\
      --sample-size 5 --output s3://marin-us-east-02a/marin/rl-data/atlas/tau-repair/<version>/grading-proof.jsonl
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.datasets.nemotron_ultra.components import (
    TAU_REPAIR_RELEASES,
    TAU_REPAIR_LOCK_URL,
)
from experiments.post_training.task_curation.judges import RelayJudgeEndpoint
from experiments.post_training.task_curation.prebuilt import import_taskcompendium_cohort
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage, judge_spec
from taskcompendium.runtime.local import LocalGraderMachines
from taskcompendium.runtime.task_grading import grade_task

logger = logging.getLogger(__name__)

CANDIDATE_REPLY = (
    "I can help with that. Could you confirm the account email or phone number on file so I can verify "
    "your identity before we continue?"
)
EMPTY_CONTROL_MEMORY_MB = 8192
PROOF_OUTCOMES = ("graded", "submission_failure")


def sample_tasks(cohort, size: int) -> Iterator[TaskSpec]:
    """Validate the pinned cohort and yield its first ``size`` tasks in published order."""
    local = Path(import_taskcompendium_cohort(cohort))
    yielded = 0
    for batch in pq.ParquetFile(local).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            if yielded >= size:
                return
            yield TaskSpec.model_validate_json(row["task_spec"])
            yielded += 1
    if yielded < size:
        raise ValueError(f"{cohort.repo}@{cohort.revision} has {yielded} tasks, expected {size}")


def empty_control(task: TaskSpec) -> GradeResult:
    """Install the grader environment from the pinned lock in a fresh machine and grade nothing."""
    environment = task.grader.environment
    if environment is None or environment.packages_lock != TAU_REPAIR_LOCK_URL:
        raise ValueError(f"{task.id}: the grader must declare the repaired packages lock")
    from taskcompendium.runtime.grading import grade_empty_in_sandbox

    factory, machine_spec = LocalGraderMachines().machine(environment, EMPTY_CONTROL_MEMORY_MB)
    result = asyncio.run(grade_empty_in_sandbox(task, factory, machine_spec))
    lock = StoragePath(environment.packages_lock)
    return replace(
        result,
        detail={
            **(result.detail or {}),
            "packages_lock": environment.packages_lock,
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        },
    )


def judge_verdict(task: TaskSpec, connection) -> GradeResult:
    """Grade one real candidate reply through the production judge path."""
    judge = judge_spec(task.grader)
    if judge is None:
        raise ValueError(f"{task.id}: the grader must declare a judge spec")
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=CANDIDATE_REPLY))),
        stop_reason=None,
    )
    result = grade_task(task, attempt, judge_connection=connection)
    return replace(
        result,
        detail={
            **(result.detail or {}),
            "judge_model": judge.model,
            "judge_temperature": judge.temperature,
        },
    )


def record(source_id: str, task: TaskSpec, kind: str, result: GradeResult) -> dict:
    return {
        "source_id": source_id,
        "task_id": task.id,
        "task_sha256": hashlib.sha256(task.model_dump_json().encode()).hexdigest(),
        "proof": kind,
        "outcome": result.status.value,
        "reward": result.reward,
        "error": result.error,
        "detail": result.detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--output", required=True, help="JSONL evidence path, local or object storage")
    parser.add_argument("--relay-job", default=DEFAULT_GLM_RELAY_JOB)
    parser.add_argument("--api-key-env", default=GLM_BULK_TOKEN_ENV)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    api_key = os.environ[args.api_key_env]
    records = []
    for blend, cohort in TAU_REPAIR_RELEASES.items():
        source_id = cohort.dataset.name
        tasks = list(sample_tasks(cohort, args.sample_size))
        logger.info("%s: proving %d tasks from %s@%s", blend, len(tasks), cohort.repo, cohort.revision)
        for task in tasks:
            records.append(record(source_id, task, "empty_control_lock_install", empty_control(task)))
        with RelayJudgeEndpoint(args.relay_job, "glm-5.3", api_key).connect() as connection:
            for task in tasks:
                records.append(record(source_id, task, "judge_end_to_end", judge_verdict(task, connection)))
    failed = [entry for entry in records if entry["outcome"] not in PROOF_OUTCOMES]
    with StoragePath(args.output).open("w") as stream:
        for entry in records:
            stream.write(json.dumps(entry) + "\n")
    for entry in records:
        logger.info("%s %s %s reward=%s", entry["proof"], entry["task_id"][:72], entry["outcome"], entry["reward"])
    if failed:
        raise SystemExit(f"{len(failed)} proof steps did not grade: {failed[:3]}")
    print(f"proof complete: {len(records)} steps, output {args.output}")


if __name__ == "__main__":
    main()
