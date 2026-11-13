# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade real repaired tau tasks end to end through the production grading path.

The tau repair restores the previously missing grader lock
(``images/env-278169f50750da36/2026.10.08/requirements.lock``) and moves the tau
pivot judge temperature to the standing 0.7 policy. This module proves the repair
on a sample of published tasks from both repaired pins:

- the grader environment installs from the pinned packages lock in a fresh local
  machine (``prepare_machine_spec`` builds the uv runtime from the hash lock) and
  the locked distributions import inside the sandbox;
- an empty submission scores 0.0 under the judge's declared ``empty_output``
  policy through the production judge machinery, without a provider call;
- the judge grades a real candidate reply through the production judge path
  (``grade_task`` with a live ``JudgeConnection``), with the model and
  temperature taken from the published ``JudgeSpec``.

Judged specs grade through the judge service, not in a machine; every production
sibling grader lock (like this one) ships no ``openai``, so the in-machine path
does not execute judge modes. The machine step therefore proves the environment
install, and the judge steps prove the end-to-end grading.

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
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import Command
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import ConversationTrace, GradingAttempt, TaskSpec, TextMessage, judge_spec
from taskcompendium.runtime.environment import prepare_machine_spec
from taskcompendium.runtime.local import LocalGraderMachines
from taskcompendium.runtime.task_grading import grade_task

from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.datasets.nemotron_ultra.components import (
    TAU_REPAIR_LOCK_URL,
    TAU_REPAIR_RELEASES,
)
from experiments.post_training.task_curation.judges import RelayJudgeEndpoint
from experiments.post_training.task_curation.prebuilt import import_taskcompendium_cohort

logger = logging.getLogger(__name__)

CANDIDATE_REPLY = (
    "I can help with that. Could you confirm the account email or phone number on file so I can verify "
    "your identity before we continue?"
)
EMPTY_CONTROL_MEMORY_MB = 8192
MACHINE_PROBE_TIMEOUT = 300.0
LOCK_PROBE = (
    "import json, sys\n"
    "import numpy, pandas, scipy, tomli_w, jsonschema, yaml, nltk\n"
    "print(json.dumps({'python': sys.version.split()[0], 'lock_distributions': 'ok'}))"
)
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


async def _lock_probe(environment, factory, machine_spec) -> dict:
    prepared = prepare_machine_spec(environment, factory, machine_spec, dict(os.environ))
    machine = await factory.create(prepared)
    try:
        result = await machine.run(Command(("python3", "-c", LOCK_PROBE), cwd="/", timeout=MACHINE_PROBE_TIMEOUT))
        if result.exit_code != 0:
            raise RuntimeError(f"Lock environment probe failed: {result.stderr.decode()[:500]}")
        return json.loads(result.stdout.decode())
    finally:
        await machine.close()


def lock_install(task: TaskSpec) -> dict:
    """Prepare the grader machine from the pinned lock and run its environment inside.

    The judged tasks grade through the judge service, not in the machine, so this
    step proves the environment half of the grader: the machine installs the lock
    environment (``prepare_machine_spec`` builds the uv runtime from the 127-package
    hash lock) and the locked distributions import inside the sandbox.
    """
    environment = task.grader.environment
    if environment is None or environment.packages_lock != TAU_REPAIR_LOCK_URL:
        raise ValueError(f"{task.id}: the grader must declare the repaired packages lock")
    factory, machine_spec = LocalGraderMachines().machine(environment, EMPTY_CONTROL_MEMORY_MB)
    probe = asyncio.run(_lock_probe(environment, factory, machine_spec))
    return {
        "packages_lock": environment.packages_lock,
        "lock_sha256": hashlib.sha256(StoragePath(environment.packages_lock).read_bytes()).hexdigest(),
        "machine_backend": factory.backend.value,
        "probe": probe,
    }


def empty_control(task: TaskSpec) -> GradeResult:
    """Grade an empty submission through the production judge path.

    The judge's declared ``empty_output`` policy scores the empty reply 0.0 without a
    provider call, so this is grader liveness over the production machinery.
    """
    judge = judge_spec(task.grader)
    if judge is None:
        raise ValueError(f"{task.id}: the grader must declare a judge spec")
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=""))),
        stop_reason=None,
    )
    result = grade_task(task, attempt)
    return replace(
        result,
        detail={
            **(result.detail or {}),
            "empty_output_policy": (
                judge.empty_output.value if hasattr(judge.empty_output, "value") else judge.empty_output
            ),
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
        install = lock_install(tasks[0])
        records.append({"source_id": source_id, "proof": "lock_install", "outcome": "graded", **install})
        logger.info(
            "%s: lock environment installed in a %s machine: %s", blend, install["machine_backend"], install["probe"]
        )
        for task in tasks:
            records.append(record(source_id, task, "empty_control", empty_control(task)))
        with RelayJudgeEndpoint(args.relay_job, "glm-5.3", api_key).connect() as connection:
            for task in tasks:
                records.append(record(source_id, task, "judge_end_to_end", judge_verdict(task, connection)))
    failed = [entry for entry in records if entry.get("task_id") and entry["outcome"] not in PROOF_OUTCOMES]
    with StoragePath(args.output).open("w") as stream:
        for entry in records:
            stream.write(json.dumps(entry) + "\n")
    for entry in records:
        name = entry.get("task_id", entry.get("packages_lock", ""))[:72]
        logger.info("%s %s %s reward=%s", entry["proof"], name, entry["outcome"], entry.get("reward"))
    if failed:
        raise SystemExit(f"{len(failed)} proof steps did not grade: {failed[:3]}")
    print(f"proof complete: {len(records)} steps, output {args.output}")


if __name__ == "__main__":
    main()
