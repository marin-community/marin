# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a complete closure cohort through the ordinary RolloutEngine harness."""

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rolloutengine.chat import ChatModel
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.tools import ToolHarness
from rolloutengine.models import (
    GradedRollout,
    GradedRolloutInterrupted,
    Model,
    ModelRequest,
    ModelTurn,
    RolloutLimits,
    TaskRolloutSpec,
    TraceMode,
)
from taskcompendium.models import TaskSpec

from experiments.post_training.structure_closure.rollout_audit import (
    EVALUATION_RECORD,
    EvaluationRecord,
    evaluation_request,
)
from experiments.post_training.structure_closure.task_data import load_validation, task_digest


async def evaluate_task(
    task: TaskSpec, model: Model, *, limits: RolloutLimits, trace: TraceMode, sampling: dict[str, Any]
) -> EvaluationRecord:
    """Retain the actual request and generic rollout, including interrupted attempts."""
    if limits.max_turns != 1:
        raise ValueError("Closure evaluation requires a one-turn rollout")
    request = None
    expected = evaluation_request(task, sampling)

    async def complete(actual: ModelRequest) -> ModelTurn:
        nonlocal request
        if actual.messages != expected.messages:
            raise ValueError("Harness conditioning differs from the frozen task")
        request = replace(actual, options=expected.options)
        return await model(request)

    engine = RolloutEngine({}, {"closure": ToolHarness(command_timeout=None)})
    spec = TaskRolloutSpec(
        task=task, harness="closure", task_machine=None, verifier_machine=None, limits=limits, trace=trace
    )
    engine.check(spec)
    try:
        result = await engine.run_task(spec, complete)
    except GradedRolloutInterrupted as error:
        result = GradedRollout(error.rollout, error.grade)
    return EvaluationRecord(task.id, task_digest(task), request, result)


async def evaluate_cohort(
    tasks: Sequence[TaskSpec], model: Model, *, limits: RolloutLimits, concurrency: int, sampling: dict[str, Any]
) -> tuple[EvaluationRecord, ...]:
    """Run every validation task once with bounded concurrency, preserving input order."""
    if concurrency <= 0:
        raise ValueError("Evaluation concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def run(task: TaskSpec) -> EvaluationRecord:
        async with semaphore:
            return await evaluate_task(task, model, limits=limits, trace=TraceMode.MESSAGES, sampling=sampling)

    return tuple(await asyncio.gather(*(run(task) for task in tasks)))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limits", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = load_validation(args.tasks)
    limits = RolloutLimits.model_validate_json(args.limits.read_text())
    sampling = json.loads(args.sampling.read_text())
    with args.output.open("x") as output:
        async with AsyncOpenAI(base_url=args.endpoint, api_key=os.environ["OPENAI_API_KEY"], max_retries=0) as client:
            model = ChatModel(client, args.model, max_tokens=sampling["max_tokens"])
            records = await evaluate_cohort(tasks, model, limits=limits, concurrency=args.concurrency, sampling=sampling)
        for record in records:
            output.write(EVALUATION_RECORD.dump_json(record).decode() + "\n")
    print(json.dumps({"expected": len(tasks), "recorded": len(records)}))


if __name__ == "__main__":
    asyncio.run(main())
