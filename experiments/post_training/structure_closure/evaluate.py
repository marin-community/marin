# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a complete closure cohort through the ordinary RolloutEngine harness."""

import argparse
import asyncio
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import (
    GradedRollout,
    GradedRolloutInterrupted,
    Model,
    ModelRequest,
    ModelTurn,
    RolloutLimits,
    TaskRolloutSpec,
    TokenUsage,
    TraceMode,
)
from taskcompendium.models import TaskSpec
from transformers import PreTrainedTokenizerFast

from experiments.post_training.structure_closure.rollout_audit import (
    EVALUATION_RECORD,
    EvaluationRecord,
    evaluation_request,
)
from experiments.post_training.structure_closure.task_data import load_validation, task_digest


def completion_model(client: AsyncOpenAI, *, model: str, tokenizer: PreTrainedTokenizerFast) -> Model:
    """Render public messages locally and retain unparsed completions, including delimiters."""

    async def complete(request: ModelRequest) -> ModelTurn:
        if request.prompt_token_ids is not None or request.continuation is not None:
            raise ValueError("Text completion evaluation requires public message conditioning")
        options = dict(request.options)
        template = dict(options.pop("extra_body"))
        template.update(template.pop("chat_template_kwargs"))
        rendered = cast(
            str,
            tokenizer.apply_chat_template(
                list(request.messages), tools=options.pop("tools", None), tokenize=False, **template
            ),
        )
        response = await client.completions.create(model=model, prompt=rendered, stream=False, n=1, **options)
        usage = response.usage
        if usage is None:
            raise ValueError("The completion endpoint reported no token usage")
        choice = response.choices[0]
        return ModelTurn(
            {"role": "assistant", "content": choice.text},
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason=choice.finish_reason,
            text=choice.text,
            metadata={
                "response_id": response.id,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            },
            usage=TokenUsage(usage.prompt_tokens, usage.completion_tokens),
        )

    return complete


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
        turn = await model(request)
        if turn.message.get("reasoning_content"):
            raise ValueError("Closure evaluation requires raw text, before reasoning-parser transformations")
        return turn

    engine = RolloutEngine({}, {"closure": ShellHarness(command_timeout=None)})
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
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--limits", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = load_validation(args.tasks)
    limits = RolloutLimits.model_validate_json(args.limits.read_text())
    sampling = json.loads(args.sampling.read_text())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.initializer, local_files_only=True)
    with args.output.open("x") as output:
        async with AsyncOpenAI(base_url=args.endpoint, api_key=os.environ["OPENAI_API_KEY"], max_retries=0) as client:
            model = completion_model(client, model=args.model, tokenizer=tokenizer)
            records = await evaluate_cohort(tasks, model, limits=limits, concurrency=args.concurrency, sampling=sampling)
        for record in records:
            output.write(EVALUATION_RECORD.dump_json(record).decode() + "\n")
    print(json.dumps({"expected": len(tasks), "recorded": len(records)}))


if __name__ == "__main__":
    asyncio.run(main())
