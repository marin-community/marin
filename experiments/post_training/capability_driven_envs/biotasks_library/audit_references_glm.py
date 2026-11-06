# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently recompute frozen BioTasks references with a blinded GLM audit."""

import argparse
import asyncio
import base64
import json
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from functools import partial
from typing import Literal

import fsspec
from pydantic import BaseModel, ConfigDict, Field
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge_adapters.campaign import OpenAIModelPort, campaign_worker
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

DEFAULT_RELAY_JOB = "/muchanem/glm53-relay-tok-02a"
DEFAULT_RELAY_SERVICE = "glm-5.3-tok"
DEFAULT_TASKS = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/"
    "campaigns/biotasks/combined/v1/snapshots/complete-1/taskspecs.jsonl"
)
DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/"
    "campaigns/biotasks/combined/v1/audits/reference-recomputation-glm-v1/results.jsonl"
)


class ReferenceAudit(BaseModel):
    """One independent assessment of a frozen reference answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    independently_computed_answer_json: str
    reference_assessment: Literal["confirmed", "incorrect", "ambiguous", "insufficient_information"]
    issue_category: Literal[
        "none",
        "arithmetic",
        "rounding_or_tie_rule",
        "schema_or_key_contract",
        "resource_or_instruction_contradiction",
        "scientific_interpretation",
        "other",
    ]
    blocking_issues: tuple[str, ...] = ()
    rationale: str
    confidence: float = Field(ge=0, le=1)


def _load_rows(uri: str) -> list[dict[str, object]]:
    with fsspec.open(uri, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = []
    for row in rows:
        task = row["task"]
        resources = []
        for resource in task["resources"]["worker"]:
            source = resource["source"]
            if source["kind"] != "inline_file":
                raise ValueError(f"audit only supports inline resources: {row['task_id']}")
            resources.append(
                {
                    "path": resource["path"],
                    "content": base64.b64decode(source["content_base64"]).decode("utf-8", errors="replace"),
                }
            )
        result.append(
            {
                "task_id": row["task_id"],
                "task_digest": row["task_digest"],
                "instruction": "\n".join(event["content"] for event in task["context"]["events"]),
                "resources_json": json.dumps(resources, sort_keys=True),
                "expected_json": json.dumps(task["grader"]["parameters"]["expected"], sort_keys=True),
            }
        )
    return result


def _prompt(row: dict[str, object]) -> str:
    return f"""Independently audit one objective computational-biology task's frozen reference answer. Recompute every
field from the instruction and solver-visible resources before comparing with the proposed reference. Do not assume the
reference is correct. Check arithmetic, filtering, sorting, rounding, tie rules, units, biological interpretation, and
whether the exact JSON key contract is sufficiently specified. The resources are complete and small; do not use outside
facts or invent missing rules.

Return independently_computed_answer_json as one valid JSON object encoded as a string. Mark:
- confirmed only when every reference field follows uniquely from the task;
- incorrect when a recomputation disagrees;
- ambiguous when multiple answers satisfy the written contract;
- insufficient_information when required input or a rule is absent.

Task ID: {row["task_id"]}
Instruction:
<instruction>
{row["instruction"]}
</instruction>
Solver-visible resources:
<resources>
{row["resources_json"]}
</resources>
Proposed frozen reference:
<reference>
{row["expected_json"]}
</reference>
"""


async def _audit(row: dict[str, object], model: ModelClient, policy: LLMPolicy) -> dict[str, object]:
    prompt = _prompt(row)
    result = await model.complete_structured(
        (Message(role="user", content=prompt),),
        ReferenceAudit,
        policy,
    )
    return {
        "task_id": row["task_id"],
        "task_digest": row["task_digest"],
        "model": model.model,
        "audit": result.value.model_dump(mode="json"),
        "prompt": prompt,
        "completions": [
            {
                "content": completion.content,
                "reasoning": completion.reasoning,
                "finish_reason": str(completion.finish_reason),
                "usage": asdict(completion.usage),
                "attempts": [asdict(attempt) for attempt in completion.attempts],
            }
            for completion in result.completions
        ],
    }


def run_shard(
    rows: Iterable[dict[str, object]], shard_info: object, *, port: OpenAIModelPort
) -> Iterator[dict[str, object]]:
    del shard_info

    async def run() -> list[dict[str, object]]:
        results = []
        async with campaign_worker(port) as worker:
            for row in rows:
                try:
                    results.append(await _audit(row, worker.model, worker.policy))
                except Exception as error:
                    results.append(
                        {
                            "task_id": row["task_id"],
                            "task_digest": row["task_digest"],
                            "model": worker.model.model,
                            "audit_error": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc()[-8000:],
                            },
                        }
                    )
        return results

    yield from asyncio.run(run())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument("--api-key-env", default=GLM_BULK_TOKEN_ENV)
    parser.add_argument("--relay-job", default=DEFAULT_RELAY_JOB)
    parser.add_argument("--relay-service", default=DEFAULT_RELAY_SERVICE)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")

    rows = _load_rows(args.tasks)
    port = OpenAIModelPort(
        args.model,
        args.api_key_env,
        relay_job=args.relay_job,
        relay_service=args.relay_service,
    )
    pipeline = (
        Dataset.from_list(rows)
        .reshard(len(rows))
        .map_shard(partial(run_shard, port=port))
        .reshard(1)
        .write_jsonl(args.output)
    )
    with ZephyrContext(
        max_workers=min(args.max_workers, len(rows)),
        name="biotasks-reference-audit-glm",
        heartbeat_timeout=1200,
    ) as context:
        result = context.execute(pipeline)
    print(f"tasks={len(rows)}")
    print(f"output={args.output}")
    print(result)


if __name__ == "__main__":
    main()
