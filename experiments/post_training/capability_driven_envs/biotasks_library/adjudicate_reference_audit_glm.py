# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Second-pass adjudication of non-confirmed BioTasks reference audits.

A model's second opinion is triage evidence, not a verified correction. Preserve
first-pass contradictions and audit errors for subsequent deterministic review.
"""

import argparse
import asyncio
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

from experiments.post_training.capability_driven_envs.biotasks_library.audit_references_glm import (
    DEFAULT_OUTPUT as FIRST_OUTPUT,
)
from experiments.post_training.capability_driven_envs.biotasks_library.audit_references_glm import (
    DEFAULT_TASKS,
    _load_rows,
)
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/"
    "campaigns/biotasks/combined/v1/audits/reference-recomputation-glm-v1/adjudication.jsonl"
)


class Adjudication(BaseModel):
    """Independent check of the original answer and the first audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_assessment: Literal["confirmed", "incorrect", "ambiguous", "insufficient_information"]
    first_audit_assessment: Literal["correct", "incorrect", "uncertain"]
    corrected_answer_json: str | None
    issue_category: Literal[
        "none",
        "arithmetic",
        "rounding_or_tie_rule",
        "instruction_ambiguity",
        "resource_ambiguity",
        "missing_information",
        "output_schema",
        "other",
    ]
    blocking_issues: tuple[str, ...] = ()
    rationale: str
    confidence: float = Field(ge=0, le=1)


def load_disputed(tasks_uri: str, audit_uri: str) -> list[dict[str, object]]:
    """Join non-confirmed audit results to their frozen task inputs by digest."""
    tasks = {row["task_id"]: row for row in _load_rows(tasks_uri)}
    with fsspec.open(audit_uri, "rt") as source:
        audits = [json.loads(line) for line in source if line.strip()]
    if len(audits) != len(tasks) or len({row["task_id"] for row in audits}) != len(audits):
        raise ValueError("First audit must have exactly one row per frozen task")
    disputed = []
    for audit in audits:
        task = tasks[audit["task_id"]]
        if audit["task_digest"] != task["task_digest"]:
            raise ValueError(f"Task digest mismatch: {audit['task_id']}")
        if audit.get("audit", {}).get("reference_assessment") == "confirmed":
            continue
        disputed.append({**task, "first_audit": audit.get("audit"), "first_audit_error": audit.get("audit_error")})
    return disputed


def _prompt(row: dict[str, object]) -> str:
    return f"""Independently adjudicate this frozen computational-biology task. Recompute every required output
from the instruction and solver-visible resources, then compare with the frozen reference. The first auditor may have
made mistakes: inspect its explanation for contradictions, incorrect counts, mistaken indexing or rounding. Do not
accept either answer on authority. Give concrete input rows and calculations for every alleged discrepancy.

Classify confirmed only if every field is determined and correct; incorrect if any required field is objectively
wrong; ambiguous if multiple answers follow the wording; insufficient_information if inputs cannot determine the
answer. Equivalent JSON numbers (0 and 0.0) are not different values. If incorrect, corrected_answer_json must be
a complete JSON object; otherwise it may be null. If unable to verify, use 'uncertain' for first_audit_assessment.

Task ID: {row["task_id"]}
Instruction: {row["instruction"]}
Solver-visible resources: {row["resources_json"]}
Frozen reference: {row["expected_json"]}
First auditor's finding: {json.dumps(row["first_audit"], ensure_ascii=False)}
First auditor's error, if any: {json.dumps(row["first_audit_error"], ensure_ascii=False)}
"""


async def _adjudicate(row: dict[str, object], model: ModelClient, policy: LLMPolicy) -> dict[str, object]:
    prompt = _prompt(row)
    result = await model.complete_structured((Message(role="user", content=prompt),), Adjudication, policy)
    return {
        "task_id": row["task_id"],
        "task_digest": row["task_digest"],
        "model": model.model,
        "adjudication": result.value.model_dump(mode="json"),
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
                    results.append(await _adjudicate(row, worker.model, worker.policy))
                except Exception as error:
                    results.append(
                        {
                            "task_id": row["task_id"],
                            "task_digest": row["task_digest"],
                            "model": worker.model.model,
                            "adjudication_error": {
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
    parser.add_argument("--first-audit", default=FIRST_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument("--api-key-env", default=GLM_BULK_TOKEN_ENV)
    parser.add_argument("--relay-job", default="/muchanem/glm53-relay-tok-02a")
    parser.add_argument("--relay-service", default="glm-5.3-tok")
    args = parser.parse_args()
    rows = load_disputed(args.tasks, args.first_audit)
    if not rows or args.max_workers < 1:
        parser.error("expected disputed tasks and positive --max-workers")
    port = OpenAIModelPort(args.model, args.api_key_env, relay_job=args.relay_job, relay_service=args.relay_service)
    pipeline = (
        Dataset.from_list(rows)
        .reshard(len(rows))
        .map_shard(partial(run_shard, port=port))
        .reshard(1)
        .write_jsonl(args.output)
    )
    with ZephyrContext(
        max_workers=min(args.max_workers, len(rows)),
        name="biotasks-reference-audit-adjudication",
        heartbeat_timeout=1200,
    ) as context:
        result = context.execute(pipeline)
    print(f"tasks={len(rows)} output={args.output}")
    print(result)


if __name__ == "__main__":
    main()
