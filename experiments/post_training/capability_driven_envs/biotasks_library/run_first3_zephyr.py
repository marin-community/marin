# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run first-three BioTasks ideas through Taskforge and an independent GLM review."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from functools import partial
from pathlib import Path

from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from taskforge_adapters.campaign import OpenAIModelPort, campaign_worker

from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import (
    Assessment,
    BioTaskIdea,
    ideas,
)
from experiments.post_training.capability_driven_envs.biotasks_library.taskforge_flow import run_idea
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.taskforge_zephyr import TaskforgeZephyrConfig, run_taskforge_zephyr

DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks/first3-objective-v1/results.jsonl"
)
DEFAULT_RELAY_SERVICE = "glm-5.3-tok"
REVIEW_RUBRIC_VERSION = "biotasks-utility-v1"

REVIEW_POLICY = LLMPolicy(
    max_tokens=2048,
    reasoning_effort=ReasoningEffort.LOW,
    temperature=0.0,
    top_p=1.0,
    max_continuations=0,
)


def review_prompt(idea: BioTaskIdea, taskforge_result: dict[str, object]) -> str:
    """Render a review packet without exposing the author's assessment to the reviewer."""
    public_idea = idea.model_dump(mode="json", exclude={"author_assessment"})
    task = None
    if "task_json" in taskforge_result:
        task = json.loads(str(taskforge_result["task_json"]))
    packet = {
        "idea": public_idea,
        "taskforge_terminal": taskforge_result["terminal"],
        "task": task,
    }
    return f"""You are reviewing a proposed computational-biology reward task. The author assessment is deliberately
hidden. Score each dimension from 1 (poor) to 5 (excellent):
- information_sufficiency: a solver can answer from solver-visible resources alone;
- biological_substance: the task tests biology or biologically meaningful analysis, not incidental arithmetic;
- scientific_validity: conclusions follow from the evidence and preserve limitations;
- objective_verifiability: the frozen answer and grader are deterministic and justified;
- hack_resistance: no answer leakage, obvious filename shortcut, ambiguous contract, or grader exploit;
- practical_usefulness: useful training signal at reasonable runtime and complexity.
List concrete blocking issues. recommended_action must be accept, revise, or reject. Confidence is 0 to 1. A
Taskforge structural acceptance is not evidence of scientific quality. A null proposal can correctly merit rejection
as a task even when its curation decision is sound.

Review rubric version: {REVIEW_RUBRIC_VERSION}
Review packet:
{json.dumps(packet, indent=2, sort_keys=True)}
"""


async def _run_one(row: dict[str, object], port: OpenAIModelPort) -> dict[str, object]:
    idea = BioTaskIdea.model_validate_json(str(row["idea_json"]))
    with tempfile.TemporaryDirectory(prefix="biotasks-first3-") as temporary:
        taskforge_result = await run_idea(Path(temporary), idea)
    prompt = review_prompt(idea, taskforge_result)
    async with campaign_worker(port) as worker:
        review = await worker.model.complete_structured(
            [{"role": "user", "content": prompt}],
            Assessment,
            REVIEW_POLICY,
        )
    return {
        **taskforge_result,
        "author_assessment": idea.author_assessment.model_dump(mode="json"),
        "glm_review": review.value.model_dump(mode="json"),
        "glm_review_provenance": {
            "model": port.model,
            "rubric_version": REVIEW_RUBRIC_VERSION,
            "policy": asdict(REVIEW_POLICY),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "completions": [asdict(completion) for completion in review.completions],
        },
    }


def run_shard(
    rows: Iterable[dict[str, object]],
    shard_info: object,
    *,
    port: OpenAIModelPort,
) -> Iterator[dict[str, object]]:
    del shard_info
    for row in rows:
        yield asyncio.run(_run_one(row, port))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument("--api-key-env", default=GLM_BULK_TOKEN_ENV)
    parser.add_argument("--relay-job", default=DEFAULT_GLM_RELAY_JOB)
    parser.add_argument("--relay-service", default=DEFAULT_RELAY_SERVICE)
    parser.add_argument("--base-url")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    port = (
        OpenAIModelPort(args.model, args.api_key_env, base_url=args.base_url)
        if args.base_url
        else OpenAIModelPort(
            args.model,
            args.api_key_env,
            relay_job=args.relay_job,
            relay_service=args.relay_service,
        )
    )
    rows = [{"idea_json": idea.model_dump_json()} for idea in ideas()]
    run_taskforge_zephyr(
        rows,
        run_shard=partial(run_shard, port=port),
        output=args.output,
        config=TaskforgeZephyrConfig(
            name="biotasks-first3-objective-v1",
            max_workers=args.max_workers,
            logical_shards=len(rows),
            region=None,
            preemptible=False,
        ),
    )


if __name__ == "__main__":
    main()
