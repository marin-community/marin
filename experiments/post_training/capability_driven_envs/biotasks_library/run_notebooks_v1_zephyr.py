# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Author and independently review the complete BioTasks notebooks/v1 cohort."""

import argparse
import asyncio
import hashlib
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from functools import partial
from pathlib import Path

import fsspec
from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from taskforge.llm.structured import IncompleteOutput, StructuredOutputError
from taskforge_adapters.campaign import OpenAIModelPort, campaign_worker

from experiments.post_training.capability_driven_envs.biotasks_library.notebooks_v1 import (
    GeneratedPortfolio,
    SourceRecord,
    extract_source_text,
    generation_prompt,
    hydrate_ideas,
)
from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import Assessment, BioTaskIdea
from experiments.post_training.capability_driven_envs.biotasks_library.run_first3_zephyr import review_prompt
from experiments.post_training.capability_driven_envs.biotasks_library.taskforge_flow import run_idea
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.taskforge_zephyr import TaskforgeZephyrConfig, run_taskforge_zephyr

ROOT = "s3://marin-us-east-02a/marin4sc/taskforge/task-library"
DEFAULT_CATALOG = f"{ROOT}/campaigns/biotasks/reviewed57-a342f41d-v1/source-catalog.jsonl"
DEFAULT_OUTPUT = f"{ROOT}/campaigns/biotasks/notebooks/v1/source-results.jsonl"
DEFAULT_RELAY_JOB = "/muchanem/glm53-relay-tok-02a"
DEFAULT_RELAY_SERVICE = "glm-5.3-tok"

GENERATION_POLICY = LLMPolicy(
    max_tokens=8192,
    reasoning_effort=ReasoningEffort.LOW,
    temperature=0.2,
    top_p=0.95,
    max_continuations=0,
)
REVIEW_POLICY = LLMPolicy(
    max_tokens=2048,
    reasoning_effort=ReasoningEffort.LOW,
    temperature=0.0,
    top_p=1.0,
    max_continuations=0,
)


def read_catalog(uri: str) -> list[dict[str, object]]:
    """Read and validate the pinned source catalog."""
    with fsspec.open(uri, "rt") as stream:
        rows = [SourceRecord.model_validate_json(line).model_dump(mode="json") for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"source catalog is empty: {uri}")
    digests = [str(row["source_sha256"]) for row in rows]
    if len(set(digests)) != len(digests):
        raise ValueError("source catalog contains duplicate objects")
    return rows


def _read_source(source: SourceRecord) -> bytes:
    with fsspec.open(source.object_uri, "rb") as stream:
        return stream.read()


async def _review(
    idea: BioTaskIdea,
    taskforge_result: dict[str, object],
    *,
    model: object,
    port: OpenAIModelPort,
) -> dict[str, object]:
    prompt = review_prompt(idea, taskforge_result)
    base = {
        **taskforge_result,
        "author_assessment": idea.author_assessment.model_dump(mode="json"),
    }
    try:
        review = await model.complete_structured(
            [{"role": "user", "content": prompt}],
            Assessment,
            REVIEW_POLICY,
        )
    except (IncompleteOutput, StructuredOutputError) as error:
        return {
            **base,
            "glm_review_error": {
                "type": type(error).__name__,
                "message": str(error),
                "completions": [asdict(completion) for completion in error.completions],
            },
        }
    return {
        **base,
        "glm_review": review.value.model_dump(mode="json"),
        "glm_review_provenance": {
            "model": port.model,
            "policy": asdict(REVIEW_POLICY),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "completions": [asdict(completion) for completion in review.completions],
        },
    }


async def _run_source(row: dict[str, object], port: OpenAIModelPort) -> dict[str, object]:
    source = SourceRecord.model_validate(row)
    source_text = extract_source_text(source, _read_source(source))
    prompt = generation_prompt(source, source_text)
    source_record = {
        "source": source.model_dump(mode="json"),
        "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "source_text_characters": len(source_text),
    }
    async with campaign_worker(port) as worker:
        prompts = [
            prompt,
            prompt + "\n\nSTRICT COMPACT RETRY: Return at most two ideas and at most 8,000 total resource characters. "
            "Keep each instruction and assessment concise.",
        ]
        failures: list[dict[str, object]] = []
        generated = None
        used_prompt = prompt
        for candidate_prompt in prompts:
            try:
                generated = await worker.model.complete_structured(
                    [{"role": "user", "content": candidate_prompt}],
                    GeneratedPortfolio,
                    GENERATION_POLICY,
                )
                used_prompt = candidate_prompt
                break
            except (IncompleteOutput, StructuredOutputError) as error:
                failures.append(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "completions": [asdict(completion) for completion in error.completions],
                    }
                )
        if generated is None:
            return {**source_record, "status": "generation_failed", "generation_failures": failures, "items": []}

        ideas: list[BioTaskIdea] = []
        hydration_errors: list[dict[str, str]] = []
        if generated.value.disposition == "no_objective_task":
            ideas = hydrate_ideas(source, generated.value)
        else:
            for candidate in generated.value.ideas:
                candidate_portfolio = generated.value.model_copy(update={"ideas": (candidate,)})
                try:
                    ideas.extend(hydrate_ideas(source, candidate_portfolio))
                except ValueError as error:
                    hydration_errors.append({"family": candidate.family, "error": str(error)})

        items: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="biotasks-notebooks-v1-") as temporary:
            root = Path(temporary)
            for idea in ideas:
                taskforge_result = await run_idea(root, idea)
                items.append(await _review(idea, taskforge_result, model=worker.model, port=port))
    return {
        **source_record,
        "status": "completed" if not hydration_errors else "completed_with_hydration_rejections",
        "portfolio": generated.value.model_dump(mode="json"),
        "generation_provenance": {
            "model": port.model,
            "policy": asdict(GENERATION_POLICY),
            "prompt_sha256": hashlib.sha256(used_prompt.encode()).hexdigest(),
            "prompt": used_prompt,
            "completions": [asdict(completion) for completion in generated.completions],
            "prior_failures": failures,
        },
        "hydration_errors": hydration_errors,
        "items": items,
    }


def run_shard(
    rows: Iterable[dict[str, object]],
    shard_info: object,
    *,
    port: OpenAIModelPort,
) -> Iterator[dict[str, object]]:
    del shard_info
    for row in rows:
        yield asyncio.run(_run_source(row, port))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=GLM_MODEL)
    parser.add_argument("--api-key-env", default=GLM_BULK_TOKEN_ENV)
    parser.add_argument("--relay-job", default=DEFAULT_RELAY_JOB)
    parser.add_argument("--relay-service", default=DEFAULT_RELAY_SERVICE)
    parser.add_argument("--base-url")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--limit", type=int)
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
    rows = read_catalog(args.catalog)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    run_taskforge_zephyr(
        rows,
        run_shard=partial(run_shard, port=port),
        output=args.output,
        config=TaskforgeZephyrConfig(
            name="biotasks-notebooks-v1",
            max_workers=args.max_workers,
            logical_shards=len(rows),
            region=None,
            preemptible=False,
        ),
    )


if __name__ == "__main__":
    main()
