# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ten managed GLM agents review a pull request, and a lead agent reviews their reviews (#10165).

A Zephyr pipeline shards the ten review attempts over its workers. Each worker runs its shard with one
``ManagedAgent`` over one ``RolloutEngine``, so the attempts of a shard run at the same time. A second pipeline runs
the lead reviewer, which gets the ten reviews and checks them against the code. All agents use one harness, the
reviewer of ``review_agent``, and each review has its own focus.

Submit on Iris to a cluster with a GLM-5.3 relay, for example cw-us-east-02a:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --priority interactive \\
        --cpu 1 --memory 2g -e GLM_API_TOKEN "$GLM_API_TOKEN" \\
        -- python -m experiments.post_training.managed_agent.review_pr \\
        --pull-request 10203 --base <base commit> --head <head commit> --output s3://<bucket>/<directory>
"""

import argparse
import asyncio
import json
import logging
import os
import urllib.request
from collections.abc import Iterator
from typing import Any

import openai
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.agent import agent_harness
from rolloutengine.contracts import AttemptEnded, ModelTurnCompleted, RolloutEvent
from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import RolloutLimits
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.worker_context import zephyr_worker_ctx

from experiments.post_training.glm import (
    DEFAULT_GLM_RELAY_JOB,
    GLM_MODEL,
    GLM_REQUEST_ATTEMPTS,
    GLM_REQUEST_TIMEOUT,
    resolve_glm_base_url,
)
from experiments.post_training.managed_agent.chat_model import ChatModel
from experiments.post_training.managed_agent.managed_agent import AgentOutcome, ManagedAgent, agent_spec, agent_task
from experiments.post_training.managed_agent.review_agent import PullRequest, reviewer

logger = logging.getLogger(__name__)

REPOSITORY_URL = "https://github.com/marin-community/marin"
API_URL = "https://api.github.com/repos/marin-community/marin"
GLM_TOKEN_ENV = "GLM_API_TOKEN"
PULL_REQUEST_KEY = "pull_request"
RELAY_JOB_KEY = "relay_job"
REVIEW_TURNS_KEY = "review_turns"
REVIEWER = "reviewer"
WORKER_RESOURCES = ResourceConfig(cpu=1, ram="2g", disk="10g")

FOCUSES = (
    "the recorder: its limits, its deadlines, and the end of an attempt",
    "the agent harness: concurrent completion calls, cancellation, and agent shutdown",
    "the public API: names, signatures, and what a harness author must know",
    "the exact-token contract and the messages trace",
    "failure handling: the operation that an interrupted attempt reports, and what its rollout keeps",
    "grading: the task's grader, the host judge, and output capture",
    "the tests: whether they test behavior, and which behavior has no test",
    "the reference docs and docstrings: whether each sentence agrees with the code",
    "simplicity: code, options, and types that the pull request can delete",
    "downstream impact: what MarinSkyRL and other callers must change, and what can break without an error",
)


def _limits(max_turns: int) -> RolloutLimits:
    # GLM reasons before it answers, so one turn can take minutes. The tools read local files, so they are fast.
    return RolloutLimits(
        max_turns=max_turns,
        max_total_tokens=None,
        model_turn_timeout=900,
        tool_turn_timeout=120,
        total_turn_timeout=3600,
        attempt_timeout=4200,
        verifier_timeout=60,
        cleanup_timeout=60,
    )


LEAD_LIMITS = _limits(max_turns=16)


def fetch_pull_request(number: int, base: str, head: str) -> PullRequest:
    """The pull request's title and description from the GitHub API, and its diff between two pinned commits."""
    agent = {"User-Agent": "marin-managed-agent-poc"}
    description_request = urllib.request.Request(
        f"{API_URL}/pulls/{number}", headers={**agent, "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(description_request, timeout=60) as response:
        description = json.load(response)
    diff_request = urllib.request.Request(f"{REPOSITORY_URL}/compare/{base}...{head}.diff", headers=agent)
    with urllib.request.urlopen(diff_request, timeout=120) as response:
        diff = response.read().decode()
    return PullRequest(number, description["title"], description["body"] or "", diff)


def review_instructions(pull_request: PullRequest, focus: str) -> str:
    return (
        f"Review pull request #{pull_request.number} of {REPOSITORY_URL}: {pull_request.title}\n\n"
        f"Focus on {focus}. Other reviewers cover the other topics.\n\n"
        f"The pull request's description:\n\n{pull_request.body}"
    )


def lead_instructions(pull_request: PullRequest, reviews: list[dict[str, Any]]) -> str:
    sections = "\n\n".join(
        f"## {review['id']} (focus: {review['focus']})\n\n{review['answer'] or '(The reviewer gave no review.)'}"
        for review in reviews
    )
    return (
        f"You lead the review of pull request #{pull_request.number} of {REPOSITORY_URL}: {pull_request.title}\n\n"
        f"{len(reviews)} reviewers, each with one focus, wrote the reviews below. Check each finding against the code "
        "with your tools. Merge the findings that describe the same problem, reject the findings that the code does "
        "not support, and rank the rest by severity. Your answer is the final review in Markdown: a summary "
        "paragraph, then the confirmed findings, each with its severity, file:line, problem, fix, and the reviewers "
        "who found it, then the rejected findings, each with one sentence that says why.\n\n"
        f"{sections}"
    )


def glm_model(relay_job: str) -> ChatModel:
    client = openai.AsyncOpenAI(
        base_url=resolve_glm_base_url(relay_job),
        api_key=os.environ[GLM_TOKEN_ENV],
        timeout=GLM_REQUEST_TIMEOUT,
        max_retries=GLM_REQUEST_ATTEMPTS,
    )
    return ChatModel(client, GLM_MODEL, max_tokens=32_768, reasoning_effort="medium")


class ProgressLog:
    """Log each model turn and the end of each attempt, so that the job log shows the progress of every agent."""

    async def on_event(self, event: RolloutEvent) -> None:
        payload = event.payload
        if isinstance(payload, ModelTurnCompleted):
            tools = [call["function"]["name"] for call in payload.message.get("tool_calls") or ()]
            logger.info(
                "%s turn %d: %.0f s, %s, tools %s",
                event.run_id,
                payload.turn_index,
                payload.duration,
                payload.usage,
                tools,
            )
        elif isinstance(payload, AttemptEnded):
            logger.info("%s ended: %s %s", event.run_id, payload.stop_reason or payload.operation, payload.error or "")


def result(item: dict[str, Any], outcome: AgentOutcome) -> dict[str, Any]:
    """The record of one attempt: the answer, and what the rollout shows about how the agent got there."""
    rollout = outcome.rollout
    steps = [step for conversation in rollout.conversations for step in conversation.steps] if rollout else []
    usage = [step.turn.usage for step in steps if step.turn.usage is not None]
    return {
        "id": outcome.id,
        "focus": item["focus"],
        "answer": outcome.answer,
        "stop_reason": None if rollout is None else rollout.stop_reason,
        "conversations": 0 if rollout is None else len(rollout.conversations),
        "model_turns": len(steps),
        "tool_calls": sum(len(step.turn.message.get("tool_calls") or ()) for step in steps),
        "prompt_tokens": sum(turn_usage.prompt_tokens for turn_usage in usage),
        "completion_tokens": sum(turn_usage.completion_tokens for turn_usage in usage),
        "failure": outcome.failure,
    }


async def _run_attempts(
    items: list[dict[str, Any]], pull_request: PullRequest, model: ChatModel, limits: RolloutLimits
) -> list[dict[str, Any]]:
    harness = agent_harness(reviewer(pull_request, max_turns=limits.max_turns))
    engine = RolloutEngine({}, {REVIEWER: harness}, observer=ProgressLog())
    manager = ManagedAgent(engine, model, concurrency=len(items))
    by_id = {item["id"]: item for item in items}
    specs = [agent_spec(agent_task(item["id"], item["instructions"]), REVIEWER, limits) for item in items]
    return [result(by_id[outcome.id], outcome) async for outcome in manager.run(specs)]


def _run_shard(items: Iterator[dict[str, Any]], limits: RolloutLimits) -> Iterator[dict[str, Any]]:
    attempts = list(items)
    # A reshard to more shards than items leaves some shards empty.
    if not attempts:
        return
    worker = zephyr_worker_ctx()
    model = glm_model(worker.get_shared(RELAY_JOB_KEY))
    yield from asyncio.run(_run_attempts(attempts, worker.get_shared(PULL_REQUEST_KEY), model, limits))


def run_reviews(items: Iterator[dict[str, Any]], _shard: ShardInfo) -> Iterator[dict[str, Any]]:
    """A Zephyr shard of reviews: one ManagedAgent runs all of them at the same time."""
    return _run_shard(items, _limits(zephyr_worker_ctx().get_shared(REVIEW_TURNS_KEY)))


def run_lead_review(items: Iterator[dict[str, Any]], _shard: ShardInfo) -> Iterator[dict[str, Any]]:
    return _run_shard(items, LEAD_LIMITS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a pull request with managed GLM agents")
    parser.add_argument("--pull-request", type=int, required=True)
    parser.add_argument("--base", required=True, help="The base commit of the diff")
    parser.add_argument("--head", required=True, help="The head commit of the diff; the job bundle must hold it")
    parser.add_argument("--output", required=True, help="The directory for reviews.jsonl and lead_review.md")
    parser.add_argument("--relay-job", default=DEFAULT_GLM_RELAY_JOB, help="The GLM relay job of this cluster")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reviews", type=int, default=len(FOCUSES), help="The first N focuses, for a smoke run")
    parser.add_argument("--review-turns", type=int, default=20, help="The model turns of each review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pull_request = fetch_pull_request(args.pull_request, args.base, args.head)
    logger.info("Pull request #%d: %d changed files", pull_request.number, len(pull_request.file_diffs()))
    context = ZephyrContext(max_workers=args.workers, resources=WORKER_RESOURCES, name="managed-agent-review")
    context.put(PULL_REQUEST_KEY, pull_request)
    context.put(RELAY_JOB_KEY, args.relay_job)
    context.put(REVIEW_TURNS_KEY, args.review_turns)

    items = [
        {"id": f"review-{index:02d}", "focus": focus, "instructions": review_instructions(pull_request, focus)}
        for index, focus in enumerate(FOCUSES[: args.reviews])
    ]
    reviews_pipeline = Dataset.from_list(items).reshard(args.workers).map_shard(run_reviews)
    reviews = sorted(
        context.execute(reviews_pipeline, name=f"pr-{pull_request.number}-reviews").results, key=lambda r: r["id"]
    )
    for review in reviews:
        logger.info("%s", json.dumps({key: value for key, value in review.items() if key != "answer"}))

    lead_item = {"id": "lead-review", "focus": "all", "instructions": lead_instructions(pull_request, reviews)}
    lead_pipeline = Dataset.from_list([lead_item]).map_shard(run_lead_review)
    [lead] = context.execute(lead_pipeline, name=f"pr-{pull_request.number}-lead-review").results
    logger.info("%s", json.dumps({key: value for key, value in lead.items() if key != "answer"}))

    output = StoragePath(args.output)
    (output / "reviews.jsonl").write_text("".join(json.dumps(record) + "\n" for record in [*reviews, lead]))
    (output / "lead_review.md").write_text(lead["answer"])
    logger.info("Wrote %s and %s", output / "reviews.jsonl", output / "lead_review.md")
    print(lead["answer"], flush=True)


if __name__ == "__main__":
    main()
