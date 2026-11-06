# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one GLM-5.3 trial per combined BioTask through the generic OpenAI-compatible runner."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Self

import fsspec
import openai
from fray.types import ResourceConfig
from iris.client.client import iris_ctx
from iris.cluster.types import JobName
from rolloutengine.chat import ChatModel
from rolloutengine.models import ModelRequest, ModelTurn, RolloutLimits
from shellbox.machine import Backend, CommandSemantics
from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.sandbox.factories import MachineHost
from taskforge.spec.draft import SHELLSIM_CPYTHON
from taskforge.stage50_validate.evidence import Evidence
from taskforge.stage50_validate.outcome import Graded, TrialKind
from taskforge.stage50_validate.preflight import preflight
from taskforge.stage50_validate.trials import (
    EngineSettings,
    RetryBackoff,
    TrialPlan,
    rollout_steps,
    run_trials,
    task_digest,
)
from taskforge_adapters.backends import machine_factories
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

IRIS_CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"
DEFAULT_RELAY_JOB = "/muchanem/glm53-relay-tok-02a"
DEFAULT_RELAY_SERVICE = "glm-5.3-tok"
DEFAULT_TASKS = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/"
    "campaigns/biotasks/combined/v1/snapshots/complete-1/taskspecs.jsonl"
)
DEFAULT_OUTPUT = (
    "s3://marin-us-east-02a/marin4sc/taskforge/task-library/"
    "campaigns/biotasks/combined/v1/calibration/glm-5.3-openai-shellsim-v1/results.jsonl"
)
MAX_TOKENS = 4096
TEMPERATURE = 0.0
TOP_P = 1.0
LIMITS = RolloutLimits(
    max_turns=8,
    max_total_tokens=8192,
    model_turn_timeout=600,
    tool_turn_timeout=180,
    total_turn_timeout=1200,
    attempt_timeout=1500,
    verifier_timeout=None,
    cleanup_timeout=60,
)
COMMAND_TIMEOUT = 120.0


@dataclass(frozen=True)
class CalibrationConfig:
    relay_job: str = DEFAULT_RELAY_JOB
    relay_service: str = DEFAULT_RELAY_SERVICE
    base_url: str | None = None
    api_key_env: str = GLM_BULK_TOKEN_ENV
    model: str = GLM_MODEL
    max_retries: int = 1


@dataclass(frozen=True)
class OpenAIEndpointRolloutModel:
    """Bind the generic OpenAI chat transport to Taskforge's per-attempt model protocol."""

    chat: ChatModel
    temperature: float
    top_p: float

    def for_step(self, step: str) -> Self:
        del step
        return self

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        options = {"temperature": self.temperature, "top_p": self.top_p, **request.options}
        return await self.chat(replace(request, options=options))


def _resolve_base_url(config: CalibrationConfig) -> str:
    if config.base_url is not None:
        return config.base_url.rstrip("/")
    client = iris_ctx().client
    if client is None:
        raise RuntimeError("Iris client is unavailable inside the worker")
    endpoints = client.resolver_for_job(JobName.from_string(config.relay_job)).resolve(config.relay_service).endpoints
    if not endpoints:
        raise RuntimeError("the GLM relay has no registered token-routed endpoint")
    base_url = endpoints[0].url.rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _execution_task(task: TaskSpec) -> TaskSpec:
    requirements = task.environment_requirements.model_copy(
        update={
            "command_semantics": CommandSemantics.SHELL_SIMULATOR,
            "docker_image": None,
            "package_spec": SHELLSIM_CPYTHON,
        }
    )
    return task.model_copy(update={"environment_requirements": requirements})


def _tokens(outcome: object) -> tuple[int, int]:
    rollout = getattr(outcome, "rollout", None)
    if rollout is None:
        return 0, 0
    usage = [step.turn.usage for step in rollout_steps(rollout) if step.turn.usage is not None]
    return sum(item.prompt_tokens for item in usage), sum(item.completion_tokens for item in usage)


def _summary(outcome: object) -> dict[str, object]:
    tokens_in, tokens_out = _tokens(outcome)
    rollout = getattr(outcome, "rollout", None)
    common: dict[str, object] = {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "turns": 0 if rollout is None else len(rollout_steps(rollout)),
        "stop_reason": None if rollout is None else rollout.stop_reason,
    }
    if isinstance(outcome, Graded):
        return {
            **common,
            "status": "graded",
            "reward": outcome.reward,
            "correct": outcome.reward == 1.0,
            "candidate_code_error": outcome.candidate_code_error,
        }
    return {
        **common,
        "status": "ungraded",
        "reward": None,
        "correct": None,
        "cause": str(getattr(outcome, "cause", "unknown")),
        "detail": str(getattr(outcome, "detail", ""))[-3000:],
    }


async def _calibrate(task: TaskSpec, config: CalibrationConfig, root: Path) -> dict[str, object]:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"worker lacks {config.api_key_env}")
    controller_url = os.environ.get(IRIS_CONTROLLER_URL_ENV)
    if not controller_url:
        raise RuntimeError(f"worker lacks {IRIS_CONTROLLER_URL_ENV}")

    frozen_task_digest = digest(task)
    execution_task = _execution_task(task)
    factories = machine_factories((Backend.SHELLSIM,), controller_url)
    settings = EngineSettings(factories, COMMAND_TIMEOUT, LIMITS)
    lowered = await preflight(execution_task, MachineHost.IRIS, settings)
    ledger = JsonlLedger(root / "ledger")
    plan = TrialPlan(
        workpiece_id=task.id,
        round=0,
        kind=TrialKind.SOLVER,
        k=1,
        max_retries=config.max_retries,
        retry_backoff=RetryBackoff(initial=2, maximum=30, factor=2, jitter=0.1),
        evidence_dir=root,
        ledger=ledger,
        first_attempt=0,
    )
    base_url = _resolve_base_url(config)
    started = time.monotonic()
    async with openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=600,
        max_retries=3,
    ) as client:
        model = OpenAIEndpointRolloutModel(
            ChatModel(client, config.model, MAX_TOKENS, context_window=262144),
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        outcomes = await run_trials(lowered.spec, plan, settings, model)
    elapsed_seconds = time.monotonic() - started

    attempts = [json.loads(path.read_text()) for path in sorted((root / str(TrialKind.SOLVER)).rglob("attempt-*.json"))]
    ledger_path = ledger.path_for(task.id)
    ledger_entries = [entry.to_json() for entry in read_entries(ledger_path)] if ledger_path.is_file() else []
    outcome = outcomes[0]
    stats = Evidence({TrialKind.SOLVER: tuple(outcomes)}).reward_stats(TrialKind.SOLVER)
    policy = {
        "transport": "openai_chat_completions",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    return {
        "campaign_format": "biotasks-openai-calibration-v1",
        "campaign_id": "biotasks/combined/v1/calibration/glm-5.3-openai-shellsim-v1",
        "trial_type": "ordinary_solver",
        "task_id": task.id,
        "frozen_task_digest": frozen_task_digest,
        "execution_task_digest": task_digest(lowered.spec, COMMAND_TIMEOUT),
        "execution_backend": Backend.SHELLSIM.value,
        "task_source": task.source.model_dump(mode="json"),
        "task_family": task.tags[-1],
        "model": config.model,
        "endpoint_kind": "openai_compatible",
        "relay_job": config.relay_job,
        "relay_service": config.relay_service,
        "explicit_base_url": config.base_url,
        "policy": policy,
        "policy_digest": digest(policy),
        "limits": LIMITS.model_dump(mode="json"),
        "elapsed_seconds": elapsed_seconds,
        "graded": stats.graded,
        "solved": stats.solved,
        "mean_reward": stats.mean_reward,
        "trial": _summary(outcome),
        "attempts": attempts,
        "ledger": ledger_entries,
    }


def run_task_shard(
    rows: Iterable[dict[str, object]], shard_info: object, *, config: CalibrationConfig
) -> Iterator[dict[str, object]]:
    del shard_info
    for row in rows:
        task = TaskSpec.model_validate_json(str(row["task_json"]))
        with tempfile.TemporaryDirectory(prefix=f"biotasks-glm-{task.id}-") as temporary:
            yield asyncio.run(_calibrate(task, config, Path(temporary)))


def _load_tasks(uri: str) -> list[TaskSpec]:
    with fsspec.open(uri, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    tasks = [TaskSpec.model_validate(row["task"]) for row in rows]
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("TaskSpec input contains duplicate task IDs")
    return tasks


def run_zephyr(tasks: list[TaskSpec], config: CalibrationConfig, output: str, max_workers: int) -> object:
    rows = [{"task_json": task.model_dump_json()} for task in tasks]
    pipeline = (
        Dataset.from_list(rows)
        .reshard(len(rows))
        .map_shard(partial(run_task_shard, config=config))
        .reshard(1)
        .write_jsonl(output)
    )
    with ZephyrContext(
        max_workers=min(max_workers, len(rows)),
        resources=ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="2g", disk="8g", preemptible=False),
        name="biotasks-combined-glm-openai-calibration",
        heartbeat_timeout=1800,
    ) as context:
        return context.execute(pipeline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Run only the first N TaskSpecs for an infrastructure smoke")
    parser.add_argument("--relay-job", default=DEFAULT_RELAY_JOB)
    parser.add_argument("--relay-service", default=DEFAULT_RELAY_SERVICE)
    parser.add_argument("--base-url", help="Explicit OpenAI-compatible /v1 URL; otherwise resolve the relay service")
    parser.add_argument("--api-key-env", default=GLM_BULK_TOKEN_ENV)
    parser.add_argument("--model", default=GLM_MODEL)
    args = parser.parse_args()
    if args.max_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-workers and --limit must be positive")
    tasks = _load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    config = CalibrationConfig(
        relay_job=args.relay_job,
        relay_service=args.relay_service,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
    )
    result = run_zephyr(tasks, config, args.output, args.max_workers)
    print(f"tasks={len(tasks)}")
    print(f"output={args.output}")
    print(result)


if __name__ == "__main__":
    main()
