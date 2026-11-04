# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run trial-level CTT solver calibration for the Kang IFN-beta demo tasks on Zephyr."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from fray.types import ResourceConfig
from rolloutengine.models import RolloutLimits
from shellbox.machine import Backend, CommandSemantics
from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig, ReasoningEffort
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import GlmRolloutModel
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
from taskforge_adapters.glm_client import FixedEndpoint, GlmClient, GlmEndpoint, Pool, RelayEndpoint
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.kang_ifnb_tasks.generate_tasks import generate
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

IRIS_CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"
DEFAULT_RELAY_JOB = "/muchanem/glm53-relay-tok-02a"
DEFAULT_RELAY_SERVICE = "glm-5.3-tok"
POLICY = LLMPolicy(
    max_tokens=4096,
    reasoning_effort=ReasoningEffort.LOW,
    temperature=0.7,
    top_p=0.95,
    max_continuations=0,
)
PROVIDER = LLMProviderConfig(
    model=GLM_MODEL,
    max_attempts=4,
    hold_timeout=900,
    backoff_initial=1,
    backoff_maximum=30,
    backoff_factor=2,
    stall_timeout=300,
    max_connections=16,
    structured_repairs=None,
    context_window=262144,
)
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
class CTTConfig:
    relay_job: str = DEFAULT_RELAY_JOB
    relay_service: str = DEFAULT_RELAY_SERVICE
    relay_base_url: str | None = None
    token_env: str = GLM_BULK_TOKEN_ENV
    model: str = GLM_MODEL
    k: int = 3
    max_retries: int = 1
    shellsim: bool = False


def _execution_task(task: TaskSpec, shellsim: bool) -> TaskSpec:
    if not shellsim:
        return task
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


def _summary(outcome: object, trial: int) -> dict[str, object]:
    tokens_in, tokens_out = _tokens(outcome)
    rollout = getattr(outcome, "rollout", None)
    common: dict[str, object] = {
        "trial": trial,
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


async def _calibrate(task: TaskSpec, config: CTTConfig, root: Path) -> dict[str, object]:
    token = os.environ.get(config.token_env)
    if not token:
        raise RuntimeError(f"worker lacks {config.token_env}")
    controller_url = os.environ.get(IRIS_CONTROLLER_URL_ENV)
    if not controller_url:
        raise RuntimeError(f"worker lacks {IRIS_CONTROLLER_URL_ENV}")
    if config.model != GLM_MODEL:
        raise ValueError(f"relay supports {GLM_MODEL}, not {config.model}")

    frozen_task_digest = digest(task)
    execution_task = _execution_task(task, config.shellsim)
    backend = Backend.SHELLSIM if config.shellsim else Backend.GVISOR
    factories = machine_factories((backend,), controller_url)
    settings = EngineSettings(factories, COMMAND_TIMEOUT, LIMITS)
    lowered = await preflight(execution_task, MachineHost.IRIS, settings)
    ledger_root = root / "ledger"
    ledger = JsonlLedger(ledger_root)
    plan = TrialPlan(
        workpiece_id=task.id,
        round=0,
        kind=TrialKind.SOLVER,
        k=config.k,
        max_retries=config.max_retries,
        retry_backoff=RetryBackoff(initial=2, maximum=30, factor=2, jitter=0.1),
        evidence_dir=root,
        ledger=ledger,
        first_attempt=0,
    )
    endpoint = (
        FixedEndpoint(GlmEndpoint(config.relay_base_url, token, Pool.BULK))
        if config.relay_base_url is not None
        else RelayEndpoint(config.relay_job, token, Pool.BULK, config.relay_service)
    )
    started = time.monotonic()
    async with GlmClient(endpoint, PROVIDER) as client:
        model = GlmRolloutModel(client, POLICY, CallLedger(ledger, task.id, 0, str(TrialKind.SOLVER)))
        outcomes = await run_trials(lowered.spec, plan, settings, model)
    elapsed_seconds = time.monotonic() - started

    attempts = [json.loads(path.read_text()) for path in sorted((root / str(TrialKind.SOLVER)).rglob("attempt-*.json"))]
    ledger_path = ledger.path_for(task.id)
    ledger_entries = [entry.to_json() for entry in read_entries(ledger_path)] if ledger_path.is_file() else []
    evidence = Evidence({TrialKind.SOLVER: tuple(outcomes)})
    stats = evidence.reward_stats(TrialKind.SOLVER)
    return {
        "campaign_format": "kang-ifnb-ctt-trials-v1",
        "campaign_id": "kang-ifnb-demo-ctt-glm53-k3",
        "trial_type": "ordinary_solver",
        "task_id": task.id,
        "frozen_task_digest": frozen_task_digest,
        "task_digest": task_digest(lowered.spec, COMMAND_TIMEOUT),
        "execution_backend": backend.value,
        "task_source": task.source.model_dump(mode="json"),
        "task_family": task.tags[-1],
        "model": config.model,
        "relay_job": config.relay_job,
        "relay_service": config.relay_service,
        "relay_base_url": config.relay_base_url,
        "policy": asdict(POLICY),
        "policy_digest": digest(POLICY),
        "limits": LIMITS.model_dump(mode="json"),
        "k": config.k,
        "elapsed_seconds": elapsed_seconds,
        "graded": stats.graded,
        "solved": stats.solved,
        "mean_reward": stats.mean_reward,
        "trials": [_summary(outcome, index) for index, outcome in enumerate(outcomes)],
        "attempts": attempts,
        "ledger": ledger_entries,
    }


def run_task_shard(
    rows: Iterable[dict[str, object]], shard_info: object, *, config: CTTConfig
) -> Iterator[dict[str, object]]:
    del shard_info
    for row in rows:
        task = TaskSpec.model_validate_json(str(row["task_json"]))
        with tempfile.TemporaryDirectory(prefix=f"kang-ctt-{task.id}-") as temporary:
            yield asyncio.run(_calibrate(task, config, Path(temporary)))


def run_zephyr(tasks: list[TaskSpec], config: CTTConfig, output: str, max_workers: int) -> object:
    rows = [{"task_json": task.model_dump_json()} for task in tasks]
    dataset = Dataset.from_list(rows).reshard(len(rows))
    pipeline = dataset.map_shard(partial(run_task_shard, config=config)).reshard(1).write_jsonl(output)
    with ZephyrContext(
        max_workers=min(max_workers, len(rows)),
        resources=ResourceConfig(cpu=1, ram="4g", disk="16g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="2g", disk="8g", preemptible=False),
        name="kang-ifnb-ctt-calibration",
        heartbeat_timeout=1800,
    ) as context:
        return context.execute(pipeline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--relay-job", default=DEFAULT_RELAY_JOB)
    parser.add_argument("--relay-service", default=DEFAULT_RELAY_SERVICE)
    parser.add_argument(
        "--relay-base-url",
        help="Optional OpenAI-compatible /v1 URL, for example an Iris capability bridge to a cross-cluster relay",
    )
    parser.add_argument(
        "--shellsim",
        action="store_true",
        help="Run solver tools in ShellSim instead of launching gVisor sandboxes",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.count, args.k, args.max_workers) < 1:
        parser.error("--count, --k, and --max-workers must be positive")
    tasks = generate(args.counts.read_bytes(), source_revision=args.source_revision, count=args.count)
    config = CTTConfig(
        relay_job=args.relay_job,
        relay_service=args.relay_service,
        relay_base_url=args.relay_base_url,
        k=args.k,
        shellsim=args.shellsim,
    )
    result = run_zephyr(tasks, config, args.output, args.max_workers)
    print(f"output={args.output}")
    print(result)


if __name__ == "__main__":
    main()
