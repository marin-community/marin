# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate TaskSpecs for catalog capabilities with the agentic Taskforge: one GLM agent for each capability.

The agent and its reviewer use GLM, and the solver rollouts use DeepSeek by default. The agent's shell runs in a
bubblewrap sandbox, so the Iris job needs ``--container-profile container_profile_privileged``. ``--pool`` selects
the GLM bearer token as in ``generate_tasks``, and ``PARALLEL_API_KEY`` holds the key of the agent's web tools.
Inside an Iris task::

    uv run --package marin-taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.generate_tasks_agentic \\
        --output-dir s3://<bucket>/<prefix> --relay-job /muchanem/glm53-relay --capability d43.culinary.scaling

``--ideas`` reads the ideas from a JSONL file of ``CapabilityIdea`` records instead of the catalog, for example the
output of ``analyze_failed_runs``.

For each capability the driver writes ``<capability id>/result.json`` (the agent's report, failure, tokens and tool
spend), ``<capability id>/agent.json`` (the agent's conversations), and ``<capability id>/tasks/<task id>/task.json``
and ``evidence.json`` for each accepted task. ``summary.json`` lists the results of all capabilities.
"""

import asyncio
import dataclasses
import json
import logging
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
import openai
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.chat import ChatModel
from rolloutengine.models import RolloutLimits
from shellbox.backends.local.machine import LocalMachineFactory
from taskforge.agentic.agent import AgentResult, generate_tasks
from taskforge.agentic.config import AgenticConfig, ToolBudget
from taskforge.agentic.gate import Gate
from taskforge.agentic.parallel import ParallelWeb
from taskforge.capability import CapabilityIdea
from taskforge.llm.glm_client import EndpointSource, GlmClient, Pool
from taskforge.llm.policy import LLMPolicy

from experiments.post_training.capability_driven_envs.catalog import example_catalog_path
from experiments.post_training.capability_driven_envs.generate_tasks import (
    GLM_PROVIDER,
    endpoint_source,
    selected_ideas,
)
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_MODEL

logger = logging.getLogger(__name__)

PARALLEL_API_KEY_ENV = "PARALLEL_API_KEY"
DEEPSEEK_MODEL = "deepseek-v4.1-flash"
"""Served beside GLM on the same relay and tokens; a model of another family makes the solver rollouts independent."""
SUMMARY_FILE = "summary.json"
MAX_OUTPUT_TOKENS = 32768
GATE = Gate(min_solver_runs=8, solver_band=(0.1, 0.8), max_format_failure_rate=0.25, min_adversary_runs=4)
COMMAND_TIMEOUT = 30.0
SOLVER_LIMITS = RolloutLimits(
    max_turns=30,
    max_total_tokens=None,
    model_turn_timeout=900,
    tool_turn_timeout=120,
    total_turn_timeout=None,
    attempt_timeout=3600,
    verifier_timeout=120,
    cleanup_timeout=60,
)


def tool_budget(tasks: int) -> ToolBudget:
    """A budget that permits about five rounds of evidence for each task, and research for the idea."""
    return ToolBudget(
        solver_rollouts=64 * tasks,
        judge_calls=4 * tasks + 2,
        web_calls=10 + 2 * tasks,
    )


def agent_limits(turns: int) -> RolloutLimits:
    return RolloutLimits(
        max_turns=turns,
        max_total_tokens=None,
        model_turn_timeout=1800,
        tool_turn_timeout=3600,
        total_turn_timeout=None,
        attempt_timeout=6 * 3600,
        verifier_timeout=None,
        cleanup_timeout=60,
    )


def workspace_factory() -> LocalMachineFactory:
    """A bubblewrap sandbox whose shell has this environment's Python, and with it taskcompendium and verifyit.

    It mounts the directory of the virtual environment read-only, for example ``/app`` with ``/app/.venv`` on Iris.
    Bubblewrap needs namespaces, which an Iris task has only under the privileged container profile.
    """
    venv = Path(sys.prefix)
    return LocalMachineFactory(read_only=(venv.parent,), bin_dirs=(venv / "bin",))


def chat_model(base_url: str, token: str, model: str, reasoning_effort: str) -> ChatModel:
    """A chat model on the relay. DeepSeek switches its reasoning on with ``thinking``, GLM sets an effort."""
    client = openai.AsyncOpenAI(base_url=base_url, api_key=token, timeout=1800, max_retries=5)
    template = {"thinking": True} if model == DEEPSEEK_MODEL else {"reasoning_effort": reasoning_effort}
    return ChatModel(client, model, MAX_OUTPUT_TOKENS, extra_body={"chat_template_kwargs": template})


async def generate_all(
    source: EndpointSource,
    ideas: Sequence[CapabilityIdea],
    *,
    tasks: int,
    turns: int,
    agent_reasoning_effort: str,
    solver_model: str,
    solver_reasoning_effort: str,
    concurrency: int,
    output: StoragePath,
) -> list[dict[str, Any]]:
    """Run one agent for each idea, at most ``concurrency`` at a time, and write each result when it finishes."""
    endpoint = source.current()
    agent = chat_model(endpoint.base_url, endpoint.token, GLM_MODEL, agent_reasoning_effort)
    solver = chat_model(endpoint.base_url, endpoint.token, solver_model, solver_reasoning_effort)
    slots = asyncio.Semaphore(concurrency)
    async with GlmClient(source, GLM_PROVIDER) as judge, ParallelWeb(os.environ[PARALLEL_API_KEY_ENV]) as web:
        config = AgenticConfig(
            agent_model=agent,
            workspace_factory=workspace_factory(),
            solver_model=solver,
            judge=judge,
            judge_policy=LLMPolicy(),
            web=web,
            agent_limits=agent_limits(turns),
            solver_limits=SOLVER_LIMITS,
            gate=GATE,
            budget=tool_budget(tasks),
            command_timeout=COMMAND_TIMEOUT,
            max_concurrent_rollouts=8,
        )

        async def one(idea: CapabilityIdea) -> dict[str, Any]:
            # Each session sends its whole conversation on every turn, so a cache key of its own lets the relay
            # reuse the prefix.
            cache_key = {"prompt_cache_key": f"taskforge-agentic-{idea.capability_id}-{uuid.uuid4().hex[:8]}"}
            model = dataclasses.replace(agent, extra_body={**agent.extra_body, **cache_key})
            async with slots:
                result = await generate_tasks(idea, tasks, dataclasses.replace(config, agent_model=model))
            logger.info("%s: %d tasks, failure %s", idea.capability_id, len(result.tasks), result.failure)
            return write_result(output, result)

        return await asyncio.gather(*(one(idea) for idea in ideas))


def read_ideas(url: str) -> list[CapabilityIdea]:
    """The ``CapabilityIdea`` records of a JSONL file, one on each line."""
    lines = StoragePath(url).read_text().splitlines()
    return [CapabilityIdea.model_validate_json(line) for line in lines if line.strip()]


def write_result(output: StoragePath, result: AgentResult) -> dict[str, Any]:
    """Write the files of one idea under ``output`` and return its line of the summary."""
    directory = output / result.idea.capability_id
    for accepted in result.tasks:
        task_directory = directory / "tasks" / accepted.task.id
        _write(task_directory / "task.json", accepted.task.model_dump_json(indent=2))
        _write(task_directory / "evidence.json", json.dumps(asdict(accepted.evidence), indent=2))
        _write(task_directory / "review.json", accepted.review.model_dump_json(indent=2))
    conversations = [list(conversation.messages) for conversation in result.rollout.conversations]
    _write(directory / "agent.json", json.dumps(conversations, indent=1, default=str))
    summary = {
        "capability_id": result.idea.capability_id,
        "tasks": [accepted.task.id for accepted in result.tasks],
        "report": result.report,
        "failure": result.failure,
        "stop_reason": result.rollout.stop_reason,
        "agent_tokens": result.agent_tokens,
        "spend": asdict(result.spend),
    }
    _write(directory / "result.json", json.dumps(summary, indent=2))
    return summary


def _write(path: StoragePath, text: str) -> None:
    if path.scheme is None:
        Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@click.command()
@click.option("--output-dir", required=True, help="A local directory or an object-store URL for the outputs.")
@click.option(
    "--catalog",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The capability catalog JSON file. Default: the package's data/example_catalog.json.",
)
@click.option("--ideas", "ideas_url", help="A JSONL file of CapabilityIdea records, in place of --catalog.")
@click.option("--capability", "capabilities", multiple=True, help="Only this capability id; repeatable.")
@click.option("--tasks", type=click.IntRange(min=1), default=4, show_default=True, help="Tasks for each capability.")
@click.option("--base-url", help="The GLM router's OpenAI-compatible base URL, ending in /v1.")
@click.option("--relay-job", help=f"The Iris GLM relay job, e.g. {DEFAULT_GLM_RELAY_JOB} (inside Iris only).")
@click.option("--pool", type=click.Choice([pool.value for pool in Pool]), default=Pool.BULK.value, show_default=True)
@click.option("--turns", type=click.IntRange(min=1), default=200, show_default=True, help="Agent turns per idea.")
# A GLM agent turn at medium effort can spend the whole output budget on reasoning, and a turn cut at the limit ends
# the session, so the agent reasons at low effort by default.
@click.option("--agent-reasoning-effort", default="low", show_default=True, help="GLM reasoning effort of the agent.")
@click.option(
    "--solver-model",
    type=click.Choice([DEEPSEEK_MODEL, GLM_MODEL]),
    default=DEEPSEEK_MODEL,
    show_default=True,
    help="The model of the solver and adversary rollouts.",
)
@click.option("--solver-reasoning-effort", default="medium", show_default=True, help="GLM reasoning effort of solvers.")
@click.option("--concurrency", type=click.IntRange(min=1), default=4, show_default=True, help="Ideas at once.")
def main(
    output_dir: str,
    catalog: Path | None,
    ideas_url: str | None,
    capabilities: tuple[str, ...],
    tasks: int,
    base_url: str | None,
    relay_job: str | None,
    pool: str,
    turns: int,
    agent_reasoning_effort: str,
    solver_model: str,
    solver_reasoning_effort: str,
    concurrency: int,
) -> None:
    """Generate TaskSpecs for catalog capabilities, with one GLM agent for each capability."""
    configure_logging(logging.INFO)
    if PARALLEL_API_KEY_ENV not in os.environ:
        raise click.ClickException(f"{PARALLEL_API_KEY_ENV} is not set")
    source = endpoint_source(base_url, relay_job, Pool(pool))
    if ideas_url is not None and catalog is not None:
        raise click.UsageError("give at most one of --ideas and --catalog")
    if ideas_url is None:
        ideas = list(selected_ideas(catalog or example_catalog_path(), capabilities).values())
    else:
        ideas = [idea for idea in read_ideas(ideas_url) if not capabilities or idea.capability_id in capabilities]
    output = StoragePath(output_dir)
    summary = asyncio.run(
        generate_all(
            source,
            ideas,
            tasks=tasks,
            turns=turns,
            agent_reasoning_effort=agent_reasoning_effort,
            solver_model=solver_model,
            solver_reasoning_effort=solver_reasoning_effort,
            concurrency=concurrency,
            output=output,
        )
    )
    _write(output / SUMMARY_FILE, json.dumps(summary, indent=2))
    click.echo(output / SUMMARY_FILE)


if __name__ == "__main__":
    main()
