# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Put the tasks of the staged generator through the gate of the agentic Taskforge, to compare the two on one idea set.

For each idea, ``generate_tasks.generate_task`` writes ``--tasks`` task programs with one GLM call each, and runs them.
Then a scripted session of ``taskforge.agentic.agent.generate_tasks``, with no model of its own, takes each TaskSpec
that a program wrote through the evidence and the review that the agent's tasks get: the grader's own positive
candidate as the correct control, a wrong control, 8 solver and 4 adversary rollouts, and ``submit`` with the
program. Thus the GLM tokens of the staged run are the tokens of the generator and of the reviewer. ``submit`` runs
the program again, so the session's workspace is a bubblewrap sandbox with Python, and the job needs
``--container-profile container_profile_privileged``::

    uv run --package marin-taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.compare_staged \\
        --output-dir s3://<bucket>/<prefix> --relay-job /muchanem/glm53-relay --ideas s3://<bucket>/<ideas>.jsonl

For each idea the driver writes ``<capability id>/staged/<workpiece id>/`` with the GLM completions, the program and
its task, ``<capability id>/tasks/<task id>/`` for each accepted task as ``generate_tasks_agentic`` does, and
``<capability id>/result.json``. ``summary.json`` lists the results of all ideas.
"""

import asyncio
import dataclasses
import json
import logging
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import click
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rolloutengine.models import ModelRequest, ModelTurn, TokenUsage
from taskcompendium.models import TaskSpec, VerifyitGrader
from taskforge.agentic.agent import generate_tasks
from taskforge.agentic.config import AgenticConfig, ToolBudget
from taskforge.agentic.parallel import ParallelWeb
from taskforge.capability import CapabilityIdea
from taskforge.content_hash import pretty_json
from taskforge.llm.glm_client import EndpointSource, GlmClient, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.task_program import PROGRAM_FILE, TASK_FILE
from verifyit.grade import negative_candidate, positive_candidate
from verifyit.spec import StructuredExactSpec, spec_from_table

from experiments.post_training.capability_driven_envs.generate_tasks import (
    GLM_PROVIDER,
    GeneratedTask,
    endpoint_source,
    generate_task,
)
from experiments.post_training.capability_driven_envs.generate_tasks_agentic import (
    COMMAND_TIMEOUT,
    DEEPSEEK_MODEL,
    GATE,
    SOLVER_LIMITS,
    SUMMARY_FILE,
    _write,
    agent_limits,
    chat_model,
    read_ideas,
    workspace_factory,
    write_result,
)
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB

logger = logging.getLogger(__name__)

PROGRAM_TIMEOUT = 300.0
WRITE_CHUNK = 32_768
"""Characters of a task file in one shell command, well below the limit of one argument."""
WRONG_ANSWER = "0"
"""The wrong control of a grader that has no negative candidate of its own."""


def tool_calls(task_files: dict[str, tuple[TaskSpec, str]]) -> list[tuple[str, dict[str, object]]]:
    """The tool calls that write each task and its program, give the task the evidence of the gate, submit it with
    the program, and then finish."""
    calls: list[tuple[str, dict[str, object]]] = [("shell", {"command": "mkdir -p tasks programs"})]
    for path, (task, source) in task_files.items():
        program = f"programs/{task.id}.py"
        for target, text in ((path, task.model_dump_json()), (program, source)):
            for start in range(0, len(text), WRITE_CHUNK):
                command = f"printf '%s' {shlex.quote(text[start : start + WRITE_CHUNK])} >> {target}"
                calls.append(("shell", {"command": command}))
        right, wrong = controls(task)
        if right is not None:
            calls.append(("grade", {"path": path, "answer": right, "expect": "pass"}))
        calls.append(("grade", {"path": path, "answer": wrong, "expect": "fail"}))
        calls.append(("rollout", {"paths": [path], "n": GATE.min_solver_runs, "role": "solver"}))
        calls.append(("rollout", {"paths": [path], "n": GATE.min_adversary_runs, "role": "adversary"}))
        calls.append(("submit", {"path": path, "program": program}))
    calls.append(("finish", {"report": "The scripted session submitted every task that the staged programs wrote."}))
    return calls


def controls(task: TaskSpec) -> tuple[str | None, str]:
    """The answer that the grader must accept, when the grader gives one, and an answer that it must reject."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader):
        return None, WRONG_ANSWER
    spec = spec_from_table({"mode": grader.mode, **grader.parameters})
    right = json.dumps(spec.expected) if isinstance(spec, StructuredExactSpec) else positive_candidate(spec)
    return right, negative_candidate(spec) or WRONG_ANSWER


@dataclass
class ScriptedSession:
    """An agent model that makes the calls of ``calls`` in order, one in each turn."""

    calls: list[tuple[str, dict[str, object]]]
    turns: int = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        name, arguments = self.calls[self.turns]
        self.turns += 1
        call = {
            "id": f"call-{self.turns}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        return ModelTurn(message, (), (), None, "tool_calls", usage=TokenUsage(0, 0))


def generator_tokens(generated: Sequence[GeneratedTask]) -> int:
    return sum(
        completion.usage.prompt_tokens + completion.usage.completion_tokens
        for workpiece in generated
        for completion in workpiece.completions
    )


async def compare_idea(
    idea: CapabilityIdea, tasks: int, glm: GlmClient, config: AgenticConfig, output: StoragePath
) -> dict[str, Any]:
    """Generate ``tasks`` staged tasks for ``idea``, validate them in one scripted session, and write the results."""
    started = time.monotonic()
    workpieces = [f"{idea.capability_id}--{sample:02d}" for sample in range(tasks)]
    generated = await asyncio.gather(
        *(generate_task(glm, idea, workpiece, LLMPolicy(), PROGRAM_TIMEOUT) for workpiece in workpieces)
    )
    directory = output / idea.capability_id
    for workpiece in generated:
        staged = directory / "staged" / workpiece.workpiece_id
        _write(staged / "completions.json", pretty_json(workpiece.completions))
        if workpiece.source is not None:
            _write(staged / PROGRAM_FILE, workpiece.source)
        if workpiece.run is not None and workpiece.run.task is not None:
            _write(staged / TASK_FILE, workpiece.run.task.model_dump_json(indent=2))
    task_files = {
        f"tasks/{workpiece.workpiece_id}.json": (workpiece.run.task, workpiece.source)
        for workpiece in generated
        if workpiece.run is not None and workpiece.run.task is not None and workpiece.source is not None
    }
    calls = tool_calls(task_files)
    session = ScriptedSession(calls)
    # The session ends at its last call, or earlier when ``tasks`` tasks are accepted.
    script_config = dataclasses.replace(config, agent_model=session, agent_limits=agent_limits(len(calls) + 1))
    result = await generate_tasks(idea, tasks, script_config)
    summary = write_result(output, result, time.monotonic() - started)
    summary.update(
        staged_tasks=len(task_files),
        staged_failures={workpiece.workpiece_id: workpiece.failure for workpiece in generated if workpiece.failure},
        generator_tokens=generator_tokens(generated),
    )
    _write(directory / "result.json", json.dumps(summary, indent=2))
    logger.info("%s: %d of %d staged tasks accepted", idea.capability_id, len(result.tasks), len(task_files))
    return summary


async def compare_all(
    source: EndpointSource, ideas: Sequence[CapabilityIdea], tasks: int, concurrency: int, output: StoragePath
) -> list[dict[str, Any]]:
    endpoint = source.current()
    solver = chat_model(endpoint.base_url, endpoint.token, DEEPSEEK_MODEL, "medium")
    slots = asyncio.Semaphore(concurrency)
    async with GlmClient(source, GLM_PROVIDER) as glm, ParallelWeb("unused") as web:
        config = AgenticConfig(
            agent_model=ScriptedSession([]),
            workspace_factory=workspace_factory(),
            solver_model=solver,
            judge=glm,
            judge_policy=LLMPolicy(),
            web=web,
            agent_limits=agent_limits(1),
            solver_limits=SOLVER_LIMITS,
            gate=GATE,
            budget=ToolBudget(
                solver_rollouts=(GATE.min_solver_runs + GATE.min_adversary_runs) * tasks,
                judge_calls=2 * tasks,
                web_calls=0,
            ),
            command_timeout=COMMAND_TIMEOUT,
            max_concurrent_rollouts=8,
        )

        async def one(idea: CapabilityIdea) -> dict[str, Any]:
            async with slots:
                return await compare_idea(idea, tasks, glm, config, output)

        return await asyncio.gather(*(one(idea) for idea in ideas))


@click.command()
@click.option("--output-dir", required=True, help="A local directory or an object-store URL for the outputs.")
@click.option("--ideas", "ideas_url", required=True, help="A JSONL file of CapabilityIdea records.")
@click.option("--tasks", type=click.IntRange(min=1), default=4, show_default=True, help="Staged tasks for each idea.")
@click.option("--base-url", help="The GLM router's OpenAI-compatible base URL, ending in /v1.")
@click.option("--relay-job", help=f"The Iris GLM relay job, e.g. {DEFAULT_GLM_RELAY_JOB} (inside Iris only).")
@click.option("--pool", type=click.Choice([pool.value for pool in Pool]), default=Pool.BULK.value, show_default=True)
@click.option("--concurrency", type=click.IntRange(min=1), default=8, show_default=True, help="Ideas at once.")
def main(
    output_dir: str, ideas_url: str, tasks: int, base_url: str | None, relay_job: str | None, pool: str, concurrency: int
) -> None:
    """Validate the staged generator's tasks with the agentic gate, for the same ideas as an agentic run."""
    configure_logging(logging.INFO)
    source = endpoint_source(base_url, relay_job, Pool(pool))
    output = StoragePath(output_dir)
    summary = asyncio.run(compare_all(source, read_ideas(ideas_url), tasks, concurrency, output))
    _write(output / SUMMARY_FILE, json.dumps(summary, indent=2))
    click.echo(output / SUMMARY_FILE)


if __name__ == "__main__":
    main()
