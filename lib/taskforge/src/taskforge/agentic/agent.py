# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Make tasks for one capability idea with one managed agent.

``generate_tasks`` runs the agent as one ``RolloutEngine.run_task`` attempt: the shell harness serves a shell on the
workspace machine of ``AgenticConfig.workspace_factory`` and the Taskforge tools of ``tools``, and the agent task has
no grader. The agent orders its own steps (research, build programs, controls, rollouts), and ``submit`` decides
which tasks it accepts: the checks, the evidence rules of ``gate``, and an independent review. Run many ideas at the
same time with an ``asyncio`` task group or a Zephyr map.
"""

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import Rollout, RolloutInterrupted, TaskRolloutSpec, TraceMode
from shellbox.machine import Backend, PackageSpec
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ProviderRequirement,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import MODE_DESCRIPTORS

from taskforge.agentic import example_build
from taskforge.agentic.config import AgenticConfig
from taskforge.agentic.prompts import AGENT_PROMPT, USER_PROMPT
from taskforge.agentic.tools import (
    AGENT_TOOLS,
    MAX_EMPTY_BUDGET_CALLS,
    WORKDIR,
    AcceptedTask,
    TaskforgeTools,
    ToolSpend,
    machine_runtime,
    rollout_tokens,
)
from taskforge.capability import CapabilityIdea, capability_idea_record
from taskforge.content_hash import pretty_json

AGENT_HARNESS = "taskforge-agent"
AGENTIC_REVISION = "1"
"""The revision of the agent task in the ``Source`` of each agent attempt."""
WORKSPACE_SEMANTICS = {Backend.LOCAL: CommandSemantics.LINUX_PROCESS, Backend.SHELLSIM: CommandSemantics.SHELL_SIMULATOR}
"""The command semantics of the agent's shell on each supported workspace backend."""
NOT_TEXT_MODES = frozenset({"predicted_action", "structure_closure"})
"""In-process verifyit modes that grade function calls or a prefill continuation, not a final answer."""


@dataclass(frozen=True)
class AgentResult:
    """The tasks that the agent made for one idea, with the agent's conversation and what it used."""

    idea: CapabilityIdea
    tasks: tuple[AcceptedTask, ...]
    report: str | None
    """The agent's report from ``finish``, or ``None`` when the session stopped otherwise."""
    rollout: Rollout
    """The agent's attempt: its conversation and its tool calls."""
    agent_tokens: int
    spend: ToolSpend
    failure: str | None
    """The failed operation and its cause, when the session failed. The tasks that it accepted before stay valid."""


async def generate_tasks(idea: CapabilityIdea, n: int, config: AgenticConfig) -> AgentResult:
    """Make up to ``n`` different tasks for ``idea``. A session that makes fewer gives a report that says why."""
    tools = TaskforgeTools(idea, n, config)
    harness = ShellHarness(command_timeout=config.command_timeout, tools={AGENT_TOOLS: tools})
    workspace = config.workspace_factory
    engine = RolloutEngine({workspace.backend: workspace}, {AGENT_HARNESS: harness})
    spec = TaskRolloutSpec(
        task=agent_task(idea, n, config),
        harness=AGENT_HARNESS,
        task_machine=machine_runtime(workspace.backend),
        verifier_machine=None,
        limits=config.agent_limits,
        trace=TraceMode.MESSAGES,
    )
    failure = None
    try:
        rollout = (await engine.run_task(spec, config.agent_model)).rollout
    except RolloutInterrupted as error:
        # A failed session, for example at a model outage, keeps the tasks that the gate accepted before it.
        rollout = error.rollout
        failure = f"{error.operation}: {error.rollout.failure}"
    return AgentResult(idea, tuple(tools.accepted), tools.report, rollout, rollout_tokens(rollout), tools.spend, failure)


def agent_task(idea: CapabilityIdea, n: int, config: AgenticConfig) -> TaskSpec:
    """The agent's own task: the prompts, and a workspace with the idea, the example program and the schema."""
    gate, budget = config.gate, config.budget
    low, high = gate.solver_band
    system = AGENT_PROMPT.format(
        n=n,
        min_solver_runs=gate.min_solver_runs,
        low=low,
        high=high,
        max_format_failures=f"{gate.max_format_failure_rate:.0%}",
        min_adversary_runs=gate.min_adversary_runs,
        solver_rollouts=budget.solver_rollouts,
        judge_calls=budget.judge_calls,
        web_calls=budget.web_calls,
        max_empty_budget_calls=MAX_EMPTY_BUDGET_CALLS,
    )
    workspace = WORKDIR.relative_to("/")
    files = (
        inline_resource(f"{workspace}/idea.json", pretty_json(capability_idea_record(idea)).encode()),
        inline_resource(f"{workspace}/example_build.py", Path(example_build.__file__).read_bytes()),
        inline_resource(
            f"{workspace}/reference/taskspec.schema.json", json.dumps(TaskSpec.model_json_schema(), indent=1).encode()
        ),
        inline_resource(f"{workspace}/reference/verifyit_specs.md", verifyit_reference().encode()),
    )
    return TaskSpec(
        id=f"taskforge-agentic-{idea.capability_id}",
        context=ConversationInput(
            events=(
                TextMessage(role="system", content=system),
                TextMessage(role="user", content=USER_PROMPT.format(n=n, capability_id=idea.capability_id)),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=WORKSPACE_SEMANTICS[config.workspace_factory.backend],
            # An empty package spec runs the shell on what the factory mounts, for example a Python environment.
            package_spec=PackageSpec(),
            working_directory=str(WORKDIR),
            tool_providers={"taskforge": ProviderRequirement(action_interface=AGENT_TOOLS, initial_state={})},
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="The gate of the Taskforge tools decides which tasks the agent made"),
        source=Source(
            dataset="taskforge-agentic",
            revision=idea.catalog_version,
            row=idea.capability_id,
            importer_revision=AGENTIC_REVISION,
        ),
        resources=ResourceGroups(worker=files),
    )


def verifyit_reference() -> str:
    """The signature and the documentation of each verifyit spec that grades an answer in process."""
    lines = [
        "# verifyit specs that grade a final answer in process",
        "",
        "Make a grader with `taskcompendium.grader.verifyit_package(spec)`. Put its `grader` in `TaskSpec.grader` and "
        "its `resources` in `TaskSpec.resources.verifier`. The `output` field of a spec is not used for these answers.",
    ]
    for mode, descriptor in MODE_DESCRIPTORS.items():
        if descriptor.candidate is None or mode in NOT_TEXT_MODES:
            continue
        spec = descriptor.spec_type
        lines += ["", f"## {mode}: {spec.__name__}{inspect.signature(spec)}"]
        doc = inspect.getdoc(spec) or ""
        if not doc.startswith(f"{spec.__name__}("):  # A dataclass without a docstring repeats its signature.
            lines += ["", doc]
    return "\n".join(lines) + "\n"
