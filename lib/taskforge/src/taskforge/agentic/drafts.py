# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draft one task from one task proposal with a short agent session, and no model besides the agent.

``draft_task`` gives the agent the proposal, the example build program and the TaskSpec references, and two tools:
``submit``, which runs only the checks of ``taskforge.agentic.tools`` that need no model, and ``finish``. Nothing
grades, rolls out or reviews the task: a later step validates the drafts. A draft keeps its build program and the
workspace that the program reads, so ``task_instances`` can make more tasks of its kind from other task ids.
"""

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError, validate_call
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness, ToolCallResult, ToolContext
from rolloutengine.models import Model, Rollout, RolloutInterrupted, RolloutLimits, TaskRolloutSpec, TraceMode
from shellbox.machine import Backend, Machine, MachineFactory, PackageSpec
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    Grader,
    NoGrader,
    PlainText,
    ProviderRequirement,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from verifyit.modes.grade_judge import JudgeConnection

from taskforge.agentic import grader_examples
from taskforge.agentic.agent import SHELL_OUTPUT_BYTES, example_program, verifyit_reference
from taskforge.agentic.prompts import DRAFT_PROMPT, DRAFT_USER_PROMPT, NO_SOLVER_MACHINE, SOLVER_MACHINES
from taskforge.agentic.tools import (
    COMMAND_SEMANTICS,
    SOLVER_HARNESS,
    WORKDIR,
    ToolError,
    keyword_grader_problems,
    machine_runtime,
    malformed_files,
    program_problems,
    read_task,
    rollout_tokens,
    solver_spec,
    unsupported,
    workspace_snapshot,
)

DRAFT_TOOLS = "taskforge-drafts:v1"
"""The action interface of the draft tools, in the agent task's ``tool_providers``."""
DRAFT_HARNESS = "taskforge-draft"
DRAFT_REVISION = "2"
"""The revision of the draft agent task in the ``Source`` of each session."""
DRAFT_MAX_ANSWER_SHARE = 0.625
"""The largest share of the variants of a draft's program that can have one reference answer. It is looser than
the agentic tools' share, so that a task with a small answer space passes: 8 random variants of a four-option task
put 6 on one letter about 1% of the time. The later validation step judges whether such a task is worth training
on."""
VERIFIER_BACKEND = Backend.LOCAL
"""The machines of graders that run in an environment: a bubblewrap sandbox that installs the grader's packages."""


class Environment(StrEnum):
    """What a proposal's solver works in. Container proposals bring an image of their own, which drafts cannot run."""

    REASONING = "reasoning"
    SHELLSIM = "shellsim"
    CONTAINER = "container"


class SolverMachine(StrEnum):
    NONE = "none"
    SHELLSIM = "shellsim"
    CPYTHON = "cpython"
    PACKAGES = "packages"


SOLVER_BACKENDS = {
    SolverMachine.NONE: Backend.SHELLSIM,
    SolverMachine.SHELLSIM: Backend.SHELLSIM,
    SolverMachine.CPYTHON: Backend.LOCAL,
    SolverMachine.PACKAGES: Backend.LOCAL,
}
"""The solver backend of each solver machine. A task of a proposal with no machine runs on none, so its backend
only sets the command semantics that ``submit`` expects of a task that declares a shell anyway."""


@dataclass(frozen=True)
class DraftProposal:
    """The header fields of a curriculum proposal that a draft uses, and the proposal document as written."""

    id: str
    environment: Environment
    solver_machine: SolverMachine
    verification: str
    source_hash: str
    """The hash of the capability record that the proposal was written from."""
    document: str = field(repr=False)

    @classmethod
    def from_json(cls, text: str) -> "DraftProposal":
        header = json.loads(text)["header"]
        environment = Environment(header["environment"])
        if environment is Environment.CONTAINER:
            raise ValueError(f"{header['id']}: container proposals have no solver machine for drafts")
        machine = SolverMachine(header["solver_machine"])
        return cls(header["id"], environment, machine, header["verification"], header["source"]["hash"], text)

    @property
    def solver_backend(self) -> Backend:
        return SOLVER_BACKENDS[self.solver_machine]

    @property
    def task_id(self) -> str:
        """A file-safe id for the draft: the proposal id ``capability/slot`` with ``--`` for the slash."""
        return self.id.replace("/", "--")


@dataclass(frozen=True)
class DraftConfig:
    """The models, machines and limits of draft sessions. One config can serve many proposals at the same time."""

    agent_model: Model
    workspace_factory: MachineFactory
    """The agent's machine: a ``LocalMachineFactory`` (bubblewrap) whose mounts give its shell Python with
    taskcompendium and verifyit."""
    solver_factories: Mapping[Backend, MachineFactory]
    """The solver machines that ``RolloutEngine.check`` checks a task against, by backend, with the
    ``VERIFIER_BACKEND`` machines of graders that run in an environment. No solver or grader runs."""
    judge: JudgeConnection
    """The judge that a judge grader would use. ``check`` needs it, and no draft calls it."""
    agent_limits: RolloutLimits
    solver_limits: RolloutLimits
    command_timeout: float


@dataclass(frozen=True)
class Draft:
    """A task that ``submit`` accepted, its build program, and a gzipped tar of the workspace that the program
    reads."""

    task: TaskSpec
    program: str
    workspace: bytes = field(repr=False)


@dataclass(frozen=True)
class DraftResult:
    proposal: DraftProposal
    draft: Draft | None
    report: str | None
    """The agent's report from ``finish``, when it gave up on the proposal."""
    rollout: Rollout
    agent_tokens: int
    submits: int
    failure: str | None
    """The failed operation and its cause, when the session failed."""


def _tool(name: str, description: str, properties: dict[str, Any]) -> FunctionDefinition:
    parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return FunctionDefinition(name=name, description=description, parameters=parameters)


TOOL_DEFINITIONS = (
    _tool(
        "submit",
        "Submit the task with the build program that wrote it. The checks need no model.",
        {
            "path": {"type": "string", "description": "The path of the TaskSpec JSON file in the workspace."},
            "program": {
                "type": "string",
                "description": (
                    "The path of the build program in the workspace: `python3 <program> <task id> <output path>` "
                    "(or `sh <program> ...` for a .sh file) must write this TaskSpec again."
                ),
            },
        },
    ),
    _tool(
        "finish",
        "Stop without a task, with a report on why the proposal cannot become one.",
        {"report": {"type": "string"}},
    ),
)


class DraftTools:
    """The tool provider of one draft session, and its result: ``draft``, ``report`` and ``submits``."""

    def __init__(self, proposal: DraftProposal, config: DraftConfig):
        self.proposal = proposal
        self.config = config
        backend = proposal.solver_backend
        self.backend = backend
        backends = {backend, VERIFIER_BACKEND}
        self.solver_engine = RolloutEngine(
            {name: config.solver_factories[name] for name in backends},
            {SOLVER_HARNESS: ShellHarness(command_timeout=config.command_timeout)},
        )
        self.draft: Draft | None = None
        self.report: str | None = None
        self.submits = 0

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return TOOL_DEFINITIONS

    async def open(self, context: ToolContext) -> "DraftSession":
        if context.machine is None:
            raise ValueError("The draft tools need the agent's workspace machine")
        return DraftSession(self, context.machine)

    def problems(self, task: TaskSpec) -> list[str]:
        """What keeps ``task`` from being a draft: the checks of the agentic tools, with judges, script graders and
        graders with an environment allowed."""
        limits = self.config.solver_limits
        spec = solver_spec(task, self.backend, limits, judge=self.config.judge, verifier_backend=VERIFIER_BACKEND)
        semantics = COMMAND_SEMANTICS[self.backend]
        return unsupported(task, self.solver_engine, spec, semantics, draft_grader_problems) + malformed_files(task)


def draft_grader_problems(grader: Grader) -> list[str]:
    """The grader rule of drafts: any grader but none, and no keyword grader."""
    if isinstance(grader, NoGrader):
        return ["The task needs a grader"]
    return keyword_grader_problems(grader)


class DraftSession:
    """The tools of one draft attempt, on the agent's workspace machine."""

    def __init__(self, tools: DraftTools, machine: Machine):
        self.tools = tools
        self.machine = machine
        handlers: dict[str, Callable[..., Awaitable[ToolCallResult]]] = {"submit": self.submit, "finish": self.finish}
        assert set(handlers) == {tool.name for tool in TOOL_DEFINITIONS}
        self.handlers = {name: validate_call(handler) for name, handler in handlers.items()}

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        handler = self.handlers.get(name)
        if handler is None:
            return _reply({"error": f"Unknown tool: {name}"})
        try:
            return await handler(**arguments)
        except ValidationError as error:
            return _reply({"error": f"Invalid arguments of {name}: {error}"})
        except ToolError as error:
            return _reply({"error": str(error)})

    async def close(self) -> None:
        pass

    async def submit(self, path: str, program: str) -> ToolCallResult:
        self.tools.submits += 1
        timeout = self.tools.config.command_timeout
        task = await read_task(self.machine, path, timeout)
        problems = self.tools.problems(task)
        if not problems:
            problems = await program_problems(
                self.machine, program, task, self.tools.problems, timeout, max_answer_share=DRAFT_MAX_ANSWER_SHARE
            )
        if problems:
            return _reply({"accepted": False, "problems": problems})
        self.tools.draft = Draft(task, program, await workspace_snapshot(self.machine))
        return _reply({"accepted": True}, done=True)

    async def finish(self, report: str) -> ToolCallResult:
        self.tools.report = report
        return _reply({"finished": True}, done=True)


async def draft_task(proposal: DraftProposal, config: DraftConfig) -> DraftResult:
    """Draft the task of ``proposal``. A session without a draft gives the agent's report or its failure."""
    tools = DraftTools(proposal, config)
    harness = ShellHarness(
        command_timeout=config.command_timeout, output_limit_bytes=SHELL_OUTPUT_BYTES, tools={DRAFT_TOOLS: tools}
    )
    workspace = config.workspace_factory
    engine = RolloutEngine({workspace.backend: workspace}, {DRAFT_HARNESS: harness})
    spec = TaskRolloutSpec(
        task=draft_agent_task(proposal, workspace.backend, config.agent_limits),
        harness=DRAFT_HARNESS,
        task_machine=machine_runtime(workspace.backend),
        verifier_machine=None,
        limits=config.agent_limits,
        trace=TraceMode.MESSAGES,
    )
    failure = None
    try:
        rollout = (await engine.run_task(spec, config.agent_model)).rollout
    except RolloutInterrupted as error:
        rollout = error.rollout
        failure = f"{error.operation}: {error.rollout.failure}"
    return DraftResult(proposal, tools.draft, tools.report, rollout, rollout_tokens(rollout), tools.submits, failure)


def draft_agent_task(proposal: DraftProposal, workspace_backend: Backend, agent_limits: RolloutLimits) -> TaskSpec:
    """The agent's own task: the prompts, and a workspace with the proposal, the example program and the schema."""
    backend = proposal.solver_backend
    machine = NO_SOLVER_MACHINE if proposal.solver_machine is SolverMachine.NONE else SOLVER_MACHINES[backend]
    system = DRAFT_PROMPT.format(
        solver_machine=machine, shell_output_kib=SHELL_OUTPUT_BYTES // 1024, turns=agent_limits.max_turns
    )
    workspace = WORKDIR.relative_to("/")
    files = (
        inline_resource(f"{workspace}/proposal.json", proposal.document.encode()),
        inline_resource(f"{workspace}/example_build.py", example_program(backend)),
        inline_resource(
            f"{workspace}/reference/taskspec.schema.json", json.dumps(TaskSpec.model_json_schema(), indent=1).encode()
        ),
        inline_resource(f"{workspace}/reference/verifyit_specs.md", verifyit_reference().encode()),
        inline_resource(f"{workspace}/reference/grader_examples.py", Path(grader_examples.__file__).read_bytes()),
    )
    return TaskSpec(
        id=f"taskforge-draft-{proposal.task_id}",
        context=ConversationInput(
            events=(
                TextMessage(role="system", content=system),
                TextMessage(role="user", content=DRAFT_USER_PROMPT.format(proposal_id=proposal.id)),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=COMMAND_SEMANTICS[workspace_backend],
            # An empty package spec runs the shell on what the factory mounts, for example a Python environment.
            package_spec=PackageSpec(),
            working_directory=str(WORKDIR),
            tool_providers={"taskforge": ProviderRequirement(action_interface=DRAFT_TOOLS, initial_state={})},
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="submit decides whether the agent made a draft"),
        source=Source(
            dataset="taskforge-agentic-drafts",
            revision=proposal.source_hash,
            row=proposal.id,
            importer_revision=DRAFT_REVISION,
        ),
        resources=ResourceGroups(worker=files),
    )


def _reply(value: Mapping[str, Any], *, done: bool = False) -> ToolCallResult:
    return ToolCallResult(json.dumps(value, default=str), done=done)
