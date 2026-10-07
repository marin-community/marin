# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The builder SDK: the ``Build`` context a builder program's ``build(b)`` receives.

A builder program is a module that defines memoized steps (``@step(role)``) and an
``async def build(b: Build) -> BuildOutput``. The program returns the task and its controls; it
never marks itself complete. ``run_build`` checks the output, adds provenance, and records the
draft. ``sdk_reference`` renders this module's public surface from its docstrings for the author.
"""

import inspect
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel
from rolloutengine.cleanup import Cleanup
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutInterrupted, SuppliedState
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.machines import task_machine
from shellbox.machine import Machine, MachineFactory
from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec
from taskcompendium.execution import TaskExecution
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import AnswerType, Source, TaskSpec, VerifierSpec
from taskcompendium.submission import SubmissionConvention, submission_compatibility

from taskforge.build import step as step_module
from taskforge.build.infrastructure import host_checked_factories, infrastructure_failure
from taskforge.build.step import (
    CURRENT_STEP,
    SDK_VERSION,
    Blob,
    Resource,
    Step,
    StepCache,
    StepRole,
    step,
)
from taskforge.canonical import canonical_json, digest
from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.llm.agent import AgentRun, AgentTool, run_agent
from taskforge.llm.agent import shell_tool as agent_shell_tool
from taskforge.llm.client import Completion, GlmClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger
from taskforge.llm.structured import StructuredTool, complete_structured
from taskforge.proposal.model import TaskProposal, render
from taskforge.spec import controls as controls_module
from taskforge.spec import draft as draft_module
from taskforge.spec.controls import Control

MACHINE_CLEANUP_TIMEOUT = 120.0
"""Seconds ``Build.machine`` and ``Build.try_grader`` wait for a machine to close; a failed close is
logged, not raised."""
DOCKER_IMAGE_REQUIREMENTS = (
    "A Docker task image (EnvironmentKind.DOCKER) must provide `sh` and `setsid` (util-linux, or busybox "
    "with its setsid applet): shellbox's Docker backend starts every command under setsid and refuses an image "
    "without it, so distroless and scratch images cannot run a task."
)
NUMERIC_LITERALS = (
    "A numeric answer's expected value (verifyit `NumericSpec.expected`) is a literal string: an integer, "
    'decimal, scientific-notation number or integer fraction such as "42", "-0.125", "1.5e3" or "1/8". '
    'It is never a float or an expression (not 0.125, not "sqrt(2)").'
)
HOST_FAILURES = (
    "When the host fails (no factory for the machine kind, the factory cannot schedule a machine, the machine "
    "host is unreachable), `b.machine` and `b.try_grader` raise `BuildInfrastructureFailure`. Let it propagate: "
    "the build is retried without a revision. A failing image build, setup or healthcheck is the program's."
)
TRY_GRADER_SOURCE = "taskforge.try_grader"
"""``Source.dataset`` of the provisional task ``Build.try_grader`` grades against."""


class BuildFailure(Exception):
    """The program decided the task cannot be built as written; ``step`` names where."""

    def __init__(self, message: str, step: str | None):
        super().__init__(message if step is None else f"{step}: {message}")
        self.message = message
        self.step = step


@dataclass(frozen=True)
class Grader:
    """What a GRADER step returns: the verifier and the evidence it was prototyped on.

    Attributes:
        verifier: The task verifier (``b.spec.shell_verifier`` or ``b.spec.answer_verifier``).
        answer_contract: The exact output format the solver must follow, for the instructions.
        reference_reply: A correct final reply, used to prototype the grader.
        reference_files: Files a correct solver leaves in the workspace, if the grader reads any.
        secret_values: Answer strings the instructions must not contain.
    """

    verifier: VerifierSpec
    answer_contract: str
    reference_reply: str
    reference_files: tuple[EnvironmentFile, ...] = ()
    secret_values: tuple[str, ...] = ()


GRADED_RESOURCE_PREFIX = "try_grader/"
"""Resource name prefix of the candidates ``Build.try_grader`` graded."""


@dataclass(frozen=True)
class GradedCandidate:
    """A candidate ``Build.try_grader`` graded: its final reply and workspace files (by path)."""

    reply: str
    files: tuple[EnvironmentFile, ...] = ()


@dataclass(frozen=True)
class BuildOutput:
    """What ``build(b)`` returns. The verifier must come from a GRADER step and the controls
    from a CONTROLS step; ``run_build`` checks both.

    ``execution`` is the ``TaskExecution`` passed to ``spec.assemble`` for ``task``: deadlines,
    the agent user, and each stage's files, setup and healthcheck. ``TaskExecution()`` sets none.

    ``convention`` is the ``taskcompendium.submission`` convention the solver submits under, for
    example ``PlainText(id="plain_text")``: RolloutEngine appends its submission instruction to the
    task prompt and extracts the answer with it. Reference replies and control replies follow it.
    It must be compatible with the task (``submission_compatibility``); a task whose answer is the
    machine state submits nothing through it. Like ``execution``, it is not part of the TaskSpec.
    """

    task: TaskSpec
    execution: TaskExecution
    convention: SubmissionConvention
    controls: tuple[Control, ...]


@dataclass(frozen=True)
class BuildServices:
    """Everything a build reaches outside its own directory.

    Attributes:
        client: The shared GLM client.
        policy: Sampling policy for every model call in the build; part of every step key.
        factories: Machine factories by environment kind, for ``Build.machine``.
        ledger: Where steps, model calls, and tool calls are recorded.
        web_tools: Web search and fetch tools (``taskforge.llm.web.web_tools``), or empty.
    """

    client: GlmClient
    policy: LLMPolicy
    factories: Mapping[EnvironmentKind, MachineFactory]
    ledger: Ledger
    web_tools: tuple[AgentTool, ...] = ()


def record_completions(fields: SpanFields, completions: Sequence[Completion]) -> None:
    """Fill an ``LLM_CALL`` span with the summed token usage, last finish reason and request count."""
    fields.tokens_in = sum(c.usage.prompt_tokens for c in completions)
    fields.tokens_out = sum(c.usage.completion_tokens for c in completions)
    fields.tokens_reasoning = sum(c.usage.reasoning_tokens for c in completions)
    fields.finish_reason = completions[-1].finish_reason
    fields.attrs["requests"] = str(len(completions))


class BuildLLM:
    """GLM access for steps. Every call is recorded in the ledger under the running step."""

    def __init__(self, client: GlmClient, policy: LLMPolicy, ledger: Ledger, item_id: str, round: int):  # noqa: A002
        self.client = client
        self.policy = policy
        self._ledger = ledger
        self._item_id = item_id
        self._round = round

    @property
    def digest(self) -> str:
        """The policy digest that every step key includes."""
        return digest({"model": self.client.endpoint.model, "policy": self.policy})

    def _step(self) -> str:
        frame = CURRENT_STEP.get()
        return "program" if frame is None else frame.name

    async def complete(self, messages: Sequence[Message]) -> Completion:
        """One chat completion (no tools) with the build policy."""
        with span(self._ledger, EntryKind.LLM_CALL, item_id=self._item_id, round=self._round, step=self._step()) as f:
            completion = await self.client.complete(messages, self.policy)
            f.model = self.client.endpoint.model
            record_completions(f, (completion,))
        return completion

    async def structured[T: BaseModel](self, messages: Sequence[Message], output_type: type[T], name: str) -> T:
        """Force one call of a strict tool named ``name`` whose arguments ``output_type`` validates.

        One repair request follows an invalid reply; a second invalid reply raises
        ``StructuredOutputError``. Put checks the model can fix in ``output_type`` validators.
        """
        tool = StructuredTool(name=name, description=inspect.getdoc(output_type) or name, output_type=output_type)
        with span(self._ledger, EntryKind.LLM_CALL, item_id=self._item_id, round=self._round, step=self._step()) as f:
            f.model = self.client.endpoint.model
            f.attrs["tool"] = name
            result = await complete_structured(self.client, messages, self.policy, tool)
            record_completions(f, result.completions)
        return result.value

    async def agent(self, messages: Sequence[Message], tools: Sequence[AgentTool], max_turns: int) -> AgentRun:
        """Run the Taskforge agent loop (``taskforge.llm.agent.run_agent``) with ``tools``."""
        record = CallLedger(ledger=self._ledger, item_id=self._item_id, round=self._round, step=self._step())
        return await run_agent(self.client, self.policy, messages, tools, max_turns, record)


class Build:
    """The context a builder program and each of its steps receive as ``b``.

    Attributes:
        proposal: The accepted proposal being built.
        item_id: The item's file-safe id.
        llm: Model access (``complete``, ``structured``, ``agent``).
        spec: ``taskforge.spec.draft``: ``environment``, ``file``, ``shell_verifier``,
            ``answer_verifier``, ``assemble`` and friends.
        controls: ``taskforge.spec.controls``: ``Control``, ``Expectation``, ``Transcript``,
            ``Workspace``, ``reply``, ``shell_turn``.
        ledger: The build ledger.
        scratch: A private directory for this build; not part of any output.
    """

    def __init__(
        self,
        proposal: TaskProposal,
        item_id: str,
        services: BuildServices,
        cache: StepCache,
        scratch: Path,
        round: int,  # noqa: A002 - matches LedgerEntry.round
    ):
        self.proposal = proposal
        self.item_id = item_id
        self.llm = BuildLLM(services.client, services.policy, services.ledger, item_id, round)
        self.spec = draft_module
        self.controls = controls_module
        self.ledger = services.ledger
        self.scratch = scratch
        self.round = round
        self.resources: list[Resource] = []
        self._services = services
        self._factories = host_checked_factories(services.factories)
        self._cache = cache

    @property
    def proposal_text(self) -> str:
        """The proposal's canonical text: front matter plus body."""
        return render(self.proposal)

    @property
    def research(self) -> tuple[AgentTool, ...]:
        """Web search and fetch tools for ``llm.agent``; fails the build when none are configured."""
        self.check(bool(self._services.web_tools), "web research is not configured for this build")
        return self._services.web_tools

    async def run_step(self, step: Step, arguments: Mapping[str, object]) -> Any:
        return await self._cache.run(
            step,
            arguments,
            self,
            proposal_digest=self.proposal.digest,
            policy_digest=self.llm.digest,
            ledger=self.ledger,
            round=self.round,
            emitted=self.resources,
        )

    def check(self, condition: bool, message: str) -> None:
        """Fail the build with ``BuildFailure(message)`` unless ``condition`` holds."""
        if not condition:
            raise self.failure(message)

    def failure(self, message: str) -> BuildFailure:
        """A ``BuildFailure(message)`` naming the running step, for ``raise b.failure(...)``."""
        frame = CURRENT_STEP.get()
        return BuildFailure(message, None if frame is None else frame.name)

    def emit(self, name: str, content: bytes) -> Blob:
        """Store ``content`` content-addressed and record it as resource ``name`` of this build."""
        blob = self._cache.blobs.put(content)
        resource = Resource(name=name, blob=blob)
        frame = CURRENT_STEP.get()
        if frame is None:
            self.resources.append(resource)
        else:
            frame.resources.append(resource)
        return blob

    def read(self, blob: Blob) -> bytes:
        """The content of a blob this build or an earlier one stored."""
        return self._cache.blobs.get(blob)

    @asynccontextmanager
    async def machine(self, environment: EnvironmentSpec) -> AsyncIterator[Machine]:
        """``async with b.machine(environment) as machine``: a prepared machine (files installed,
        setup and healthcheck run).

        It is created and closed exactly as RolloutEngine would for a task attempt. Use it to
        prototype fixtures and graders; ``shell_tool(machine)`` gives ``llm.agent`` a shell in it.

        Raises:
            BuildInfrastructureFailure: the host failed to create or drive the machine.
        """
        self.check(environment.kind != EnvironmentKind.NULL, "a null environment has no machine")
        cleanup = Cleanup(MACHINE_CLEANUP_TIMEOUT)
        async with task_machine(environment, self._factories, cleanup) as machine:
            assert machine is not None
            yield machine

    def shell_tool(self, machine: Machine, timeout: float = 300.0) -> AgentTool:
        """RolloutEngine's ``shell(command)`` tool over ``machine``, for ``llm.agent``."""
        return agent_shell_tool(machine, timeout=timeout, output_limit_bytes=64 * 1024)

    async def try_grader(
        self,
        environment: EnvironmentSpec,
        verifier: VerifierSpec,
        answer_type: AnswerType,
        convention: SubmissionConvention,
        instruction: str,
        reply: str,
        workspace: Sequence[EnvironmentFile] = (),
    ) -> GradeResult:
        """Grade one candidate (a final ``reply`` plus ``workspace`` files) the way the engine would.

        RolloutEngine prepares a machine for ``environment``, installs the ``workspace`` files at
        their absolute paths (as root, after setup), and grades ``instruction`` and ``reply`` as a
        two-message conversation with ``verifier`` under ``convention``: any verifier and answer type
        ``spec.assemble`` accepts, with a reply in the form ``convention`` extracts (the final
        assistant text). No model is called.

        This prototypes a grader while it is being written: on its reference answer, an empty
        answer, and wrong answers you invent for the purpose. Every graded candidate is recorded,
        and ``run_build`` fails a build whose controls include one (other than the reference and
        the empty answer): controls are written, not graded; ``validate`` replays them.

        Raises:
            BuildFailure: ``spec.assemble`` rejects the task these arguments describe, or
                ``convention`` cannot carry ``answer_type`` to ``verifier``.
            BuildInfrastructureFailure: the host failed to create or drive a grading machine.
        """
        candidate = GradedCandidate(reply=reply, files=tuple(sorted(workspace, key=lambda f: f.path)))
        self.emit(f"{GRADED_RESOURCE_PREFIX}{digest(candidate)}.json", canonical_json(candidate).encode())
        execution = TaskExecution()
        try:
            task = draft_module.assemble(
                task_id=f"{self.item_id}.try_grader",
                instruction=instruction,
                answer_type=answer_type,
                environment=environment,
                verifier=verifier,
                source=Source(
                    dataset=TRY_GRADER_SOURCE,
                    revision=self.proposal.digest,
                    row=self.item_id,
                    importer_revision=SDK_VERSION,
                ),
                execution=execution,
            )
        except ValueError as error:
            raise self.failure(f"try_grader: {error}") from error
        if answer_type not in draft_module.MACHINE_ANSWER_TYPES:
            compatibility = submission_compatibility(task, convention)
            self.check(
                compatibility.compatible, f"try_grader: convention {convention.id!r}: {'; '.join(compatibility.reasons)}"
            )
        engine = ShellboxRolloutEngine(
            _no_model,
            self._factories,
            max_turns=1,
            command_timeout=MACHINE_CLEANUP_TIMEOUT,
            cleanup_timeout=MACHINE_CLEANUP_TIMEOUT,
            convention=convention,
        )
        state = SuppliedState(
            messages=({"role": "user", "content": instruction}, {"role": "assistant", "content": reply}),
            files=candidate.files,
        )
        try:
            return await engine.grade_state(task, state, execution=execution)
        except RolloutInterrupted as error:
            failure = infrastructure_failure(error)
            if failure is None:
                raise
            raise failure from error


async def _no_model(request: ModelRequest) -> ModelTurn:
    raise AssertionError("ShellboxRolloutEngine.grade_state never calls the model")


SDK_EXPORTS: dict[str, object] = {
    "Build": Build,
    "BuildOutput": BuildOutput,
    "TaskExecution": TaskExecution,
    "BuildFailure": BuildFailure,
    "Grader": Grader,
    "Blob": Blob,
    "Resource": Resource,
    "StepRole": StepRole,
    "step": step,
    "spec": draft_module,
    "controls": controls_module,
}
"""Names every builder program starts with."""


def _describe(name: str, obj: Any, indent: str = "") -> list[str]:
    if inspect.isclass(obj) and issubclass(obj, Enum):
        members = ", ".join(f"{member.name} = {member.value!r}" for member in obj)
        return [f"{indent}enum {name}: {members}", ""]
    doc = inspect.getdoc(obj) or ""
    head = f"{indent}{'class ' if inspect.isclass(obj) else ''}{name}{inspect.signature(obj)}"
    return [head, *(f"{indent}    {line}" for line in doc.splitlines()), ""]


def _module_reference(title: str, module: ModuleType, names: Sequence[str]) -> list[str]:
    lines = [f"## {title} (`{module.__name__}`)", "", inspect.getdoc(module) or "", ""]
    for name in names:
        lines += _describe(name, getattr(module, name))
    return lines


def sdk_reference() -> str:
    """The builder SDK reference given to the program author, generated from docstrings."""
    lines = ["# Builder SDK reference", "", inspect.getdoc(inspect.getmodule(Build)) or "", ""]
    lines += ["## Build context (`b`)", ""]
    lines += _describe("Build", Build)
    for name, member in inspect.getmembers(Build):
        if name.startswith("_") or name == "run_step":
            continue
        if isinstance(member, property):
            lines += [
                f"  b.{name}  (property)",
                *(f"      {line}" for line in (inspect.getdoc(member) or "").splitlines()),
                "",
            ]
        else:
            lines += _describe(f"b.{name}", member, "  ")
    for name in ("complete", "structured", "agent"):
        lines += _describe(f"b.llm.{name}", getattr(BuildLLM, name), "  ")
    for name in ("Grader", "BuildOutput", "BuildFailure"):
        lines += _describe(name, SDK_EXPORTS[name])
    lines += ["## Machines and answers", "", f"- {DOCKER_IMAGE_REQUIREMENTS}", f"- {NUMERIC_LITERALS}"]
    lines += [f"- {HOST_FAILURES}", ""]
    lines += _module_reference("Steps", step_module, ("step", "StepRole", "Blob", "Resource"))
    lines += _module_reference(
        "Task spec helpers, available as `spec`",
        draft_module,
        ("environment", "file", "shell_command", "shell_verifier", "reward_file", "answer_verifier", "assemble"),
    )
    lines += _module_reference(
        "Controls, available as `controls`",
        controls_module,
        (
            "Control",
            "ControlKind",
            "ControlCategory",
            "ControlConcern",
            "Expectation",
            "Transcript",
            "Workspace",
            "reply",
            "shell_turn",
            "validate_controls",
        ),
    )
    lines += [
        "Allowed categories per control kind:",
        *(
            f"- {kind.name}: {', '.join(sorted(c.name for c in categories))}"
            for kind, categories in controls_module.CATEGORIES.items()
        ),
        "Allowed concerns per control category:",
        *(
            f"- {category.name}: {', '.join(sorted(c.name for c in concerns))}"
            for category, concerns in controls_module.CONCERNS.items()
        ),
        "Every stage needs controls with each of these concerns: "
        f"{', '.join(sorted(c.name for c in controls_module.REQUIRED_CONCERNS_PER_STAGE))}.",
        f"Negative controls and graded malformed controls need reward_max <= {controls_module.REJECTION_CEILING}.",
        "",
    ]
    return "\n".join(lines)
