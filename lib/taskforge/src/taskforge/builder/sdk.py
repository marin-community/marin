# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The builder SDK: the ``Build`` context a builder program's ``build(b)`` receives.

A builder program is a module that defines memoized steps (``@step(role)``) and an
``async def build(b: Build) -> BuildOutput``. The program returns the task, its lowered form and
its controls; it never marks itself complete. ``run_build`` checks the output, adds provenance, and
records the draft. ``sdk_reference`` renders this module's public surface from its docstrings for
the author.
"""

import inspect
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutInterrupted, SuppliedState
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.machines import prepare_machine
from rolloutengine.spec import LoweredTaskSpec, TaskSessionSpec
from shellbox.machine import Machine, MachineFactory
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import AnswerFormat, AnswerType, EnvironmentRequirements, Source, TaskResource, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from taskforge.builder import step as step_module
from taskforge.builder.infrastructure import (
    BuildInfrastructureFailure,
    InfrastructureCause,
    host_checked_factories,
    infrastructure_failure,
)
from taskforge.builder.step import (
    CURRENT_STEP,
    SDK_VERSION,
    Blob,
    Resource,
    Step,
    StepCache,
    StepRole,
    step,
)
from taskforge.content_hash import canonical_json, digest
from taskforge.ledger.records import Ledger
from taskforge.llm.agent import AgentRun, AgentTool, run_agent
from taskforge.llm.agent import shell_tool as agent_shell_tool
from taskforge.llm.client import Completion, GlmClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger, recorded_complete, recorded_structured
from taskforge.llm.structured import StructuredTool
from taskforge.proposal.model import TaskProposal, render
from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.images import BUILD_LIMITS, DockerBuild, ImageBuilder, check_build_limits
from taskforge.spec import controls as controls_module
from taskforge.spec import draft as draft_module
from taskforge.spec.controls import Control
from taskforge.spec.draft import MachineSettings

MACHINE_CLEANUP_TIMEOUT = 120.0
"""Seconds ``Build.machine`` and ``Build.try_grader`` wait for a machine to close."""
TRY_GRADER_MACHINE = draft_module.machine(startup_timeout=600.0)
"""The task and verifier machine settings ``Build.try_grader`` uses unless the program passes its own."""
TRY_GRADER_VERIFIER_TIMEOUT = 600.0
"""Seconds ``Build.try_grader`` allows one grading, verifier machine start included."""
IMAGE_REPOSITORY_PREFIX = "taskforge-tasks"
"""Registry repository under which ``Build.publish_image`` pushes an item's task image."""
DOCKER_IMAGE_REQUIREMENTS = (
    "A task image (`b.publish_image(DockerBuild(...))`) must provide `sh` and `setsid` (util-linux, or busybox "
    "with its setsid applet): shellbox's Docker backend starts every command under setsid and refuses an image "
    "without it, so distroless and scratch images cannot run a task. On Iris the image runs under gVisor, which "
    "cannot switch users: leave the image's user as root."
)
NUMERIC_LITERALS = (
    "A numeric answer's expected value (verifyit `NumericSpec.expected`) is a literal string: an integer, "
    'decimal, scientific-notation number or integer fraction such as "42", "-0.125", "1.5e3" or "1/8". '
    'It is never a float or an expression (not 0.125, not "sqrt(2)").'
)
HOST_FAILURES = (
    "When the host fails (no factory for the machine backend, no image builder, the factory cannot schedule a "
    "machine, the machine host is unreachable), `b.machine`, `b.publish_image` and `b.try_grader` raise "
    "`BuildInfrastructureFailure`. Let it propagate: the build is retried without a revision. A failing image "
    "build, setup command or grader is the program's."
)
GRADERS = (
    "Graders: prefer `spec.answer_grader(spec)` with a verifyit mode that grades in process (exact, numeric, mcq, "
    "math, ifeval, json_schema, xml_elements, csv_columns, structured_exact, predicted_action); it takes no "
    "verifier machine. Any other check is a program in a separate verifier machine: `spec.python_grader` "
    "(`python3 /tests/grade.py`) or `spec.script_grader` (any command), with "
    "`environment=spec.grader_environment(image)`: the task image for a task with one, otherwise Taskforge's "
    "grader-base image (CPython 3.12, standard library, `sh`, root). The program reads the extracted answer at "
    "`/app/answer.txt` (`answer_path=spec.ANSWER_PATH`; `None` for a file or workspace-state answer), each "
    "captured `output_paths` file at its own absolute path, its private files under `/tests` (`config` at "
    "`/tests/config.json`) and the conversation at `/tests/conversation.json`; it never reads stdin, and its "
    "last stdout line is the reward (`StdoutReward`). A grader in a verifier machine takes a `verifier_machine` "
    "in `b.lower`; an in-process grader takes none. The task's `answer_format` (`spec.assemble`) is how the "
    "answer is requested and read."
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
    """What a GRADER step returns: the grader and the evidence it was prototyped on.

    Attributes:
        package: The task's grader (``spec.answer_grader``, ``spec.python_grader`` or
            ``spec.script_grader``), passed to ``spec.assemble`` as ``grader``.
        answer_contract: The exact output format the solver must follow, for the instructions.
        reference_reply: A correct final reply, used to prototype the grader.
        reference_files: Files a correct solver leaves in the task machine (``spec.file``, paths
            relative to the machine root), if the grader reads any.
        secret_values: Answer strings the instructions must not contain.
    """

    package: GraderPackage
    answer_contract: str
    reference_reply: str
    reference_files: tuple[TaskResource, ...] = ()
    secret_values: tuple[str, ...] = ()


GRADED_RESOURCE_PREFIX = "try_grader/"
"""Resource name prefix of the candidates ``Build.try_grader`` graded."""


@dataclass(frozen=True)
class GradedCandidate:
    """A candidate ``Build.try_grader`` graded: its final reply and workspace files (by path)."""

    reply: str
    files: tuple[TaskResource, ...] = ()


def file_set(files: Sequence[TaskResource]) -> frozenset[tuple[str, bytes]]:
    """``files`` as (path, content) pairs, so two file lists compare by what they install."""
    return frozenset((resource.path, resource_bytes(resource)) for resource in files)


@dataclass(frozen=True)
class BuildOutput:
    """What ``build(b)`` returns. The grader must come from a GRADER step and the controls
    from a CONTROLS step; ``run_build`` checks both.

    ``lowered`` is what ``b.lower(task, ...)`` returned for ``task``: the machine settings (backend,
    network, limits, user, startup timeout) and the session (turn budget, deadlines, verifier
    timeout) RolloutEngine runs it with. Validation replaces the session's turn budget and deadlines
    with its own; the verifier timeout and machine settings stand.

    The task carries its answer format (``spec.assemble(answer_format=...)``): RolloutEngine appends
    its submission instruction to the task prompt and extracts the answer with it. Reference replies
    and control replies follow it.
    """

    task: TaskSpec
    lowered: LoweredTaskSpec
    controls: tuple[Control, ...]


@dataclass(frozen=True)
class BuildServices:
    """Everything a build reaches outside its own directory.

    Attributes:
        client: The shared GLM client.
        policy: Sampling policy for every model call in the build; part of every step key.
        host: Where the build runs; ``spec.lower`` picks machine backends for it.
        factories: Machine factories keyed by shellbox ``Backend`` value, as
            ``taskforge.sandbox.factories.machine_factories(host, ...)`` returns them.
        images: Publishes task images by digest, or ``None`` when this host cannot.
        ledger: Where steps, model calls, and tool calls are recorded.
        web_tools: Web search and fetch tools (``taskforge.llm.web.web_tools``), or empty.
    """

    client: GlmClient
    policy: LLMPolicy
    host: MachineHost
    factories: Mapping[str, MachineFactory]
    images: ImageBuilder | None
    ledger: Ledger
    web_tools: tuple[AgentTool, ...] = ()


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

    def _record(self) -> CallLedger:
        frame = CURRENT_STEP.get()
        step = "program" if frame is None else frame.name
        return CallLedger(ledger=self._ledger, item_id=self._item_id, round=self._round, step=step)

    async def complete(self, messages: Sequence[Message]) -> Completion:
        """One chat completion (no tools) with the build policy."""
        return await recorded_complete(self.client, messages, self.policy, {}, self._record(), {})

    async def structured[T: BaseModel](self, messages: Sequence[Message], output_type: type[T], name: str) -> T:
        """Force one call of a strict tool named ``name`` whose arguments ``output_type`` validates.

        One repair request follows an invalid reply; a second invalid reply raises
        ``StructuredOutputError``. Put checks the model can fix in ``output_type`` validators.
        """
        tool = StructuredTool(name=name, description=inspect.getdoc(output_type) or name, output_type=output_type)
        result = await recorded_structured(self.client, messages, self.policy, tool, self._record(), {"tool": name})
        return result.value

    async def agent(self, messages: Sequence[Message], tools: Sequence[AgentTool], max_turns: int) -> AgentRun:
        """Run the Taskforge agent loop (``taskforge.llm.agent.run_agent``) with ``tools``."""
        return await run_agent(self.client, self.policy, messages, tools, max_turns, self._record())


class Build:
    """The context a builder program and each of its steps receive as ``b``.

    Attributes:
        proposal: The accepted proposal being built.
        item_id: The item's file-safe id.
        llm: Model access (``complete``, ``structured``, ``agent``).
        spec: ``taskforge.spec.draft``: ``requirements``, ``machine``, ``session``, ``file``,
            ``answer_grader``, ``python_grader``, ``script_grader``, ``grader_environment``,
            ``assemble`` and friends.
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
        self._factories = host_checked_factories(services.factories, services.host)
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

    async def publish_image(self, build: DockerBuild) -> str:
        """Build ``build`` for linux/amd64, push it, and return the digest-pinned reference.

        Pass the result as ``spec.requirements(image=...)``; call it inside a step so the reference
        is memoized with the step's output.

        Raises:
            BuildFailure: the build context exceeds ``BUILD_LIMITS``.
            BuildInfrastructureFailure: this host has no image builder.
        """
        images = self._services.images
        if images is None:
            raise BuildInfrastructureFailure(InfrastructureCause.NO_IMAGE_BUILDER, "this host cannot publish images")
        try:
            check_build_limits(build, BUILD_LIMITS)
        except ValueError as error:
            raise self.failure(f"publish_image: {error}") from error
        return await images.publish(build, self.image_repository)

    @property
    def image_repository(self) -> str:
        """The registry repository ``publish_image`` pushes this item's images under."""
        name = re.sub(r"[^a-z0-9]+", "-", self.item_id.lower()).strip("-")
        return f"{IMAGE_REPOSITORY_PREFIX}/{name}"

    def lower(
        self,
        task: TaskSpec,
        *,
        task_machine: MachineSettings | None,
        verifier_machine: MachineSettings | None,
        session: TaskSessionSpec,
    ) -> LoweredTaskSpec:
        """``spec.lower`` for this host and its machine factories: what ``BuildOutput.lowered`` holds.

        ``task_machine`` is the task machine's settings (``spec.machine``), ``None`` exactly when the
        task has none; ``verifier_machine`` likewise, given exactly when the grader has an environment
        (``spec.python_grader``, ``spec.script_grader``, or ``spec.answer_grader`` with one).

        Raises:
            BuildFailure: ``spec.lower`` or RolloutEngine rejects the lowered task.
        """
        try:
            return draft_module.lower(
                task,
                host=self._services.host,
                task_machine=task_machine,
                verifier_machine=verifier_machine,
                session=session,
                factories=self._factories,
            )
        except (ValueError, NotImplementedError) as error:
            raise self.failure(f"lower: {error}") from error

    @asynccontextmanager
    async def machine(
        self, requirements: EnvironmentRequirements, machine: MachineSettings, files: Sequence[TaskResource] = ()
    ) -> AsyncIterator[Machine]:
        """``async with b.machine(requirements, machine, files) as m``: a prepared machine (``files``
        installed relative to the root, setup commands run).

        It is created and closed exactly as RolloutEngine would for a task attempt. Use it to
        prototype fixtures and graders; ``shell_tool(m)`` gives ``llm.agent`` a shell in it.

        Raises:
            BuildInfrastructureFailure: the host failed to create or drive the machine.
        """
        runtime = draft_module.machine_runtime(machine, requirements, self._services.host)
        async with prepare_machine(
            requirements, runtime, tuple(files), self._factories, cleanup_timeout=MACHINE_CLEANUP_TIMEOUT
        ) as prepared:
            yield prepared

    def shell_tool(self, machine: Machine, timeout: float = 300.0) -> AgentTool:
        """RolloutEngine's ``shell(command)`` tool over ``machine``, for ``llm.agent``."""
        return agent_shell_tool(machine, timeout=timeout, output_limit_bytes=64 * 1024)

    async def try_grader(
        self,
        requirements: EnvironmentRequirements | None,
        grader: GraderPackage,
        answer_type: AnswerType,
        answer_format: AnswerFormat,
        instruction: str,
        reply: str,
        *,
        files: Sequence[TaskResource] = (),
        workspace: Sequence[TaskResource] = (),
        output_paths: Sequence[str] = (),
        machine: MachineSettings = TRY_GRADER_MACHINE,
    ) -> GradeResult:
        """Grade one candidate (a final ``reply`` plus ``workspace`` files) the way the engine would.

        RolloutEngine prepares the task machine for ``requirements`` with the agent-visible
        ``files`` (none when ``requirements`` is ``None``), installs the ``workspace`` files relative
        to the machine root (as root, after setup), and grades ``instruction`` and ``reply`` as a
        two-message conversation with ``grader`` and ``answer_format``: any grader, answer type and
        format ``spec.assemble`` accepts, with a reply in the form ``answer_format`` extracts (the
        final assistant text for ``PlainText``). A grader with an environment runs in its own
        verifier machine and reads the ``output_paths`` captured from the task machine. ``machine``
        sets both machines. No model is called.

        This prototypes a grader while it is being written: on its reference answer, an empty
        answer, and wrong answers you invent for the purpose. Every graded candidate is recorded,
        and ``run_build`` fails a build whose controls include one (other than the reference and
        the empty answer): controls are written, not graded; ``validate`` replays them.

        Raises:
            BuildFailure: ``spec.assemble`` or ``spec.lower`` rejects the task these arguments
                describe, including an ``answer_format`` that cannot carry ``answer_type`` to ``grader``.
            BuildInfrastructureFailure: the host failed to create or drive a grading machine.
        """
        candidate = GradedCandidate(reply=reply, files=tuple(sorted(workspace, key=lambda f: f.path)))
        self.emit(f"{GRADED_RESOURCE_PREFIX}{digest(candidate)}.json", canonical_json(candidate).encode())
        try:
            task = draft_module.assemble(
                task_id=f"{self.item_id}.try_grader",
                instruction=instruction,
                answer_type=answer_type,
                answer_format=answer_format,
                grader=grader,
                source=Source(
                    dataset=TRY_GRADER_SOURCE,
                    revision=self.proposal.digest,
                    row=self.item_id,
                    importer_revision=SDK_VERSION,
                ),
                environment=requirements,
                files=files,
                output_paths=output_paths,
            )
        except ValueError as error:
            raise self.failure(f"try_grader: {error}") from error
        lowered = self.lower(
            task,
            task_machine=None if requirements is None else machine,
            verifier_machine=None if draft_module.grading_environment(task) is None else machine,
            session=draft_module.session(
                max_turns=1,
                model_turn_timeout=None,
                command_timeout=None,
                tool_turn_timeout=None,
                total_turn_timeout=None,
                attempt_timeout=None,
                verifier_timeout=TRY_GRADER_VERIFIER_TIMEOUT,
                cleanup_timeout=MACHINE_CLEANUP_TIMEOUT,
            ),
        )
        engine = ShellboxRolloutEngine(_no_model, self._factories)
        state = SuppliedState(
            messages=({"role": "user", "content": instruction}, {"role": "assistant", "content": reply}),
            resources=candidate.files,
        )
        try:
            return await engine.grade_state(lowered, state)
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
    "BuildFailure": BuildFailure,
    "DockerBuild": DockerBuild,
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
    for name in ("Grader", "BuildOutput", "BuildFailure", "DockerBuild"):
        lines += _describe(name, SDK_EXPORTS[name])
    lines += ["## Machines, graders and answers", "", f"- {GRADERS}", f"- {DOCKER_IMAGE_REQUIREMENTS}"]
    lines += [f"- {NUMERIC_LITERALS}", f"- {HOST_FAILURES}", ""]
    lines += _module_reference("Steps", step_module, ("step", "StepRole", "Blob", "Resource"))
    lines += _module_reference(
        "Task spec helpers, available as `spec`",
        draft_module,
        (
            "requirements",
            "machine",
            "Resources",
            "session",
            "lower",
            "file",
            "answer_grader",
            "grader_environment",
            "grading_environment",
            "python_grader",
            "script_grader",
            "reward_file",
            "assemble",
        ),
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
        "A task needs controls with each of these concerns: "
        f"{', '.join(sorted(c.name for c in controls_module.REQUIRED_CONCERNS))}.",
        f"Negative controls and graded malformed controls need reward_max <= {controls_module.REJECTION_CEILING}.",
        "",
    ]
    return "\n".join(lines)
