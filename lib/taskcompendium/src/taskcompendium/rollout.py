# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One model/tool/grader loop for serialized tasks."""

import asyncio
import json
import math
import tarfile
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Protocol
from uuid import uuid4

from harbor_config.env import resolve_env_vars
from shellbox.image import DockerfileSource
from shellbox.image import RegistryImage as ShellboxRegistryImage
from shellbox.machine import (
    Command,
    DockerImage,
    ExitReason,
    Machine,
    MachineFactory,
    MachineSpec,
    NetworkPolicy,
    ShellSimBuiltins,
)

from taskcompendium.chat import assistant_message, chat_conversation
from taskcompendium.environment import (
    ArtifactKind,
    DockerBuild,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    ExitCodeReward,
    FileReward,
    HealthcheckSpec,
    LocalImage,
    MissingArtifactPolicy,
    RegistryImage,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.grading import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    StageRewardStrategy,
    StageVerifierSpec,
    TaskSpec,
    TaskStage,
    VerifierKind,
)
from taskcompendium.submission import AnswerFormat, SubmissionConvention, chat_request, conversation_messages
from taskcompendium.verifier_registry import grade_answer, validate_task_verifiers

MISSING_FILE_EXIT = 44


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    prefix_token_ids: tuple[int, ...]
    assistant_message_index: int | None


@dataclass(frozen=True)
class ModelTurn:
    """Exact tokens and the parsed message from one inference request."""

    message: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    stop_reason: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationLimitReached(Exception):
    """The rendered prompt leaves no permitted generation budget."""

    def __init__(self, prompt_token_ids: tuple[int, ...]):
        self.prompt_token_ids = prompt_token_ids
        super().__init__("The prompt reached the configured generation limit")


@dataclass(frozen=True)
class RolloutFailure:
    """Safe failure fields for consumers of a rollout record."""

    exception_type: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutData:
    task_id: str
    messages: tuple[dict[str, Any], ...]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    grade: GradeResult
    stop_reason: str
    steps: tuple["RolloutStep", ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    failure: RolloutFailure | None = None


class RolloutOperation(StrEnum):
    ATTEMPT = "attempt"
    START = "start"
    PREPARE = "prepare"
    MODEL = "model"
    ADVANCE = "advance"
    GRADE = "grade"


class RolloutInterrupted(RuntimeError):
    """An execution failure with the last completed rollout and original exception cause."""

    def __init__(self, rollout: RolloutData, operation: RolloutOperation):
        self.rollout = rollout
        self.operation = operation
        super().__init__(f"Rollout interrupted during {operation}")


class RolloutContractError(ValueError):
    """Rollout evidence violates the exact-token contract."""


@dataclass(frozen=True)
class Transition:
    """One task transition after a model response."""

    done: bool
    observations: tuple[dict[str, Any], ...] = ()
    reset_conversation: tuple[dict[str, Any], ...] | None = None
    reward: float | None = None
    token_rewards: tuple[float, ...] | None = None
    token_credit: tuple[float, ...] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    grade: GradeResult | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutStep:
    turn: ModelTurn
    transition: Transition
    response_end: int
    messages: tuple[dict[str, Any], ...]


class TaskSession(Protocol):
    """Task operations without model inference or token processing."""

    async def prepare(self) -> dict[str, Any]: ...

    async def advance(self, turn: ModelTurn) -> Transition: ...

    async def grade(self, messages: tuple[dict[str, Any], ...]) -> GradeResult: ...

    async def close(self) -> None: ...


class RolloutEngine(Protocol):
    """A blocking task iterator that runs on a worker-owned thread."""

    def generate(self, tasks: Iterator[TaskSpec]) -> Iterator[RolloutData]: ...


async def _install_files(machine: Machine, files: tuple[EnvironmentFile, ...]) -> None:
    with TemporaryDirectory(prefix="rollout-files-") as directory:
        for index, file in enumerate(files):
            source = Path(directory) / str(index)
            source.write_bytes(file.content)
            source.chmod(file.mode)
            await machine.upload(source, file.path)


def _machine_command(command: EnvironmentCommand) -> Command:
    return Command(
        argv=command.argv,
        cwd=command.cwd,
        env=resolve_env_vars(command.env),
        timeout=command.timeout,
        user=command.user,
    )


async def _wait_for_healthcheck(machine: Machine, healthcheck: HealthcheckSpec) -> None:
    loop = asyncio.get_running_loop()
    grace_end = loop.time() + healthcheck.start_period
    failures = 0
    while True:
        in_grace = loop.time() < grace_end
        result = await machine.run(_machine_command(healthcheck.command))
        if result.exit_code == 0:
            return
        if not in_grace:
            failures += 1
            if failures >= healthcheck.retries:
                raise RuntimeError(f"Environment healthcheck failed after {failures} attempts")
        await asyncio.sleep(healthcheck.start_interval if in_grace else healthcheck.interval)


@asynccontextmanager
async def task_machine(environment: EnvironmentSpec, factories: Mapping[EnvironmentKind, MachineFactory]):
    """Release the machine after model or verifier failure and cancellation."""
    if environment.kind == EnvironmentKind.NULL:
        yield None
        return
    async with AsyncExitStack() as resources:
        if environment.kind == EnvironmentKind.SHELLSIM:
            source = ShellSimBuiltins()
        elif isinstance(environment.image, LocalImage):
            source = DockerImage(environment.image.reference)
        elif isinstance(environment.image, RegistryImage):
            source = ShellboxRegistryImage(environment.image.reference)
        else:
            assert isinstance(environment.image, DockerBuild)
            directory = Path(resources.enter_context(TemporaryDirectory(prefix="rollout-build-")))
            for file in environment.image.files:
                path = directory / file.path.lstrip("/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file.content)
                path.chmod(file.mode)
            source = DockerfileSource(directory, directory / environment.image.dockerfile.lstrip("/"))
        async with asyncio.timeout(environment.startup_timeout):
            machine = await factories[environment.kind].create(
                MachineSpec(
                    source=source,
                    workdir=environment.workdir,
                    env=resolve_env_vars(environment.env),
                    network=NetworkPolicy.ALLOW if environment.network else NetworkPolicy.DENY,
                    memory_mb=environment.memory_mb,
                    cpus=environment.cpus,
                    storage_mb=environment.storage_mb,
                    gpus=environment.gpus,
                    startup_timeout=environment.startup_timeout,
                )
            )
            resources.push_async_callback(_close_machine, machine)
            await _install_files(machine, environment.files)
            for command in environment.setup:
                result = await machine.run(_machine_command(command))
                if result.reason == ExitReason.TIMED_OUT:
                    raise TimeoutError("Environment setup command timed out")
                if result.exit_code != 0:
                    raise RuntimeError(f"Environment setup failed: {result.reason}, exit={result.exit_code}")
            if environment.healthcheck is not None:
                await _wait_for_healthcheck(machine, environment.healthcheck)
        yield machine


async def _close_machine(machine: Machine) -> None:
    # A total-attempt deadline can expire during cleanup after a startup timeout.
    cleanup = asyncio.create_task(machine.close())
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise


async def grade_rollout(
    task: TaskSpec,
    convention: SubmissionConvention,
    messages: tuple[dict[str, Any], ...],
    machine: Machine | None,
    factories: Mapping[EnvironmentKind, MachineFactory],
) -> GradeResult:
    """Grade the final transcript and task filesystem without model access to private files."""
    if task.verifier.kind != VerifierKind.SHELL:
        conversation = chat_conversation(list(messages))
        return await asyncio.to_thread(grade_answer, task, convention, conversation, machine)
    if machine is None:
        raise ValueError("Shell grading requires a task machine")
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    for command in verifier.collect:
        result = await machine.run(_machine_command(command))
        if result.exit_code != 0:
            return GradeResult(
                Outcome.INFRA_ERROR, None, "Cannot collect grading inputs", failure=GradingFailure.EXECUTION
            )
    if verifier.environment is None:
        return await _shell_grade(verifier, messages, machine)
    async with task_machine(verifier.environment, factories) as grading_machine:
        assert grading_machine is not None
        with TemporaryDirectory(prefix="rollout-artifacts-") as directory:
            for index, artifact in enumerate(verifier.artifacts):
                path = Path(directory) / str(index)
                if await _download_artifact(machine, artifact, path, verifier.timeout):
                    await grading_machine.upload(path, artifact.target)
        return await _shell_grade(verifier, messages, grading_machine)


async def _download_artifact(machine: Machine, artifact: VerifierArtifact, target: Path, timeout: float) -> bool:
    """Download an artifact. Return false only when its missing-file policy permits omission."""
    kind = artifact.kind
    if kind == ArtifactKind.AUTO or artifact.missing == MissingArtifactPolicy.SKIP:
        result = await machine.run(
            Command(
                argv=(
                    "sh",
                    "-c",
                    'if [ -d "$1" ]; then printf directory; elif [ -f "$1" ]; then printf file; '
                    f"else exit {MISSING_FILE_EXIT}; fi",
                    "artifact-kind",
                    artifact.source,
                ),
                timeout=timeout,
                user="0",
            )
        )
        if result.exit_code == MISSING_FILE_EXIT and artifact.missing == MissingArtifactPolicy.SKIP:
            return False
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot inspect grading artifact {artifact.source}: exit={result.exit_code}")
        kind = ArtifactKind(result.stdout.decode())
    if kind == ArtifactKind.DIRECTORY:
        target.mkdir()
    if not artifact.exclude or kind != ArtifactKind.DIRECTORY:
        await machine.download(artifact.source, target)
        return True
    remote_archive = f"/tmp/taskcompendium-artifact-{uuid4().hex}.tar"
    try:
        result = await machine.run(
            Command(
                argv=(
                    "tar",
                    "-cf",
                    remote_archive,
                    *(f"--exclude={pattern}" for pattern in artifact.exclude),
                    "-C",
                    artifact.source,
                    ".",
                ),
                timeout=timeout,
                user="0",
            )
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot archive grading artifact {artifact.source}: exit={result.exit_code}")
        archive_path = target.with_suffix(".tar")
        await machine.download(remote_archive, archive_path)
        with tarfile.open(archive_path) as archive:
            archive.extractall(target, filter="data")
    finally:
        removed = await machine.run(Command(argv=("rm", "-f", remote_archive), timeout=timeout, user="0"))
        if removed.exit_code != 0:
            raise RuntimeError(f"Cannot remove grading artifact archive: exit={removed.exit_code}")
    return True


async def _shell_grade(
    verifier: ShellVerifierSpec, messages: tuple[dict[str, Any], ...], machine: Machine
) -> GradeResult:
    if isinstance(verifier.reward, FileReward):
        paths = tuple(file.path for file in verifier.reward.files)
        directories = tuple(sorted({str(PurePosixPath(path).parent) for path in paths}))
        for argv in (("mkdir", "-p", *directories), ("rm", "-f", *paths)):
            prepared = await machine.run(Command(argv=argv, timeout=verifier.timeout, user=verifier.user))
            if prepared.exit_code != 0:
                return GradeResult(
                    Outcome.INFRA_ERROR, None, "Cannot prepare private reward files", failure=GradingFailure.EXECUTION
                )
    await _install_files(machine, verifier.files)
    result = await machine.run(
        Command(
            argv=verifier.argv,
            env=resolve_env_vars(verifier.env),
            stdin=json.dumps(messages).encode(),
            timeout=verifier.timeout,
            user=verifier.user,
        )
    )
    diagnostics = {
        "stdout": result.stdout.decode(errors="replace"),
        "stderr": result.stderr.decode(errors="replace"),
        "exit_code": result.exit_code,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
    }
    if result.reason == ExitReason.TIMED_OUT:
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Grader command timed out",
            diagnostics=diagnostics,
            failure=GradingFailure.TIMEOUT,
        )
    if isinstance(verifier.reward, ExitCodeReward):
        passed = result.exit_code == 0
        return GradeResult(Outcome.GRADED, float(passed), passed=passed, diagnostics=diagnostics)
    if isinstance(verifier.reward, FileReward):
        return await _file_grade(machine, verifier.reward, verifier.timeout, diagnostics, verifier.user)
    if result.exit_code != 0 or result.stdout_truncated:
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            f"Grader command failed: {result.reason}, exit={result.exit_code}",
            diagnostics=diagnostics,
            failure=GradingFailure.EXECUTION,
        )
    try:
        reward = float(result.stdout.decode().strip())
    except (UnicodeError, ValueError):
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Grader stdout must contain one finite numeric reward",
            diagnostics=diagnostics,
            failure=GradingFailure.INVALID_REWARD,
        )
    if not math.isfinite(reward):
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Grader returned a nonfinite reward",
            diagnostics=diagnostics,
            failure=GradingFailure.INVALID_REWARD,
        )
    return GradeResult(Outcome.GRADED, reward, diagnostics=diagnostics)


async def _file_grade(
    machine: Machine, specification: FileReward, timeout: float, diagnostics: dict[str, Any], user: str | None
) -> GradeResult:
    for file in specification.files:
        result = await machine.run(
            Command(
                argv=(
                    "sh",
                    "-c",
                    f'if [ -f "$1" ]; then cat "$1"; else exit {MISSING_FILE_EXIT}; fi',
                    "reward-file",
                    file.path,
                ),
                timeout=timeout,
                user=user,
            )
        )
        if result.exit_code == MISSING_FILE_EXIT:
            continue
        if result.exit_code != 0 or result.stdout_truncated:
            return GradeResult(
                Outcome.INFRA_ERROR,
                None,
                f"Cannot read reward file: {file.path}",
                diagnostics=diagnostics,
                failure=GradingFailure.EXECUTION,
            )
        if not result.stdout.strip():
            return GradeResult(
                Outcome.INFRA_ERROR,
                None,
                f"Empty reward file: {file.path}",
                diagnostics=diagnostics,
                failure=GradingFailure.EMPTY_REWARD,
            )
        try:
            values = json.loads(result.stdout) if file.format == RewardFileFormat.JSON else None
            value = values[file.key] if values is not None else result.stdout.decode()
            if isinstance(value, bool):
                raise ValueError("A boolean is not a numeric reward")
            reward = float(value)
            if not math.isfinite(reward):
                raise ValueError("Nonfinite reward")
        except (UnicodeError, ValueError, TypeError, KeyError) as error:
            return GradeResult(
                Outcome.INFRA_ERROR,
                None,
                f"Invalid reward file {file.path}: {error}",
                diagnostics=diagnostics,
                failure=GradingFailure.INVALID_REWARD,
            )
        return GradeResult(
            Outcome.GRADED,
            reward,
            passed=None if specification.pass_above is None else reward > specification.pass_above,
            diagnostics={
                **diagnostics,
                "rewards": (
                    {
                        key: float(value)
                        for key, value in values.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    }
                    | {file.key: reward}
                    if isinstance(values, dict)
                    else {"reward": reward}
                ),
            },
        )
    return GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "Grader did not write a reward file",
        diagnostics=diagnostics,
        failure=GradingFailure.MISSING_REWARD,
    )


def rollout_request(task: TaskSpec, convention: SubmissionConvention) -> dict[str, Any]:
    """Prepare only the public task fields for inference."""
    if task.environment.interaction is not None or task.answer_type in (AnswerType.FILE, AnswerType.STATE):
        request = {"messages": conversation_messages(task.context)}
    else:
        request = chat_request(task, convention)
    if task.environment.kind != EnvironmentKind.NULL:
        if task.final_tools.functions:
            raise ValueError("Executable tasks expose the Shellbox shell tool only")
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a shell command in the task workspace. Files persist between commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
    return request


class ShellboxTaskSession:
    """Execute shell calls and grade the final task state."""

    def __init__(
        self,
        task: TaskSpec,
        machine: Machine | None,
        convention: SubmissionConvention,
        command_timeout: float,
        factories: Mapping[EnvironmentKind, MachineFactory],
        stage: TaskStage | None = None,
    ):
        self.task = task
        self.machine = machine
        self.convention = convention
        self.command_timeout = command_timeout
        self.factories = factories
        self.stage = stage

    async def prepare(self) -> dict[str, Any]:
        if self.task.environment_requirements.action_interfaces:
            raise ValueError("No executable action interfaces are configured")
        available = set() if self.machine is None else {"shell", "filesystem"}
        if not set(self.task.environment_requirements.capabilities) <= available:
            raise ValueError("The task environment does not supply its required capabilities")
        if self.stage is not None:
            assert self.machine is not None
            if self.stage.workdir_files:
                result = await self.machine.run(
                    Command(("pwd",), user=self.task.agent_user, timeout=self.command_timeout)
                )
                if result.exit_code != 0:
                    raise RuntimeError("Cannot find the stage working directory")
                workdir = result.stdout.decode().strip()
                await _install_files(
                    self.machine,
                    tuple(
                        file.model_copy(update={"path": f"{workdir.rstrip('/')}{file.path}"})
                        for file in self.stage.workdir_files
                    ),
                )
            for command in self.stage.setup:
                result = await self.machine.run(_machine_command(command))
                if result.exit_code != 0:
                    raise RuntimeError(f"Task stage {self.stage.name} setup failed: exit={result.exit_code}")
            if self.stage.healthcheck is not None:
                await _wait_for_healthcheck(self.machine, self.stage.healthcheck)
        return rollout_request(self.task, self.convention)

    async def advance(self, turn: ModelTurn) -> Transition:
        message = assistant_message(turn.message)
        if self.machine is None or not isinstance(message, AssistantToolCalls) or turn.stop_reason == "length":
            return Transition(done=True)
        observations = []
        for call in message.calls:
            if call.name != "shell" or set(call.arguments) != {"command"}:
                raise ValueError("Executable tasks require shell(command: string) calls")
            command = call.arguments["command"]
            if not isinstance(command, str):
                raise ValueError("Shell command must be a string")
            result = await self.machine.run(
                Command(argv=("sh", "-c", command), timeout=self.command_timeout, user=self.task.agent_user)
            )
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(
                        {
                            "stdout": result.stdout.decode(errors="replace"),
                            "stderr": result.stderr.decode(errors="replace"),
                            "exit_code": result.exit_code,
                            "reason": result.reason.value,
                            "truncated": result.stdout_truncated or result.stderr_truncated,
                        }
                    ),
                }
            )
        return Transition(done=False, observations=tuple(observations))

    async def grade(self, messages: tuple[dict[str, Any], ...]) -> GradeResult:
        return await grade_rollout(self.task, self.convention, messages, self.machine, self.factories)

    async def close(self) -> None:
        # The engine owns the machine and closes it after the session.
        pass


def _empty_rollout(task: TaskSpec) -> RolloutData:
    return RolloutData(
        task.id,
        tuple(conversation_messages(task.context)),
        (),
        (),
        (),
        (),
        GradeResult(Outcome.UNAVAILABLE, None, "Execution has no final grade"),
        "error",
    )


async def _remove_stage_grader(stage: TaskStage, machine: Machine) -> None:
    if stage.verifier.kind != VerifierKind.SHELL:
        return
    verifier = ShellVerifierSpec.model_validate_json(stage.verifier.parameters_json)
    if verifier.environment is not None:
        return
    paths = [file.path for file in verifier.files]
    if isinstance(verifier.reward, FileReward):
        paths.extend(file.path for file in verifier.reward.files)
    if not paths:
        return
    result = await machine.run(Command(("rm", "-f", *paths), user="0", timeout=verifier.timeout))
    if result.exit_code != 0:
        raise RuntimeError("Cannot remove private stage verifier files")


def _combined_stage_grade(grades: list[GradeResult], strategy: StageRewardStrategy) -> GradeResult:
    final = grades[-1]
    if strategy == StageRewardStrategy.FINAL:
        return final
    valid = [(grade, grade.reward) for grade in grades if grade.status == Outcome.GRADED and grade.reward is not None]
    if not valid:
        return final
    reward = sum(value for _, value in valid) / len(valid)
    components = [grade.diagnostics.get("rewards", {"reward": value}) for grade, value in valid]
    rewards = {key: sum(values.get(key, 0.0) for values in components) / len(valid) for key in set().union(*components)}
    return GradeResult(
        Outcome.GRADED,
        reward,
        passed=reward > 0,
        diagnostics={**final.diagnostics, "rewards": rewards},
    )


class ShellboxRolloutEngine:
    """Generate exact-token rollouts on one owner thread.

    Each task has an isolated machine. Another thread can request cancellation.
    """

    def __init__(
        self,
        model: Callable[[ModelRequest], Awaitable[ModelTurn]],
        factories: Mapping[EnvironmentKind, MachineFactory],
        *,
        max_turns: int,
        command_timeout: float,
        convention: SubmissionConvention,
        sessions: Mapping[str, Callable[[TaskSpec], TaskSession]] | None = None,
    ):
        if max_turns < 1 or command_timeout <= 0:
            raise ValueError("Rollout limits must be positive")
        self.model = model
        self.factories = factories
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.convention = convention
        self.sessions = {} if sessions is None else sessions
        self._execution_lock = threading.Lock()
        self._active: asyncio.Task | None = None
        self._cancelled = False

    def generate(self, tasks: Iterator[TaskSpec]) -> Iterator[RolloutData]:
        """Run tasks on the caller's thread and yield each completed rollout."""
        with asyncio.Runner() as runner:
            for task in tasks:
                yield runner.run(self._run_cancellable(task))

    def cancel(self) -> None:
        """Request cancellation from another thread and prevent new task execution."""
        with self._execution_lock:
            if self._cancelled:
                return
            self._cancelled = True
            if self._active is not None:
                self._active.get_loop().call_soon_threadsafe(self._active.cancel)

    async def _run_cancellable(self, task: TaskSpec) -> RolloutData:
        with self._execution_lock:
            if self._cancelled:
                raise asyncio.CancelledError
            self._active = asyncio.current_task()
        try:
            return await self._run(task)
        finally:
            with self._execution_lock:
                self._active = None

    async def _run(self, task: TaskSpec) -> RolloutData:
        """Run one task and release its session and machine after failure."""
        validate_task_verifiers(task)
        deadline = asyncio.timeout(task.attempt_timeout)
        async with AsyncExitStack() as resources:
            try:
                async with deadline:
                    return await self._run_task(task, resources)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                raise RolloutInterrupted(_empty_rollout(task), RolloutOperation.ATTEMPT) from error

    async def _run_task(self, task: TaskSpec, resources: AsyncExitStack) -> RolloutData:
        convention = self.convention
        if task.answer_type == AnswerType.NATIVE_ACTION:
            convention = SubmissionConvention(id="final-action", answer_format=AnswerFormat.FINAL_ACTION)
        try:
            machine = await resources.enter_async_context(task_machine(task.environment, self.factories))
        except Exception as error:
            raise RolloutInterrupted(_empty_rollout(task), RolloutOperation.START) from error
        if task.stages:
            assert machine is not None
            return await self._run_stages(task, machine, convention)
        if task.environment.interaction is None:
            session = ShellboxTaskSession(task, machine, convention, self.command_timeout, self.factories)
        else:
            session = self.sessions[task.environment.interaction](task)
        resources.push_async_callback(session.close)
        return await self._run_session(task, session)

    async def _run_stages(self, task: TaskSpec, machine: Machine, convention: SubmissionConvention) -> RolloutData:
        specification = StageVerifierSpec.model_validate_json(task.verifier.parameters_json)
        record = None
        grades = []
        stage_names = []
        last_graded_step = None
        interruption = None
        for stage in task.stages:
            phase = task.model_copy(
                update={
                    "context": stage.context or task.context,
                    "verifier": stage.verifier,
                    "stages": (),
                    "agent_timeout": task.agent_timeout if stage.agent_timeout is None else stage.agent_timeout,
                    "agent_user": task.agent_user if stage.agent_user is None else stage.agent_user,
                }
            )
            initial_steps = 0 if record is None else len(record.steps)
            initial_tokens = 0 if record is None else len(record.response_token_ids)
            session = ShellboxTaskSession(phase, machine, convention, self.command_timeout, self.factories, stage)
            try:
                record = await self._run_session(phase, session, record)
            except RolloutInterrupted as error:
                record = error.rollout
                interruption = error
            finally:
                await session.close()
                await _remove_stage_grader(stage, machine)
            grade = record.grade
            grades.append(grade)
            stage_names.append(stage.name)
            steps = tuple(
                (
                    replace(step, transition=replace(step.transition, grade=grade, reward=0.0))
                    if index >= initial_steps
                    else step
                )
                for index, step in enumerate(record.steps)
            )
            record = replace(record, steps=steps)
            if grade.status == Outcome.SKIPPED and interruption is None:
                continue
            if grade.status != Outcome.GRADED:
                record = replace(
                    record,
                    loss_mask=record.loss_mask[:initial_tokens] + (0,) * (len(record.loss_mask) - initial_tokens),
                )
                break
            last_graded_step = len(record.steps) - 1
            rewards = grade.diagnostics.get("rewards", {"reward": grade.reward})
            if any(rewards.get(key, -math.inf) < minimum for key, minimum in stage.minimum_rewards.items()):
                break
            if interruption is not None:
                break
        assert record is not None
        final = _combined_stage_grade(grades, specification.strategy)
        if last_graded_step is not None and final.status == Outcome.GRADED:
            steps = list(record.steps)
            last = steps[last_graded_step]
            steps[last_graded_step] = replace(last, transition=replace(last.transition, reward=final.reward))
            record = replace(record, steps=tuple(steps))
        record = replace(
            record,
            grade=replace(
                final,
                diagnostics={
                    **final.diagnostics,
                    "stages": [
                        {
                            "name": name,
                            "status": grade.status.value,
                            "reward": grade.reward,
                            "passed": grade.passed,
                            "error": grade.error,
                            "rewards": grade.diagnostics.get("rewards", {}),
                        }
                        for name, grade in zip(stage_names, grades, strict=True)
                    ],
                },
            ),
        )
        if interruption is not None:
            raise RolloutInterrupted(record, interruption.operation) from interruption.__cause__
        return record

    async def _run_session(self, task: TaskSpec, session: TaskSession, prefix: RolloutData | None = None) -> RolloutData:
        completed = _empty_rollout(task)
        if prefix is not None:
            completed = replace(prefix, grade=completed.grade)
        try:
            request = await session.prepare()
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.PREPARE) from error
        messages = request.pop("messages")
        if prefix is not None:
            messages = [*prefix.messages, *prefix.steps[-1].transition.observations, *messages]
        if prefix is None:
            completed = replace(completed, messages=tuple(messages))
        prompt: tuple[int, ...] = ()
        tokens: tuple[int, ...] = ()
        masks: tuple[int, ...] = ()
        logprobs: tuple[float, ...] | None = ()
        steps: list[RolloutStep] = []
        assistant_index = None
        if prefix is not None:
            prompt = prefix.prompt_token_ids
            tokens = prefix.prompt_token_ids + prefix.response_token_ids
            masks = prefix.loss_mask
            logprobs = prefix.logprobs
            steps = list(prefix.steps)
            assistant_index = len(prefix.messages) - 1
        initial_step_count = len(steps)
        stop_reason = "max_turns"
        deadline = None if task.agent_timeout is None else asyncio.get_running_loop().time() + task.agent_timeout
        for index in range(self.max_turns):
            try:
                async with asyncio.timeout_at(deadline):
                    turn = await self.model(ModelRequest(tuple(messages), request, tokens, assistant_index))
            except GenerationLimitReached as limit:
                stop_reason = "length"
                if len(steps) == initial_step_count:
                    return replace(
                        completed,
                        prompt_token_ids=prompt if prefix is not None else limit.prompt_token_ids,
                        grade=GradeResult(
                            Outcome.UNAVAILABLE, None, "Generation limit reached before the first response"
                        ),
                        stop_reason=stop_reason,
                    )
                messages = list(steps[-1].messages)
                break
            except RolloutContractError:
                raise
            except Exception as error:
                if len(steps) > initial_step_count:
                    try:
                        grade = await session.grade(completed.messages)
                    except Exception as grading_error:
                        raise RolloutInterrupted(completed, RolloutOperation.GRADE) from grading_error
                    completed = replace(completed, grade=grade)
                raise RolloutInterrupted(completed, RolloutOperation.MODEL) from error
            if turn.logprobs is not None and len(turn.logprobs) != len(turn.response_token_ids):
                raise RolloutContractError("Model logprobs must align with response tokens")
            if not turn.response_token_ids:
                raise RolloutContractError("Model response contains no token evidence")
            if assistant_index is None:
                prompt = turn.prompt_token_ids
                tokens = prompt
            if turn.prompt_token_ids[: len(tokens)] != tokens:
                raise RolloutContractError("Model transport changed the served token prefix")
            observation_count = len(turn.prompt_token_ids) - len(tokens)
            masks += (0,) * observation_count + (1,) * len(turn.response_token_ids)
            if logprobs is not None:
                logprobs = None if turn.logprobs is None else logprobs + (0.0,) * observation_count + turn.logprobs
            tokens = turn.prompt_token_ids + turn.response_token_ids
            assistant_index = len(messages)
            messages.append(turn.message)
            try:
                async with asyncio.timeout_at(deadline):
                    transition = await session.advance(turn)
            except RolloutContractError:
                raise
            except Exception as error:
                raise RolloutInterrupted(completed, RolloutOperation.ADVANCE) from error
            for values in (transition.token_rewards, transition.token_credit):
                if values is not None and len(values) != len(turn.response_token_ids):
                    raise RolloutContractError("Transition rewards and credit must align with model response tokens")
            stop_reason = turn.stop_reason
            if transition.reset_conversation is not None and index + 1 < self.max_turns:
                messages = list(transition.reset_conversation)
                prompt = tokens = masks = ()
                logprobs = ()
                steps.clear()
                completed = replace(
                    completed,
                    messages=tuple(messages),
                    prompt_token_ids=(),
                    response_token_ids=(),
                    loss_mask=(),
                    logprobs=(),
                    steps=(),
                    metrics={},
                )
                assistant_index = None
                continue
            steps.append(RolloutStep(turn, transition, len(tokens) - len(prompt) - 1, tuple(messages)))
            completed = RolloutData(
                task.id,
                tuple(messages),
                prompt,
                tokens[len(prompt) :],
                masks,
                logprobs,
                completed.grade,
                stop_reason,
                tuple(steps),
                transition.metrics,
            )
            if transition.done or stop_reason == "length":
                break
            if index + 1 == self.max_turns:
                stop_reason = "max_turns"
                break
            messages.extend(transition.observations)
        try:
            grade = await session.grade(tuple(messages))
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.GRADE) from error
        return replace(completed, grade=grade, stop_reason=stop_reason)
