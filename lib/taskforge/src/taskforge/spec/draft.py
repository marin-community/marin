# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a TaskCompendium ``TaskSpec`` from builder outputs and lower it for RolloutEngine.

These helpers own the serialized details a builder should not repeat: verifier
``parameters_json``, resource groups, and the capability requirements of a task
machine. ``assemble`` adds the checks TaskSpec itself does not make and returns
a spec that survives a JSON round trip.

A ``TaskSpec`` holds the semantic task only. How it runs (machine backend,
limits, user, deadlines) is the ``LoweredTaskSpec`` RolloutEngine executes, and
``lower`` alone builds one, choosing each machine's backend for the host. A
lowered spec is therefore bound to the host whose factories it names.

Only ``_presentation`` builds the presented context and concrete tools, and
only ``lower`` builds a ``LoweredTaskSpec``, so moving either is a single-site
change.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import JsonValue
from rolloutengine.lowering import SHELLBOX_SESSION, lower_task
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import Backend, MachineFactory, NetworkPolicy
from taskcompendium.grader import GraderPackage, script_package
from taskcompendium.grading import validate_verifier, verifier_descriptor
from taskcompendium.grading_contract import supports_candidate_mode
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.shell_verifier import (
    ExitCodeReward,
    FileReward,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    StdoutReward,
    VerifierArtifact,
    VerifierCommand,
)
from verifyit.spec import Mode, Spec, mode_of

from taskforge.sandbox.factories import MachineHost, container_backend

SHELL_CAPABILITY = "shell"
FILESYSTEM_CAPABILITY = "filesystem"

type Reward = StdoutReward | ExitCodeReward | FileReward

TEXT_ANSWER_TYPES = frozenset({AnswerType.TEXT, AnswerType.NUMBER})
ANSWER_TYPES_BY_KIND: dict[str, frozenset[AnswerType]] = {
    Mode.EXACT: TEXT_ANSWER_TYPES | {AnswerType.JSON},
    Mode.NUMERIC: TEXT_ANSWER_TYPES,
    Mode.MCQ: TEXT_ANSWER_TYPES,
    Mode.STRUCTURED_EXACT: frozenset({AnswerType.JSON}),
    Mode.PREDICTED_ACTION: frozenset({AnswerType.NATIVE_ACTION}),
}
"""The answer types each candidate-mode answer grader can grade (TaskCompendium's submission envelopes)."""
MACHINE_ANSWER_TYPES = frozenset({AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE})
SCRIPT_ANSWER_TYPES = TEXT_ANSWER_TYPES | {AnswerType.FILE, AnswerType.WORKSPACE_STATE}
"""The answer types a host-run script grader reads: the extracted text answer or captured output files."""
SCRIPT_KIND = Mode.SCRIPT.value
SHELL_KIND = "shell"
SKIPPED_KIND = "skipped"
PRIVATE_ROOTS = (PurePosixPath("/tests"), PurePosixPath("/logs/verifier"))
"""Where RolloutEngine installs private grader files and writes grader logs; no submission may land there."""


@dataclass(frozen=True)
class Resources:
    """Machine limits. ``None`` leaves the limit to the machine factory."""

    memory_mb: int | None = None
    cpus: int | None = None
    storage_mb: int | None = None
    gpus: int = 0


@dataclass(frozen=True)
class MachineSettings:
    """A ``MachineRuntimeSpec`` without its backend; ``lower`` picks the backend for this host."""

    network: NetworkPolicy
    resources: Resources
    user: str | None
    startup_timeout: float | None
    cleanup_timeout: float | None


@dataclass(frozen=True)
class _Presentation:
    """The TaskSpec fields that describe how the task is presented."""

    context: ConversationInput
    final_tools: tuple[FunctionDefinition, ...]


def file(path: str, content: str, mode: int = 0o644) -> TaskResource:
    """A UTF-8 text file at ``path`` relative to the machine root.

    ``"workspace/numbers.txt"`` lands at ``/workspace/numbers.txt``; an absolute
    path fails TaskResource validation.
    """
    return TaskResource(path=path, source=inline_resource(path, content.encode()).source, mode=f"{mode:o}")


def requirements(
    *, image: str | None, setup: Sequence[str] = (), workdir: str = "/workspace", env: Mapping[str, str] | None = None
) -> EnvironmentRequirements:
    """The task machine's contents, all visible to the agent.

    ``image`` is a digest-pinned registry reference (an ``ImageBuilder`` output)
    or ``None`` for ShellSim's built-in filesystem. ``setup`` commands run with
    ``sh -c`` as user ``0`` under the machine's startup timeout; a builder that
    needs a readiness check puts a wait loop there. Agent-visible files go to
    ``assemble(files=)``.
    """
    return EnvironmentRequirements(
        capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY),
        docker_image=image,
        working_directory=workdir,
        setup_commands=tuple(setup),
        environment_variables=dict(env or {}),
    )


def machine(
    *,
    startup_timeout: float | None,
    network: bool = False,
    resources: Resources = Resources(),
    user: str | None = None,
    cleanup_timeout: float | None = None,
) -> MachineSettings:
    """Deployment settings for the task machine or a shell grader's machine.

    ``user`` is the user model commands (or the shell grader) run as; ``None``
    keeps the image's user.
    """
    return MachineSettings(
        network=NetworkPolicy.ALLOW if network else NetworkPolicy.DENY,
        resources=resources,
        user=user,
        startup_timeout=startup_timeout,
        cleanup_timeout=cleanup_timeout,
    )


def session(
    *,
    max_turns: int,
    model_turn_timeout: float | None,
    command_timeout: float | None,
    tool_turn_timeout: float | None,
    total_turn_timeout: float | None,
    attempt_timeout: float | None,
    verifier_timeout: float | None,
    cleanup_timeout: float,
) -> TaskSessionSpec:
    """The turn budget and deadlines of one attempt in RolloutEngine's shellbox session."""
    return TaskSessionSpec(
        task_session=SHELLBOX_SESSION,
        max_turns=max_turns,
        model_turn_timeout=model_turn_timeout,
        command_timeout=command_timeout,
        tool_turn_timeout=tool_turn_timeout,
        total_turn_timeout=total_turn_timeout,
        attempt_timeout=attempt_timeout,
        verifier_timeout=verifier_timeout,
        cleanup_timeout=cleanup_timeout,
    )


def reward_file(
    path: str, file_format: RewardFileFormat, key: str = "reward", pass_above: float | None = None
) -> FileReward:
    """Read the score from one reward file the grader writes."""
    return FileReward(files=(RewardFile(path=path, format=file_format, key=key),), pass_above=pass_above)


def shell_verifier(
    argv: Sequence[str],
    reward: Reward,
    *,
    image: str,
    files: Sequence[TaskResource] = (),
    collect: Sequence[VerifierCommand] = (),
    artifacts: Sequence[VerifierArtifact] = (),
    env: Mapping[str, str] | None = None,
) -> GraderPackage:
    """A grader command run in a separate verifier machine started from ``image``.

    For a Docker task ``image`` is the task image. ``files`` are private and
    installed under ``/tests`` after the final model response. ``collect``
    commands run in the task machine and ``artifacts`` are copied from it into
    the verifier machine; an artifact a control may delete needs an explicit
    kind, not ``AUTO``. The deadline is the session's ``verifier_timeout`` and
    the user is the verifier machine's.
    """
    parameters = ShellVerifierSpec(argv=tuple(argv), reward=reward, collect=tuple(collect), artifacts=tuple(artifacts))
    verifier = VerifierSpec(
        kind=SHELL_KIND,
        parameters_json=parameters.model_dump_json(),
        environment_requirements=EnvironmentRequirements(docker_image=image, environment_variables=dict(env or {})),
    )
    return GraderPackage(verifier, tuple(files))


def script_verifier(script: str, config: Mapping[str, JsonValue], *, timeout: float) -> GraderPackage:
    """A Python grader that verifyit's ``script`` mode runs on the host after the attempt.

    ``script`` is ``grader.py`` and ``config`` its private ``config.json``. The
    grader reads captured output under ``$VERIFYIT_WORKSPACE`` (the text answer at
    ``answer.txt``, output paths under ``captured/``), its inputs under
    ``$VERIFYIT_TESTS_DIR``, writes ``$VERIFYIT_LOGS_DIR/verdict.json`` and exits 0.
    It runs under the host's ``python3`` with the standard library only.
    """
    return script_package(script.encode(), dict(config), timeout=timeout)


def answer_verifier(spec: Spec) -> GraderPackage:
    """A verifyit candidate-mode grader of the final answer, graded in process.

    Raises:
        ValueError: ``spec``'s mode is not a candidate mode (exact, numeric, mcq,
            predicted_action, structured_exact).
    """
    mode = mode_of(spec).value
    if not supports_candidate_mode(mode):
        raise ValueError(f"{mode!r} is not a candidate-mode answer grader")
    return GraderPackage(verifier_descriptor(spec))


def assemble(
    task_id: str,
    instruction: str,
    answer_type: AnswerType,
    grader: GraderPackage,
    source: Source,
    *,
    environment: EnvironmentRequirements | None,
    files: Sequence[TaskResource] = (),
    output_paths: Sequence[str] = (),
    system: str | None = None,
    final_tools: Sequence[FunctionDefinition] = (),
    tags: Sequence[str] = (),
) -> TaskSpec:
    """Build a TaskSpec and reject one that RolloutEngine could not grade as intended.

    ``environment`` is the task machine (``requirements(...)``) or ``None`` for a
    task without one. ``files`` are the agent-visible files installed relative to
    the machine root; the grader's files stay private.

    Raises:
        ValueError: TaskSpec validation failed, files or output paths were given
            without a task machine, the grader is invalid or does not fit the
            answer type or task machine, an output path lies in a private
            grading root, or private grader content is agent-visible.
    """
    if environment is None and (files or output_paths):
        raise ValueError("Files and output paths require a task machine")
    if environment is not None and SHELL_CAPABILITY not in environment.capabilities:
        raise ValueError(f"A task machine needs the {SHELL_CAPABILITY!r} capability")
    presentation = _presentation(instruction, system, final_tools)
    spec = TaskSpec(
        id=task_id,
        context=presentation.context,
        environment_requirements=environment or EnvironmentRequirements(),
        final_tools=presentation.final_tools,
        output_paths=tuple(output_paths),
        answer_type=answer_type,
        verifier=grader.verifier,
        source=source,
        resources=ResourceGroups(worker=tuple(files), verifier=grader.resources),
        tags=tuple(tags),
    )
    _validate_grader(spec.verifier)
    _check_grader(spec)
    for path in spec.output_paths:
        candidate = PurePosixPath(path)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Output path {path!r} must be absolute and normalized")
        if any(candidate.is_relative_to(root) for root in PRIVATE_ROOTS):
            raise ValueError(f"Output path {path!r} overlaps private grading files")
    visible = {resource_bytes(item) for item in spec.resources.all + spec.resources.worker}
    leaked = sorted(item.path for item in spec.resources.verifier if resource_bytes(item) in visible)
    if leaked:
        raise ValueError(f"Private verifier file content is also agent-visible: {leaked}")
    restored = TaskSpec.model_validate_json(spec.model_dump_json())
    if restored != spec:
        raise ValueError(f"Task {task_id!r} does not survive a JSON round trip")
    return restored


def lower(
    task: TaskSpec,
    *,
    host: MachineHost,
    task_machine: MachineSettings | None,
    verifier_machine: MachineSettings | None,
    session: TaskSessionSpec,
    factories: Mapping[str, MachineFactory],
) -> LoweredTaskSpec:
    """The ``LoweredTaskSpec`` that runs ``task`` on ``host``'s ``factories``.

    Every Taskforge ``LoweredTaskSpec`` is built here. Each machine's backend is
    ShellSim when its requirements name no image, else the host's container
    backend. Script and answer graders run in process, so only a shell grader
    takes a verifier machine.

    Raises:
        ValueError: A machine is given where the task has none or missing where it
            has one, or RolloutEngine rejects the lowered spec.
        NotImplementedError: RolloutEngine cannot run the lowered spec.
    """
    if (task_machine is None) != (SHELL_CAPABILITY not in task.environment_requirements.capabilities):
        raise ValueError("A task machine is given exactly when the task has the shell capability")
    if (verifier_machine is None) != (task.verifier.kind != SHELL_KIND):
        raise ValueError("A verifier machine is given exactly when the task has a shell grader")
    runtime = TaskRuntimeSpec(
        task_machine=_machine_runtime(task_machine, task.environment_requirements, host),
        verifier_machine=_machine_runtime(verifier_machine, task.verifier.environment_requirements, host),
    )
    return lower_task(task, runtime, session, factories=factories, sessions={})


def _machine_runtime(
    settings: MachineSettings | None, requirements: EnvironmentRequirements, host: MachineHost
) -> MachineRuntimeSpec | None:
    if settings is None:
        return None
    backend = Backend.SHELLSIM if requirements.docker_image is None else container_backend(host)
    return MachineRuntimeSpec(
        backend=backend.value,
        network=settings.network,
        cpus=settings.resources.cpus,
        memory_mb=settings.resources.memory_mb,
        storage_mb=settings.resources.storage_mb,
        gpus=settings.resources.gpus,
        user=settings.user,
        startup_timeout=settings.startup_timeout,
        cleanup_timeout=settings.cleanup_timeout,
    )


def _presentation(instruction: str, system: str | None, final_tools: Sequence[FunctionDefinition]) -> _Presentation:
    return _Presentation(context=_conversation(instruction, system), final_tools=tuple(final_tools))


def _conversation(instruction: str, system: str | None = None) -> ConversationInput:
    system_events = () if system is None else (TextMessage(role="system", content=system),)
    return ConversationInput(events=(*system_events, TextMessage(role="user", content=instruction)))


def _validate_grader(verifier: VerifierSpec) -> None:
    if verifier.kind == SHELL_KIND:
        ShellVerifierSpec.model_validate_json(verifier.parameters_json)
        return
    if verifier.kind == SKIPPED_KIND:
        if not isinstance(json.loads(verifier.parameters_json).get("reason"), str):
            raise ValueError("A skipped grader needs a reason")
        return
    validate_verifier(verifier)


def _check_grader(spec: TaskSpec) -> None:
    kind = spec.verifier.kind
    answer_type = spec.answer_type
    if kind == SKIPPED_KIND:
        return
    if kind == SHELL_KIND:
        if spec.environment_requirements.docker_image is None:
            raise ValueError("A shell verifier requires an image-backed task machine")
        return
    if kind == SCRIPT_KIND:
        if answer_type not in SCRIPT_ANSWER_TYPES:
            raise ValueError(f"A script verifier cannot grade a {answer_type.value!r} answer")
        if answer_type in MACHINE_ANSWER_TYPES and not spec.output_paths:
            raise ValueError(f"A script verifier grading a {answer_type.value!r} answer needs output paths")
        return
    if answer_type in MACHINE_ANSWER_TYPES:
        raise ValueError(f"A {answer_type.value!r} answer requires a script or shell verifier, not {kind!r}")
    allowed = ANSWER_TYPES_BY_KIND.get(kind)
    if allowed is None:
        raise ValueError(f"Taskforge does not grade {kind!r} verifiers")
    if answer_type not in allowed:
        names = " or ".join(sorted(item.value for item in allowed))
        raise ValueError(f"A {kind!r} verifier requires a {names} answer, not {answer_type.value!r}")
