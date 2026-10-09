# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a TaskCompendium ``TaskSpec`` from builder outputs and lower it for RolloutEngine.

These helpers own the serialized details a builder should not repeat: the
grader model, resource groups, and the capability requirements of a task
machine. ``assemble`` adds the checks TaskSpec itself does not make and returns
a spec that survives a JSON round trip.

A ``TaskSpec`` holds the semantic task only. How it runs (machine backend,
limits, user, deadlines) is the ``LoweredTaskSpec`` RolloutEngine executes, and
``lower`` alone builds one, choosing each machine's backend for the host. A
lowered spec is therefore bound to the host whose factories it names.

Only ``_presentation`` builds the presented context, concrete tools and answer
format, and only ``lower`` builds a ``LoweredTaskSpec``, so moving either is a
single-site change.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import JsonValue
from rolloutengine.lowering import SHELLBOX_SESSION, lower_task
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import Backend, MachineFactory, NetworkPolicy
from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    AnswerFormat,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    FunctionDefinition,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    StdoutReward,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.submission import require_submission_compatibility
from verifyit.candidate import IN_PROCESS_MODES
from verifyit.spec import Spec, mode_of

from taskforge.sandbox.factories import MachineHost, container_backend, grading_environment
from taskforge.sandbox.images import GRADER_BASE_IMAGE

SHELL_CAPABILITY = "shell"
FILESYSTEM_CAPABILITY = "filesystem"

type Reward = StdoutReward | ExitCodeReward | FileReward

ANSWER_PATH = "/app/answer.txt"
"""Where a script grader finds the extracted answer of a text, number, JSON or native-action task."""
PYTHON_GRADER_ARGV = ("python3", "/tests/grade.py")
PYTHON_GRADER_FILES = frozenset({"grade.py", "config.json"})
MACHINE_ANSWERS = frozenset({AnswerType.FILE, AnswerType.WORKSPACE_STATE})
"""Answers the grader reads from the task machine's files rather than from the conversation."""


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
    answer_format: AnswerFormat


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


def grader_environment(task_image: str | None, *, env: Mapping[str, str] | None = None) -> EnvironmentRequirements:
    """The verifier machine of a script grader: the task image, or Taskforge's grader base without one.

    A Docker task grades in a fresh machine from its own image, so the grader sees the task's tools. A
    ShellSim task, or a task without a machine, grades in ``GRADER_BASE_IMAGE`` (CPython 3.12 with
    the standard library, ``sh``, ``setsid`` and ``tar``, as root in ``/app``).
    """
    return EnvironmentRequirements(
        docker_image=GRADER_BASE_IMAGE if task_image is None else task_image, environment_variables=dict(env or {})
    )


def script_grader(
    argv: Sequence[str],
    reward: Reward,
    *,
    environment: EnvironmentRequirements,
    answer_path: str | None,
    timeout: float,
    files: Sequence[TaskResource] = (),
    cwd: str = "/app",
    env: Mapping[str, str] | None = None,
    collect: Sequence[VerifierCommand] = (),
    artifacts: Sequence[VerifierArtifact] = (),
) -> GraderPackage:
    """A grader command run in a separate verifier machine built from ``environment``.

    ``environment`` names a digest-pinned image (``grader_environment``) or, on Iris, a packages
    lock that RolloutEngine builds on its local backend. The verifier machine receives ``files``
    under ``/tests``, the task's agent-visible files at ``/``, the captured ``output_paths`` at
    their own absolute paths, the conversation as JSON chat messages at
    ``/tests/conversation.json``, and the extracted answer at ``answer_path``, which is ``None``
    for a file or workspace-state answer. ``collect`` commands run as root in the task machine and
    ``artifacts`` are copied from it first; an artifact a control may delete needs an explicit
    kind, not ``AUTO``. ``argv`` runs in ``cwd`` as the verifier machine's user and must finish
    within ``timeout`` and the session's ``verifier_timeout``.
    """
    grader = ScriptGrader(
        argv=tuple(argv),
        cwd=cwd,
        env=dict(env or {}),
        environment=environment,
        collect=tuple(collect),
        artifacts=tuple(artifacts),
        answer_path=answer_path,
        reward=reward,
        timeout=timeout,
    )
    return GraderPackage(grader, tuple(files))


def python_grader(
    script: str,
    config: Mapping[str, JsonValue],
    *,
    environment: EnvironmentRequirements,
    answer_path: str | None,
    timeout: float,
    files: Sequence[TaskResource] = (),
) -> GraderPackage:
    """A Python program graded as ``python3 /tests/grade.py`` in a verifier machine.

    ``script`` is ``/tests/grade.py`` and ``config`` its private ``/tests/config.json``; ``files``
    are further private inputs, paths relative to ``/tests``. The program reads the extracted
    answer at ``answer_path`` (``ANSWER_PATH`` for a text answer, ``None`` for a file answer) and
    captured output files at their absolute paths, never stdin, and prints the reward as its last
    stdout line.

    Raises:
        ValueError: a file is named ``grade.py`` or ``config.json``, or two files share a path.
    """
    paths = [resource.path for resource in files]
    if len(set(paths)) != len(paths) or PYTHON_GRADER_FILES & set(paths):
        raise ValueError(f"Python grader files repeat a path or shadow grade.py or config.json: {sorted(paths)}")
    resources = (
        inline_resource("grade.py", script.encode()),
        inline_resource("config.json", json.dumps(dict(config), sort_keys=True).encode()),
        *files,
    )
    return script_grader(
        PYTHON_GRADER_ARGV,
        StdoutReward(),
        environment=environment,
        answer_path=answer_path,
        timeout=timeout,
        files=resources,
    )


def answer_grader(spec: Spec, *, environment: EnvironmentRequirements | None = None) -> GraderPackage:
    """A verifyit grader of the final answer.

    Without ``environment`` the mode must be one verifyit grades in process (exact, numeric, mcq,
    math, ifeval, json_schema, xml_elements, csv_columns, structured_exact, predicted_action). Any
    other mode runs the verifyit command in a verifier machine built from ``environment``, whose
    image must contain verifyit.

    Raises:
        ValueError: the mode does not grade in process and no environment was given.
    """
    mode = mode_of(spec)
    if environment is None and mode not in IN_PROCESS_MODES:
        raise ValueError(
            f"verifyit mode {mode.value!r} does not grade in process; give it a grading environment with "
            "verifyit installed, or use script_grader"
        )
    return verifyit_package(spec, environment=environment)


def assemble(
    task_id: str,
    instruction: str,
    answer_type: AnswerType,
    answer_format: AnswerFormat,
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

    ``answer_format`` says how the model is asked for its final answer and how the answer is read
    from its conversation; a file or workspace-state task carries one too, which grading never
    reads. ``environment`` is the task machine (``requirements(...)``) or ``None`` for a task
    without one. ``files`` are the agent-visible files installed relative to the machine root; the
    grader's files stay private.

    Raises:
        ValueError: TaskSpec validation failed, files or output paths were given without a task
            machine, the answer format cannot carry the answer for this grader, a file or
            workspace-state answer has no output path or artifact to reach the grader, an output
            path is not normalized, or private grader content is agent-visible.
    """
    if environment is None and (files or output_paths):
        raise ValueError("Files and output paths require a task machine")
    if environment is not None and SHELL_CAPABILITY not in environment.capabilities:
        raise ValueError(f"A task machine needs the {SHELL_CAPABILITY!r} capability")
    if answer_type in MACHINE_ANSWERS and not output_paths and not _reads_task_machine(grader):
        raise ValueError(f"A {answer_type.value!r} answer needs output paths or grader artifacts to reach the grader")
    presentation = _presentation(instruction, system, final_tools, answer_format)
    spec = TaskSpec(
        id=task_id,
        context=presentation.context,
        environment_requirements=environment or EnvironmentRequirements(),
        final_tools=presentation.final_tools,
        output_paths=tuple(output_paths),
        answer_type=answer_type,
        answer_format=presentation.answer_format,
        grader=grader.grader,
        source=source,
        resources=ResourceGroups(worker=tuple(files), verifier=grader.resources),
        tags=tuple(tags),
    )
    if spec.answer_type in CONVERSATION_ANSWERS:
        # As the Shellbox session does at start: only an answer read from the conversation has a format to check.
        require_submission_compatibility(spec)
    for path in spec.output_paths:
        candidate = PurePosixPath(path)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Output path {path!r} must be absolute and normalized")
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

    Every Taskforge ``LoweredTaskSpec`` is built here. Each machine's backend is ShellSim when its
    requirements name neither an image nor a packages lock, the local backend for a lock alone, and
    the host's container backend for an image. A grader with an environment (a script grader, or a
    verifyit grader that does not grade in process) takes a verifier machine; no other grader does.

    Raises:
        ValueError: A machine is given where the task has none or missing where it has one, or
            RolloutEngine rejects the lowered spec.
        NotImplementedError: RolloutEngine cannot run the lowered spec.
    """
    if (task_machine is None) != (SHELL_CAPABILITY not in task.environment_requirements.capabilities):
        raise ValueError("A task machine is given exactly when the task has the shell capability")
    grading = grading_environment(task)
    if (verifier_machine is None) != (grading is None):
        raise ValueError("A verifier machine is given exactly when the task's grader has an environment")
    runtime = TaskRuntimeSpec(
        task_machine=(
            None if task_machine is None else machine_runtime(task_machine, task.environment_requirements, host)
        ),
        verifier_machine=(
            None if verifier_machine is None or grading is None else machine_runtime(verifier_machine, grading, host)
        ),
    )
    return lower_task(task, runtime, session, factories=factories, sessions={})


def machine_runtime(
    settings: MachineSettings, requirements: EnvironmentRequirements, host: MachineHost
) -> MachineRuntimeSpec:
    """The ``MachineRuntimeSpec`` of a machine with ``requirements`` on ``host``.

    ``lower`` builds every task and verifier machine with it; a builder prototyping a machine
    outside a lowered task uses it too, so the backend choice stays in one place.
    """
    return MachineRuntimeSpec(
        backend=machine_backend(requirements, host).value,
        network=settings.network,
        cpus=settings.resources.cpus,
        memory_mb=settings.resources.memory_mb,
        storage_mb=settings.resources.storage_mb,
        gpus=settings.resources.gpus,
        user=settings.user,
        startup_timeout=settings.startup_timeout,
        cleanup_timeout=settings.cleanup_timeout,
    )


def machine_backend(requirements: EnvironmentRequirements, host: MachineHost) -> Backend:
    """ShellSim without an image or lock, the local backend for a lock alone, else the host's container backend."""
    if requirements.docker_image is not None:
        return container_backend(host)
    if requirements.packages_lock is not None:
        return Backend.LOCAL
    return Backend.SHELLSIM


def _reads_task_machine(grader: GraderPackage) -> bool:
    return isinstance(grader.grader, ScriptGrader) and bool(grader.grader.artifacts or grader.grader.collect)


def _presentation(
    instruction: str, system: str | None, final_tools: Sequence[FunctionDefinition], answer_format: AnswerFormat
) -> _Presentation:
    return _Presentation(
        context=_conversation(instruction, system), final_tools=tuple(final_tools), answer_format=answer_format
    )


def _conversation(instruction: str, system: str | None = None) -> ConversationInput:
    system_events = () if system is None else (TextMessage(role="system", content=system),)
    return ConversationInput(events=(*system_events, TextMessage(role="user", content=instruction)))
