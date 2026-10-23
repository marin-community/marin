# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a TaskCompendium ``TaskSpec`` from builder outputs and lower it for RolloutEngine.

These helpers own the serialized details a builder should not repeat: the
grader model, resource groups, and the capability requirements of a task
machine. ``assemble`` adds the checks TaskSpec itself does not make and returns
a spec that survives a JSON round trip.

A ``TaskSpec`` holds the semantic task only. How it runs (machine backend,
limits, user, deadlines, harness) is the ``TaskRolloutSpec`` RolloutEngine executes, and
``lower`` alone builds one, choosing each machine's backend for the host. A
lowered spec is therefore bound to the host whose factories it names, and its
task may name the software differently from the ``TaskSpec``: on a host with
the local backend, ``GRADER_BASE_IMAGE`` becomes ``LOCAL_CPYTHON``. A package
machine always lowers onto the local backend, so on a host without one ``check``
rejects it (``host_requirements``).

Every Taskforge spec names ``TOOL_HARNESS``, RolloutEngine's shell harness, and every Taskforge engine
comes from ``rollout_engine``, which registers that harness with its command timeout: the timeout of one
shell command belongs to the harness, not to the spec.

Only ``lower`` builds a ``TaskRolloutSpec``, so moving it is a single-site change.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from pydantic import JsonValue
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from shellbox.machine import Backend, MachineFactory, NetworkPolicy, PackageSpec
from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    GRADER_ROOTS,
    AnswerFormat,
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    FunctionDefinition,
    Grader,
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
    under_grader_root,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.submission import require_submission_compatibility
from verifyit.candidate import IN_PROCESS_MODES
from verifyit.spec import Spec, mode_of

from taskforge.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT, MachineHost, container_backend, grading_environment
from taskforge.sandbox.images import GRADER_BASE_IMAGE
from taskforge.spec.verdict import VERDICT_REWARD

SHELL_CAPABILITY = "shell"
TOOL_HARNESS = "tools"
"""The harness every Taskforge spec selects: a shell tool on the task machine, or a plain conversation without one."""
NO_MACHINE = EnvironmentRequirements()
"""The environment of a task without a machine: the solver answers in the conversation, with no tools."""
FILESYSTEM_CAPABILITY = "filesystem"
SHELLSIM_CPYTHON = PackageSpec(pypi_packages=())
"""ShellSim's CPython 3.13 and no PyPI packages; a bare ``PackageSpec()`` would have no ``python3``."""

LOCAL_CPYTHON = PackageSpec(pypi_packages=())
"""``GRADER_BASE_IMAGE``'s software on the local backend: a uv-managed CPython 3.12 and no PyPI packages."""
LOCAL_READ_ONLY = "read_only"
"""shellbox's ``SystemDirectories.READ_ONLY``, the local factory's default mode for system directories."""
LOCAL_COPY_ON_WRITE = "copy_on_write"
"""shellbox's ``SystemDirectories.COPY_ON_WRITE``, the mode ``sandbox.factories.local_factory`` sets."""
LOCAL_UNWRITABLE_ROOTS: Mapping[str, tuple[str, ...]] = {
    LOCAL_READ_ONLY: ("/bin", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/sbin", "/sys", "/usr"),
    LOCAL_COPY_ON_WRITE: ("/sys",),
}
"""The host directories a local machine cannot write, by its factory's system-directory mode.

A machine that must write under one of them keeps its image (``host_requirements``). Mirrors shellbox's
Linux-only enum; ``tests/spec/test_draft.py`` checks they agree.
"""


type Reward = StdoutReward | ExitCodeReward | FileReward

ANSWER_PATH = "/app/answer.txt"
"""Where a script grader finds the extracted answer of a text, number, JSON or native-action task."""
GRADER_RESOURCE_ROOT = "/tests"
"""Where RolloutEngine installs ``resources.verifier`` on the verifier machine; agent-visible files are staged
at ``/``."""
PYTHON_GRADER_ARGV = ("python3", f"{GRADER_RESOURCE_ROOT}/grade.py")
PYTHON_GRADER_FILES = frozenset({"grade.py", "config.json"})
MACHINE_ANSWERS = frozenset({AnswerType.FILE, AnswerType.WORKSPACE_STATE})
"""Answers the grader reads from the task machine's files rather than from the conversation."""


@dataclass(frozen=True)
class MachineLimits:
    """A machine's CPU, memory, storage and GPU limits. ``None`` leaves the limit to the machine factory."""

    memory_mb: int | None = None
    cpus: int | None = None
    storage_mb: int | None = None
    gpus: int = 0


@dataclass(frozen=True)
class MachineSettings:
    """A ``MachineRuntimeSpec`` without its backend; ``lower`` picks the backend for this host."""

    network: NetworkPolicy
    limits: MachineLimits
    user: str | None
    startup_timeout: float | None
    cleanup_timeout: float | None


def file(path: str, content: str, mode: int = 0o644) -> TaskResource:
    """A UTF-8 text file at ``path`` relative to the machine root.

    ``"workspace/numbers.txt"`` lands at ``/workspace/numbers.txt``; an absolute
    path fails TaskResource validation.
    """
    return TaskResource(path=path, source=inline_resource(path, content.encode()).source, mode=f"{mode:o}")


def requirements(
    *,
    image: str | None,
    packages: Sequence[str] = (),
    setup: Sequence[str] = (),
    workdir: str = "/workspace",
    env: Mapping[str, str] | None = None,
) -> EnvironmentRequirements:
    """The task machine's contents, all visible to the agent.

    ``image`` is a digest-pinned registry reference (an ``ImageBuilder`` output), whose commands
    run as Linux processes, or ``None`` for ShellSim's simulated shell and built-in filesystem with
    the catalog's CPython (``SHELLSIM_CPYTHON``). ``packages`` instead names PyPI requirements
    (``"numpy"``, ``"pandas==2.2.3"``) for a machine without an image: Linux processes with CPython
    3.12 and those packages, installed from wheels when the machine starts, with no image to build.
    Such a machine runs on the local backend only (``host_requirements``). ``setup`` commands run with
    ``sh -c`` as user ``0`` under the machine's startup timeout; a builder that
    needs a readiness check puts a wait loop there. Agent-visible files go to
    ``assemble(files=)``.

    Raises:
        ValueError: both ``image`` and ``packages`` are given, or a package is not one requirement
            (``check_packages``).
    """
    if image is not None and packages:
        raise ValueError("A task machine names an image or packages, not both")
    check_packages(packages)
    if packages:
        semantics, package_spec = CommandSemantics.LINUX_PROCESS, PackageSpec(pypi_packages=tuple(packages))
    elif image is None:
        semantics, package_spec = CommandSemantics.SHELL_SIMULATOR, SHELLSIM_CPYTHON
    else:
        semantics, package_spec = CommandSemantics.LINUX_PROCESS, None
    return EnvironmentRequirements(
        capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY),
        command_semantics=semantics,
        docker_image=image,
        package_spec=package_spec,
        working_directory=workdir,
        setup_commands=tuple(setup),
        environment_variables=dict(env or {}),
    )


def check_packages(packages: Sequence[str]) -> None:
    """Raise ``ValueError`` unless each entry is one PyPI requirement without pip options.

    The entries become lines of a uv requirements file, where a word starting with ``-`` anywhere on
    the line, as in ``" -r other.txt"``, is read as an option.
    """
    malformed = [
        entry
        for entry in packages
        if not entry.strip() or any(word.startswith("-") for word in entry.split()) or {"\n", "\r"} & set(entry)
    ]
    if malformed:
        raise ValueError(f"Packages are PyPI requirements, one per entry, without pip options: {malformed}")


def machine(
    *,
    startup_timeout: float | None,
    network: bool = False,
    limits: MachineLimits = MachineLimits(),
    user: str | None = None,
    cleanup_timeout: float | None = None,
) -> MachineSettings:
    """Deployment settings for the task machine or a shell grader's machine.

    ``user`` is the user model commands (or the shell grader) run as; ``None``
    keeps the image's user.
    """
    return MachineSettings(
        network=NetworkPolicy.ALLOW if network else NetworkPolicy.DENY,
        limits=limits,
        user=user,
        startup_timeout=startup_timeout,
        cleanup_timeout=cleanup_timeout,
    )


GRADER_TIMEOUT = 300.0
"""Default command timeout for generated script graders, and budget for in-process grading."""
MACHINE_PREPARATION_TIMEOUT = 300.0
GRADING_MACHINE = machine(startup_timeout=SANDBOX_SCHEDULING_TIMEOUT + MACHINE_PREPARATION_TIMEOUT)
"""Machine settings allowing scheduling plus resource installation and setup."""


def verifier_timeout(grader: Grader) -> float:
    """Allow the verifier's scheduling budget and its own command timeout before ending grading."""
    return SANDBOX_SCHEDULING_TIMEOUT + (grader.timeout if isinstance(grader, ScriptGrader) else GRADER_TIMEOUT)


def rollout_limits(
    *,
    max_turns: int,
    model_turn_timeout: float | None,
    tool_turn_timeout: float | None,
    total_turn_timeout: float | None,
    attempt_timeout: float | None,
    verifier_timeout: float | None,
    cleanup_timeout: float,
) -> RolloutLimits:
    """The turn budget and deadlines of one attempt; the shell command timeout is the harness's (``rollout_engine``)."""
    return RolloutLimits(
        max_turns=max_turns,
        # No per-attempt token budget on trials: a trial is bounded by its deadlines and its turn budget,
        # and a cap here would end an attempt the evidence needs whole.
        max_total_tokens=None,
        model_turn_timeout=model_turn_timeout,
        tool_turn_timeout=tool_turn_timeout,
        total_turn_timeout=total_turn_timeout,
        attempt_timeout=attempt_timeout,
        verifier_timeout=verifier_timeout,
        cleanup_timeout=cleanup_timeout,
    )


def rollout_engine(factories: Mapping[str, MachineFactory], *, command_timeout: float | None) -> RolloutEngine:
    """A ``RolloutEngine`` on ``factories`` with RolloutEngine's shell harness registered as ``TOOL_HARNESS``.

    ``command_timeout`` bounds each shell command of a turn; ``None`` leaves it to the turn deadlines.
    ``RolloutEngine.check`` requires it to be shorter than a spec's ``tool_turn_timeout`` when both are set.
    """
    return RolloutEngine(factories, {TOOL_HARNESS: ShellHarness(command_timeout=command_timeout)})


def check(spec: TaskRolloutSpec, factories: Mapping[str, MachineFactory]) -> None:
    """Reject ``spec`` as ``RolloutEngine.check`` does on ``factories`` under the shell harness.

    The check is the engine's own, before any machine starts. The harness it checks against has no command
    timeout; the engine that runs ``spec`` checks its own again.

    Raises:
        ValueError: RolloutEngine rejects the spec.
        NotImplementedError: RolloutEngine cannot run the spec.
    """
    rollout_engine(factories, command_timeout=None).check(spec)


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
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=GRADER_BASE_IMAGE if task_image is None else task_image,
        environment_variables=dict(env or {}),
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
    within ``timeout`` and the attempt's ``RolloutLimits.verifier_timeout``.
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
    are further private inputs, paths relative to ``/tests``. No private file may have the same
    bytes as a solver-visible file (``assemble`` refuses it); the verifier machine already holds the
    solver-visible files at their own paths, so the program reads them there. The program reads the extracted
    answer at ``answer_path`` (``ANSWER_PATH`` for a text answer, ``None`` for a file answer) and
    captured output files at their absolute paths, never stdin, and writes its verdict
    (``spec.verdict.verdict_json``: the reward and one result per check) to ``spec.verdict.GRADER_VERDICT_PATH``.

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
        VERDICT_REWARD,
        environment=environment,
        answer_path=answer_path,
        timeout=timeout,
        files=resources,
    )


def answer_grader(spec: Spec, *, environment: EnvironmentRequirements | None = None) -> GraderPackage:
    """A verifyit grader of the final answer.

    Without ``environment`` the mode must be one verifyit grades in process (exact, numeric, mcq,
    math, ifeval, json_schema, xml_elements, csv_columns, structured_exact, predicted_action). Any
    other mode except the LLM judge runs ``python -c 'import verifyit ...'`` in a verifier machine
    built from ``environment``, whose image must contain verifyit. ``GRADER_BASE_IMAGE`` does not.
    RolloutEngine grades an LLM judge (``JudgeSpec``) on the host through the judge service whatever
    ``environment`` says, and ``grade_candidate`` serves no judge, so a judge cannot be prototyped at
    build time: grade with deterministic checks in ``python_grader`` instead.

    Raises:
        ValueError: the mode does not grade in process and no environment was given, or the
            environment is ``GRADER_BASE_IMAGE``.
    """
    mode = mode_of(spec)
    if environment is None and mode not in IN_PROCESS_MODES:
        raise ValueError(
            f"verifyit mode {mode.value!r} does not grade in process; give it a grading environment with "
            "verifyit installed, or use script_grader"
        )
    if environment is not None and environment.docker_image == GRADER_BASE_IMAGE:
        raise ValueError(
            f"verifyit mode {mode.value!r} cannot run on GRADER_BASE_IMAGE, which has no verifyit (it fails with "
            "\"No module named 'verifyit'\"); grade_candidate calls no model either, so write deterministic checks "
            "with python_grader"
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
    reads. ``environment`` is the task machine (``requirements(...)``), or ``NO_MACHINE`` or ``None``
    for a task without one. ``files`` are the agent-visible files installed relative to the machine
    root; the grader's files stay private, so none of them may have the same bytes as an
    agent-visible file.

    Raises:
        ValueError: TaskSpec validation failed, files or output paths were given without a task
            machine, the answer format cannot carry the answer for this grader, a file or
            workspace-state answer has no output path or artifact to reach the grader, or private
            grader content is agent-visible. TaskSpec validation covers output paths that are not
            absolute and normalized or that reach the grader's private directories.
    """
    if environment == NO_MACHINE:
        environment = None
    if environment is None and (files or output_paths):
        raise ValueError("Files and output paths require a task machine")
    if environment is not None and SHELL_CAPABILITY not in environment.capabilities:
        raise ValueError(f"A task machine needs the {SHELL_CAPABILITY!r} capability")
    if answer_type in MACHINE_ANSWERS and not output_paths and not _reads_task_machine(grader):
        raise ValueError(f"A {answer_type.value!r} answer needs output paths or grader artifacts to reach the grader")
    spec = TaskSpec(
        id=task_id,
        context=_conversation(instruction, system),
        environment_requirements=environment or EnvironmentRequirements(),
        final_tools=tuple(final_tools),
        output_paths=tuple(output_paths),
        answer_type=answer_type,
        answer_format=answer_format,
        grader=grader.grader,
        source=source,
        resources=ResourceGroups(worker=tuple(files), verifier=grader.resources),
        tags=tuple(tags),
    )
    if spec.answer_type in CONVERSATION_ANSWERS:
        # As the shell harness does at start: only an answer read from the conversation has a format to check.
        require_submission_compatibility(spec)
    # A private file whose bytes the solver can read is no longer private: the verifier machine
    # already sees the agent-visible files at their own paths, so the grader reads them there.
    visible = {resource_bytes(item) for item in _agent_visible(spec)}
    leaked = sorted(item.path for item in spec.resources.verifier if resource_bytes(item) in visible)
    if leaked:
        raise ValueError(f"Private verifier file content is also agent-visible: {leaked}")
    # Grading stages agent-visible files after the verifier's, so one under a grader root would replace a
    # private grader file or plant a verdict.
    reserved = sorted(item.path for item in _agent_visible(spec) if under_grader_root(f"/{item.path}"))
    if reserved:
        raise ValueError(f"Agent-visible files lie under the grader roots {GRADER_ROOTS}: {reserved}")
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
    limits: RolloutLimits,
    factories: Mapping[str, MachineFactory],
) -> TaskRolloutSpec:
    """The ``TaskRolloutSpec`` that runs ``task`` on ``host``'s ``factories`` under ``TOOL_HARNESS``.

    Every Taskforge ``TaskRolloutSpec`` is built here. Each machine runs the requirements
    ``host_requirements`` gives for ``factories``, the same rule for the task machine and the verifier
    machine, on the backend ``machine_backend`` picks for them; the lowered task carries those
    requirements and each machine's ``MachineRuntimeSpec.backend`` names its backend. A grader with an
    environment (a script grader, or a verifyit grader that is neither in process nor a judge) takes a
    verifier machine; no other grader does. The spec keeps the engine's default exact-token trace; a
    caller can select ``TraceMode.MESSAGES`` when token evidence is unnecessary.

    Raises:
        ValueError: A machine is given where the task has none or missing where it has one, or
            RolloutEngine rejects the lowered spec (``check``).
        NotImplementedError: RolloutEngine cannot run the lowered spec.
    """
    if (task_machine is None) != (SHELL_CAPABILITY not in task.environment_requirements.capabilities):
        raise ValueError("A task machine is given exactly when the task has the shell capability")
    if (verifier_machine is None) != (grading_environment(task) is None):
        raise ValueError("A verifier machine is given exactly when the task's grader has an environment")
    task = _hosted_task(task, factories)
    grading = grading_environment(task)
    spec = TaskRolloutSpec(
        task=task,
        harness=TOOL_HARNESS,
        task_machine=(
            None if task_machine is None else machine_runtime(task_machine, task.environment_requirements, host)
        ),
        verifier_machine=(
            None if verifier_machine is None or grading is None else machine_runtime(verifier_machine, grading, host)
        ),
        limits=limits,
    )
    check(spec, factories)
    return spec


def machine_runtime(
    settings: MachineSettings, requirements: EnvironmentRequirements, host: MachineHost
) -> MachineRuntimeSpec:
    """The ``MachineRuntimeSpec`` of a machine with ``requirements`` on ``host``.

    Pass requirements already resolved for the available factories with ``host_requirements``.
    """
    return MachineRuntimeSpec(
        backend=machine_backend(requirements, host).value,
        network=settings.network,
        cpus=settings.limits.cpus,
        memory_mb=settings.limits.memory_mb,
        storage_mb=settings.limits.storage_mb,
        gpus=settings.limits.gpus,
        user=settings.user,
        startup_timeout=settings.startup_timeout,
        cleanup_timeout=settings.cleanup_timeout,
    )


def machine_backend(requirements: EnvironmentRequirements, host: MachineHost) -> Backend:
    """The host's container backend for an image, the local backend for Linux processes without one
    (a packages lock or a package spec), else ShellSim."""
    if requirements.docker_image is not None:
        return container_backend(host)
    if requirements.command_semantics == CommandSemantics.LINUX_PROCESS:
        return Backend.LOCAL
    return Backend.SHELLSIM


@runtime_checkable
class SystemDirectoriesFactory(Protocol):
    """A machine factory that says how its machines see the host's system directories, as the local one does."""

    system_directories: str


def host_requirements(
    requirements: EnvironmentRequirements, factories: Mapping[str, MachineFactory], writes: Sequence[str] = ()
) -> EnvironmentRequirements:
    """``requirements`` as a host with ``factories`` runs them.

    With a local factory, a machine on ``GRADER_BASE_IMAGE`` runs ``LOCAL_CPYTHON`` on the local
    backend, unless one of the absolute paths in ``writes`` (everything staged or written on the
    machine) falls under ``LOCAL_UNWRITABLE_ROOTS`` for the factory's ``system_directories``, in which
    case it keeps the image on the host's container backend, as it does without a local factory. A
    copy-on-write factory, the one Taskforge builds on Iris, leaves only ``/sys`` out, so a task shipping
    files to ``/usr`` runs locally; a factory that does not state
    its mode is taken as read-only, shellbox's default. A package machine (``requirements(packages=...)``)
    runs on the local backend only, its packages installed into a uv venv at machine start; on a
    host without a local factory, ``RolloutEngine.check`` rejects it as an unknown factory. Every
    other requirement is unchanged.
    """
    local = factories.get(Backend.LOCAL.value)
    if requirements.docker_image != GRADER_BASE_IMAGE or local is None:
        return requirements
    mode = local.system_directories if isinstance(local, SystemDirectoriesFactory) else LOCAL_READ_ONLY
    if any(_under(path, LOCAL_UNWRITABLE_ROOTS[mode]) for path in writes):
        return requirements
    return requirements.model_copy(update={"docker_image": None, "package_spec": LOCAL_CPYTHON})


def _under(path: str, roots: Sequence[str]) -> bool:
    parts = PurePosixPath(path).parts
    return any(parts[: len(PurePosixPath(root).parts)] == PurePosixPath(root).parts for root in roots)


def _agent_visible(task: TaskSpec) -> tuple[TaskResource, ...]:
    return (*task.resources.all, *task.resources.worker)


def _hosted_task(task: TaskSpec, factories: Mapping[str, MachineFactory]) -> TaskSpec:
    """``task`` with its task machine's and its grader's requirements as ``host_requirements`` gives them.

    A machine's writes are every path something is staged at or written to on it: the task machine gets
    the agent-visible files, its working directory and the output paths; the verifier also gets the
    private resources, the captured outputs and, for a script grader, its working directory, its artifacts and
    the extracted answer at its ``answer_path``.
    """
    requirements = task.environment_requirements
    visible_writes = [f"/{resource.path}" for resource in _agent_visible(task)]
    task_writes = [*visible_writes, *task.output_paths]
    if requirements.working_directory is not None:
        task_writes.append(requirements.working_directory)
    update: dict[str, object] = {"environment_requirements": host_requirements(requirements, factories, task_writes)}
    grading = grading_environment(task)
    if grading is not None:
        grader_writes = [
            *(f"{GRADER_RESOURCE_ROOT}/{resource.path}" for resource in task.resources.verifier),
            *visible_writes,
            *task.output_paths,
        ]
        if isinstance(task.grader, ScriptGrader):
            grader_writes += [task.grader.cwd, *(artifact.target for artifact in task.grader.artifacts)]
            if task.grader.answer_path is not None:
                grader_writes.append(task.grader.answer_path)
        update["grader"] = task.grader.model_copy(
            update={"environment": host_requirements(grading, factories, grader_writes)}
        )
    return task.model_copy(update=update)


def _reads_task_machine(grader: GraderPackage) -> bool:
    return isinstance(grader.grader, ScriptGrader) and bool(grader.grader.artifacts or grader.grader.collect)


def _conversation(instruction: str, system: str | None = None) -> ConversationInput:
    system_events = () if system is None else (TextMessage(role="system", content=system),)
    return ConversationInput(events=(*system_events, TextMessage(role="user", content=instruction)))
