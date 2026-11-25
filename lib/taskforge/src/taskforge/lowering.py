# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, spec/draft.py.

"""Bind authored tasks to Taskforge's machine factories and RolloutEngine limits."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import DEFAULT_SHELL_TOOL, ShellHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from rolloutengine.validation import task_machine_reason
from shellbox.machine import Backend, MachineFactory, NetworkPolicy, PackageSpec
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    CommandSemantics,
    EnvironmentRequirements,
    Grader,
    GraderMachine,
    ScriptGrader,
    TaskSpec,
    VerifyitGrader,
    judge_spec,
)
from taskcompendium.runtime.shell import resolve_shell_tool
from taskcompendium.submission import require_submission_compatibility
from verifyit.grade import InvalidTask
from verifyit.judge_contract import validate_judge_spec
from verifyit.modes.grade_judge import JudgeConnection

from taskforge.model.controls import Candidate
from taskforge.policy import GRADER_BASE_IMAGE, SOURCE_IMAGES, SOURCE_TASK_ID, SOURCE_TASK_SOURCE, Policy
from taskforge.resources.sandbox.factories import (
    DEFAULT_WORKDIR,
    SANDBOX_SCHEDULING_TIMEOUT,
    grading_environment,
)

SHELL_CAPABILITY = "shell"
TOOL_HARNESS = "tools"
"""The harness every Taskforge spec selects: a shell tool on the task machine, or a plain conversation without one."""


class InvalidTaskSpec(ValueError):
    """Authored task requirements are inconsistent or unsupported; return them to the author."""


def check_authored(task: TaskSpec) -> None:
    """Raise ``InvalidTaskSpec`` for the authored defects RolloutEngine reports only as ``ValueError``.

    Runs before machine admission, so the defect returns to the author instead of holding the stage.
    """
    try:
        judge = judge_spec(task.grader)
        if judge is not None:
            validate_judge_spec(judge)
        resolve_shell_tool(task, DEFAULT_SHELL_TOOL)
        if task.answer_type in CONVERSATION_ANSWERS:
            require_submission_compatibility(task)
    except (ValueError, InvalidTask) as error:
        raise InvalidTaskSpec(str(error)) from error
    if SHELL_CAPABILITY not in task.environment_requirements.capabilities and (reason := task_machine_reason(task)):
        raise InvalidTaskSpec(reason)


@dataclass(frozen=True, slots=True)
class MachineLimits:
    """A machine's CPU, memory, storage and GPU limits. ``None`` leaves the limit to the machine factory."""

    memory_mb: int | None = None
    cpus: int | None = None
    storage_mb: int | None = None
    gpus: int = 0


@dataclass(frozen=True, slots=True)
class MachineSettings:
    """A ``MachineRuntimeSpec`` without its backend; ``lower`` picks the backend for this host."""

    network: NetworkPolicy
    limits: MachineLimits
    user: str | None
    startup_timeout: float | None
    cleanup_timeout: float | None


def requirements(
    *,
    image: str | None,
    packages: Sequence[str] = (),
    setup: Sequence[str] = (),
    workdir: str = DEFAULT_WORKDIR,
    env: Mapping[str, str] | None = None,
) -> EnvironmentRequirements:
    """Declare an author, adversary or task environment."""
    if image is not None and packages:
        raise ValueError("A task machine names an image or packages, not both")
    if packages:
        semantics, package_spec = CommandSemantics.LINUX_PROCESS, PackageSpec(pypi_packages=tuple(packages))
    elif image is None:
        semantics, package_spec = CommandSemantics.SHELL_SIMULATOR, PackageSpec(pypi_packages=())
    else:
        semantics, package_spec = CommandSemantics.LINUX_PROCESS, None
    return EnvironmentRequirements(
        capabilities=(SHELL_CAPABILITY, "filesystem"),
        command_semantics=semantics,
        docker_image=image,
        package_spec=package_spec,
        working_directory=workdir,
        setup_commands=tuple(setup),
        environment_variables=dict(env or {}),
    )


IN_PROCESS_GRADER_TIMEOUT = 300.0
"""Deadline for graders without a declared command timeout; script graders own their timeout."""
MACHINE_PREPARATION_TIMEOUT = 300.0
GRADING_MACHINE = MachineSettings(
    NetworkPolicy.DENY, MachineLimits(), None, SANDBOX_SCHEDULING_TIMEOUT + MACHINE_PREPARATION_TIMEOUT, None
)
"""Machine settings allowing scheduling plus resource installation and setup."""


def verifier_timeout(grader: Grader) -> float:
    """Allow the verifier's scheduling budget and its own command timeout before ending grading."""
    return SANDBOX_SCHEDULING_TIMEOUT + grader.timeout if isinstance(grader, ScriptGrader) else IN_PROCESS_GRADER_TIMEOUT


def rollout_engine(factories: Mapping[str, MachineFactory], *, command_timeout: float | None) -> RolloutEngine:
    """A ``RolloutEngine`` on ``factories`` with RolloutEngine's shell harness registered as ``TOOL_HARNESS``.

    ``command_timeout`` bounds each shell command of a turn; ``None`` leaves it to the turn deadlines.
    ``RolloutEngine.check`` requires it to be shorter than a spec's ``tool_turn_timeout`` when both are set.
    """
    return RolloutEngine(factories, {TOOL_HARNESS: ShellHarness(command_timeout=command_timeout)})


def lower(
    task: TaskSpec,
    *,
    task_machine: MachineSettings | None,
    verifier_machine: MachineSettings | None,
    limits: RolloutLimits,
    factories: Mapping[str, MachineFactory],
    judge_connection: JudgeConnection | None = None,
) -> TaskRolloutSpec:
    """The ``TaskRolloutSpec`` that runs ``task`` on ``factories`` under ``TOOL_HARNESS``.

    Supply a task machine exactly when the task offers shell tools, and a verifier machine exactly
    when its grader uses a separate machine. Backend selection and software requirements are bound
    to this host. Validation checks factory support without starting a machine. The spec keeps the
    engine's exact-token trace; callers can select ``TraceMode.MESSAGES`` for text-only evidence.

    Raises:
        ValueError: A machine is given where the task has none or missing where it has one, or
            RolloutEngine rejects the lowered spec (``check``).
        NotImplementedError: RolloutEngine cannot run the lowered spec.
    """
    if (task_machine is None) != (SHELL_CAPABILITY not in task.environment_requirements.capabilities):
        raise ValueError("A task machine is given exactly when the task has the shell capability")
    if (verifier_machine is None) != (grading_environment(task) is None):
        raise ValueError("A verifier machine is given exactly when the task's grader uses a separate machine")
    task = _hosted_task(task)
    grading = grading_environment(task)
    spec = TaskRolloutSpec(
        task=task,
        harness=TOOL_HARNESS,
        judge_connection=judge_connection,
        task_machine=(None if task_machine is None else machine_runtime(task_machine, task.environment_requirements)),
        verifier_machine=(
            None if verifier_machine is None or grading is None else machine_runtime(verifier_machine, grading)
        ),
        limits=limits,
    )
    rollout_engine(factories, command_timeout=None).check(spec)
    return spec


def machine_runtime(settings: MachineSettings, requirements: EnvironmentRequirements) -> MachineRuntimeSpec:
    """The ``MachineRuntimeSpec`` of a machine with ``requirements``.

    Pass requirements already resolved for the available factories with ``host_requirements``.
    """
    backend = machine_backend(requirements)
    return MachineRuntimeSpec(
        backend=backend.value,
        network=settings.network,
        cpus=None if backend is Backend.SHELLSIM else settings.limits.cpus,
        memory_mb=settings.limits.memory_mb,
        storage_mb=settings.limits.storage_mb,
        gpus=settings.limits.gpus,
        user=settings.user,
        startup_timeout=settings.startup_timeout,
        cleanup_timeout=settings.cleanup_timeout,
    )


def machine_backend(requirements: EnvironmentRequirements) -> Backend:
    """Choose Iris for images and ShellSim for simulated commands."""
    if requirements.docker_image is not None:
        return Backend.GVISOR
    if requirements.command_semantics == CommandSemantics.LINUX_PROCESS:
        raise InvalidTaskSpec("Linux process requirements need a container image on Iris")
    return Backend.SHELLSIM


def host_requirements(requirements: EnvironmentRequirements) -> EnvironmentRequirements:
    """Normalize the working directory without changing the requested software."""
    if requirements.command_semantics is not None and not requirements.working_directory:
        return requirements.model_copy(update={"working_directory": DEFAULT_WORKDIR})
    return requirements


def _hosted_task(task: TaskSpec) -> TaskSpec:
    """Normalize task and grader working directories."""
    update: dict[str, object] = {"environment_requirements": host_requirements(task.environment_requirements)}
    grading = grading_environment(task)
    if grading is not None:
        grader = task.grader
        assert isinstance(grader, ScriptGrader | VerifyitGrader) and isinstance(grader.placement, GraderMachine)
        placement = grader.placement.model_copy(update={"environment": host_requirements(grading)})
        update["grader"] = grader.model_copy(update={"placement": placement})
    return task.model_copy(update=update)


@dataclass(frozen=True, slots=True)
class LoweringSettings:
    """Machine settings and budgets shared by every lowering of a run.

    Attributes:
        source_images: The container images a task from another source brings, from policy
            ``sandbox.source_images``, which the driver sets from the source task. Its task and grader
            machines may use them besides the pinned image and ShellSim; empty for generated tasks.
        source_identity: A source task's ``id`` and its ``source`` as JSON, from policy ``source.task_id``
            and ``source.task_source``, which the driver sets; a revision must keep both, since they join
            it back to its source. ``None`` for generated tasks.
    """

    task_machine: MachineSettings
    verifier_machine: MachineSettings
    limits: RolloutLimits
    judge_model: str = ""
    source_images: frozenset[str] = frozenset()
    source_identity: tuple[str, str] | None = None


def lowering_settings(policy: Policy) -> LoweringSettings:
    """Share machine selection and attempt budgets between author grading and validation."""
    limits = RolloutLimits(
        max_turns=None,
        max_total_tokens=None,
        model_turn_timeout=policy.setting("solver.turn_timeout", float),
        tool_turn_timeout=policy.setting("solver.tool_timeout", float),
        total_turn_timeout=None,
        attempt_timeout=policy.setting("solver.attempt_timeout", float),
        verifier_timeout=None,
        cleanup_timeout=policy.setting("solver.cleanup_timeout", float),
    )
    memory_mb = policy.setting("sandbox.memory_mb", int)
    if memory_mb <= 0:
        raise ValueError("sandbox.memory_mb must be positive")
    settings = replace(GRADING_MACHINE, limits=MachineLimits(memory_mb=memory_mb, cpus=1))
    task_id = policy.setting(SOURCE_TASK_ID, str)
    return LoweringSettings(
        settings,
        settings,
        limits,
        policy.setting("judge.model", str),
        frozenset(policy.setting(SOURCE_IMAGES, tuple)),
        (task_id, policy.setting(SOURCE_TASK_SOURCE, str)) if task_id else None,
    )


def check_identity(task: TaskSpec, settings: LoweringSettings) -> None:
    """Refuse a revision of a source task that changes its ``id`` or ``source``."""
    if settings.source_identity is None:
        return
    task_id, source = settings.source_identity
    if (task.id, task.source.model_dump_json()) != (task_id, source):
        raise InvalidTaskSpec(
            f"A task from another source keeps its id {task_id!r} and its source {source}; "
            f"this task has id {task.id!r} and source {task.source.model_dump_json()}"
        )


def check_images(task: TaskSpec, settings: LoweringSettings) -> None:
    """Refuse a container image the run does not admit.

    A task from another source may use only its source's own images, the pinned image included only when
    the source used it. A generated task's grader may use the pinned image; its own image is left to the
    machine factory.
    """
    admitted = tuple(sorted(settings.source_images)) or (GRADER_BASE_IMAGE,)
    grader = grading_environment(task)
    if grader is not None and grader.docker_image not in (None, *admitted):
        raise InvalidTaskSpec(
            f"grader image must be {' or '.join(admitted)}, or null for ShellSim; got {grader.docker_image}"
        )
    image = task.environment_requirements.docker_image
    if settings.source_images and image not in (None, *admitted):
        raise InvalidTaskSpec(f"task image must be {' or '.join(admitted)}, or null for ShellSim; got {image}")


def lower_with_settings(
    task: TaskSpec, settings: LoweringSettings, factories: Mapping[str, MachineFactory]
) -> TaskRolloutSpec:
    """Lower with shared budgets and only the machines the task requires."""
    check_authored(task)
    judge = judge_spec(task.grader)
    connection = None
    if judge is not None:
        if isinstance(task.grader, ScriptGrader):
            raise InvalidTaskSpec("Script judge graders are not supported: use a host VerifyitGrader with mode=judge")
        if not settings.judge_model:
            raise InvalidTaskSpec("Judge grading is disabled: set policy judge.model or submit a deterministic grader")
        if judge.model and judge.model != settings.judge_model:
            raise InvalidTaskSpec(
                f"Judge model must be empty or {settings.judge_model!r}, as selected by policy judge.model"
            )
        connection = JudgeConnection(
            "http://forge-judge.invalid/v1", "inference-effect", default_model=settings.judge_model
        )
    check_identity(task, settings)
    check_images(task, settings)
    timeout = verifier_timeout(task.grader)
    if judge is not None:
        timeout = max(timeout, settings.limits.model_turn_timeout or 0)
    limits = settings.limits.model_copy(update={"verifier_timeout": timeout})
    return lower(
        task,
        task_machine=settings.task_machine if SHELL_CAPABILITY in task.environment_requirements.capabilities else None,
        verifier_machine=settings.verifier_machine if grading_environment(task) is not None else None,
        limits=limits,
        factories=factories,
        judge_connection=connection,
    )


def check_candidates(task: TaskSpec, candidates: Sequence[Candidate]) -> None:
    """Reject authored files when lowering would provide no solver task machine."""
    if SHELL_CAPABILITY not in task.environment_requirements.capabilities and any(
        candidate.files for candidate in candidates
    ):
        raise ValueError("Candidate files require a task machine")
