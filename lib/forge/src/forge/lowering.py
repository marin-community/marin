# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, spec/draft.py.

"""Bind authored tasks to Forge's machine factories and RolloutEngine limits."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from rolloutengine.validation import InvalidTaskSpec, check_grader
from shellbox.machine import DEFAULT_WORKDIR, Backend, MachineFactory, NetworkPolicy, PackageSpec
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    Grader,
    ScriptGrader,
    TaskSpec,
    judge_spec,
)
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import RUBRIC_REFERENCE

from forge.model.controls import Candidate
from forge.model.proposal import Verification
from forge.policy import GRADER_BASE_IMAGE, Policy
from forge.resources.clef import CLEF_MODEL
from forge.resources.sandbox.factories import (
    SANDBOX_SCHEDULING_TIMEOUT,
    grading_environment,
)

SHELL_CAPABILITY = "shell"
TOOL_HARNESS = "tools"
"""The harness every Taskforge spec selects: a shell tool on the task machine, or a plain conversation without one."""


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
    when its grader declares an environment. Backend selection and software requirements are bound
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
        raise ValueError("A verifier machine is given exactly when the task's grader has an environment")
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
        update["grader"] = task.grader.model_copy(update={"environment": host_requirements(grading)})
    return task.model_copy(update=update)


@dataclass(frozen=True, slots=True)
class LoweringSettings:
    task_machine: MachineSettings
    verifier_machine: MachineSettings
    limits: RolloutLimits
    judge_model: str = ""


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
    return LoweringSettings(settings, settings, limits, policy.setting("judge.model", str))


def lower_with_settings(
    task: TaskSpec, settings: LoweringSettings, factories: Mapping[str, MachineFactory]
) -> TaskRolloutSpec:
    """Lower with shared budgets and only the machines the task requires."""
    check_grader(task.grader)
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
        if settings.judge_model == CLEF_MODEL and (
            grading_environment(task) is not None or judge.rubric != RUBRIC_REFERENCE or not judge.question.strip()
        ):
            raise InvalidTaskSpec(CLEF_GRADER)
        connection = JudgeConnection(
            "http://forge-judge.invalid/v1", "inference-effect", default_model=settings.judge_model
        )
    grader = grading_environment(task)
    if grader is not None and grader.docker_image not in (None, GRADER_BASE_IMAGE):
        raise InvalidTaskSpec(
            f"grader image must be {GRADER_BASE_IMAGE} or null for ShellSim; got {grader.docker_image}"
        )
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


CLEF_GRADER = (
    "The Clef decision model grades a host VerifyitGrader (no environment) with mode=judge, rubric=reference, "
    "the task question plus its rubric in question, and the gold answer in references"
)


def check_verification(task: TaskSpec, verification: Verification, settings: LoweringSettings) -> None:
    """Require the grader kind the proposal's verification names.

    ``decision_model`` needs a Clef-judged grader and fails closed (``ValueError``, a host refusal) when policy
    selects no Clef judge; ``answer_in_box`` and ``code`` need a deterministic grader. Other kinds are unchecked.

    Raises:
        InvalidTaskSpec: The authored grader does not match the verification.
        ValueError: The verification is ``decision_model`` and policy ``judge.model`` is not Clef.
    """
    judge = judge_spec(task.grader) is not None
    if verification is Verification.DECISION_MODEL:
        if settings.judge_model != CLEF_MODEL:
            raise ValueError(
                f"decision_model grading needs policy judge.model {CLEF_MODEL!r}, not {settings.judge_model!r}"
            )
        if not judge:
            raise InvalidTaskSpec(
                f"Verification is decision_model, so a script or deterministic grader is refused. {CLEF_GRADER}"
            )
    elif verification in (Verification.ANSWER_IN_BOX, Verification.CODE) and judge:
        raise InvalidTaskSpec(f"Verification is {verification}: submit a deterministic grader, not a judge")


def check_candidates(task: TaskSpec, candidates: Sequence[Candidate]) -> None:
    """Reject authored files when lowering would provide no solver task machine."""
    if SHELL_CAPABILITY not in task.environment_requirements.capabilities and any(
        candidate.files for candidate in candidates
    ):
        raise ValueError("Candidate files require a task machine")
