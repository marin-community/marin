# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind authored tasks to Taskforge's machine factories and RolloutEngine limits."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import DEFAULT_SHELL_TOOL, ShellHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from rolloutengine.validation import task_machine_reason
from shellbox.machine import Backend, MachineFactory, NetworkPolicy, PackageSpec
from taskcompendium.decision_model import DecisionModelConnection, DecisionModelRequest
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    CommandSemantics,
    DecisionModelGrader,
    EnvironmentRequirements,
    Grader,
    ScriptGrader,
    TaskSpec,
    judge_spec,
)
from taskcompendium.runtime.shell import resolve_shell_tool
from taskcompendium.submission import require_submission_compatibility
from verifyit.grade import InvalidTask
from verifyit.judge_contract import validate_judge_spec
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import RUBRIC_ANSWERS, RUBRICS

from taskforge.llm.policy import ReasoningEffort
from taskforge.model.controls import Candidate
from taskforge.policy import GRADER_BASE_IMAGE, Policy
from taskforge.resources.sandbox.factories import (
    DEFAULT_WORKDIR,
    SANDBOX_SCHEDULING_TIMEOUT,
    grading_environment,
)

SHELL_CAPABILITY = "shell"
FIXED_JUDGE_SETTINGS = {"temperature": 0.0, "chat_template_kwargs": {}}
"""Values accepted for judge settings omitted from the author reference."""
JUDGE_RUBRICS = sorted(RUBRICS - {RUBRIC_ANSWERS})
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
            if fixed := [name for name, value in FIXED_JUDGE_SETTINGS.items() if getattr(judge, name) != value]:
                raise ValueError(
                    f"Leave judge {', '.join(fixed)} unset: the Taskforge judge samples greedily and takes no chat "
                    "template arguments"
                )
            if judge.reasoning_effort and judge.reasoning_effort not in ReasoningEffort:
                raise ValueError(f"Leave judge reasoning_effort unset or use one of {', '.join(ReasoningEffort)}")
            if judge.rubric == RUBRIC_ANSWERS:
                raise ValueError(
                    f"The Taskforge judge cannot grade the {RUBRIC_ANSWERS} rubric: it samples at the judge's default "
                    f"rather than greedily; use one of {JUDGE_RUBRICS}"
                )
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
    decision_model: DecisionModelConnection | None = None,
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
        decision_model=decision_model,
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
    decision_model: str = field(default="", metadata={"serialize": False})
    """``decision_model.model``, empty when disabled; replay keys carry it only where one is configured."""
    judge_wait: float = field(default=0.0, metadata={"serialize": False})
    """``provider.unavailable_wait`` plus a stall, within which a judge call succeeds or fails."""
    decision_model_wait: float = field(default=0.0, metadata={"serialize": False})
    """``decision_model.unavailable_wait``, after which a read raises its typed outage. Neither wait enters a replay
    key here: each reaches the spec's ``verifier_timeout`` only for the grader it bounds."""


def lowering_settings(policy: Policy) -> LoweringSettings:
    """Share machine selection and attempt budgets between author grading and validation."""
    limits = RolloutLimits(
        max_turns=None,
        max_total_tokens=None,
        model_turn_timeout=None,  # the model client owns a turn: its time limit, stalls and availability window
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
    judge, decision_model = policy.setting("judge.model", str), policy.setting("decision_model.model", str)
    judge_wait = policy.setting("provider.unavailable_wait", float) + policy.setting("provider.stall_timeout", float)
    wait = policy.setting("decision_model.unavailable_wait", float)
    return LoweringSettings(settings, settings, limits, judge, decision_model, judge_wait, wait)


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
    decision_model = None
    if isinstance(task.grader, DecisionModelGrader):
        if not settings.decision_model:
            raise InvalidTaskSpec(
                "Decision-model grading is disabled: set policy decision_model.model or use another grader"
            )
        decision_model = DecisionModelConnection(settings.decision_model, unbound_read)
    grader = grading_environment(task)
    if grader is not None and grader.docker_image not in (None, GRADER_BASE_IMAGE):
        raise InvalidTaskSpec(
            f"grader image must be {GRADER_BASE_IMAGE} or null for ShellSim; got {grader.docker_image}"
        )
    timeout = verifier_timeout(task.grader)
    if judge is not None:  # the judge call's own availability window ends first
        timeout = max(timeout, settings.judge_wait)
    if decision_model is not None:  # the read's typed outage, then the grade's finalisation
        timeout += settings.decision_model_wait
    limits = settings.limits.model_copy(update={"verifier_timeout": timeout})
    return lower(
        task,
        task_machine=settings.task_machine if SHELL_CAPABILITY in task.environment_requirements.capabilities else None,
        verifier_machine=settings.verifier_machine if grading_environment(task) is not None else None,
        limits=limits,
        factories=factories,
        judge_connection=connection,
        decision_model=decision_model,
    )


async def unbound_read(_request: DecisionModelRequest) -> float:
    raise RuntimeError("A decision-model read is bound at grade setup, by the installed decision model")


def check_candidates(task: TaskSpec, candidates: Sequence[Candidate]) -> None:
    """Reject authored files when lowering would provide no solver task machine."""
    if SHELL_CAPABILITY not in task.environment_requirements.capabilities and any(
        candidate.files for candidate in candidates
    ):
        raise ValueError("Candidate files require a task machine")
