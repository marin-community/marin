# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Source: Taskforge integration 9807c5cfb3, spec/draft.py.

"""Bind authored tasks to Forge's machine factories and RolloutEngine limits."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from rolloutengine.validation import InvalidTaskSpec, check_grader
from shellbox.machine import DEFAULT_WORKDIR, Backend, MachineFactory, NetworkPolicy, PackageSpec
from taskcompendium.grading import DecisionModelConnection
from taskcompendium.models import (
    CommandSemantics,
    DecisionModelGrader,
    EnvironmentRequirements,
    Grader,
    ScriptGrader,
    TaskSpec,
    VerifyitGrader,
    judge_spec,
)
from verifyit.modes.grade_judge import JudgeConnection

from forge.model.controls import Candidate
from forge.model.proposal import Verification
from forge.policy import GRADER_BASE_IMAGE, Policy
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


LOCAL_SANDBOX = os.environ.get("FORGE_LOCAL_SANDBOX") == "1"
"""THROWAWAY (2026-11-09): run grader-base Linux machines under the driver's own bwrap instead of Iris pods."""
LOCAL_CPYTHON = PackageSpec(pypi_packages=())

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
        # THROWAWAY: local (bwrap) machines run without per-machine cgroups in the Iris pod, so no cpu/memory limit.
        cpus=None if backend in (Backend.SHELLSIM, Backend.LOCAL) else settings.limits.cpus,
        memory_mb=None if backend is Backend.LOCAL else settings.limits.memory_mb,
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
        if LOCAL_SANDBOX and requirements.package_spec is not None:
            return Backend.LOCAL
        raise InvalidTaskSpec("Linux process requirements need a container image on Iris")
    return Backend.SHELLSIM


def host_requirements(requirements: EnvironmentRequirements) -> EnvironmentRequirements:
    """Normalize the working directory without changing the requested software."""
    if LOCAL_SANDBOX and requirements.docker_image == GRADER_BASE_IMAGE:
        requirements = requirements.model_copy(
            update={"docker_image": None, "package_spec": requirements.package_spec or LOCAL_CPYTHON}
        )
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
    decision_model: str = ""
    """The decision model behind decision_model graders (``decision_model.model``); empty disables them."""


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
    return LoweringSettings(
        settings, settings, limits, policy.setting("judge.model", str), policy.setting("decision_model.model", str)
    )


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
        connection = JudgeConnection(
            "http://forge-judge.invalid/v1", "inference-effect", default_model=settings.judge_model
        )
    decision_model = None
    if isinstance(task.grader, DecisionModelGrader):
        if not settings.decision_model:
            raise ValueError("Decision-model grading is disabled: policy decision_model.model is empty")
        decision_model = UNBOUND_DECISION_MODEL
    grader = grading_environment(task)
    if grader is not None and grader.docker_image not in (None, GRADER_BASE_IMAGE):
        raise InvalidTaskSpec(
            f"grader image must be {GRADER_BASE_IMAGE} or null for ShellSim; got {grader.docker_image}"
        )
    timeout = verifier_timeout(task.grader)
    if judge is not None or decision_model is not None:
        timeout = max(timeout, settings.limits.model_turn_timeout or 0)
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


ANSWER_IN_BOX_MODES = frozenset({"exact", "numeric", "math", "mcq"})
"""VerifyIt's generic final-answer matches: exact, numeric, math and choice (``mcq``)."""


def check_verification(task: TaskSpec, verification: Verification, settings: LoweringSettings) -> None:
    """Require the grader kind the proposal's verification names.

    ``answer_in_box`` is a host VerifyIt exact, numeric, math or choice match. ``code`` is a deterministic
    script or VerifyIt check. ``decision_model`` is a ``DecisionModelGrader`` and fails closed (``ValueError``,
    a host refusal) when policy names no decision model. None of them is an LLM judge; other kinds are unchecked.

    Raises:
        InvalidTaskSpec: The authored grader does not match the verification.
        ValueError: The verification is ``decision_model`` and policy ``decision_model.model`` is empty.
    """
    grader = task.grader
    decision = isinstance(grader, DecisionModelGrader)
    if verification is Verification.DECISION_MODEL:
        if not settings.decision_model:
            raise ValueError("decision_model proposals need policy decision_model.model; it is empty")
        if not decision:
            raise InvalidTaskSpec(
                'Verification is decision_model: the grader must be {"kind": "decision_model", ...} with deliverable, '
                "rubric, reference and cut; script, VerifyIt and LLM-judge graders are refused"
            )
    elif verification is Verification.ANSWER_IN_BOX:
        if not (
            isinstance(grader, VerifyitGrader) and grader.environment is None and grader.mode in ANSWER_IN_BOX_MODES
        ):
            raise InvalidTaskSpec(
                "Verification is answer_in_box: the grader must be a host VerifyitGrader (no environment) with mode "
                "exact, numeric, math or mcq; script, code, decision-model and LLM-judge graders are refused"
            )
    elif verification is Verification.CODE and (decision or judge_spec(grader) is not None):
        raise InvalidTaskSpec(
            "Verification is code: the grader must be a deterministic script or VerifyIt check; "
            "decision-model and LLM-judge graders are refused"
        )


async def _unbound_decision_model(_grader: DecisionModelGrader, _answer: str) -> float:
    raise RuntimeError("The decision model is bound by forge.resources.decision_model.DecisionModelEngine")


UNBOUND_DECISION_MODEL = DecisionModelConnection(_unbound_decision_model)
"""Lets RolloutEngine check a decision_model spec; ``DecisionModelEngine`` binds the real one per run."""


def check_candidates(task: TaskSpec, candidates: Sequence[Candidate]) -> None:
    """Reject authored files when lowering would provide no solver task machine."""
    if SHELL_CAPABILITY not in task.environment_requirements.capabilities and any(
        candidate.files for candidate in candidates
    ):
        raise ValueError("Candidate files require a task machine")
