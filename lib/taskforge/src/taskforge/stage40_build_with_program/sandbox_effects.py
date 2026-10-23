# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The effects a build step may have: a prepared machine, a graded candidate, a calibrated grader, a published image.

Each function uses the host through ``RLEnvBuildServices``. The machine factories are wrapped
with ``infrastructure.host_checked_factories`` first, so a host fault raises
``BuildInfrastructureFailure`` and a fault of the program's own choices stays an ordinary error or a
``BuildFailure``.

``grade_candidate`` grades a candidate through ``RolloutEngine.run_task`` under the shell harness, the
path a validation control takes. The candidate's workspace files are written by one shell call each and
its reply ends the conversation (``spec.controls.candidate_turns``). The engine grades only a rollout
with a model turn. Scripted grading uses the messages trace and records no synthetic token ids.

A decision-model grader (``spec.decision``) is graded as validation grades it: the engine runs the candidate
with the grader swapped to ``NoGrader``, and the host's decision client (``RLEnvBuildServices.decision``)
grades the final conversation (``decision.run_decision_trial``). ``calibrate_decision_grader`` reads the
reference and its equivalent variants that way and sets the grader's bar, then reads the fooling answers
(``sdk.fooling_negatives``) under it and refuses a grader that credits one.

``build_machine`` prepares a machine through RolloutEngine's own ``prepare_machine``, so files, setup
commands, the execution user and the startup deadline behave as in a task attempt. It accepts
requirements and resources before a build has assembled a complete task.

A deadline that expires while a machine starts or a candidate is graded is the host's
(``InfrastructureCause.SANDBOX_DEADLINE``): a grader command past its own timeout comes back as a
result, so the deadline only expires when the sandbox is slow to start, transfer or answer. A grade
whose grader command timed out (``GradingFailure.TIMEOUT``) is graded once more on fresh machines,
because one slow exec cannot be told from a slow grader; only the second result counts.
"""

import re
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from rolloutengine.engine import RolloutEngine
from rolloutengine.machines import prepare_machine
from rolloutengine.models import Model, RolloutInterrupted, TaskRolloutSpec, TraceMode
from shellbox.machine import Machine
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import AnswerFormat, AnswerType, EnvironmentRequirements, Source, TaskResource

from taskforge.sandbox.factories import grading_environment
from taskforge.sandbox.image_builder import BuildFailed, ImageJobFailed
from taskforge.sandbox.images import BUILD_LIMITS, DockerBuild, check_build_limits
from taskforge.spec import draft
from taskforge.spec.controls import ScriptedModel, candidate_turns
from taskforge.spec.decision import (
    REFERENCE_CONTROL,
    DecisionClient,
    DecisionGrader,
    known_correct_check,
    run_decision_trial,
)
from taskforge.spec.draft import GRADING_MACHINE, MachineSettings, machine_runtime
from taskforge.stage40_build_with_program.context import BuildFailure, BuildInfrastructureFailure, RLEnvBuildContext
from taskforge.stage40_build_with_program.infrastructure import (
    InfrastructureCause,
    deadline_failure,
    host_checked_factories,
    infrastructure_failure,
)
from taskforge.stage40_build_with_program.outputs import GradedCandidate
from taskforge.stage40_build_with_program.sdk import (
    ANSWER_FORMAT,
    fooling_negatives,
    fooling_problem,
    graded_candidate,
)
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import SDK_VERSION

MACHINE_CLEANUP_TIMEOUT = 120.0
"""Seconds ``build_machine`` and ``grade_candidate`` wait for a machine to close."""
GRADE_CANDIDATE_SOURCE = "taskforge.grade_candidate"
"""``Source.dataset`` of the provisional task ``grade_candidate`` grades against."""
IMAGE_REPOSITORY_PREFIX = "taskforge-tasks"
"""Registry repository under which ``publish_image`` pushes a workpiece's task image."""
CALIBRATION_INSTRUCTION = "(instruction)"
"""The instruction of the provisional task ``calibrate_decision_grader`` reads its candidates against."""
KNOWN_CORRECT_ADVICE = (
    "Either the rubric rejects a correct answer (state the equivalent forms it must accept) or a variant is not "
    "equivalent to the reference (replace it)"
)
"""How to fix a grader whose known-correct candidates read below "partially matches"."""


@asynccontextmanager
async def build_machine(
    services: RLEnvBuildServices,
    requirements: EnvironmentRequirements,
    settings: MachineSettings,
    files: Sequence[TaskResource] = (),
) -> AsyncIterator[Machine]:
    """``async with build_machine(services, requirements, settings, files) as m``: a prepared machine.

    The machine starts as a task attempt's would, on the requirements ``spec.draft.host_requirements``
    gives for the host, with ``files`` installed relative to the root and the setup commands run. A
    step uses it to try fixtures or a grader program; ``llm.agent.shell_tool(m, ...)`` gives an agent
    loop a shell in it. Leaving the block closes it.

    Raises:
        BuildInfrastructureFailure: the host failed to create or drive the machine, or the machine did
            not start within ``settings.startup_timeout`` (``SANDBOX_DEADLINE``).
    """
    factories = host_checked_factories(services.factories, services.host)
    hosted = draft.host_requirements(requirements, factories)
    runtime = machine_runtime(settings, hosted, services.host)
    async with AsyncExitStack() as owned:
        try:
            prepared = await owned.enter_async_context(
                prepare_machine(
                    hosted, tuple(files), factories[runtime.backend], runtime, cleanup_timeout=MACHINE_CLEANUP_TIMEOUT
                )
            )
        except TimeoutError as error:
            raise BuildInfrastructureFailure(
                InfrastructureCause.SANDBOX_DEADLINE,
                f"machine start: the sandbox did not answer within the {settings.startup_timeout:.0f}-second "
                "startup deadline",
            ) from error
        yield prepared


async def grade_candidate(
    context: RLEnvBuildContext,
    requirements: EnvironmentRequirements,
    grader: GraderPackage | DecisionGrader,
    answer_type: AnswerType,
    answer_format: AnswerFormat,
    instruction: str,
    reply: str,
    *,
    files: Sequence[TaskResource] = (),
    workspace: Sequence[TaskResource] = (),
    output_paths: Sequence[str] = (),
    settings: MachineSettings = GRADING_MACHINE,
) -> GradeResult:
    """Grade one candidate, a final ``reply`` and the ``workspace`` files it leaves, as validation would.

    The grader receives a provisional task: ``instruction`` as its only message, the task's
    environment ``requirements`` with the agent-visible ``files``, ``grader`` with ``answer_type``
    and ``answer_format``, and ``output_paths``. Pass the task's own requirements: a task machine
    (``draft.requirements(...)``), or ``draft.NO_MACHINE`` for a task without one. The
    grading starts a task machine exactly when they have the shell capability, as validation's
    lowering does, so a step passes the ``requirements`` step's output whatever machine it declares.
    The candidate writes each ``workspace`` file with a shell call, paths relative to the machine
    root, then replies with ``reply`` in the form ``answer_format`` extracts. A grader with an
    environment runs in its own verifier machine. ``settings`` applies to both machines. No model is
    called, except the host's decision model for a ``DecisionGrader``: the candidate runs with the
    grader swapped to ``NoGrader`` and the host grades its final conversation, as a validation trial
    is graded.

    A grader step calls this on its reference answer, on the empty answer and on wrong answers it
    invents, and keeps every candidate in ``Grader.prototyped``.

    An empty ``reply`` comes back either as ``Outcome.SUBMISSION_FAILURE`` with reward 0.0, when
    TaskCompendium refuses the submission before any grader runs, or as ``Outcome.GRADED`` with the
    reward an empty candidate earns under the grader's empty-output policy, usually 0.0. A caller
    counts both as a rejection: the grader step of ``template.standard`` fails an empty answer only
    when it comes back ``GRADED`` above the grader's declared rejection ceiling.

    A grade whose grader command timed out (``GradingFailure.TIMEOUT``) is graded once more on fresh
    machines and the second grade is returned, so ``INFRA_ERROR`` with that failure means the grader
    was slow twice.

    Raises:
        BuildFailure: ``spec.draft.assemble`` or ``spec.draft.lower`` rejects the provisional task,
            including an ``answer_format`` that cannot carry ``answer_type`` to ``grader``, or
            ``workspace`` is given without a task machine.
        BuildInfrastructureFailure: the host failed to create or drive a grading machine, a machine
            start or the grading outlasted its deadline (``SANDBOX_DEADLINE``), or the grader is a
            ``DecisionGrader`` and the host has no decision client (``NO_DECISION_MODEL``).
    """
    return await grade_candidate_on_host(
        context.services,
        context.workpiece_id,
        requirements,
        grader,
        answer_type,
        answer_format,
        instruction,
        reply,
        files=files,
        workspace=workspace,
        output_paths=output_paths,
        settings=settings,
    )


async def grade_candidate_on_host(
    services: RLEnvBuildServices,
    workpiece_id: str,
    requirements: EnvironmentRequirements,
    grader: GraderPackage | DecisionGrader,
    answer_type: AnswerType,
    answer_format: AnswerFormat,
    instruction: str,
    reply: str,
    *,
    files: Sequence[TaskResource] = (),
    workspace: Sequence[TaskResource] = (),
    output_paths: Sequence[str] = (),
    settings: MachineSettings = GRADING_MACHINE,
) -> GradeResult:
    """Grade a candidate using the host resources, without a build model or sampling policy."""
    client = _decision_client(services) if isinstance(grader, DecisionGrader) else None
    try:
        task = draft.assemble(
            task_id=f"{workpiece_id}.grade_candidate",
            instruction=instruction,
            answer_type=answer_type,
            answer_format=answer_format,
            grader=grader,
            source=Source(
                dataset=GRADE_CANDIDATE_SOURCE,
                revision=SDK_VERSION,
                row=workpiece_id,
                importer_revision=SDK_VERSION,
            ),
            environment=requirements,
            files=files,
            output_paths=output_paths,
        )
    except ValueError as error:
        raise BuildFailure(f"grade_candidate: {error}") from error
    machine = draft.SHELL_CAPABILITY in task.environment_requirements.capabilities
    if workspace and not machine:
        raise BuildFailure("grade_candidate: workspace files need a task machine")
    turns = candidate_turns(reply, sorted(workspace, key=lambda resource: resource.path))
    factories = host_checked_factories(services.factories, services.host)
    try:
        spec = draft.lower(
            task,
            host=services.host,
            task_machine=settings if machine else None,
            verifier_machine=None if grading_environment(task) is None else settings,
            limits=draft.rollout_limits(
                max_turns=len(turns),
                model_turn_timeout=None,
                tool_turn_timeout=None,
                total_turn_timeout=None,
                attempt_timeout=None,
                verifier_timeout=draft.verifier_timeout(task.grader),
                cleanup_timeout=MACHINE_CLEANUP_TIMEOUT,
            ),
            factories=factories,
        )
    except (ValueError, NotImplementedError) as error:
        raise BuildFailure(f"grade_candidate: {error}") from error
    engine = draft.rollout_engine(factories, command_timeout=None)
    spec = spec.model_copy(update={"trace": TraceMode.MESSAGES})
    model = ScriptedModel(turns, context_assistant_turns=0)
    grade = await _graded(engine, spec, model, client)
    if grade.failure is GradingFailure.TIMEOUT:
        grade = await _graded(engine, spec, model, client)
    return grade


@dataclass(frozen=True)
class DecisionCalibration:
    """``calibrate_decision_grader``'s result: the grader, why it is refused (``None`` when it is not), the reads.

    Attributes:
        grader: The calibrated grader, or the grader as given when it is refused.
        problem: Why the known-correct candidates (``decision.known_correct_check``) or the fooling answers
            (``sdk.fooling_problem``) refuse it, or ``None``.
        candidates: Every candidate read, in order (the reference, the variants, then, once the bar is set, the
            empty answer and the fooling answers), for ``Grader.prototyped``.
    """

    grader: DecisionGrader
    problem: str | None
    candidates: tuple[GradedCandidate, ...]


async def calibrate_decision_grader(
    context: RLEnvBuildContext,
    grader: DecisionGrader,
    variants: Sequence[str],
    *,
    fluent_wrong_reply: str,
    hedged_reply: str,
) -> DecisionCalibration:
    """Read ``grader``'s reference and the equivalent ``variants`` as validation would, set the bar, and read the
    fooling answers under it.

    Each candidate is graded by ``grade_candidate`` on a provisional task without a machine, since the
    decision model reads the final reply only. ``decision.known_correct_check`` then refuses the grader or
    gives its acceptance level, which is stored with the reads in the returned grader's ``calibration``.
    Under that bar the empty answer is read, then the five fooling answers ``sdk.fooling_negatives`` builds
    from the reference, ``fluent_wrong_reply`` and ``hedged_reply``; their levels go in the grader's
    ``fooling``, and ``sdk.fooling_problem`` refuses the grader when one earns more than the rejection
    ceiling. ``run_build`` refuses a task whose decision grader lacks these reads, so a program ships the
    grader this returns.

    Raises:
        BuildFailure: ``fluent_wrong_reply`` or ``hedged_reply`` is blank.
        BuildInfrastructureFailure: the host has no decision client (``NO_DECISION_MODEL``).
    """
    if not fluent_wrong_reply.strip() or not hedged_reply.strip():
        raise BuildFailure("calibrate_decision_grader: fluent_wrong_reply and hedged_reply must be non-empty replies")
    replies = {REFERENCE_CONTROL: grader.reference, **{f"variant:{i}": v for i, v in enumerate(variants, 1)}}
    levels: dict[str, float] = {}
    candidates: list[GradedCandidate] = []
    for name, reply in replies.items():
        grade = await _read(context, grader, reply)
        candidates.append(graded_candidate(reply, (), grade))
        levels[name] = _level(grade)
    check = known_correct_check(levels)
    if check.problem is not None:
        problem = f"{check.problem}. {KNOWN_CORRECT_ADVICE}"
        return DecisionCalibration(grader, problem, tuple(candidates))
    calibrated = DecisionGrader.model_validate(
        {**grader.model_dump(), "acceptance_level": check.acceptance_level, "calibration": levels}
    )
    candidates.append(graded_candidate("", (), await _read(context, calibrated, "")))
    fooling: dict[str, float] = {}
    for negative in fooling_negatives(grader.reference, fluent_wrong_reply, hedged_reply):
        grade = await _read(context, calibrated, negative.reply)
        candidates.append(graded_candidate(negative.reply, (), grade))
        fooling[negative.kind.value] = _level(grade)
    calibrated = DecisionGrader.model_validate({**calibrated.model_dump(), "fooling": fooling})
    problem = fooling_problem(calibrated)
    if problem is not None:
        return DecisionCalibration(grader, problem, tuple(candidates))
    return DecisionCalibration(calibrated, None, tuple(candidates))


async def _read(context: RLEnvBuildContext, grader: DecisionGrader, reply: str) -> GradeResult:
    return await grade_candidate(
        context, draft.NO_MACHINE, grader, AnswerType.TEXT, ANSWER_FORMAT, CALIBRATION_INSTRUCTION, reply
    )


def _level(grade: GradeResult) -> float:
    """The level a decision grade read; a candidate it could not read (an empty reply) reads "missing"."""
    if grade.status is not Outcome.GRADED or grade.detail is None:
        return 0.0
    return float(grade.detail["level"])


def _decision_client(services: RLEnvBuildServices) -> DecisionClient:
    client = services.decision
    if client is None:
        raise BuildInfrastructureFailure(
            InfrastructureCause.NO_DECISION_MODEL, "this host has no decision model to grade a decision-model grader"
        )
    return client


async def _graded(
    engine: RolloutEngine, spec: TaskRolloutSpec, model: Model, decision: DecisionClient | None
) -> GradeResult:
    """One rollout of ``spec`` on fresh machines, host-graded for a decision grader; a host failure or a
    sandbox deadline raises as infrastructure."""
    try:
        if decision is None:
            result = await engine.run_task(spec, model)
        else:
            result = await run_decision_trial(engine, spec, model, decision)
    except RolloutInterrupted as error:
        failure = infrastructure_failure(error) or deadline_failure(error)
        if failure is None:
            raise
        raise failure from error
    return result.grade


def image_repository(workpiece_id: str) -> str:
    """The registry repository ``publish_image`` pushes ``workpiece_id``'s images under."""
    return f"{IMAGE_REPOSITORY_PREFIX}/{re.sub(r'[^a-z0-9]+', '-', workpiece_id.lower()).strip('-')}"


async def publish_image(context: RLEnvBuildContext, build: DockerBuild) -> str:
    """Build ``build`` for linux/amd64, push it under the workpiece's repository, and return the pinned reference.

    A step passes the result to ``spec.draft.requirements(image=...)`` and returns it, so the
    reference is memoized with the step's output.

    Raises:
        BuildFailure: the build context exceeds ``BUILD_LIMITS``, or the image failed to build from it
            (``image_builder.BuildFailed``).
        BuildInfrastructureFailure: this host has no image builder (``NO_IMAGE_BUILDER``), or a build or
            push job failed for a host reason (``IMAGE_JOB_FAILED``, from ``image_builder.ImageJobFailed``).
    """
    return await publish_image_on_host(context.services, context.workpiece_id, build)


async def publish_image_on_host(services: RLEnvBuildServices, workpiece_id: str, build: DockerBuild) -> str:
    """Publish the workpiece's image using the host's image builder."""
    images = services.images
    if images is None:
        raise BuildInfrastructureFailure(InfrastructureCause.NO_IMAGE_BUILDER, "this host cannot publish images")
    try:
        check_build_limits(build, BUILD_LIMITS)
    except ValueError as error:
        raise BuildFailure(f"publish_image: {error}") from error
    try:
        return await images.publish(build, image_repository(workpiece_id))
    except BuildFailed as error:
        raise BuildFailure(f"publish_image: {error}") from error
    except ImageJobFailed as error:
        raise BuildInfrastructureFailure(InfrastructureCause.IMAGE_JOB_FAILED, f"publish_image: {error}") from error
