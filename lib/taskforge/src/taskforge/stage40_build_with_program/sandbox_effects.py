# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The sandbox effects a build step may have: a prepared machine, a graded candidate, a published image.

Each function uses the host through ``RLEnvBuildServices``. The machine factories are wrapped
with ``infrastructure.host_checked_factories`` first, so a host fault raises
``BuildInfrastructureFailure`` and a fault of the program's own choices stays an ordinary error or a
``BuildFailure``.

``grade_candidate`` grades a candidate through ``RolloutEngine.run``, the path a validation
control takes. The candidate's workspace files are written by one shell call each and its reply
ends the conversation (``spec.controls.candidate_turns``). The engine requires token evidence for
every turn, and grading reads none, so the model serving those turns makes up ids that extend each
request's prefix by one token. A registered task session would still need such a model: the engine
builds the task's ``TaskGrader`` inside ``run``, hands it to the session through
``TaskSessionContext``, and requests at least one model turn before the session grades.

``build_machine`` prepares a machine through RolloutEngine's own ``_prepare_machine``, so files, setup
commands, the execution user and the startup deadline behave as in a task attempt. RolloutEngine
exposes no public form of it.

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
from typing import Any

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutInterrupted
from rolloutengine.engine import RolloutEngine
from rolloutengine.machines import _prepare_machine
from rolloutengine.spec import LoweredTaskSpec
from shellbox.machine import Machine
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult, GradingFailure
from taskcompendium.models import AnswerFormat, AnswerType, EnvironmentRequirements, Source, TaskResource

from taskforge.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT, grading_environment
from taskforge.sandbox.images import BUILD_LIMITS, DockerBuild, check_build_limits
from taskforge.spec import draft
from taskforge.spec.controls import candidate_turns
from taskforge.spec.draft import MachineSettings, machine_runtime
from taskforge.stage40_build_with_program.context import BuildFailure, BuildInfrastructureFailure, RLEnvBuildContext
from taskforge.stage40_build_with_program.infrastructure import (
    InfrastructureCause,
    deadline_failure,
    host_checked_factories,
    infrastructure_failure,
)
from taskforge.stage40_build_with_program.sdk import GRADER_TIMEOUT
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import SDK_VERSION

MACHINE_CLEANUP_TIMEOUT = 120.0
"""Seconds ``build_machine`` and ``grade_candidate`` wait for a machine to close."""
GRADE_CANDIDATE_PREPARATION_TIMEOUT = 300.0
"""Seconds a grading machine's start allows past the factory's scheduling budget, for its files and setup commands."""
GRADE_CANDIDATE_MACHINE = draft.machine(startup_timeout=SANDBOX_SCHEDULING_TIMEOUT + GRADE_CANDIDATE_PREPARATION_TIMEOUT)
"""The task and verifier machine settings ``grade_candidate`` uses unless the step passes its own.

The startup deadline outlasts ``SANDBOX_SCHEDULING_TIMEOUT``, as the verifier deadline does, so a
sandbox that never starts surfaces as the factory's typed ``SCHEDULING_TIMEOUT`` rather than as an
untyped startup deadline; the margin is the machine's preparation once it is scheduled.
"""
GRADE_CANDIDATE_VERIFIER_TIMEOUT = SANDBOX_SCHEDULING_TIMEOUT + GRADER_TIMEOUT
"""Seconds ``grade_candidate`` allows one grading: the verifier machine's scheduling budget, then the grader's own.

The sum keeps the deadline behind the factory's scheduling timeout, so a sandbox that never starts
surfaces as the factory's ``SCHEDULING_TIMEOUT`` host failure rather than as this deadline.
"""
GRADE_CANDIDATE_SOURCE = "taskforge.grade_candidate"
"""``Source.dataset`` of the provisional task ``grade_candidate`` grades against."""
IMAGE_REPOSITORY_PREFIX = "taskforge-tasks"
"""Registry repository under which ``publish_image`` pushes an item's task image."""


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
            prepared = await _prepare_machine(
                hosted, runtime, tuple(files), factories, _Cleanup(MACHINE_CLEANUP_TIMEOUT), owned
            )
        except TimeoutError as error:
            raise BuildInfrastructureFailure(
                InfrastructureCause.SANDBOX_DEADLINE,
                f"machine start: the sandbox did not answer within the {settings.startup_timeout:.0f}-second "
                "startup deadline",
            ) from error
        assert prepared is not None, "a machine runtime always prepares a machine"
        yield prepared


@dataclass(frozen=True)
class _CandidateModel:
    """Serves a candidate's assistant turns in order, with made-up token ids that extend each prefix."""

    turns: tuple[dict[str, Any], ...]

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        message = self.turns[sum(entry["role"] == "assistant" for entry in request.messages)]
        prompt = (*request.prefix_token_ids, len(request.prefix_token_ids))
        stop = "tool_calls" if "tool_calls" in message else "stop"
        return ModelTurn(message, prompt, (len(prompt),), None, stop, text=message["content"])


async def grade_candidate(
    context: RLEnvBuildContext,
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
    settings: MachineSettings = GRADE_CANDIDATE_MACHINE,
) -> GradeResult:
    """Grade one candidate, a final ``reply`` and the ``workspace`` files it leaves, as validation would.

    The grader receives a provisional task: ``instruction`` as its only message, the task machine of
    ``requirements`` with the agent-visible ``files`` (no machine when ``requirements`` is
    ``None``), ``grader`` with ``answer_type`` and ``answer_format``, and ``output_paths``. The
    candidate writes each ``workspace`` file with a shell call, paths relative to the machine root,
    then replies with ``reply`` in the form ``answer_format`` extracts. A grader with an environment
    runs in its own verifier machine. ``settings`` applies to both machines. No model is called.

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
        BuildInfrastructureFailure: the host failed to create or drive a grading machine, or a machine
            start or the grading outlasted its deadline (``SANDBOX_DEADLINE``).
    """
    if workspace and requirements is None:
        raise BuildFailure("grade_candidate: workspace files need a task machine")
    try:
        task = draft.assemble(
            task_id=f"{context.workpiece_id}.grade_candidate",
            instruction=instruction,
            answer_type=answer_type,
            answer_format=answer_format,
            grader=grader,
            source=Source(
                dataset=GRADE_CANDIDATE_SOURCE, revision=SDK_VERSION, row=context.workpiece_id, importer_revision=SDK_VERSION
            ),
            environment=requirements,
            files=files,
            output_paths=output_paths,
        )
    except ValueError as error:
        raise BuildFailure(f"grade_candidate: {error}") from error
    turns = candidate_turns(reply, sorted(workspace, key=lambda resource: resource.path))
    factories = host_checked_factories(context.services.factories, context.services.host)
    try:
        lowered = draft.lower(
            task,
            host=context.services.host,
            task_machine=None if requirements is None else settings,
            verifier_machine=None if grading_environment(task) is None else settings,
            session=draft.session(
                max_turns=len(turns),
                model_turn_timeout=None,
                command_timeout=None,
                tool_turn_timeout=None,
                total_turn_timeout=None,
                attempt_timeout=None,
                verifier_timeout=GRADE_CANDIDATE_VERIFIER_TIMEOUT,
                cleanup_timeout=MACHINE_CLEANUP_TIMEOUT,
            ),
            factories=factories,
        )
    except (ValueError, NotImplementedError) as error:
        raise BuildFailure(f"grade_candidate: {error}") from error
    engine = RolloutEngine(_CandidateModel(turns), factories)
    grade = await _graded(engine, lowered)
    if grade.failure is GradingFailure.TIMEOUT:
        grade = await _graded(engine, lowered)
    return grade


async def _graded(engine: RolloutEngine, lowered: LoweredTaskSpec) -> GradeResult:
    """One rollout of ``lowered`` on fresh machines; a host failure or a sandbox deadline raises as infrastructure."""
    try:
        rollout = await engine.run(lowered)
    except RolloutInterrupted as error:
        failure = infrastructure_failure(error) or deadline_failure(error)
        if failure is None:
            raise
        raise failure from error
    return rollout.grade


def image_repository(workpiece_id: str) -> str:
    """The registry repository ``publish_image`` pushes ``workpiece_id``'s images under."""
    return f"{IMAGE_REPOSITORY_PREFIX}/{re.sub(r'[^a-z0-9]+', '-', workpiece_id.lower()).strip('-')}"


async def publish_image(context: RLEnvBuildContext, build: DockerBuild) -> str:
    """Build ``build`` for linux/amd64, push it under the item's repository, and return the pinned reference.

    A step passes the result to ``spec.draft.requirements(image=...)`` and returns it, so the
    reference is memoized with the step's output.

    Raises:
        BuildFailure: the build context exceeds ``BUILD_LIMITS``.
        BuildInfrastructureFailure: this host has no image builder.
    """
    images = context.services.images
    if images is None:
        raise BuildInfrastructureFailure(InfrastructureCause.NO_IMAGE_BUILDER, "this host cannot publish images")
    try:
        check_build_limits(build, BUILD_LIMITS)
    except ValueError as error:
        raise BuildFailure(f"publish_image: {error}") from error
    return await images.publish(build, image_repository(context.workpiece_id))
