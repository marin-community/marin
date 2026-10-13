# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The sandbox effects a build step may have: a prepared machine, a graded candidate, a published image.

Each function uses the host through ``RLEnvBuildServices``. The machine factories are wrapped
with ``infrastructure.host_checked_factories`` first, so a host fault raises
``BuildInfrastructureFailure`` and a fault of the program's own choices stays an ordinary error or a
``BuildFailure``.

``try_grader`` grades a candidate through ``RolloutEngine.run``, the path a validation
control takes. The candidate's workspace files are written by one shell call each and its reply
ends the conversation (``spec.controls.candidate_turns``). The engine requires token evidence for
every turn, and grading reads none, so the model serving those turns makes up ids that extend each
request's prefix by one token. A registered task session would still need such a model: the engine
builds the task's ``TaskGrader`` inside ``run``, hands it to the session through
``TaskSessionContext``, and requests at least one model turn before the session grades.

``build_machine`` prepares a machine through RolloutEngine's own ``_prepare_machine``, so files, setup
commands, the execution user and the startup deadline behave as in a task attempt. RolloutEngine
exposes no public form of it.
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
from shellbox.machine import Machine
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import AnswerFormat, AnswerType, EnvironmentRequirements, Source, TaskResource

from taskforge.sandbox.factories import grading_environment
from taskforge.sandbox.images import BUILD_LIMITS, DockerBuild, check_build_limits
from taskforge.spec import draft
from taskforge.spec.controls import candidate_turns
from taskforge.spec.draft import MachineSettings, machine_runtime
from taskforge.stage40_build.context import BuildFailure, BuildInfrastructureFailure, RLEnvBuildContext
from taskforge.stage40_build.infrastructure import InfrastructureCause, host_checked_factories, infrastructure_failure
from taskforge.stage40_build.services import RLEnvBuildServices
from taskforge.stage40_build.step_store import SDK_VERSION

MACHINE_CLEANUP_TIMEOUT = 120.0
"""Seconds ``build_machine`` and ``try_grader`` wait for a machine to close."""
TRY_GRADER_MACHINE = draft.machine(startup_timeout=600.0)
"""The task and verifier machine settings ``try_grader`` uses unless the step passes its own."""
TRY_GRADER_VERIFIER_TIMEOUT = 600.0
"""Seconds ``try_grader`` allows one grading, verifier machine start included."""
TRY_GRADER_SOURCE = "taskforge.try_grader"
"""``Source.dataset`` of the provisional task ``try_grader`` grades against."""
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

    The machine starts as a task attempt's would, with ``files`` installed relative to the root and
    the setup commands run. A step uses it to try fixtures or a grader program;
    ``llm.agent.shell_tool(m, ...)`` gives an agent loop a shell in it. Leaving the block closes it.

    Raises:
        BuildInfrastructureFailure: the host failed to create or drive the machine.
    """
    runtime = machine_runtime(settings, requirements, services.host)
    factories = host_checked_factories(services.factories, services.host)
    async with AsyncExitStack() as owned:
        prepared = await _prepare_machine(
            requirements, runtime, tuple(files), factories, _Cleanup(MACHINE_CLEANUP_TIMEOUT), owned
        )
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


async def try_grader(
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
    settings: MachineSettings = TRY_GRADER_MACHINE,
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

    An empty ``reply`` usually returns ungraded. TaskCompendium refuses the submission before any
    grader runs, and the result carries ``Outcome.SUBMISSION_FAILURE``, reward 0.0 and the error
    "Text submission requires nonempty assistant content without tool calls". A caller counts that
    status as a rejection: the grader step of ``template.standard`` fails an empty answer only when
    it comes back ``GRADED`` above ``REJECTION_CEILING``.

    Raises:
        BuildFailure: ``spec.draft.assemble`` or ``spec.draft.lower`` rejects the provisional task,
            including an ``answer_format`` that cannot carry ``answer_type`` to ``grader``, or
            ``workspace`` is given without a task machine.
        BuildInfrastructureFailure: the host failed to create or drive a grading machine.
    """
    if workspace and requirements is None:
        raise BuildFailure("try_grader: workspace files need a task machine")
    try:
        task = draft.assemble(
            task_id=f"{context.item_id}.try_grader",
            instruction=instruction,
            answer_type=answer_type,
            answer_format=answer_format,
            grader=grader,
            source=Source(
                dataset=TRY_GRADER_SOURCE, revision=SDK_VERSION, row=context.item_id, importer_revision=SDK_VERSION
            ),
            environment=requirements,
            files=files,
            output_paths=output_paths,
        )
    except ValueError as error:
        raise BuildFailure(f"try_grader: {error}") from error
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
                verifier_timeout=TRY_GRADER_VERIFIER_TIMEOUT,
                cleanup_timeout=MACHINE_CLEANUP_TIMEOUT,
            ),
            factories=factories,
        )
    except (ValueError, NotImplementedError) as error:
        raise BuildFailure(f"try_grader: {error}") from error
    try:
        rollout = await RolloutEngine(_CandidateModel(turns), factories).run(lowered)
    except RolloutInterrupted as error:
        failure = infrastructure_failure(error)
        if failure is None:
            raise
        raise failure from error
    return rollout.grade


def image_repository(item_id: str) -> str:
    """The registry repository ``publish_image`` pushes ``item_id``'s images under."""
    return f"{IMAGE_REPOSITORY_PREFIX}/{re.sub(r'[^a-z0-9]+', '-', item_id.lower()).strip('-')}"


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
    return await images.publish(build, image_repository(context.item_id))
