# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a build program to a ``TaskDraft``.

``run_program`` walks the program's run order, ending with the sink. For each step it builds the
step's key and calls ``memoized``, which loads a stored output when one exists and otherwise calls
the step and stores what it returns. A step named in ``invalidate`` runs and replaces its memo even
when one exists. Each step runs on a copy of the build context whose ``step`` is the step's name,
and records one ``STEP`` ledger span tagged with its key digest and cache status.

A step that fails because of the host raises ``BuildInfrastructureFailure``
(``infrastructure.py``), even when the program caught the host's error and raised its own. No
revision of the program can fix the host, so the caller retries or abandons the workpiece without
asking the author. A step that runs past ``STEP_TIMEOUT`` raises ``BuildExpired`` naming the step:
whether the step or Taskforge was slow is unknown, so the caller holds the workpiece rather than asking for
a revision.

``run_build`` computes the model part of every key (``build_model_digest``), runs the program for
one proposal, applies the research requirement, refuses a task whose machine is not the solver
machine the proposal declares (``check_solver_machine``), and writes the sink's ``TaskSpec`` with
the build's ``WorkpieceBuildProvenance`` to ``<round_dir>/draft/``. Validation lowers the task for its own
host and reads any step's stored output, such as the grader's reference answer, through
``step_output``.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any

from pydantic import BaseModel, TypeAdapter
from rigging.filesystem.atomic import write_bytes_atomic
from taskcompendium.models import TaskSpec

from taskforge.content_hash import digest
from taskforge.ledger.records import EntryKind, span
from taskforge.llm.client import Completion, ModelClient, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.stage10_proposal.model import SolverMachine, TaskProposal
from taskforge.stage30_author.program import PROPOSAL_INPUT, RLEnvBuildProgram, RLEnvBuildSink, RLEnvBuildStep
from taskforge.stage40_build_with_program.context import (
    BuildExpired,
    BuildFailure,
    BuildInfrastructureFailure,
    ExpiredClock,
    RLEnvBuildContext,
)
from taskforge.stage40_build_with_program.infrastructure import infrastructure_failure, machine_available
from taskforge.stage40_build_with_program.outputs import SOURCES_STEP, Sources
from taskforge.stage40_build_with_program.sdk import solver_machine
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage40_build_with_program.step_store import (
    SDK_VERSION,
    CacheStatus,
    RLEnvBuildStepKey,
    RLEnvBuildStepMemo,
    RLEnvBuildStepStore,
    memoized,
    recomputed,
)

DRAFT_DIR = "draft"
TASK_FILE = "task.json"
PROVENANCE_FILE = "provenance.json"
BUILD_STAGE = "build"
"""The ``step`` of the ``STAGE`` ledger span ``run_build`` records around a build."""
STEP_TIMEOUT = 1800.0
"""Seconds one step may run before the build expires (``BuildExpired``); a step that never returns would
otherwise hold its workpiece for good."""


class ResearchRequirement(StrEnum):
    """Whether ``run_build`` requires the sources step to hold a finding for every research item."""

    ENFORCED = "enforced"
    OFF = "off"


@dataclass(frozen=True)
class RLEnvBuildStepOutcome:
    """A step's result in one build.

    Attributes:
        status: ``HIT`` when this build loaded the step's stored output, ``MISS`` when it ran the
            step and stored the output, ``INVALIDATED`` when the caller asked for a rerun.
        memo: The memo the build loaded or wrote.
    """

    status: CacheStatus
    memo: RLEnvBuildStepMemo


@dataclass(frozen=True)
class WorkpieceBuildProvenance:
    """The provenance of an attempt to build a workpiece.

    It holds the build's inputs by digest and the outcome of each step. ``run_build`` writes it to
    ``provenance.json`` beside the task.

    Attributes:
        workpiece_id: The workpiece the caller built.
        proposal_digest: ``content_hash.digest`` of the proposal.
        program_digest: ``RLEnvBuildProgram.digest``.
        sdk_version: ``SDK_VERSION`` during the build.
        model: The model id, ``ModelClient.model``.
        model_digest: The model part of every step's key: ``build_model_digest`` of the model, the
            sampling policy and the services settings, or the caller's override.
        round: The build round.
        steps: One outcome per step in run order, the sink last. Each pairs the stored memo of
            the step's output with this build's status for it. A driver reads cache reuse from
            this list, and ``step_output`` reads a step's output through it.
    """

    workpiece_id: str
    proposal_digest: str
    program_digest: str
    sdk_version: str
    model: str
    model_digest: str
    round: int
    steps: tuple[RLEnvBuildStepOutcome, ...]


@dataclass(frozen=True)
class TaskDraft:
    task: TaskSpec
    provenance: WorkpieceBuildProvenance


def build_model_digest(model: ModelClient, sampling: LLMPolicy, services: RLEnvBuildServices) -> str:
    """The digest of everything besides a step's source and inputs that changes what the step returns.

    It covers the model id, every sampling field, the reasoning replay of agent loops and their
    limits, so a build under another value of any of them reruns every step. The machines, images,
    web tools and ledger of ``services`` stay out: they change where a step runs, not what it
    returns.
    """
    return digest(
        {
            "model": model.model,
            "sampling": sampling,
            "reasoning_replay": services.reasoning_replay,
            "agent_limits": services.agent_limits,
        }
    )


@dataclass
class _BudgetedModel:
    """``client`` for one step, refusing the call past ``limit``.

    Every path a step has to the model goes through ``context.model``: the SDK's structured calls,
    its agent loops and a direct call alike. Each call spends one of the step's ``max_calls``, and
    the first call past the limit fails the build as the program's; ``run_program`` names the step.
    """

    client: ModelClient
    limit: int
    spent: int = 0

    @property
    def model(self) -> str:
        return self.client.model

    def _spend(self) -> None:
        if self.spent >= self.limit:
            raise BuildFailure(f"more than {self.limit} model calls in one step; keep retries within the step's budget")
        self.spent += 1

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        self._spend()
        return await self.client.complete(messages, policy, request_fields)

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        self._spend()
        return await self.client.complete_structured(messages, output_type, policy)


async def run_program(
    program: RLEnvBuildProgram,
    context: RLEnvBuildContext,
    proposal: object,
    store: RLEnvBuildStepStore,
    model_digest: str,
    invalidate: frozenset[str] = frozenset(),
) -> tuple[TaskSpec, tuple[RLEnvBuildStepOutcome, ...]]:
    """Run ``program`` and return the sink's ``TaskSpec`` with one outcome per step, in ``program.order``.

    Every step starts as soon as the steps it takes inputs from have finished, so steps that do not
    depend on each other run at the same time; the sink runs once every step has. A step's key
    holds the output digest of each input, with ``content_hash.digest`` of ``proposal`` for
    ``PROPOSAL_INPUT``. The runner passes the key to ``memoized``. When ``store`` holds a memo
    under it, the step is a ``HIT`` and the runner passes the decoded output on. Otherwise
    ``memoized`` calls the step with its context and inputs and stores the output, and the step is
    a ``MISS``. A step named in ``invalidate`` runs either way and is ``INVALIDATED``. The first
    step to fail ends the build: the steps still running are cancelled and its failure is raised
    with the traceback it left its step with, so the frames of the program that raised it stay in it.

    Raises:
        BuildFailure: a step raised one. When the step left ``step`` unset, the runner sets it to
            the step's name.
        BuildExpired: a step ran past ``STEP_TIMEOUT``, or raised one; the runner sets ``step`` as
            for ``BuildFailure``.
        BuildInfrastructureFailure: a step failed because of the host, anywhere in the cause chain
            of what it raised.
    """
    assert (
        store.workpiece_id == context.workpiece_id
    ), f"store is for {store.workpiece_id}, context for {context.workpiece_id}"
    unknown = sorted(invalidate - {*program.steps, program.sink.name})
    assert not unknown, f"invalidate names {unknown}, which are not steps of the program"
    outputs: dict[str, Any] = {PROPOSAL_INPUT: proposal}
    digests: dict[str, str] = {PROPOSAL_INPUT: digest(proposal)}
    outcomes: dict[str, RLEnvBuildStepOutcome] = {}
    raised: dict[BaseException, TracebackType | None] = {}

    async def run[T](step: RLEnvBuildStep | RLEnvBuildSink, output_type: type[T]) -> T:
        name = step.name
        key = RLEnvBuildStepKey(
            SDK_VERSION, step.source_digest, {dep: digests[dep] for dep in step.inputs}, model_digest
        )
        budgeted = _BudgetedModel(context.model, context.services.agent_limits.max_calls)
        compute = partial(
            step.fn, replace(context, step=name, model=budgeted), **{dep: outputs[dep] for dep in step.inputs}
        )
        cached = recomputed if name in invalidate else memoized
        with span(
            context.services.ledger,
            EntryKind.STEP,
            workpiece_id=context.workpiece_id,
            round=context.round,
            step=name,
            code_hash=step.source_digest,
            input_hash=key.digest,
        ) as fields:
            deadline = asyncio.timeout(STEP_TIMEOUT)
            try:
                async with deadline:
                    status, memo, output = await cached(store, name, key, output_type, compute)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                raise BuildExpired(
                    f"ran for more than {STEP_TIMEOUT:.0f} seconds without returning", ExpiredClock.STEP, name
                ) from error
            except Exception as error:
                raised[error] = error.__traceback__
                infrastructure = infrastructure_failure(error)
                if infrastructure is not None:
                    fields.attrs["infrastructure"] = infrastructure.cause
                    if isinstance(error, BuildInfrastructureFailure):
                        raise
                    raise infrastructure from error
                if isinstance(error, BuildFailure) and error.step is None:
                    raise BuildFailure(error.message, name) from error
                if isinstance(error, BuildExpired) and error.step is None:
                    raise BuildExpired(error.message, error.clock, name) from error
                raise
            fields.attrs["cache"] = status
            fields.output_hash = memo.output_digest
        outputs[name] = output
        digests[name] = memo.output_digest
        outcomes[name] = RLEnvBuildStepOutcome(status, memo)
        return output

    started: dict[str, asyncio.Task[Any]] = {}

    async def after_inputs[T](step: RLEnvBuildStep | RLEnvBuildSink, output_type: type[T]) -> T:
        for dep in step.inputs:
            if dep in started:
                await started[dep]
        return await run(step, output_type)

    try:
        async with asyncio.TaskGroup() as group:
            for name in program.order[:-1]:
                step = program.steps[name]
                started[name] = group.create_task(after_inputs(step, step.output_type))
            sink = group.create_task(after_inputs(program.sink, TaskSpec))
    except ExceptionGroup as failures:
        # One step failed and the group cancelled the rest. Raise that failure as a serial run
        # would, with its own cause chain and without the group as context. Each step awaiting the
        # failed one re-raised the same exception object, which can leave its traceback holding only
        # that await, so it is raised with the traceback its own step left it with.
        failure = failures.exceptions[0]
        raise failure.with_traceback(raised.get(failure, failure.__traceback__)) from failure.__cause__
    return sink.result(), tuple(outcomes[name] for name in program.order)


def check_research(proposal: TaskProposal, sources: Sources) -> None:
    """Raise ``BuildFailure`` unless ``sources`` holds a finding with web calls for every research item.

    A program that skipped an item, or caught the failure of its lookup, fails here.
    """
    found = {finding.index for finding in sources.findings if finding.web_calls}
    missing = [index for index in range(len(proposal.header.research)) if index not in found]
    if missing:
        raise BuildFailure(
            f"research items {missing} have no finding: call research(context, proposal, i) for every research "
            "item of the proposal and let its failure propagate",
            SOURCES_STEP,
        )


def check_solver_machine(proposal: TaskProposal, task: TaskSpec, sink: str) -> None:
    """Raise ``BuildFailure`` for the step ``sink`` unless ``task`` runs on the machine ``proposal`` declares.

    The proposal pins the solver's machine (``ProposalHeader.solver_machine``, ``None`` for a container
    proposal's own image, ``none`` for no machine), so a program that builds another one is wrong
    whatever the host.
    """
    built = solver_machine(task.environment_requirements)
    if built != proposal.header.solver_machine:
        raise BuildFailure(
            f"the task runs on {_machine_text(built)}, but the proposal declares "
            f"{_machine_text(proposal.header.solver_machine)}; build the machine the proposal declares",
            sink,
        )


def _machine_text(machine: SolverMachine | None) -> str:
    if machine is None:
        return "the task's own image"
    if machine is SolverMachine.NONE:
        return f"no machine (solver_machine {machine})"
    return f"solver_machine {machine}"


async def run_build(
    program: RLEnvBuildProgram,
    proposal: TaskProposal,
    workpiece_id: str,
    round_dir: Path,
    store_root: str,
    model: ModelClient,
    sampling: LLMPolicy,
    services: RLEnvBuildServices,
    round: int,  # noqa: A002 - the loop's round number
    *,
    research: ResearchRequirement,
    invalidate: frozenset[str] = frozenset(),
    model_digest: str | None = None,
) -> TaskDraft:
    """Run ``program`` for ``proposal`` and write the draft to ``<round_dir>/draft/``.

    Args:
        program: The build program.
        proposal: The proposal, passed to steps taking ``PROPOSAL_INPUT``.
        workpiece_id: The caller's id for the workpiece; ``check_workpiece_id`` must accept it.
        round_dir: The workpiece's directory for this round.
        store_root: The local path or URL of the step store shared by all workpieces.
        model: The model steps call.
        sampling: The sampling policy steps pass to ``model``.
        services: Machines, images, web tools, the ledger and the agent-loop settings.
        round: The build round, copied into ``WorkpieceBuildProvenance`` and the ledger.
        research: Under ``ENFORCED``, a program for a proposal that lists research needs a ``sources``
            step returning ``Sources`` with a finding for every research item; a proposal without
            research needs no such step.
        invalidate: Step names to rerun even when the store holds their output.
        model_digest: Replaces ``build_model_digest(model, sampling, services)`` in every key.

    Raises:
        BuildFailure: the research requirement fails, a step failed, or the task's machine is not the
            solver machine the proposal declares.
        BuildExpired: a step ran past ``STEP_TIMEOUT``, or raised one (the standard grader step does for a grader
            that timed out twice).
        BuildInfrastructureFailure: this host cannot provide the declared solver machine
            (``machine_available``), before any step runs, or a step failed because of the host.
        ValueError: ``check_workpiece_id`` refused ``workpiece_id``.
    """
    machine_available(services, proposal)
    researches = research is ResearchRequirement.ENFORCED and bool(proposal.header.research)
    if researches and (SOURCES_STEP not in program.steps or program.steps[SOURCES_STEP].output_type is not Sources):
        raise BuildFailure(f"a build that researches its proposal needs a {SOURCES_STEP!r} step returning Sources")
    store = RLEnvBuildStepStore(store_root, workpiece_id)
    keyed_model = build_model_digest(model, sampling, services) if model_digest is None else model_digest
    context = RLEnvBuildContext(
        workpiece_id=workpiece_id, round=round, model=model, sampling=sampling, services=services, step=BUILD_STAGE
    )
    with span(services.ledger, EntryKind.STAGE, workpiece_id=workpiece_id, round=round, step=BUILD_STAGE) as fields:
        fields.attrs["program"] = program.digest
        task, outcomes = await run_program(program, context, proposal, store, keyed_model, invalidate)
        fields.attrs["hits"] = str(sum(outcome.status is CacheStatus.HIT for outcome in outcomes))
        fields.attrs["steps"] = str(len(outcomes))
    draft = TaskDraft(
        task=task,
        provenance=WorkpieceBuildProvenance(
            workpiece_id=workpiece_id,
            proposal_digest=digest(proposal),
            program_digest=program.digest,
            sdk_version=SDK_VERSION,
            model=model.model,
            model_digest=keyed_model,
            round=round,
            steps=outcomes,
        ),
    )
    if researches:
        check_research(proposal, step_output(draft, store_root, SOURCES_STEP, Sources))
    check_solver_machine(proposal, task, program.sink.name)
    _write_draft(round_dir / DRAFT_DIR, draft)
    return draft


def step_output[T](draft: TaskDraft, store_root: str, name: str, output_type: type[T]) -> T:
    """Read the output the step ``name`` had in ``draft``'s build back from the step store.

    Raises:
        KeyError: the draft's program has no step ``name``.
        ValueError: the stored blob does not match the digest in its memo.
    """
    memos = {outcome.memo.name: outcome.memo for outcome in draft.provenance.steps}
    if name not in memos:
        raise KeyError(f"Draft {draft.provenance.workpiece_id} has no step {name!r}; its steps are {sorted(memos)}")
    return RLEnvBuildStepStore(store_root, draft.provenance.workpiece_id).read_output(memos[name], output_type)


def _write_draft(directory: Path, draft: TaskDraft) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        TASK_FILE: draft.task.model_dump_json(indent=2).encode(),
        PROVENANCE_FILE: _PROVENANCE.dump_json(draft.provenance, indent=2),
    }
    for name, content in files.items():
        write_bytes_atomic(str(directory / name), content)


def load_draft(directory: Path) -> TaskDraft:
    """Read the draft ``run_build`` wrote to ``directory``."""
    return TaskDraft(
        task=TaskSpec.model_validate_json((directory / TASK_FILE).read_bytes()),
        provenance=_PROVENANCE.validate_json((directory / PROVENANCE_FILE).read_bytes()),
    )


_PROVENANCE: TypeAdapter[WorkpieceBuildProvenance] = TypeAdapter(WorkpieceBuildProvenance)
