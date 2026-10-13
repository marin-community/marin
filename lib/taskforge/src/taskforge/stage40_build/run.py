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
revision of the program can fix the host, so the caller retries or abandons the item without
asking the author.

``run_build`` computes the model part of every key (``build_model_digest``), runs the program for
one proposal, applies the research requirement, and writes the sink's ``TaskSpec`` with the
build's ``ItemBuildProvenance`` to ``<round_dir>/draft/``. Validation lowers the task for its own
host and reads any step's stored output, such as the grader's reference answer, through
``step_output``.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from taskcompendium.models import TaskSpec

from taskforge.content_hash import digest
from taskforge.ledger.records import EntryKind, span
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.program import PROPOSAL_INPUT, RLEnvBuildProgram, RLEnvBuildSink, RLEnvBuildStep
from taskforge.stage40_build.context import BuildFailure, BuildInfrastructureFailure, RLEnvBuildContext
from taskforge.stage40_build.infrastructure import infrastructure_failure
from taskforge.stage40_build.outputs import SOURCES_STEP, Sources
from taskforge.stage40_build.services import RLEnvBuildServices
from taskforge.stage40_build.step_store import (
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
class ItemBuildProvenance:
    """The provenance of an attempt to build an item.

    It holds the build's inputs by digest and the outcome of each step. ``run_build`` writes it to
    ``provenance.json`` beside the task.

    Attributes:
        item_id: The item the caller built.
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

    item_id: str
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
    provenance: ItemBuildProvenance


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


async def run_program(
    program: RLEnvBuildProgram,
    context: RLEnvBuildContext,
    proposal: object,
    store: RLEnvBuildStepStore,
    model_digest: str,
    invalidate: frozenset[str] = frozenset(),
) -> tuple[TaskSpec, tuple[RLEnvBuildStepOutcome, ...]]:
    """Run ``program`` in order and return the sink's ``TaskSpec`` with one outcome per step.

    A step's key holds the output digest of each input, with ``content_hash.digest`` of
    ``proposal`` for ``PROPOSAL_INPUT``. The runner passes the key to ``memoized``. When ``store``
    holds a memo under it, the step is a ``HIT`` and the runner passes the decoded output on.
    Otherwise ``memoized`` calls the step with its context and inputs and stores the output, and
    the step is a ``MISS``. A step named in ``invalidate`` runs either way and is ``INVALIDATED``.
    The sink runs last, the same way.

    Raises:
        BuildFailure: a step raised one. When the step left ``step`` unset, the runner sets it to
            the step's name.
        BuildInfrastructureFailure: a step failed because of the host, anywhere in the cause chain
            of what it raised.
    """
    assert store.item_id == context.item_id, f"store is for {store.item_id}, context for {context.item_id}"
    unknown = sorted(invalidate - {*program.steps, program.sink.name})
    assert not unknown, f"invalidate names {unknown}, which are not steps of the program"
    outputs: dict[str, Any] = {PROPOSAL_INPUT: proposal}
    digests: dict[str, str] = {PROPOSAL_INPUT: digest(proposal)}
    outcomes: list[RLEnvBuildStepOutcome] = []

    async def run[T](step: RLEnvBuildStep | RLEnvBuildSink, output_type: type[T]) -> T:
        name = step.name
        key = RLEnvBuildStepKey(
            SDK_VERSION, step.source_digest, {dep: digests[dep] for dep in step.inputs}, model_digest
        )
        compute = partial(step.fn, replace(context, step=name), **{dep: outputs[dep] for dep in step.inputs})
        cached = recomputed if name in invalidate else memoized
        with span(
            context.services.recording,
            EntryKind.STEP,
            item_id=context.item_id,
            round=context.round,
            step=name,
            code_hash=step.source_digest,
            input_hash=key.digest,
        ) as fields:
            try:
                status, memo, output = await cached(store, name, key, output_type, compute)
            except Exception as error:
                infrastructure = infrastructure_failure(error)
                if infrastructure is not None:
                    fields.attrs["infrastructure"] = infrastructure.cause
                    if isinstance(error, BuildInfrastructureFailure):
                        raise
                    raise infrastructure from error
                if isinstance(error, BuildFailure) and error.step is None:
                    raise BuildFailure(error.message, name) from error
                raise
            fields.attrs["cache"] = status
            fields.output_hash = memo.output_digest
        outputs[name] = output
        digests[name] = memo.output_digest
        outcomes.append(RLEnvBuildStepOutcome(status, memo))
        return output

    for name in program.order[:-1]:
        step = program.steps[name]
        await run(step, step.output_type)
    task = await run(program.sink, TaskSpec)
    return task, tuple(outcomes)


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


async def run_build(
    program: RLEnvBuildProgram,
    proposal: TaskProposal,
    item_id: str,
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
        item_id: The caller's id for the item; ``check_item_id`` must accept it.
        round_dir: The item's directory for this round.
        store_root: The local path or URL of the step store shared by all items.
        model: The model steps call.
        sampling: The sampling policy steps pass to ``model``.
        services: Machines, images, web tools, the ledger and the agent-loop settings.
        round: The build round, copied into ``ItemBuildProvenance`` and the ledger.
        research: Under ``ENFORCED``, the program needs a ``sources`` step returning ``Sources``
            with a finding for every research item of ``proposal``.
        invalidate: Step names to rerun even when the store holds their output.
        model_digest: Replaces ``build_model_digest(model, sampling, services)`` in every key.

    Raises:
        BuildFailure: the research requirement fails, or a step failed.
        BuildInfrastructureFailure: a step failed because of the host.
        ValueError: ``check_item_id`` refused ``item_id``.
    """
    if research is ResearchRequirement.ENFORCED and (
        SOURCES_STEP not in program.steps or program.steps[SOURCES_STEP].output_type is not Sources
    ):
        raise BuildFailure(f"a build that researches its proposal needs a {SOURCES_STEP!r} step returning Sources")
    store = RLEnvBuildStepStore(store_root, item_id)
    keyed_model = build_model_digest(model, sampling, services) if model_digest is None else model_digest
    context = RLEnvBuildContext(
        item_id=item_id, round=round, model=model, sampling=sampling, services=services, step=BUILD_STAGE
    )
    with span(services.recording, EntryKind.STAGE, item_id=item_id, round=round, step=BUILD_STAGE) as fields:
        fields.attrs["program"] = program.digest
        task, outcomes = await run_program(program, context, proposal, store, keyed_model, invalidate)
        fields.attrs["hits"] = str(sum(outcome.status is CacheStatus.HIT for outcome in outcomes))
        fields.attrs["steps"] = str(len(outcomes))
    draft = TaskDraft(
        task=task,
        provenance=ItemBuildProvenance(
            item_id=item_id,
            proposal_digest=digest(proposal),
            program_digest=program.digest,
            sdk_version=SDK_VERSION,
            model=model.model,
            model_digest=keyed_model,
            round=round,
            steps=outcomes,
        ),
    )
    if research is ResearchRequirement.ENFORCED:
        check_research(proposal, step_output(draft, store_root, SOURCES_STEP, Sources))
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
        raise KeyError(f"Draft {draft.provenance.item_id} has no step {name!r}; its steps are {sorted(memos)}")
    return RLEnvBuildStepStore(store_root, draft.provenance.item_id).read_output(memos[name], output_type)


def _write_draft(directory: Path, draft: TaskDraft) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        TASK_FILE: draft.task.model_dump_json(indent=2).encode(),
        PROVENANCE_FILE: _PROVENANCE.dump_json(draft.provenance, indent=2),
    }
    for name, content in files.items():
        with atomic_rename(str(directory / name)) as temp:
            Path(temp).write_bytes(content)


def load_draft(directory: Path) -> TaskDraft:
    """Read the draft ``run_build`` wrote to ``directory``."""
    return TaskDraft(
        task=TaskSpec.model_validate_json((directory / TASK_FILE).read_bytes()),
        provenance=_PROVENANCE.validate_json((directory / PROVENANCE_FILE).read_bytes()),
    )


_PROVENANCE: TypeAdapter[ItemBuildProvenance] = TypeAdapter(ItemBuildProvenance)
