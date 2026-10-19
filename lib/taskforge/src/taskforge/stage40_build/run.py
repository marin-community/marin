# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a build program to a ``TaskDraft``.

``run_program`` walks the program's run order, ending with the sink. For each step it builds the
step's key and calls ``memoized``, which loads a stored output when one exists and otherwise calls
the step and stores what it returns.
``run_build`` runs the program for one proposal and writes the sink's ``TaskSpec`` with the
build's ``WorkpieceBuildProvenance`` to ``<round_dir>/draft/``.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from taskcompendium.models import TaskSpec

from taskforge.content_hash import canonical_json, digest
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage30_author.program import PROPOSAL_INPUT, RLEnvBuildProgram, RLEnvBuildSink, RLEnvBuildStep
from taskforge.stage40_build.context import BuildFailure, RLEnvBuildContext
from taskforge.stage40_build.step_store import (
    SDK_VERSION,
    CacheStatus,
    RLEnvBuildStepKey,
    RLEnvBuildStepMemo,
    RLEnvBuildStepStore,
)

DRAFT_DIR = "draft"
TASK_FILE = "task.json"
PROVENANCE_FILE = "provenance.json"


@dataclass(frozen=True)
class RLEnvBuildStepOutcome:
    """A step's result in one build.

    Attributes:
        status: ``HIT`` when this build loaded the step's stored output, ``MISS`` when it ran the
            step and stored the output.
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
        model_digest: The digest passed to ``run_build``: by default
            ``content_hash.digest({"model": model.model, "sampling": sampling})``, with the
            ``LLMPolicy`` hashed field by field. It enters every step's key, so a change of
            sampling reruns every step.
        round: The build round.
        steps: One outcome per step in run order, the sink last. Each pairs the stored memo of
            the step's output with this build's ``HIT`` or ``MISS`` status for it. A driver reads
            cache reuse from this list.
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


async def _memoized[T](
    store: RLEnvBuildStepStore,
    name: str,
    key: RLEnvBuildStepKey,
    output_type: type[T],
    compute: Callable[[], Awaitable[T]],
) -> tuple[CacheStatus, RLEnvBuildStepMemo, T]:
    """Load ``name``'s output under ``key`` from ``store``, or compute and store it on a miss.

    The runner's one store call: it loads first. When a memo exists, it returns ``HIT`` with the
    memo and the decoded output, leaving ``compute`` uncalled. Otherwise it awaits ``compute()``,
    stores the result, and returns ``MISS`` with the new memo and the result decoded from its
    stored form, so a consumer receives the same value on a miss as on a later hit.
    """
    loaded = store.load(key, name, output_type)
    if loaded is not None:
        memo, output = loaded
        return CacheStatus.HIT, memo, output
    computed = await compute()
    memo = store.store(key, name, computed)
    return CacheStatus.MISS, memo, TypeAdapter(output_type).validate_json(canonical_json(computed))


async def run_program(
    program: RLEnvBuildProgram,
    context: RLEnvBuildContext,
    proposal: object,
    store: RLEnvBuildStepStore,
    model_digest: str,
) -> tuple[TaskSpec, tuple[RLEnvBuildStepOutcome, ...]]:
    """Run ``program`` and return the sink's ``TaskSpec`` with one outcome per step, in ``program.order``.

    Every step starts as soon as the steps it takes inputs from have finished, so steps that do not
    depend on each other run at the same time; the sink runs once every step has. A step's key
    holds the output digest of each input, with ``content_hash.digest`` of ``proposal`` for
    ``PROPOSAL_INPUT``. When ``store`` holds a memo under the key, the step is a ``HIT`` and the
    runner passes the decoded output on. Otherwise the runner calls the step with ``context`` and
    its inputs and stores the output, and the step is a ``MISS``. The first step to fail ends the
    build: the steps still running are cancelled and its failure is raised.

    Raises:
        BuildFailure: a step raised one. When the step left ``step`` unset, the runner sets it to
            the step's name.
    """
    assert (
        store.workpiece_id == context.workpiece_id
    ), f"store is for {store.workpiece_id}, context for {context.workpiece_id}"
    outputs: dict[str, Any] = {PROPOSAL_INPUT: proposal}
    digests: dict[str, str] = {PROPOSAL_INPUT: digest(proposal)}
    outcomes: dict[str, RLEnvBuildStepOutcome] = {}

    async def run[T](step: RLEnvBuildStep | RLEnvBuildSink, output_type: type[T]) -> T:
        key = RLEnvBuildStepKey(
            SDK_VERSION, step.source_digest, {dep: digests[dep] for dep in step.inputs}, model_digest
        )
        compute = partial(step.fn, context, **{dep: outputs[dep] for dep in step.inputs})
        try:
            status, memo, output = await _memoized(store, step.name, key, output_type, compute)
        except BuildFailure as failure:
            if failure.step is not None:
                raise
            raise BuildFailure(failure.message, step.name) from failure
        outputs[step.name] = output
        digests[step.name] = memo.output_digest
        outcomes[step.name] = RLEnvBuildStepOutcome(status, memo)
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
        # would, with its own cause chain and without the group as context.
        failure = failures.exceptions[0]
        raise failure from failure.__cause__
    return sink.result(), tuple(outcomes[name] for name in program.order)


async def run_build(
    program: RLEnvBuildProgram,
    proposal: object,
    workpiece_id: str,
    round_dir: Path,
    store_root: str,
    model: ModelClient,
    sampling: LLMPolicy,
    model_digest: str,
    round: int,  # noqa: A002 - the loop's round number
) -> TaskDraft:
    """Run ``program`` for ``proposal`` and write the draft to ``<round_dir>/draft/``.

    Args:
        program: The build program.
        proposal: A JSON-serialisable value, passed to steps taking ``PROPOSAL_INPUT``.
        workpiece_id: The caller's id for the workpiece; ``check_workpiece_id`` must accept it.
        round_dir: The workpiece's directory for this round.
        store_root: The local path or URL of the step store shared by all workpieces.
        model: The model steps call.
        sampling: The sampling policy steps pass to ``model``.
        model_digest: The model part of every step key, usually ``model_digest(model, sampling)``.
        round: The build round, copied into ``WorkpieceBuildProvenance``.

    Raises:
        BuildFailure: a step failed.
        ValueError: ``check_workpiece_id`` refused ``workpiece_id``.
    """
    store = RLEnvBuildStepStore(store_root, workpiece_id)
    context = RLEnvBuildContext(workpiece_id=workpiece_id, model=model, sampling=sampling)
    task, outcomes = await run_program(program, context, proposal, store, model_digest)
    draft = TaskDraft(
        task=task,
        provenance=WorkpieceBuildProvenance(
            workpiece_id=workpiece_id,
            proposal_digest=digest(proposal),
            program_digest=program.digest,
            sdk_version=SDK_VERSION,
            model=model.model,
            model_digest=model_digest,
            round=round,
            steps=outcomes,
        ),
    )
    _write_draft(round_dir / DRAFT_DIR, draft)
    return draft


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


_PROVENANCE: TypeAdapter[WorkpieceBuildProvenance] = TypeAdapter(WorkpieceBuildProvenance)
