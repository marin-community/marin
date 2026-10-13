# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a build program to a ``TaskDraft``.

``run_program`` walks the program's run order, ending with the sink. For each step it builds the
step's key and calls ``memoized``, which loads a stored output when one exists and otherwise calls
the step and stores what it returns.
``run_build`` runs the program for one proposal and writes the sink's ``TaskSpec`` with the
build's ``ItemBuildProvenance`` to ``<round_dir>/draft/``.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from taskcompendium.models import TaskSpec

from taskforge.content_hash import digest
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
    memoized,
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
        model_digest: The digest passed to ``run_build``: by default
            ``content_hash.digest({"model": model.model, "sampling": sampling})``, with the
            ``LLMPolicy`` hashed field by field. It enters every step's key, so a change of
            sampling reruns every step.
        round: The build round.
        steps: One outcome per step in run order, the sink last. Each pairs the stored memo of
            the step's output with this build's ``HIT`` or ``MISS`` status for it. A driver reads
            cache reuse from this list.
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


async def run_program(
    program: RLEnvBuildProgram,
    context: RLEnvBuildContext,
    proposal: object,
    store: RLEnvBuildStepStore,
    model_digest: str,
) -> tuple[TaskSpec, tuple[RLEnvBuildStepOutcome, ...]]:
    """Run ``program`` in order and return the sink's ``TaskSpec`` with one outcome per step.

    A step's key holds the output digest of each input, with ``content_hash.digest`` of
    ``proposal`` for ``PROPOSAL_INPUT``. The runner passes the key to ``memoized``. When ``store``
    holds a memo under it, the step is a ``HIT`` and the runner passes the decoded output on.
    Otherwise ``memoized`` calls the step with ``context`` and its inputs and stores the output,
    and the step is a ``MISS``. The sink runs last, the same way.

    Raises:
        BuildFailure: a step raised one. When the step left ``step`` unset, the runner sets it to
            the step's name.
    """
    assert store.item_id == context.item_id, f"store is for {store.item_id}, context for {context.item_id}"
    outputs: dict[str, Any] = {PROPOSAL_INPUT: proposal}
    digests: dict[str, str] = {PROPOSAL_INPUT: digest(proposal)}
    outcomes: list[RLEnvBuildStepOutcome] = []

    async def run[T](step: RLEnvBuildStep | RLEnvBuildSink, output_type: type[T]) -> T:
        key = RLEnvBuildStepKey(
            SDK_VERSION, step.source_digest, {dep: digests[dep] for dep in step.inputs}, model_digest
        )
        compute = partial(step.fn, context, **{dep: outputs[dep] for dep in step.inputs})
        try:
            status, memo, output = await memoized(store, step.name, key, output_type, compute)
        except BuildFailure as failure:
            if failure.step is not None:
                raise
            raise BuildFailure(failure.message, step.name) from failure
        outputs[step.name] = output
        digests[step.name] = memo.output_digest
        outcomes.append(RLEnvBuildStepOutcome(status, memo))
        return output

    for name in program.order[:-1]:
        step = program.steps[name]
        await run(step, step.output_type)
    task = await run(program.sink, TaskSpec)
    return task, tuple(outcomes)


async def run_build(
    program: RLEnvBuildProgram,
    proposal: object,
    item_id: str,
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
        item_id: The caller's id for the item; ``check_item_id`` must accept it.
        round_dir: The item's directory for this round.
        store_root: The local path or URL of the step store shared by all items.
        model: The model steps call.
        sampling: The sampling policy steps pass to ``model``.
        model_digest: The model part of every step key, usually ``model_digest(model, sampling)``.
        round: The build round, copied into ``ItemBuildProvenance``.

    Raises:
        BuildFailure: a step failed.
        ValueError: ``check_item_id`` refused ``item_id``.
    """
    store = RLEnvBuildStepStore(store_root, item_id)
    context = RLEnvBuildContext(item_id=item_id, model=model, sampling=sampling)
    task, outcomes = await run_program(program, context, proposal, store, model_digest)
    draft = TaskDraft(
        task=task,
        provenance=ItemBuildProvenance(
            item_id=item_id,
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


_PROVENANCE: TypeAdapter[ItemBuildProvenance] = TypeAdapter(ItemBuildProvenance)
