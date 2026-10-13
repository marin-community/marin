# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a build program to a ``TaskDraft``.

``run_program`` walks the program's run order. For each step it builds the step's key, loads a
stored output when one exists, and otherwise calls the step and stores what it returns.
``run_build`` checks that the program's sink returns a ``TaskSpec``, runs the program for one
proposal, and writes the task with its build record to ``<round_dir>/draft/``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from taskcompendium.models import TaskSpec

from taskforge.content_hash import digest
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage30_author.program import PROPOSAL_INPUT, RLEnvBuildProgram
from taskforge.stage40_build.context import BuildFailure, RLEnvBuildContext
from taskforge.stage40_build.step_store import (
    SDK_VERSION,
    RLEnvBuildStepKey,
    RLEnvBuildStepRecord,
    RLEnvBuildStepStore,
)

DRAFT_DIR = "draft"
TASK_FILE = "task.json"
PROVENANCE_FILE = "provenance.json"


@dataclass(frozen=True)
class Provenance:
    """The inputs of a build, by digest, and what the build did with each step.

    ``run_build`` writes it to ``provenance.json`` beside the task.

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
        steps: One record per step in run order, each with its ``HIT`` or ``MISS`` status in this
            build. A driver reads cache reuse from this list.
    """

    item_id: str
    proposal_digest: str
    program_digest: str
    sdk_version: str
    model: str
    model_digest: str
    round: int
    steps: tuple[RLEnvBuildStepRecord, ...]


@dataclass(frozen=True)
class TaskDraft:
    task: TaskSpec
    provenance: Provenance


async def run_program(
    program: RLEnvBuildProgram,
    context: RLEnvBuildContext,
    proposal: object,
    store: RLEnvBuildStepStore,
    model_digest: str,
) -> tuple[Any, tuple[RLEnvBuildStepRecord, ...]]:
    """Run ``program`` in order and return the sink's output with one record per step.

    A step's key holds the output digest of each declared input, with ``content_hash.digest`` of
    ``proposal`` for ``PROPOSAL_INPUT``. When ``store`` holds a record under the key, the step is a
    ``HIT`` and the runner passes the decoded output on. Otherwise the runner calls the step with
    ``context`` and its inputs, then stores the output as a ``MISS``.

    Raises:
        BuildFailure: a step raised one. When the step left ``step`` unset, the runner sets it to
            the step's name.
    """
    assert store.item_id == context.item_id, f"store is for {store.item_id}, context for {context.item_id}"
    outputs: dict[str, Any] = {PROPOSAL_INPUT: proposal}
    digests: dict[str, str] = {PROPOSAL_INPUT: digest(proposal)}
    records: list[RLEnvBuildStepRecord] = []
    for name in program.order:
        step = program.steps[name]
        key = RLEnvBuildStepKey(
            SDK_VERSION, step.source_digest, {dep: digests[dep] for dep in step.inputs}, model_digest
        )
        loaded = store.load(key, name, step.output_type)
        if loaded is not None:
            record, output = loaded
        else:
            try:
                output = await step.fn(context, **{dep: outputs[dep] for dep in step.inputs})
            except BuildFailure as failure:
                if failure.step is not None:
                    raise
                raise BuildFailure(failure.message, name) from failure
            record = store.store(key, name, output)
        outputs[name] = output
        digests[name] = record.output_digest
        records.append(record)
    return outputs[program.sink], tuple(records)


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
        program: The build program. Its sink returns the ``TaskSpec``.
        proposal: A JSON-serialisable value, passed to steps declaring ``PROPOSAL_INPUT``.
        item_id: The caller's id for the item; ``check_item_id`` must accept it.
        round_dir: The item's directory for this round.
        store_root: The local path or URL of the step store shared by all items.
        model: The model steps call.
        sampling: The sampling policy steps pass to ``model``.
        model_digest: The model part of every step key, usually ``model_digest(model, sampling)``.
        round: The build round, copied into ``Provenance``.

    Raises:
        BuildFailure: the sink's return annotation is not ``TaskSpec``, or a step failed.
        ValueError: ``check_item_id`` refused ``item_id``.
    """
    sink = program.steps[program.sink]
    if sink.output_type is not TaskSpec:
        raise BuildFailure(f"the sink returns {sink.output_type!r}, not TaskSpec", program.sink)
    store = RLEnvBuildStepStore(store_root, item_id)
    context = RLEnvBuildContext(item_id=item_id, model=model, sampling=sampling)
    task, records = await run_program(program, context, proposal, store, model_digest)
    draft = TaskDraft(
        task=task,
        provenance=Provenance(
            item_id=item_id,
            proposal_digest=digest(proposal),
            program_digest=program.digest,
            sdk_version=SDK_VERSION,
            model=model.model,
            model_digest=model_digest,
            round=round,
            steps=records,
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


_PROVENANCE: TypeAdapter[Provenance] = TypeAdapter(Provenance)
