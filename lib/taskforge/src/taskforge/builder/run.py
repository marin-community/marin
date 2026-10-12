# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a builder program to a ``TaskDraft``: its sink's ``TaskSpec`` and the provenance of every step.

``run_program`` is the only place a program's steps are sequenced and have effects: it runs them in
``Program.order``, keys each on ``StepKey``, reuses a stored output or runs and stores the step, and
records one ``STEP`` span per step. ``run_build`` wraps it for one proposal: it refuses a program
whose sink does not return a ``TaskSpec``, records the ``STAGE`` span, and writes the draft to
``<round_dir>/draft/``. Lowering the task for RolloutEngine is validation's concern, not the build's.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from taskcompendium.models import TaskSpec

from taskforge.atomic_file import write_atomic
from taskforge.builder.sdk import BuildContext, BuildFailure
from taskforge.builder.step import (
    PROPOSAL_INPUT,
    SDK_VERSION,
    CacheStatus,
    Program,
    StepCache,
    StepKey,
    StepRecord,
)
from taskforge.ledger.records import EntryKind, Ledger, span
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import RecordedModel
from taskforge.proposal.model import TaskProposal

DRAFT_DIR = "draft"
TASK_FILE = "task.json"
PROVENANCE_FILE = "provenance.json"
PROGRAM_STEP = "program"
"""The ledger step of a model call made outside any step."""


@dataclass(frozen=True)
class Provenance:
    """Where a draft came from: inputs by digest, and every step with its cache status in this build."""

    item_id: str
    proposal_digest: str
    program_digest: str
    sdk_version: str
    model: str
    policy_digest: str
    round: int
    steps: tuple[StepRecord, ...]


@dataclass(frozen=True)
class TaskDraft:
    """A built task and its provenance."""

    task: TaskSpec
    provenance: Provenance


def item_id_for(proposal: TaskProposal) -> str:
    """A file-safe item id from the proposal id (``d01.x/3`` becomes ``d01.x--3``)."""
    return proposal.header.id.replace("/", "--")


async def run_program(
    program: Program,
    proposal: TaskProposal,
    b: BuildContext,
    cache: StepCache,
    ledger: Ledger,
    round: int,  # noqa: A002 - matches LedgerEntry.round
) -> tuple[Any, tuple[StepRecord, ...]]:
    """Run ``program`` for ``proposal``; return the sink's output and one record per step, in order.

    Each step's ``inputs`` are the output digests of the steps it takes (``proposal.digest`` for
    ``"proposal"``). A key with a stored output is a ``HIT`` and the step does not run; otherwise the
    step runs with ``b.model`` attributed to it and its output is stored. Each step is one ``STEP``
    span with ``code_hash`` its source digest, ``input_hash`` its key, ``output_hash`` its output
    digest and ``attrs["cache"]`` its status.

    Raises:
        BuildFailure: a step raised one; it carries that step's name.
    """
    outputs: dict[str, Any] = {PROPOSAL_INPUT: proposal}
    digests: dict[str, str] = {PROPOSAL_INPUT: proposal.digest}
    records: list[StepRecord] = []
    for name in program.order:
        step = program.steps[name]
        inputs = {dep: digests[dep] for dep in step.inputs}
        key = StepKey(SDK_VERSION, step.source_digest, inputs, b.model.digest)
        with span(
            ledger,
            EntryKind.STEP,
            item_id=b.item_id,
            round=round,
            step=name,
            code_hash=step.source_digest,
            input_hash=key.digest,
        ) as fields:
            loaded = cache.load(step, key)
            if loaded is not None:
                record, output = loaded
            else:
                context = replace(b, model=replace(b.model, step=name))
                try:
                    output = await step.fn(context, **{dep: outputs[dep] for dep in step.inputs})
                except BuildFailure as failure:
                    if failure.step is not None:
                        raise
                    raise BuildFailure(failure.message, name) from failure
                record = cache.store(step, key, inputs, output)
            fields.attrs["cache"] = record.status
            fields.output_hash = record.output.digest
        outputs[name] = output
        digests[name] = record.output.digest
        records.append(record)
    return outputs[program.sink], tuple(records)


async def run_build(
    program: Program,
    proposal: TaskProposal,
    round_dir: Path,
    cache: Path,
    model: ModelClient,
    sampling: LLMPolicy,
    ledger: Ledger,
    round: int = 0,  # noqa: A002 - matches LedgerEntry.round
) -> TaskDraft:
    """Run ``program`` for ``proposal`` and write its draft to ``<round_dir>/draft/``.

    Args:
        program: The builder program; its sink returns the task.
        proposal: The proposal it builds: the item id, and the ``"proposal"`` input.
        round_dir: The item's round directory; the draft goes to ``draft/`` under it.
        cache: The step cache root shared across items (``items/<id>/steps`` and ``blobs``).
        model: The model steps call, with ``sampling``; both are part of every step key.
        sampling: The sampling policy of every model call.
        ledger: Where the build, its steps and their model calls are recorded.
        round: The build round, for the ledger.

    Raises:
        BuildFailure: the sink does not return a ``TaskSpec``, or a step failed.
    """
    sink = program.steps[program.sink]
    if sink.output_type is not TaskSpec:
        raise BuildFailure(f"the sink returns {sink.output_type!r}, not TaskSpec", program.sink)
    item_id = item_id_for(proposal)
    recorded = RecordedModel(model, sampling, ledger, item_id, round, PROGRAM_STEP)
    b = BuildContext(item_id=item_id, model=recorded)
    with span(ledger, EntryKind.STAGE, item_id=item_id, round=round, step="build") as fields:
        fields.attrs["program"] = program.digest
        task, records = await run_program(program, proposal, b, StepCache(cache, item_id), ledger, round)
        fields.attrs["steps"] = str(len(records))
        fields.attrs["hits"] = str(sum(record.status == CacheStatus.HIT for record in records))
        draft = TaskDraft(
            task=task,
            provenance=Provenance(
                item_id=item_id,
                proposal_digest=proposal.digest,
                program_digest=program.digest,
                sdk_version=SDK_VERSION,
                model=model.model,
                policy_digest=recorded.digest,
                round=round,
                steps=records,
            ),
        )
        _write_draft(round_dir / DRAFT_DIR, draft)
    return draft


def _write_draft(directory: Path, draft: TaskDraft) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(directory / TASK_FILE, draft.task.model_dump_json(indent=2).encode())
    write_atomic(directory / PROVENANCE_FILE, _PROVENANCE.dump_json(draft.provenance, indent=2))


def load_draft(directory: Path) -> TaskDraft:
    """Read a draft written by ``run_build``."""
    return TaskDraft(
        task=TaskSpec.model_validate_json((directory / TASK_FILE).read_bytes()),
        provenance=_PROVENANCE.validate_json((directory / PROVENANCE_FILE).read_bytes()),
    )


_PROVENANCE: TypeAdapter[Provenance] = TypeAdapter(Provenance)
