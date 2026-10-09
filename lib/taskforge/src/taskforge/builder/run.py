# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a builder program to a ``TaskDraft``: the TaskSpec, its lowered form, its controls, and
provenance.

``run_build`` enforces the library rules a program cannot opt out of: the task's grader is the
output of a GRADER step, the controls are the output of a CONTROLS step, the two roles are separate
steps, the lowered spec carries exactly the task and passes RolloutEngine's
``validate_lowered_task`` on this host's factories, and the controls are a complete set for the task
(``validate_controls``). Controls are not replayed here. The draft is written to
``<item_dir>/draft/``.

A draft is bound to the host it was built on: ``lowered.json`` names that host's machine backends,
and validation evidence is keyed by the lowered spec.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from rolloutengine.lowering import validate_lowered_task
from rolloutengine.spec import LoweredTaskSpec
from taskcompendium.models import TaskSpec

from taskforge.atomic_file import write_atomic
from taskforge.builder.sdk import Build, BuildFailure, BuildOutput, BuildServices, Grader, file_set
from taskforge.builder.step import SDK_VERSION, CacheStatus, Resource, StepCache, StepRecord, StepRole
from taskforge.ledger.records import EntryKind, span
from taskforge.proposal.model import TaskProposal
from taskforge.spec.controls import Control, controls_json, validate_controls

DRAFT_DIR = "draft"
TASK_FILE = "task.json"
LOWERED_FILE = "lowered.json"
CONTROLS_FILE = "controls.json"
PROVENANCE_FILE = "provenance.json"
SCRATCH_DIR = "scratch"


@dataclass(frozen=True)
class Provenance:
    """Where a draft came from: inputs by digest, and every step call with its cache status."""

    item_id: str
    proposal_digest: str
    program_digest: str
    sdk_version: str
    model: str
    policy_digest: str
    round: int
    steps: tuple[StepRecord, ...]
    resources: tuple[Resource, ...]


@dataclass(frozen=True)
class TaskDraft:
    """A built task. ``lowered`` (whose ``task`` is ``task``, answer format included) is what validation
    runs."""

    task: TaskSpec
    lowered: LoweredTaskSpec
    controls: tuple[Control, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.lowered.task != self.task:
            raise ValueError(f"Draft {self.task.id!r}: the lowered spec carries a different task")


class Program(Protocol):
    """A compiled builder program."""

    @property
    def digest(self) -> str: ...

    @property
    def build(self) -> Callable[[Build], Awaitable[BuildOutput]]: ...


def item_id_for(proposal: TaskProposal) -> str:
    """A file-safe item id from the proposal id (``d01.x/3`` becomes ``d01.x--3``)."""
    return proposal.header.id.replace("/", "--")


def check_roles(output: BuildOutput, records: Sequence[StepRecord], outputs: Sequence[object]) -> None:
    """Raise ``BuildFailure`` unless the grader and controls came from separate GRADER and CONTROLS steps."""
    graders = [value for record, value in zip(records, outputs, strict=True) if record.role == StepRole.GRADER]
    controls = [value for record, value in zip(records, outputs, strict=True) if record.role == StepRole.CONTROLS]
    if not graders or not controls:
        raise BuildFailure("a build needs a GRADER step and a separate CONTROLS step", None)
    packages = [grader.package for grader in graders if isinstance(grader, Grader)]
    task = output.task
    if not any(
        package.grader == task.grader and file_set(package.resources) == file_set(task.resources.verifier)
        for package in packages
    ):
        raise BuildFailure(f"the task's {task.grader.kind} grader is not the output of a GRADER step", None)
    if not any(tuple(value) == output.controls for value in controls if isinstance(value, tuple)):
        raise BuildFailure("the task's controls are not the output of a CONTROLS step", None)


def _write_draft(directory: Path, draft: TaskDraft) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(directory / TASK_FILE, draft.task.model_dump_json(indent=2).encode())
    write_atomic(directory / LOWERED_FILE, draft.lowered.model_dump_json(indent=2).encode())
    write_atomic(directory / CONTROLS_FILE, controls_json(draft.controls))
    write_atomic(directory / PROVENANCE_FILE, _PROVENANCE.dump_json(draft.provenance, indent=2))


async def run_build(
    program: Program,
    proposal: TaskProposal,
    item_dir: Path,
    cache: Path,
    services: BuildServices,
    invalidate: Sequence[str] = (),
    round: int = 0,  # noqa: A002 - matches LedgerEntry.round
) -> TaskDraft:
    """Run ``program.build`` for ``proposal`` and return the checked draft.

    Args:
        program: The compiled builder program.
        proposal: The proposal it builds; its digest is part of every step key.
        item_dir: The item's directory; the draft lands in ``draft/``, scratch files in ``scratch/``.
        cache: The step cache root shared across items (``items/<id>/steps`` and ``blobs``).
        services: Model, machines and ledger.
        invalidate: Step names to recompute even when their key has a record.
        round: The build round, for the ledger.

    Raises:
        BuildFailure: the program failed a check, or its output breaks a library rule.
    """
    item_id = item_id_for(proposal)
    step_cache = StepCache(root=cache, item_id=item_id, invalidated=frozenset(invalidate))
    scratch = item_dir / SCRATCH_DIR
    scratch.mkdir(parents=True, exist_ok=True)
    b = Build(proposal, item_id, services, step_cache, scratch, round)
    with span(services.ledger, EntryKind.STAGE, item_id=item_id, round=round, step="build") as fields:
        fields.attrs["program"] = program.digest
        output = await program.build(b)
        if not isinstance(output, BuildOutput):
            raise BuildFailure(f"build(b) returned {type(output).__name__}, not BuildOutput", None)
        _check_output(output, step_cache, services)
        draft = TaskDraft(
            task=output.task,
            lowered=output.lowered,
            controls=output.controls,
            provenance=Provenance(
                item_id=item_id,
                proposal_digest=proposal.digest,
                program_digest=program.digest,
                sdk_version=SDK_VERSION,
                model=services.client.endpoint.model,
                policy_digest=b.llm.digest,
                round=round,
                steps=tuple(step_cache.records),
                resources=tuple(b.resources),
            ),
        )
        _write_draft(item_dir / DRAFT_DIR, draft)
        fields.attrs["hits"] = str(sum(record.status == CacheStatus.HIT for record in step_cache.records))
        fields.attrs["steps"] = str(len(step_cache.records))
    return draft


def _check_output(output: BuildOutput, step_cache: StepCache, services: BuildServices) -> None:
    """The library rules a program's output must keep: step roles, a lowering of the task its host can
    run, and well-formed controls.

    Raises:
        BuildFailure: the first rule ``output`` breaks.
    """
    outputs = [_output(step_cache, record) for record in step_cache.records]
    check_roles(output, step_cache.records, outputs)
    if output.lowered.task != output.task:
        raise BuildFailure("lowered: BuildOutput.lowered is not b.lower(...) of BuildOutput.task", None)
    try:
        validate_lowered_task(output.lowered, factories=services.factories, sessions={})
    except (ValueError, NotImplementedError) as error:
        raise BuildFailure(f"lowered: {error}", None) from error
    try:
        validate_controls(output.task, output.controls)
    except ValueError as error:
        raise BuildFailure(f"controls: {error}", None) from error


def _output(cache: StepCache, record: StepRecord) -> object:
    """A role-checked step's decoded output; other steps are not decoded."""
    if record.role == StepRole.GRADER:
        return _GRADER.validate_json(cache.blobs.get(record.output))
    if record.role == StepRole.CONTROLS:
        return _CONTROLS.validate_json(cache.blobs.get(record.output))
    return None


def load_draft(directory: Path) -> TaskDraft:
    """Read a draft written by ``run_build``.

    Raises:
        ValueError: a draft file does not parse, or ``task.json`` differs from the lowered task.
    """
    lowered = LoweredTaskSpec.model_validate_json((directory / LOWERED_FILE).read_bytes())
    return TaskDraft(
        task=TaskSpec.model_validate_json((directory / TASK_FILE).read_bytes()),
        lowered=lowered,
        controls=_CONTROLS.validate_json((directory / CONTROLS_FILE).read_bytes()),
        provenance=_PROVENANCE.validate_json((directory / PROVENANCE_FILE).read_bytes()),
    )


_PROVENANCE: TypeAdapter[Provenance] = TypeAdapter(Provenance)
_GRADER: TypeAdapter[Grader] = TypeAdapter(Grader)
_CONTROLS: TypeAdapter[tuple[Control, ...]] = TypeAdapter(tuple[Control, ...])
