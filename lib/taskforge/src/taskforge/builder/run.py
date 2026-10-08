# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a builder program to a ``TaskDraft``: the TaskSpec, its lowered form, its controls, and
provenance.

``run_build`` enforces the library rules a program cannot opt out of: the task's grader is the
output of a GRADER step, the controls are the output of a CONTROLS step, the two roles are separate
steps, the lowered spec carries exactly the task and passes RolloutEngine's
``validate_lowered_task`` on this host's factories, the submission convention can carry the task's
answer (``submission_compatibility``), and the controls are a complete set for the task
(``validate_controls``). Controls are not replayed here; ``validate`` does that. A build whose
controls include a candidate ``b.try_grader`` graded (other than a grader's reference answer or the
empty answer) fails, so a program cannot fit its controls to its grader. The draft is written to
``<item_dir>/draft/``.

A draft is bound to the host it was built on: ``lowered.json`` names that host's machine backends,
and validation evidence is keyed by the lowered spec.

A host failure during the build (``taskforge.builder.infrastructure``) raises
``BuildInfrastructureFailure`` rather than ``BuildFailure``, even when the program wrapped it.
"""

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from rolloutengine.lowering import validate_lowered_task
from rolloutengine.spec import LoweredTaskSpec
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.submission import (
    AnswerCall,
    FinalAction,
    JsonAnswer,
    JsonValueAnswer,
    PlainText,
    SubmissionConvention,
    submission_compatibility,
)

from taskforge.atomic_file import write_atomic
from taskforge.builder.infrastructure import BuildInfrastructureFailure, host_checked_factories, infrastructure_failure
from taskforge.builder.sdk import (
    GRADED_RESOURCE_PREFIX,
    Build,
    BuildFailure,
    BuildOutput,
    BuildServices,
    GradedCandidate,
    Grader,
    file_set,
)
from taskforge.builder.step import SDK_VERSION, CacheStatus, Resource, StepCache, StepRecord, StepRole
from taskforge.content_hash import pretty_json
from taskforge.ledger.records import EntryKind, span
from taskforge.proposal.model import TaskProposal
from taskforge.spec.controls import Control, Workspace, controls_json, validate_controls
from taskforge.spec.draft import MACHINE_ANSWER_TYPES

DRAFT_DIR = "draft"
TASK_FILE = "task.json"
LOWERED_FILE = "lowered.json"
CONVENTION_FILE = "convention.json"
CONTROLS_FILE = "controls.json"
PROVENANCE_FILE = "provenance.json"
SCRATCH_DIR = "scratch"
CONVENTION_TYPES: dict[str, type[SubmissionConvention]] = {
    convention.__name__: convention for convention in (PlainText, JsonAnswer, JsonValueAnswer, AnswerCall, FinalAction)
}
"""The submission conventions a draft can record, by class name (``convention.json``'s ``type``)."""


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
    """A built task. ``lowered`` (whose ``task`` is ``task``) and ``convention`` are what validation
    runs it with."""

    task: TaskSpec
    lowered: LoweredTaskSpec
    convention: SubmissionConvention
    controls: tuple[Control, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.lowered.task != self.task:
            raise ValueError(f"Draft {self.task.id!r}: the lowered spec carries a different task")


class Program(Protocol):
    """A compiled builder program; ``taskforge.builder.author.BuildProgram`` satisfies this."""

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
        package.verifier == task.verifier and file_set(package.resources) == file_set(task.resources.verifier)
        for package in packages
    ):
        raise BuildFailure(f"the task's {task.verifier.kind} grader is not the output of a GRADER step", None)
    if not any(tuple(value) == output.controls for value in controls if isinstance(value, tuple)):
        raise BuildFailure("the task's controls are not the output of a CONTROLS step", None)


def _is_graded(control: Control, graded: Sequence[GradedCandidate], exempt_replies: frozenset[str]) -> bool:
    """Whether ``control`` is one of the ``graded`` candidates.

    A workspace control matches a candidate with the same files. A reply-only transcript matches
    a candidate with the same reply and no files. A transcript with shell calls writes files no
    candidate records, so it matches on its final reply alone, unless that reply is exempt.
    """
    if isinstance(control.payload, Workspace):
        files = file_set(control.payload.files)
        return bool(files) and any(file_set(c.files) == files for c in graded)
    final = control.payload.turns[-1]
    if not isinstance(final, TextMessage):
        return False
    if control.payload.calls():
        return final.content not in exempt_replies and any(c.reply == final.content for c in graded)
    return any(c.reply == final.content and not c.files for c in graded)


def check_controls_not_graded(
    controls: Sequence[Control], graders: Sequence[Grader], graded: Sequence[GradedCandidate]
) -> None:
    """Raise ``BuildFailure`` when a control is a candidate the build graded with ``b.try_grader``.

    Each grader's reference answer and the empty answer are exempt: prototyping a grader on them
    is required, and they are also the usual positive and malformed controls.
    """
    exempt = {GradedCandidate(reply="")} | {
        GradedCandidate(reply=g.reference_reply, files=tuple(sorted(g.reference_files, key=lambda f: f.path)))
        for g in graders
    }
    candidates = [c for c in graded if c not in exempt]
    exempt_replies = frozenset({"", *(g.reference_reply for g in graders)})
    replayed = [control.id for control in controls if _is_graded(control, candidates, exempt_replies)]
    if replayed:
        raise BuildFailure(
            f"controls {replayed} are candidates b.try_grader graded during the build. Controls are written, "
            "not graded: prototype the grader on other candidates; validate replays the controls",
            None,
        )


def check_convention(task: TaskSpec, convention: SubmissionConvention) -> None:
    """Raise ``BuildFailure`` unless ``convention`` can carry ``task``'s answer.

    A task whose answer is the machine state submits nothing through a convention, so any
    recordable convention fits it.
    """
    if type(convention).__name__ not in CONVENTION_TYPES:
        raise BuildFailure(f"convention: {type(convention).__name__} is not one of {sorted(CONVENTION_TYPES)}", None)
    if task.answer_type in MACHINE_ANSWER_TYPES:
        return
    compatibility = submission_compatibility(task, convention)
    if not compatibility.compatible:
        raise BuildFailure(f"convention {convention.id!r}: {'; '.join(compatibility.reasons)}", None)


def convention_json(convention: SubmissionConvention) -> bytes:
    """``convention`` with its class name, as ``load_convention`` reads it."""
    return pretty_json({"type": type(convention).__name__, "convention": convention}).encode()


def load_convention(content: bytes) -> SubmissionConvention:
    record = json.loads(content)
    return CONVENTION_TYPES[record["type"]].model_validate(record["convention"])


def _write_draft(directory: Path, draft: TaskDraft) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(directory / TASK_FILE, draft.task.model_dump_json(indent=2).encode())
    write_atomic(directory / LOWERED_FILE, draft.lowered.model_dump_json(indent=2).encode())
    write_atomic(directory / CONVENTION_FILE, convention_json(draft.convention))
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
        services: Model, machines, ledger, and web tools.
        invalidate: Step names to recompute even when their key has a record.
        round: The build round, for the ledger.

    Raises:
        BuildFailure: the program failed a check, or its output breaks a library rule.
        BuildInfrastructureFailure: the host failed a machine the build used; anywhere in the cause
            chain of what the program raised, it wins over the program's own error.
    """
    item_id = item_id_for(proposal)
    step_cache = StepCache(root=cache, item_id=item_id, invalidated=frozenset(invalidate))
    scratch = item_dir / SCRATCH_DIR
    scratch.mkdir(parents=True, exist_ok=True)
    b = Build(proposal, item_id, services, step_cache, scratch, round)
    with span(services.ledger, EntryKind.STAGE, item_id=item_id, round=round, step="build") as fields:
        fields.attrs["program"] = program.digest
        try:
            output = await program.build(b)
        except Exception as error:
            failure = infrastructure_failure(error)
            if failure is None:
                raise
            fields.attrs["infrastructure"] = failure.cause
            if isinstance(error, BuildInfrastructureFailure):
                raise
            raise failure from error
        if not isinstance(output, BuildOutput):
            raise BuildFailure(f"build(b) returned {type(output).__name__}, not BuildOutput", None)
        outputs = [_output(step_cache, record) for record in step_cache.records]
        check_roles(output, step_cache.records, outputs)
        graded = [
            _GRADED.validate_json(b.read(r.blob)) for r in b.resources if r.name.startswith(GRADED_RESOURCE_PREFIX)
        ]
        check_controls_not_graded(output.controls, [o for o in outputs if isinstance(o, Grader)], graded)
        if output.lowered.task != output.task:
            raise BuildFailure("lowered: BuildOutput.lowered is not b.lower(...) of BuildOutput.task", None)
        try:
            validate_lowered_task(
                output.lowered, factories=host_checked_factories(services.factories, services.host), sessions={}
            )
        except (ValueError, NotImplementedError) as error:
            raise BuildFailure(f"lowered: {error}", None) from error
        check_convention(output.task, output.convention)
        try:
            validate_controls(output.task, output.controls)
        except ValueError as error:
            raise BuildFailure(f"controls: {error}", None) from error
        draft = TaskDraft(
            task=output.task,
            lowered=output.lowered,
            convention=output.convention,
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
        convention=load_convention((directory / CONVENTION_FILE).read_bytes()),
        controls=_CONTROLS.validate_json((directory / CONTROLS_FILE).read_bytes()),
        provenance=_PROVENANCE.validate_json((directory / PROVENANCE_FILE).read_bytes()),
    )


_PROVENANCE: TypeAdapter[Provenance] = TypeAdapter(Provenance)
_GRADER: TypeAdapter[Grader] = TypeAdapter(Grader)
_CONTROLS: TypeAdapter[tuple[Control, ...]] = TypeAdapter(tuple[Control, ...])
_GRADED: TypeAdapter[GradedCandidate] = TypeAdapter(GradedCandidate)
