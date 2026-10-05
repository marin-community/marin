# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a builder program to a ``TaskDraft``: the TaskSpec, its controls, and provenance.

``run_build`` enforces the library rules a program cannot opt out of: the task's verifier (each
stage's verifier for a staged task) is the output of a GRADER step, the controls are the output of
a CONTROLS step, the two roles are separate steps, and the controls are a complete set for the
task (``validate_controls``). Controls are not replayed here; ``validate`` does that. A build whose
controls include a candidate ``b.try_grader`` graded (other than a grader's reference answer or the
empty answer) fails, so a program cannot fit its controls to its grader. The draft is written to
``<item_dir>/draft/``.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from taskcompendium.environment import EnvironmentFile
from taskcompendium.models import TaskSpec, TextMessage, VerifierSpec

from taskforge.build.sdk import (
    GRADED_RESOURCE_PREFIX,
    Build,
    BuildFailure,
    BuildOutput,
    BuildServices,
    GradedCandidate,
    Grader,
)
from taskforge.build.step import SDK_VERSION, CacheStatus, Resource, StepCache, StepRecord, StepRole
from taskforge.canonical import write_atomic
from taskforge.ledger.records import EntryKind, span
from taskforge.proposal.model import TaskProposal
from taskforge.spec.controls import Control, Workspace, controls_json, validate_controls

DRAFT_DIR = "draft"
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
    task: TaskSpec
    controls: tuple[Control, ...]
    provenance: Provenance


class Program(Protocol):
    """A compiled builder program; ``taskforge.build.author.BuildProgram`` satisfies this."""

    @property
    def digest(self) -> str: ...

    @property
    def build(self) -> Callable[[Build], Awaitable[BuildOutput]]: ...


def item_id_for(proposal: TaskProposal) -> str:
    """A file-safe item id from the proposal id (``d01.x/3`` becomes ``d01.x--3``)."""
    return proposal.header.id.replace("/", "--")


def _task_graders(task: TaskSpec) -> tuple[VerifierSpec, ...]:
    return tuple(stage.verifier for stage in task.stages) or (task.verifier,)


def check_roles(output: BuildOutput, records: Sequence[StepRecord], outputs: Sequence[object]) -> None:
    """Raise ``BuildFailure`` unless the verifier and controls came from separate GRADER and CONTROLS steps."""
    graders = [value for record, value in zip(records, outputs, strict=True) if record.role == StepRole.GRADER]
    controls = [value for record, value in zip(records, outputs, strict=True) if record.role == StepRole.CONTROLS]
    if not graders or not controls:
        raise BuildFailure("a build needs a GRADER step and a separate CONTROLS step", None)
    verifiers = [grader.verifier for grader in graders if isinstance(grader, Grader)]
    for verifier in _task_graders(output.task):
        if verifier not in verifiers:
            raise BuildFailure(f"the task's {verifier.kind} verifier is not the output of a GRADER step", None)
    if not any(tuple(value) == output.controls for value in controls if isinstance(value, tuple)):
        raise BuildFailure("the task's controls are not the output of a CONTROLS step", None)


def _file_set(files: Sequence[EnvironmentFile]) -> frozenset[tuple[str, bytes]]:
    return frozenset((f.path, f.content) for f in files)


def _is_graded(control: Control, graded: Sequence[GradedCandidate], exempt_replies: frozenset[str]) -> bool:
    """Whether ``control`` is one of the ``graded`` candidates.

    A workspace control matches a candidate with the same files. A reply-only transcript matches
    a candidate with the same reply and no files. A transcript with shell calls writes files no
    candidate records, so it matches on its final reply alone, unless that reply is exempt.
    """
    if isinstance(control.payload, Workspace):
        files = _file_set(control.payload.files)
        return bool(files) and any(_file_set(c.files) == files for c in graded)
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


def _write_draft(directory: Path, draft: TaskDraft) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(directory / "task.json", draft.task.model_dump_json(indent=2).encode())
    write_atomic(directory / "controls.json", controls_json(draft.controls))
    write_atomic(directory / "provenance.json", _PROVENANCE.dump_json(draft.provenance, indent=2))


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
        outputs = [_output(step_cache, record) for record in step_cache.records]
        check_roles(output, step_cache.records, outputs)
        graded = [
            _GRADED.validate_json(b.read(r.blob)) for r in b.resources if r.name.startswith(GRADED_RESOURCE_PREFIX)
        ]
        check_controls_not_graded(output.controls, [o for o in outputs if isinstance(o, Grader)], graded)
        try:
            validate_controls(output.task, output.controls)
        except ValueError as error:
            raise BuildFailure(f"controls: {error}", None) from error
        draft = TaskDraft(
            task=output.task,
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
    """Read a draft written by ``run_build``."""
    return TaskDraft(
        task=TaskSpec.model_validate_json((directory / "task.json").read_bytes()),
        controls=_CONTROLS.validate_json((directory / "controls.json").read_bytes()),
        provenance=_PROVENANCE.validate_json((directory / "provenance.json").read_bytes()),
    )


_PROVENANCE: TypeAdapter[Provenance] = TypeAdapter(Provenance)
_GRADER: TypeAdapter[Grader] = TypeAdapter(Grader)
_CONTROLS: TypeAdapter[tuple[Control, ...]] = TypeAdapter(tuple[Control, ...])
_GRADED: TypeAdapter[GradedCandidate] = TypeAdapter(GradedCandidate)
