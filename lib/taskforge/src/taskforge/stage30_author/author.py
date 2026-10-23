# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authors of build programs, the revisions they receive, and the program manifest.

A caller awaits an author for a program with an ``AuthorContext``, the proposal, and ``None`` on
the workpiece's first call. The context names the workpiece, its round and its directory. In each
later call the caller passes the ``Revision`` describing what went wrong in the previous one. Every
call also passes the author state triage's accept attached to the workpiece, a JSON mapping that is
empty unless a consumer's triage attached findings or obligations; the package's authors ignore it.
The author returns an ``AuthoredProgram``: the ``RLEnvBuildProgram`` to build next, with the Python
source it loaded the program from. The coordinator writes that source to ``PROGRAM_FILE``, beside
``PROGRAM_MANIFEST_FILE``, in a directory of the author call, and the build stage loads the program
back from the file through ``loader.load_program_source``. A revision carries no program source:
an author that revises its own program reads the ``PROGRAM_FILE`` it wrote. ``static_author``
turns a program written by hand into an author returning no source. An author call that ends
without a program it can hand in raises ``AuthorFailure``, and the coordinator asks again with
the failure as an ``AUTHOR_FAILURE`` revision.

An author answering a ``MODIFY`` revision may swap the task's verifier kind, and returns the swap as
``AuthoredProgram.verification_change``. Such a call receives the proposal as declared, with the change
already in force, if any, on the revision; the author starts from ``change.revise(proposal)`` and may
keep, replace or undo the change. Every other call receives the proposal the build checks (the revised
one after a swap) and returns no change, so a build or author failure cannot change the kind. The caller
builds that round and every later one with ``change.revise(proposal)`` and the change, until a later
``MODIFY`` answer replaces or undoes it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from taskforge.stage10_proposal.model import OFFERED_VERIFICATIONS, TaskProposal, Verification
from taskforge.stage30_author.program import RLEnvBuildProgram

PROGRAM_MANIFEST_FILE = "program.json"
PROGRAM_FILE = "program.py"


class RevisionKind(StrEnum):
    """The reason for authoring a workpiece again."""

    NONE = "none"
    BUILD_FAILURE = "build_failure"
    AUTHOR_FAILURE = "author_failure"
    """The previous author call returned no valid program, as when the output limit cut its reply."""
    MODIFY = "modify"
    """Review or an adversary sends its findings back; the only kind that may carry a ``VerificationChange``."""


@dataclass(frozen=True)
class VerificationChange:
    """A swap of the task's verifier kind, made by a revision that answers findings.

    Attributes:
        declared: The kind the proposal declared.
        revised: The kind every later round builds, one of ``OFFERED_VERIFICATIONS`` other than ``declared``.
        reason: Why the kind changes, shown to the author.
    """

    declared: Verification
    revised: Verification
    reason: str

    def __post_init__(self) -> None:
        if self.revised is self.declared:
            raise ValueError(f"a verification change swaps the kind, but both kinds are {self.declared}")
        if self.revised not in OFFERED_VERIFICATIONS:
            raise ValueError(
                f"a verification change revises to one of {[str(kind) for kind in OFFERED_VERIFICATIONS]}, "
                f"got {self.revised}"
            )
        if not self.reason.strip():
            raise ValueError("a verification change states its reason")

    def revise(self, proposal: TaskProposal) -> TaskProposal:
        """``proposal`` verified by the ``revised`` kind, with no composite parts.

        Raises:
            ValueError: ``proposal`` does not declare the ``declared`` kind.
        """
        if proposal.header.verification is not self.declared:
            raise ValueError(
                f"the change revises a {self.declared} proposal, but the proposal declares "
                f"{proposal.header.verification}"
            )
        return replace(proposal, header=replace(proposal.header, verification=self.revised, composite=()))


@dataclass(frozen=True)
class Revision:
    """The change requested of the next round's author, with ``detail`` explaining it.

    Only a ``MODIFY`` revision may carry a ``verification_change``, the change in force on the declared
    proposal, which the author starts from: a swap answers findings, so a build or author failure cannot
    escape a grader by changing its kind.
    """

    kind: RevisionKind
    detail: str
    verification_change: VerificationChange | None = None

    def __post_init__(self) -> None:
        if self.verification_change is not None and self.kind is not RevisionKind.MODIFY:
            raise ValueError(f"only a {RevisionKind.MODIFY} revision may change the verification, not {self.kind}")


@dataclass(frozen=True)
class AuthoredProgram:
    """A program from an author, with ``source`` holding the Python text defining it.

    ``source`` is ``None`` for a program defined in Python, as ``static_author`` hands back,
    because the coordinator then has no text to write. ``verification_change`` is the swap of the
    verifier kind the program is built under, relative to the proposal the author received, or ``None``
    for that proposal's own kind; only an author answering a ``MODIFY`` revision returns one.
    """

    program: RLEnvBuildProgram
    source: str | None
    verification_change: VerificationChange | None


class AuthorFailure(Exception):
    """The author call ended without a program to hand in; the next call revises with this failure.

    The coordinator records ``AUTHOR_FAILED`` and spends a build revision on it, as on a
    ``StructuredOutputError``, so an author raises it for a failure another call may repair, as
    when its sessions ran out without a submission.
    """


@dataclass(frozen=True)
class AuthorContext:
    """The workpiece an author call writes a program for.

    Attributes:
        workpiece_id: The workpiece's id.
        round: The workpiece's round, which counts the review ``Modify`` decisions it followed.
        workpiece_dir: The workpiece's directory. The coordinator's files in it are ``events.json``,
            ``repaired/`` and, in ``rounds/<n>/``, ``authored/``, ``draft/``, ``build-failures/``,
            ``validation.json`` and ``decision.json``. An author keeps its own files in this
            directory under any other name.
    """

    workpiece_id: str
    round: int
    workpiece_dir: Path


class Author(Protocol):
    async def __call__(
        self,
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
        author_state: Mapping[str, JsonValue],
    ) -> AuthoredProgram: ...


def program_manifest(program: RLEnvBuildProgram) -> dict[str, object]:
    """Lay out ``program`` as the JSON manifest a caller writes beside the program's source, once per
    author call: the digest, the sink, the run order, and each step's inputs and source digest.

    The ``"steps"`` mapping lists the sink after the other steps.
    """
    return {
        "digest": program.digest,
        "sink": program.sink.name,
        "order": list(program.order),
        "steps": {
            step.name: {"inputs": list(step.inputs), "source_digest": step.source_digest}
            for step in (*program.steps.values(), program.sink)
        },
    }


def static_author(program: RLEnvBuildProgram) -> Author:
    """Return an author that hands back ``program``, with no source, for every call.

    It keeps the revision's verification change unchanged and never chooses one.
    """

    async def author(
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
        author_state: Mapping[str, JsonValue],
    ) -> AuthoredProgram:
        return AuthoredProgram(program, None, None if revision is None else revision.verification_change)

    return author
