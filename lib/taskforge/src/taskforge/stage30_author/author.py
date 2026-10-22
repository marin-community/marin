# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authors of build programs, the revisions they receive, and the program manifest.

A caller awaits an author for a program in round 0 with the proposal and ``None``. In each later
round the caller passes the ``Revision`` describing what went wrong in the previous one. Every call
also passes the author state triage's accept attached to the workpiece, a JSON mapping that is
empty unless a consumer's triage attached findings or obligations; the package's authors ignore it.
The author returns an ``AuthoredProgram``: the ``RLEnvBuildProgram`` to build next, with the Python
source it loaded the program from. The coordinator writes that source to ``PROGRAM_FILE``, beside
``PROGRAM_MANIFEST_FILE``, in a directory of the author call, and the build stage loads the program
back from the file through ``loader.load_program_source``. A revision carries no program source:
an author that revises its own program reads the ``PROGRAM_FILE`` it wrote. ``static_author``
turns a program written by hand into an author returning no source.

A ``MODIFY`` revision may carry a ``VerificationChange``, which swaps the task's verifier kind. The
caller passes ``change.revise(proposal)`` to the author and the build of that round and every later
one, so each reader of ``proposal.header.verification`` sees the revised kind.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
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

    Only a ``MODIFY`` revision may carry a ``verification_change``: a swap answers findings, so a
    build or author failure cannot escape a grader by changing its kind.
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
    because the coordinator then has no text to write.
    """

    program: RLEnvBuildProgram
    source: str | None


class Author(Protocol):
    async def __call__(
        self, proposal: TaskProposal, revision: Revision | None, author_state: Mapping[str, JsonValue]
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
    """Return an author that hands back ``program``, with no source, for every proposal, revision and state."""
    authored = AuthoredProgram(program, None)

    async def author(
        proposal: TaskProposal, revision: Revision | None, author_state: Mapping[str, JsonValue]
    ) -> AuthoredProgram:
        return authored

    return author
