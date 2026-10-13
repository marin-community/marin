# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authors of build programs, the revisions they receive, and the program manifest.

A caller awaits an author for a program in round 0 with the proposal and ``None``. In each later
round the caller passes the ``Revision`` describing what went wrong in the previous one, and the
author returns the ``AuthoredProgram`` to build next: the program and, when the author wrote one,
its source. The author stage stores that source as ``PROGRAM_FILE`` beside ``PROGRAM_MANIFEST_FILE``,
and the build stage loads it back through ``loader.load_program_source``. A revision carries no
program source: an author that revises its own program reads the ``PROGRAM_FILE`` it wrote.
``static_author`` turns a program written by hand into an author.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.program import RLEnvBuildProgram

PROGRAM_MANIFEST_FILE = "program.json"
PROGRAM_FILE = "program.py"


class RevisionKind(StrEnum):
    """The reason for authoring an item again."""

    NONE = "none"
    BUILD_FAILURE = "build_failure"
    MODIFY = "modify"


@dataclass(frozen=True)
class Revision:
    """The change requested of the next round's author, with ``detail`` explaining it."""

    kind: RevisionKind
    detail: str


@dataclass(frozen=True)
class AuthoredProgram:
    """A program an author returned, with the module source it was loaded from, or ``None`` for a
    program written in Python by hand."""

    program: RLEnvBuildProgram
    source: str | None


class Author(Protocol):
    async def __call__(self, proposal: TaskProposal, revision: Revision | None) -> AuthoredProgram: ...


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
    """Return an author that hands back ``program`` for every proposal and revision."""

    async def author(proposal: TaskProposal, revision: Revision | None) -> AuthoredProgram:
        return AuthoredProgram(program, None)

    return author
