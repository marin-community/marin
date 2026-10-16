# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authors of build programs, the revisions they receive, and the program manifest.

A caller asks an author for a program in round 0 with the proposal and ``None``. In each later
round the caller passes the ``Revision`` describing what went wrong in the previous one. The
author returns an ``AuthoredProgram``: the ``RLEnvBuildProgram`` to build next, with the Python
source it loaded the program from. The coordinator writes that source to ``PROGRAM_FILE`` in a
directory of the author call, and the build stage loads the program back from the file.
``static_author`` turns a program written by hand into an author returning no source.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from taskforge.stage30_author.program import RLEnvBuildProgram

PROGRAM_MANIFEST_FILE = "program.json"
PROGRAM_FILE = "program.py"


class RevisionKind(StrEnum):
    """The reason for authoring an item again."""

    NONE = "none"
    BUILD_FAILURE = "build_failure"
    AUTHOR_FAILURE = "author_failure"
    """The previous author call returned no valid program, as when the output limit cut its reply."""
    MODIFY = "modify"


@dataclass(frozen=True)
class Revision:
    """The change requested of the next round's author, with ``detail`` explaining it."""

    kind: RevisionKind
    detail: str


@dataclass(frozen=True)
class AuthoredProgram:
    """A program from an author, with ``source`` holding the Python text defining it.

    ``source`` is ``None`` for a program defined in Python, as ``static_author`` hands back,
    because the coordinator then has no text to write.
    """

    program: RLEnvBuildProgram
    source: str | None


class Author(Protocol):
    async def __call__(self, proposal: object, revision: Revision | None) -> AuthoredProgram: ...


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
    """Return an author that hands back ``program``, with no source, for every proposal and revision."""
    authored = AuthoredProgram(program, None)

    async def author(proposal: object, revision: Revision | None) -> AuthoredProgram:
        return authored

    return author
