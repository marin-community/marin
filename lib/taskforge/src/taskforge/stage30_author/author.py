# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authors of build programs, the revisions they receive, and the program manifest.

A caller asks an author for a program in round 0 with the proposal and ``None``. In each later
round the caller passes the ``Revision`` describing what went wrong in the previous one, and the
author returns the ``RLEnvBuildProgram`` to build next. ``static_author`` turns a program written
by hand into an author.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from taskforge.stage30_author.program import RLEnvBuildProgram

PROGRAM_MANIFEST_FILE = "program.json"


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


class Author(Protocol):
    def __call__(self, proposal: object, revision: Revision | None) -> RLEnvBuildProgram: ...


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

    def author(proposal: object, revision: Revision | None) -> RLEnvBuildProgram:
        return program

    return author
