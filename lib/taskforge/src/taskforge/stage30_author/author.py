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
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from taskforge.stage10_proposal.model import TaskProposal
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
    """Return an author that hands back ``program``, with no source, for every call."""
    authored = AuthoredProgram(program, None)

    async def author(
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
        author_state: Mapping[str, JsonValue],
    ) -> AuthoredProgram:
        return authored

    return author
