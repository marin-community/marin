# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authoring an item's builder program: the revision an author is asked for, and the manifest kept
in the item's round directory.

An author turns a proposal, and the revision the previous round asks for (``None`` in round 0),
into a ``Program``. ``adopt`` is the author of a caller that writes its program itself.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from taskforge.builder.step import Program
from taskforge.proposal.model import TaskProposal

PROGRAM_MANIFEST_FILE = "program.json"


class RevisionKind(StrEnum):
    """Why an item is authored again: not at all, its build failed, or review asked for a modification."""

    NONE = "none"
    BUILD_FAILURE = "build_failure"
    MODIFY = "modify"


@dataclass(frozen=True)
class Revision:
    """What the next round's author is asked to change, and why."""

    kind: RevisionKind
    detail: str


def program_manifest(program: Program) -> dict[str, object]:
    """``program``'s shape for its round directory: digest, sink, order, and each step's inputs and source digest."""
    return {
        "digest": program.digest,
        "sink": program.sink,
        "order": list(program.order),
        "steps": {
            name: {"inputs": list(step.inputs), "source_digest": step.source_digest}
            for name, step in program.steps.items()
        },
    }


def adopt(program: Program) -> Callable[[TaskProposal, Revision | None], Program]:
    """The author that returns ``program`` for every proposal and ignores the revision."""

    def author(proposal: TaskProposal, revision: Revision | None) -> Program:
        return program

    return author
