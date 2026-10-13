# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The stage functions a ``StageCoordinator`` calls, as callable Protocols, and the ``StageFunctions`` bundle.

Each Protocol takes the signature of a library function, so a caller passes the library's functions
as they are and may also call any stage directly::

    stage     Protocol    library function
    --------  ----------  ----------------------------------------------
    triage    Triage      ``stage20_triage.verdict.accept``
    author    Author      ``stage30_author.author.static_author(program)``
    build     Build       ``stage40_build.run.run_build``
    validate  Validate    a caller's function
    review    Review      a caller's function

The Protocols defined here take positional-only parameters, so a caller's function may use its own
parameter names. ``Author`` comes from ``stage30_author.author``, and the coordinator calls it with
``proposal`` and ``revision`` in that order. The coordinator writes the summary returned by
validate to ``validation.json`` and hands review the JSON read back from that file, so a resumed
review receives the same value as a review in the first process.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage20_triage.verdict import TriageDecision
from taskforge.stage30_author.author import Author
from taskforge.stage30_author.program import RLEnvBuildProgram
from taskforge.stage40_build.run import TaskDraft
from taskforge.stage60_review.decision import Decision


class Triage(Protocol):
    def __call__(self, proposal: object, /) -> TriageDecision: ...


class Build(Protocol):
    async def __call__(
        self,
        program: RLEnvBuildProgram,
        proposal: object,
        item_id: str,
        round_dir: Path,
        store_root: str,
        model: ModelClient,
        sampling: LLMPolicy,
        model_digest: str,
        round: int,  # noqa: A002 - run_build's round number
        /,
    ) -> TaskDraft: ...


class Validate(Protocol):
    """Validate ``draft`` and return a JSON-serialisable summary for review."""

    async def __call__(self, draft: TaskDraft, /) -> object: ...


class Review(Protocol):
    """Decide on the summary of a validation round."""

    def __call__(self, summary: object, /) -> Decision: ...


@dataclass(frozen=True)
class StageFunctions:
    triage: Triage
    author: Author
    build: Build
    validate: Validate
    review: Review
