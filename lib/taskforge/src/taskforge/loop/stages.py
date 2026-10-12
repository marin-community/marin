# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The stage functions a loop coordinates, as callable Protocols, and the ``Stages`` bundle a caller passes.

Each Protocol is the signature of a library function, so the library's functions satisfy them as
they are and a caller may call any stage directly, without a loop:

    =========  ===================================  ======================================
    stage      Protocol                             library function
    =========  ===================================  ======================================
    propose    ``Propose[IdeaT]``                   a ``ProposalSource``'s bound ``propose``
    triage     ``Triage``                           ``triage.verdict.accept``
    author     ``Author``                           ``builder.author.adopt(program)``
    build      ``Build``                            ``builder.run.run_build``
    validate   ``Validate``                         ``validate.run.run_validation``
    review     ``Review``                           ``review.rules.decide``
    =========  ===================================  ======================================

A caller passes its own function for any stage instead; the type checker holds it to the Protocol at
the ``Stages(...)`` call. Parameters are positional-only, so a function's parameter names are free.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from taskforge.builder.author import Revision
from taskforge.builder.run import TaskDraft
from taskforge.builder.step import Program
from taskforge.ledger.records import Ledger
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import RecordedModel
from taskforge.proposal.model import TaskProposal
from taskforge.proposal.source import ProposalBatch
from taskforge.review.decision import Decision
from taskforge.review.rules import BandRules
from taskforge.triage.verdict import TriageDecision
from taskforge.validate.calibration import CalibrationSummary
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.solver import ModelFactory, ValidationSite
from taskforge.validate.trials import EngineSettings


class Propose[IdeaT](Protocol):
    """``n`` proposal slots for one idea, calling ``model`` (recorded under the idea item, step ``propose``)."""

    async def __call__(self, idea: IdeaT, n: int, model: RecordedModel, /) -> ProposalBatch: ...


class Triage(Protocol):
    """Whether a proposal goes on to authoring."""

    def __call__(self, proposal: TaskProposal, /) -> TriageDecision: ...


class Author(Protocol):
    """The program for a proposal; ``revision`` says what the previous program failed at, ``None`` at first."""

    def __call__(self, proposal: TaskProposal, revision: Revision | None, /) -> Program: ...


class Build(Protocol):
    """Run ``program`` for ``proposal`` into a draft under ``round_dir``, memoized under ``cache``."""

    async def __call__(
        self,
        program: Program,
        proposal: TaskProposal,
        round_dir: Path,
        cache: Path,
        model: ModelClient,
        sampling: LLMPolicy,
        ledger: Ledger,
        round: int,  # noqa: A002 - matches LedgerEntry.round
        /,
    ) -> TaskDraft: ...


class Validate(Protocol):
    """One validation round of a draft, summarized."""

    async def __call__(
        self,
        draft: TaskDraft,
        policy: ValidationPolicy,
        site: ValidationSite,
        engine: EngineSettings,
        models: ModelFactory,
        /,
    ) -> CalibrationSummary: ...


class Review(Protocol):
    """The decision on one validation round's summary."""

    def __call__(self, summary: CalibrationSummary, rules: BandRules, /) -> Decision: ...


@dataclass(frozen=True)
class Stages[IdeaT]:
    """One function per stage, for ``loop.program.Loop`` to coordinate."""

    propose: Propose[IdeaT]
    triage: Triage
    author: Author
    build: Build
    validate: Validate
    review: Review
