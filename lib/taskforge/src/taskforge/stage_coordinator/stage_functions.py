# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The stage functions a ``StageCoordinator`` calls, as callable Protocols, and the ``StageFunctions`` bundle.

Each Protocol takes the signature of a library function, so a caller passes the library's functions
as they are and may also call any stage directly::

    stage         Protocol        library function
    ------------  --------------  -------------------------------------------
    triage        Triage          ``stage20_triage.preflight.preflight``
    repair        RepairProposal  a caller's function, for a triage that returns REPAIR
    author        Author          ``stage30_author.author.static_author(program)``
    load_program  LoadProgram     ``stage40_build_with_program.program_sdk.load_program``
    build         Build           ``stage40_build_with_program.run.run_build``
    validate      Validate        ``stage50_validate.round.validate_round``, closed over the host's settings
                                  and models
    review        Review          ``stage60_review.rules.default_review``

The Protocols defined here take positional-only parameters, so a caller's function may use its own
parameter names; ``Build`` passes ``run_build``'s keyword-only ``research`` and ``model_digest`` by
name. ``Author`` comes from ``stage30_author.author``, and the coordinator calls it with
``proposal`` and ``revision`` in that order. The triage stage calls ``repair`` only on a REPAIR
verdict within ``WorkpieceBudget.max_triage_repairs``, and triages the proposal it returns; under a
budget of zero it never calls it. The build stage calls ``load_program`` on the source the author
returned, read back from ``authored/<k>/program.py``. The validate stage passes the draft with the
round's ``ValidationContext``: the workpiece, the round, its directory, the proposal it built, the step
store, the MODIFY rounds left and the validation pass of the round it runs. A
caller's validate function closes over what the coordinator does not own, such as the host's
``ValidationSettings`` and the trial models::

    async def validate(draft: TaskDraft, context: ValidationContext) -> ValidationResult:
        return await validate_round(draft, context, settings, solver_model, controls_model, adversary_model)

The coordinator writes the ``ValidationResult`` returned by validate to ``validation.json``
(``write_validation``), the file's one writer. The review stage hands review the result read back from
that file (``load_validation``), so a resumed review receives the same value as a review in the first
process, with the round's ``ReviewContext``: the round, the MODIFY rounds left and the coordinator's
``BandRules``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage20_triage.verdict import Verdict
from taskforge.stage30_author.author import Author
from taskforge.stage30_author.program import RLEnvBuildProgram
from taskforge.stage40_build_with_program.run import ResearchRequirement, TaskDraft
from taskforge.stage40_build_with_program.services import RLEnvBuildServices
from taskforge.stage50_validate.round import ValidationContext, ValidationResult
from taskforge.stage60_review.decision import Decision
from taskforge.stage60_review.rules import ReviewContext


class Triage(Protocol):
    async def __call__(self, proposal: TaskProposal, /) -> Verdict: ...


class RepairProposal(Protocol):
    """Rewrite ``proposal`` to answer its REPAIR ``verdict``, for example by fixing a header defect.

    Raises:
        ProposalFormatError: no valid rewrite resulted.
    """

    async def __call__(self, proposal: TaskProposal, verdict: Verdict, /) -> TaskProposal: ...


class LoadProgram(Protocol):
    """Turn the Python source of a program into the program."""

    def __call__(self, source: str, /) -> RLEnvBuildProgram: ...


class Build(Protocol):
    async def __call__(
        self,
        program: RLEnvBuildProgram,
        proposal: TaskProposal,
        workpiece_id: str,
        round_dir: Path,
        store_root: str,
        model: ModelClient,
        sampling: LLMPolicy,
        services: RLEnvBuildServices,
        round: int,  # noqa: A002 - run_build's round number
        /,
        *,
        research: ResearchRequirement,
        model_digest: str,
    ) -> TaskDraft: ...


class Validate(Protocol):
    """Validate ``draft`` in the round ``context`` names and return the round's result for review."""

    async def __call__(self, draft: TaskDraft, context: ValidationContext, /) -> ValidationResult: ...


class Review(Protocol):
    """Decide on a validation round's ``result`` in its review ``context``."""

    async def __call__(self, result: ValidationResult, context: ReviewContext, /) -> Decision: ...


@dataclass(frozen=True)
class StageFunctions:
    triage: Triage
    repair: RepairProposal
    author: Author
    load_program: LoadProgram
    build: Build
    validate: Validate
    review: Review
