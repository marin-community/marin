# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A revision that swaps the verifier kind: the change, the revised proposal, and which revisions may carry it."""

from dataclasses import replace

import pytest

from taskforge.stage10_proposal.model import CompositePart, TaskProposal, Verification, parse, render
from taskforge.stage30_author.author import Revision, RevisionKind, VerificationChange
from tests.stage40_build_with_program.conftest import PROPOSAL

REASON = "the decision model accepted a hedged answer"
PARTS = (CompositePart(Verification.DECISION_MODEL, 0.5), CompositePart(Verification.CODE, 0.5))


def declaring(verification: Verification, composite: tuple[CompositePart, ...] = ()) -> TaskProposal:
    proposal = parse(PROPOSAL)
    return replace(proposal, header=replace(proposal.header, verification=verification, composite=composite))


@pytest.mark.parametrize(
    ("revised", "reason"),
    [(Verification.DECISION_MODEL, REASON), (Verification.COMPOSITE, REASON), (Verification.CODE, "  ")],
    ids=["same-kind", "composite", "blank-reason"],
)
def test_a_verification_change_swaps_to_another_offered_kind_for_a_stated_reason(revised, reason):
    with pytest.raises(ValueError):
        VerificationChange(Verification.DECISION_MODEL, revised, reason)


def test_revise_refuses_a_proposal_of_another_kind():
    change = VerificationChange(Verification.DECISION_MODEL, Verification.CODE, REASON)

    with pytest.raises(ValueError):
        change.revise(declaring(Verification.ANSWER_IN_BOX))


@pytest.mark.parametrize(
    ("declared", "parts"),
    [(Verification.DECISION_MODEL, ()), (Verification.COMPOSITE, PARTS)],
    ids=["single-kind", "composite"],
)
def test_the_revised_proposal_declares_the_revised_kind_and_renders_and_parses_back(declared, parts):
    proposal = declaring(declared, parts)

    revised = VerificationChange(declared, Verification.CODE, REASON).revise(proposal)

    assert (revised.header.verification, revised.header.composite) == (Verification.CODE, ())
    assert replace(revised, header=proposal.header) == proposal
    assert parse(render(revised)) == revised


@pytest.mark.parametrize("kind", [RevisionKind.BUILD_FAILURE, RevisionKind.AUTHOR_FAILURE])
def test_only_a_modify_revision_may_change_the_verification(kind):
    change = VerificationChange(Verification.DECISION_MODEL, Verification.CODE, REASON)

    with pytest.raises(ValueError):
        Revision(kind, "grader: the reference scored 0.0", change)
    assert Revision(RevisionKind.MODIFY, "the grader accepted a hedge", change).verification_change == change
