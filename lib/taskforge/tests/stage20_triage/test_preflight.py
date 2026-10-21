# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The default triage: the deterministic preflight accepts or rejects with the failing check as the reason."""

import asyncio
from dataclasses import replace

from taskforge.stage10_proposal.model import ResearchNeed
from taskforge.stage20_triage.checks import CheckStatus
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import TriageDecision, Verdict


def run(p) -> Verdict:
    return asyncio.run(preflight(p))


def test_a_well_formed_proposal_is_accepted_with_every_check_recorded(proposal):
    verdict = run(proposal)
    assert (verdict.decision, verdict.reasons, verdict.author_state) == (TriageDecision.ACCEPT, (), {})
    assert (verdict.proposal_id, verdict.proposal_digest) == (proposal.header.id, proposal.digest)
    assert {r.status for r in verdict.structural} == {CheckStatus.PASS}


def test_a_proposal_with_a_thin_section_is_rejected_with_the_failing_check(proposal, section_replaced):
    verdict = run(section_replaced(proposal, "Risks and null conditions", "None."))
    assert verdict.decision is TriageDecision.REJECT
    assert verdict.reasons == ("required_sections: '## Risks and null conditions' has 1 words, fewer than 40",)


def test_a_null_proposal_is_rejected_with_its_reason(proposal):
    header = replace(
        proposal.header,
        research_need=ResearchNeed.NOT_NEEDED,
        research=(),
        build=(),
        resources=(),
        null_reason="no gradable artifact exists here",
    )
    verdict = run(replace(proposal, header=header))
    assert verdict.decision is TriageDecision.REJECT
    assert verdict.reasons == ("null proposal: no gradable artifact exists here",)


def test_an_advisory_failure_does_not_reject(proposal, section_replaced):
    p = section_replaced(proposal, "Build plan", "Sessions follow the usual order with acceptance gates. " * 10)
    verdict = run(p)
    assert verdict.decision is TriageDecision.ACCEPT
    assert [r.name for r in verdict.structural if r.status is CheckStatus.FAIL] == ["resources_in_build_plan"]


def test_verdict_round_trips_through_json(proposal, section_replaced):
    verdict = run(section_replaced(proposal, "Risks and null conditions", "None."))
    assert Verdict.from_json(verdict.to_json()) == verdict


def test_a_verdict_carrying_author_state_round_trips_through_json(proposal):
    verdict = replace(run(proposal), author_state={"obligations": ["derive 42 in the grader"], "strict": True})
    assert Verdict.from_json(verdict.to_json()) == verdict
