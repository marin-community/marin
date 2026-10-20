# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The default triage: the deterministic preflight accepts or rejects with the failing check as the reason."""

import asyncio
from dataclasses import replace

from taskforge.stage10_proposal.model import Environment, ResearchNeed, SolverMachine, Verification
from taskforge.stage20_triage.checks import ALL_COMBINATIONS, CheckContext, CheckStatus
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import TriageDecision, Verdict


def run(p, **kwargs) -> Verdict:
    return asyncio.run(preflight(p, **kwargs))


def test_a_well_formed_proposal_is_accepted_with_every_check_recorded(proposal):
    verdict = run(proposal)
    assert (verdict.decision, verdict.reasons, verdict.author_state) == (TriageDecision.ACCEPT, (), {})
    assert (verdict.proposal_id, verdict.proposal_digest) == (proposal.header.id, proposal.digest)
    assert {r.status for r in verdict.structural} == {CheckStatus.PASS, CheckStatus.SKIP}


def test_a_malformed_proposal_is_rejected_with_the_failing_check(proposal):
    verdict = run(replace(proposal, header=replace(proposal.header, resources=("../secrets.txt",))))
    assert verdict.decision is TriageDecision.REJECT
    assert verdict.reasons == (
        "header_schema_valid: resources must be relative paths inside the build root: ['../secrets.txt']",
    )


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


def test_a_solver_machine_the_environment_cannot_take_is_rejected(proposal):
    header = replace(proposal.header, environment=Environment.SHELLSIM, solver_machine=SolverMachine.NONE)
    verdict = run(replace(proposal, header=header))
    assert verdict.decision is TriageDecision.REJECT
    [machine] = [r for r in verdict.reasons if r.startswith("solver_machine_fits: ")]
    assert "only a reasoning proposal may declare none" in machine


def test_an_advisory_failure_does_not_reject(proposal, section_replaced):
    p = section_replaced(proposal, "Build plan", "Sessions follow the usual order with acceptance gates. " * 10)
    verdict = run(p)
    assert verdict.decision is TriageDecision.ACCEPT
    assert [r.name for r in verdict.structural if r.status is CheckStatus.FAIL] == ["resources_in_build_plan"]


def test_a_narrower_context_rejects_a_pairing_it_leaves_out(proposal):
    ctx = CheckContext(allowed_combinations=ALL_COMBINATIONS - {(Environment.REASONING, Verification.CODE)})
    verdict = run(proposal, ctx=ctx)
    assert verdict.reasons == ("combination_allowed: reasoning x code is not an allowed combination",)


def test_verdict_round_trips_through_json(proposal):
    verdict = run(replace(proposal, header=replace(proposal.header, resources=("../secrets.txt",))))
    assert Verdict.from_json(verdict.to_json()) == verdict


def test_a_verdict_carrying_author_state_round_trips_through_json(proposal):
    verdict = replace(run(proposal), author_state={"obligations": ["derive 42 in the grader"], "strict": True})
    assert Verdict.from_json(verdict.to_json()) == verdict
