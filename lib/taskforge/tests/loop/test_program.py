# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The item program end to end on ShellSim: the real author and the real adversary agent loop against a
scripted router, real builds, real validation through RolloutEngine and the task's real verifier, and fakes
only at the source, the rubric and the solver model."""

import asyncio
import json
from collections import Counter
from dataclasses import replace

import pytest
from pydantic import TypeAdapter
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Machine, MachineSpec
from taskcompendium.environment import EnvironmentKind

from taskforge.build.infrastructure import InfrastructureCause
from taskforge.build.run import item_id_for
from taskforge.canonical import sha256_hex
from taskforge.ledger.records import EntryKind
from taskforge.llm.client import Completion
from taskforge.loop.events import (
    EventKind,
    Phase,
    ProposalOrigin,
    RevisionKind,
    Terminal,
    build_host_failures,
    derive_state,
)
from taskforge.loop.program import NOOP_FAILURE, idea_item_id, run_idea, run_item
from taskforge.proposal.model import render
from taskforge.review.decision import DECISION_FILE, Accept, BandOutcome, Reject, RejectKind, Repair, load_decision
from taskforge.review.rules import STAGED_BRIEF, BandChoice, BandRule, BandRules
from taskforge.triage.verdict import TriageDecision
from taskforge.validate.adversary import CONTEXT_HEADER, SUBMIT_TOOL_NAME, AdversaryRole
from taskforge.validate.attempts import load_adversary_attempt, trial_files
from taskforge.validate.calibration import FindingKind
from taskforge.validate.outcome import TrialKind

IDEA = "d00.arithmetic.products"
CORRECT = "ANSWER = 42"
NO_ANSWER = "I could not work it out."
READ = ("shell", "cat /workspace/question.txt")
TOO_EASY_ACCEPTED = BandRules(BandRule(1, BandChoice.ACCEPT), BandRule(1, BandChoice.REJECT))
COMPLETIONS = TypeAdapter(tuple[Completion, ...])


def kinds(loop, item_id: str) -> list[str]:
    return [entry.step for entry in loop.entries(item_id) if entry.kind is EntryKind.EVENT]


def events_of(loop, item_id: str, kind: EventKind):
    return [e for e in loop.entries(item_id) if e.kind is EntryKind.EVENT and e.step == kind]


def evidence_dir(loop, item_id: str, round: int):  # noqa: A002 - matches LedgerEntry.round
    (built,) = [e for e in events_of(loop, item_id, EventKind.BUILT) if e.round == round]
    return loop.root / "items" / item_id / "rounds" / str(round) / f"evidence-{built.input_hash[:12]}"


def tool_names(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools", ())}


def last_user_turn(fake_glm, request: int) -> str:
    return fake_glm.requests[request]["messages"][-1]["content"]


async def test_an_idea_item_is_accepted_and_a_relaunch_changes_nothing(loop, programs, fake_glm):
    first, second = programs.proposal(1), programs.proposal(2)
    loop.source.batches.append((first, "slot 1 front matter is not YAML", second))
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)
    policy = programs.policy()

    async with loop.services() as services:
        proposals = await run_idea(IDEA, "idea", policy, services)
        terminal = await run_item(proposals[0], ProposalOrigin.GENERATED, policy, services)
    item_id = item_id_for(first)

    assert proposals == (first, second)
    idea = idea_item_id(IDEA)
    assert kinds(loop, idea) == [EventKind.SLOT_FAILED, EventKind.PROPOSED]
    assert events_of(loop, idea, EventKind.PROPOSED)[0].attrs["items"] == f"{item_id},{item_id_for(second)}"
    assert terminal is Terminal.ACCEPTED
    assert events_of(loop, item_id, EventKind.OPENED)[0].attrs["origin"] == ProposalOrigin.GENERATED
    logged = kinds(loop, item_id)
    assert logged[:5] == [
        EventKind.OPENED,
        EventKind.TRIAGED,
        EventKind.AUTHORED,
        EventKind.BUILT,
        EventKind.CONTROLS_REPLAYED,
    ]
    assert set(logged[5:7]) == {EventKind.SOLVED, EventKind.ADVERSARIES_RUN}
    assert logged[7:] == [EventKind.DECIDED, EventKind.TERMINAL]
    evidence = evidence_dir(loop, item_id, 0)
    decision = load_decision(evidence / DECISION_FILE)
    assert isinstance(decision, Accept) and decision.summary.solve_rate == 0.5
    assert decision.band is BandOutcome.IN_BAND
    assert (evidence / "calibration.json").exists()
    (adversaries,) = events_of(loop, item_id, EventKind.ADVERSARIES_RUN)
    assert adversaries.attrs == {
        "shortcut_graded": "1",
        "shortcut_passes": "0",
        "shortcut_submissions": "0",
        "shortcut_claimed": "0",
        "context_digest": "",
        "seq": adversaries.attrs["seq"],
        "schema": adversaries.attrs["schema"],
    }
    (decided,) = events_of(loop, item_id, EventKind.DECIDED)
    assert (decided.attrs["band"], decided.attrs["solved"], decided.attrs["graded"]) == ("in_band", "2", "4")
    assert decided.attrs["solve_rate"] == "0.500"
    assert events_of(loop, item_id, EventKind.TERMINAL)[0].attrs["reason"] == "calibrated"

    async with loop.services() as services:
        assert await run_idea(IDEA, "idea", policy, services) == (first, second)
        assert await run_item(first, ProposalOrigin.GENERATED, policy, services) is Terminal.ACCEPTED
    assert loop.source.calls == 1 and len(loop.rubric.assessed) == 1
    assert programs.authored(fake_glm) == 1 and len(fake_glm.requests) == 2


async def test_an_idea_whose_batches_are_empty_is_exhausted_after_its_reproposals(loop, programs):
    loop.source.batches += [("bad front matter",), ("bad again",)]

    async with loop.services() as services:
        proposals = await run_idea(IDEA, "idea", programs.policy(max_idea_reproposals=1), services)

    assert proposals == ()
    assert kinds(loop, idea_item_id(IDEA)) == [
        EventKind.SLOT_FAILED,
        EventKind.PROPOSED,
        EventKind.SLOT_FAILED,
        EventKind.PROPOSED,
        EventKind.IDEA_EXHAUSTED,
    ]


async def test_an_idea_keeps_its_record_and_every_batch_s_calls_beside_its_proposals(loop, programs):
    first, second = programs.proposal(1), programs.proposal(2)
    loop.source.batches += [("bad front matter",), (first, "slot 1 front matter is not YAML", second)]
    loop.source.repaired = frozenset({2})

    async with loop.services() as services:
        assert await run_idea(IDEA, "idea", programs.policy(max_idea_reproposals=1), services) == (first, second)

    idea_dir = loop.root / "items" / idea_item_id(IDEA)
    assert json.loads((idea_dir / "idea.json").read_text()) == {"idea": "idea", "kind": "test"}
    empty, accepted = idea_dir / "batches" / "0", idea_dir / "batches" / "1"
    assert (empty / "slots" / "0" / "failure.txt").read_text() == "bad front matter"
    plan = accepted / "plan"
    assert json.loads((plan / "request.json").read_text()) == [{"role": "user", "content": "plan 2 slots for idea"}]
    assert [c.content for c in COMPLETIONS.validate_json((plan / "completions.json").read_text())] == ["plan"]
    slots = accepted / "slots"
    assert sorted(path.name for path in slots.iterdir()) == ["0", "1", "2"]
    assert json.loads((slots / "0" / "request.json").read_text()) == [
        {"role": "user", "content": "write slot 0 of idea"}
    ]
    assert not (slots / "0" / "repair_error.txt").exists() and not (slots / "0" / "failure.txt").exists()
    assert (slots / "1" / "failure.txt").read_text() == "slot 1 front matter is not YAML"
    assert len(COMPLETIONS.validate_json((slots / "1" / "completions.json").read_text())) == 2
    assert (slots / "2" / "repair_error.txt").read_text() == "front matter missing"
    repaired = COMPLETIONS.validate_json((slots / "2" / "completions.json").read_text())
    assert [c.content for c in repaired] == ["not a proposal", render(second)]


async def test_a_supplied_proposal_is_opened_as_supplied_and_a_relaunch_under_another_origin_is_refused(loop, programs):
    loop.rubric.decisions[:] = [TriageDecision.REJECT]
    proposal = programs.proposal()
    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services) is Terminal.REJECTED

        with pytest.raises(ValueError, match="opened as a supplied proposal"):
            await run_item(proposal, ProposalOrigin.GENERATED, programs.policy(), services)
    (opened,) = events_of(loop, item_id_for(proposal), EventKind.OPENED)
    assert opened.attrs["origin"] == ProposalOrigin.SUPPLIED


async def test_a_process_killed_mid_trials_resumes_without_repeating_finished_work(loop, programs, fake_glm):
    proposal = programs.proposal()
    item_id = item_id_for(proposal)
    programs.submit(fake_glm, programs.source())
    fake_glm.stream(content="", stall_after_first=True)  # the adversary's first turn hangs until killed
    policy = programs.policy()

    async with loop.services() as services:
        run = asyncio.create_task(run_item(proposal, ProposalOrigin.SUPPLIED, policy, services))
        while len(fake_glm.requests) < 2 or EventKind.SOLVED not in kinds(loop, item_id):
            await asyncio.sleep(0.01)
        run.cancel()  # the process dies mid-phase: no FAILED event is written
        with pytest.raises(asyncio.CancelledError):
            await run
    fake_glm.released.set()
    assert derive_state(loop.entries(item_id)).phase is Phase.TRIALS
    tokenizer_calls, solver_calls = loop.tokenizer.calls, loop.model.calls

    programs.adversary_turns(fake_glm)
    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services) is Terminal.ACCEPTED

    assert programs.authored(fake_glm) == 1 and len(loop.rubric.assessed) == 1
    assert (loop.tokenizer.calls, loop.model.calls) == (tokenizer_calls, solver_calls)
    assert kinds(loop, item_id).count(EventKind.CONTROLS_REPLAYED) == 1
    assert kinds(loop, item_id).count(EventKind.SOLVED) == 1
    adversaries = trial_files(evidence_dir(loop, item_id, 0), TrialKind.ADVERSARY)
    assert sorted(adversaries) == [f"{AdversaryRole.SHORTCUT}/0"]


async def test_a_repair_that_rebuilds_the_same_task_is_a_failed_revision_and_the_budget_rejects(
    loop, programs, fake_glm
):
    proposal = programs.proposal()
    first = programs.source()
    # Each round's adversary passes without reading the question: a repair-tier shortcut.
    unread_pass = (("submit", CORRECT, []), "SHORTCUT: the answer needs no input")
    programs.submit(fake_glm, first)
    programs.adversary_turns(fake_glm, *unread_pass)
    programs.submit(fake_glm, first)  # the repair returns the same program: a no-op
    programs.submit(fake_glm, programs.source(grader_timeout=90))
    programs.adversary_turns(fake_glm, *unread_pass)
    policy = programs.policy(max_repairs=1)

    async with loop.services() as services:
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.REJECTED
    repair = load_decision(evidence_dir(loop, item_id, 0) / DECISION_FILE)
    assert isinstance(repair, Repair)
    assert [finding.kind for finding in repair.brief.findings] == [FindingKind.SHORTCUT_PASSED]
    assert "The shortcut adversary 0 found an accepted submission" in repair.brief.failure
    assert repair.brief.failure in last_user_turn(fake_glm, 3)

    (noop,) = events_of(loop, item_id, EventKind.BUILD_FAILED)
    assert noop.round == 1 and noop.attrs["noop"] == "true"
    assert NOOP_FAILURE in last_user_turn(fake_glm, 4) and repair.brief.failure in last_user_turn(fake_glm, 4)
    revisions = [e.attrs["revision"] for e in events_of(loop, item_id, EventKind.AUTHORED)]
    assert revisions == [RevisionKind.NONE, RevisionKind.REPAIR, RevisionKind.BUILD_FAILURE]

    rejected = load_decision(evidence_dir(loop, item_id, 1) / DECISION_FILE)
    assert isinstance(rejected, Reject) and rejected.kind is RejectKind.BUDGET
    terminal_event = events_of(loop, item_id, EventKind.TERMINAL)[-1]
    assert terminal_event.attrs["kind"] == RejectKind.BUDGET


async def test_a_shortcut_found_after_the_threshold_is_noted_and_the_item_accepted(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source(lenient=True))
    probes = [("submit", f"The product is {n}.", []) for n in (40, 41, 42)]
    programs.adversary_turns(fake_glm, READ, *probes, ("submit", "ANSWER = 0", []), "SHORTCUT: any integer passes")
    loop.model.solver = (CORRECT, NO_ANSWER)

    async with loop.services() as services:
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.ACCEPTED
    (adversaries,) = events_of(loop, item_id, EventKind.ADVERSARIES_RUN)
    assert (adversaries.attrs["shortcut_passes"], adversaries.attrs["shortcut_claimed"]) == ("1", "1")
    assert adversaries.attrs["shortcut_submissions"] == "4"
    (decided,) = events_of(loop, item_id, EventKind.DECIDED)
    assert (decided.attrs["decision"], decided.attrs["notes"]) == ("accept", "shortcut_passed")
    trial = trial_files(evidence_dir(loop, item_id, 0), TrialKind.ADVERSARY)["shortcut/0"]
    assert trial.last_path is not None
    submissions = load_adversary_attempt(trial.last_path).submissions
    assert [s.passed for s in submissions] == [False, False, False, True]


async def test_a_shortcut_claim_cut_off_on_the_output_budget_is_counted_as_no_claim(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm, READ)
    fake_glm.stream(content="SHORTCUT: any integer passes because the grader", finish="length")

    async with loop.services() as services:
        await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services)
    item_id = item_id_for(proposal)

    (adversaries,) = events_of(loop, item_id, EventKind.ADVERSARIES_RUN)
    assert (adversaries.attrs["shortcut_graded"], adversaries.attrs["shortcut_claimed"]) == ("1", "0")
    calibration = json.loads((evidence_dir(loop, item_id, 0) / "calibration.json").read_text())
    assert calibration["roles"]["shortcut"]["claims"] == {"shortcut": 0, "no_shortcut": 0, "none": 1}


async def test_a_shortcut_within_the_threshold_is_repaired_with_its_candidate_control(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source(lenient=True))
    programs.adversary_turns(fake_glm, READ, ("submit", "ANSWER = 0", []), "SHORTCUT: any integer passes")
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)
    loop.model.solver = (CORRECT, NO_ANSWER)

    async with loop.services() as services:
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.ACCEPTED
    repair = load_decision(evidence_dir(loop, item_id, 0) / DECISION_FILE)
    assert isinstance(repair, Repair)
    (finding,) = repair.brief.findings
    assert finding.kind is FindingKind.SHORTCUT_PASSED
    (control,) = finding.new_controls
    assert control.author == "adversary/shortcut/0#1"
    assert control.id in repair.brief.failure and "ANSWER = 0" in repair.brief.failure
    assert repair.brief.failure in last_user_turn(fake_glm, 4)
    decided = events_of(loop, item_id, EventKind.DECIDED)
    assert [(e.attrs["decision"], e.attrs.get("findings")) for e in decided] == [
        ("repair", "shortcut_passed"),
        ("accept", None),
    ]


async def test_a_staged_draft_is_repaired_into_a_single_stage_task_without_validation(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source(staged=True))
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)

    async with loop.services() as services:
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.ACCEPTED
    assert STAGED_BRIEF in last_user_turn(fake_glm, 1)
    staged_round = evidence_dir(loop, item_id, 0)
    assert isinstance(load_decision(staged_round / DECISION_FILE), Repair)
    assert not (staged_round / "control").exists()
    built = events_of(loop, item_id, EventKind.BUILT)
    assert [e.attrs["staged"] for e in built] == ["true", "false"]


async def test_validation_retries_end_abandoned_and_a_relaunch_re_enters_with_a_fresh_budget(
    loop, programs, fake_glm, unavailable
):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)  # settled in the first pass; the retries load it from its file
    policy = programs.policy(max_validation_retries=1)
    loop.model.error = unavailable

    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services) is Terminal.ABANDONED
    item_id = item_id_for(proposal)
    decided = events_of(loop, item_id, EventKind.DECIDED)
    assert [(e.attrs["decision"], e.attrs["abandon"]) for e in decided] == [("retry", "false"), ("retry", "true")]
    assert {e.attrs["cause"] for e in decided} == {"model_unavailable"}
    assert events_of(loop, item_id, EventKind.TERMINAL)[-1].attrs["causes"] == "model_unavailable:2"

    loop.model.error = None
    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services) is Terminal.ACCEPTED

    assert len(fake_glm.requests) == 2 and programs.authored(fake_glm) == 1
    assert kinds(loop, item_id).count(EventKind.BUILT) == 1
    solver = trial_files(evidence_dir(loop, item_id, 0), TrialKind.SOLVER)
    assert all(files.attempts == 3 for files in solver.values())


async def test_triage_repairs_until_its_bound_then_rejects_without_authoring(loop, programs, fake_glm):
    loop.rubric.decisions[:] = [TriageDecision.REPAIR, TriageDecision.REPAIR]
    proposal = programs.proposal()

    async with loop.services() as services:
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(max_triage_repairs=1), services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.REJECTED
    assert kinds(loop, item_id) == [
        EventKind.OPENED,
        EventKind.TRIAGED,
        EventKind.TRIAGE_REPAIRED,
        EventKind.TRIAGED,
        EventKind.TERMINAL,
    ]
    assert len(loop.rubric.assessed) == 2 and loop.rubric.assessed[0] != loop.rubric.assessed[1]
    assert events_of(loop, item_id, EventKind.TERMINAL)[0].attrs["kind"] == RejectKind.TASK
    assert fake_glm.requests == []


async def test_an_item_over_its_output_token_budget_is_rejected_before_authoring(loop, programs, fake_glm):
    loop.rubric.tokens_out = 500
    proposal = programs.proposal()

    async with loop.services() as services:
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(output_token_budget=100), services)

    assert terminal is Terminal.REJECTED
    (closing,) = events_of(loop, item_id_for(proposal), EventKind.TERMINAL)
    assert closing.attrs["kind"] == RejectKind.BUDGET and "500" in closing.attrs["reason"]
    assert fake_glm.requests == []


async def test_an_unhandled_exception_records_failed_and_a_relaunch_re_enters(loop, programs, fake_glm):
    loop.rubric.raise_on_assess = RuntimeError("rubric store unreadable")
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)
    policy = programs.policy()

    async with loop.services() as services:
        with pytest.raises(RuntimeError, match="rubric store unreadable"):
            await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services)
    item_id = item_id_for(proposal)
    (failed,) = events_of(loop, item_id, EventKind.TERMINAL)
    assert failed.attrs["terminal"] == Terminal.FAILED and "RuntimeError" in failed.attrs["reason"]

    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services) is Terminal.ACCEPTED


class DroppingFactory:
    """A ShellSim factory whose first ``drops`` creations lose the connection to the host."""

    def __init__(self, drops: int):
        self.drops = drops
        self.inner = ShellSimMachineFactory()

    async def create(self, spec: MachineSpec) -> Machine:
        if self.drops:
            self.drops -= 1
            raise ConnectionError("connection reset by the machine host")
        return await self.inner.create(spec)


async def test_a_dropped_host_connection_rebuilds_the_program_without_telling_the_author(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)

    async with loop.services() as services:
        flaky = replace(
            services, build=replace(services.build, factories={EnvironmentKind.SHELLSIM: DroppingFactory(1)})
        )
        assert (
            await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(max_build_retries=1), flaky)
            is Terminal.ACCEPTED
        )
    item_id = item_id_for(proposal)

    (dropped,) = events_of(loop, item_id, EventKind.BUILD_INFRASTRUCTURE)
    assert (dropped.attrs["cause"], dropped.attrs["abandon"]) == (InfrastructureCause.HOST_UNREACHABLE, "false")
    assert events_of(loop, item_id, EventKind.BUILD_FAILED) == []
    assert programs.authored(fake_glm) == 1 and len(events_of(loop, item_id, EventKind.AUTHORED)) == 1


async def test_a_build_the_host_keeps_failing_is_abandoned_and_a_relaunch_rebuilds_the_same_program(
    loop, programs, fake_glm
):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)
    policy = programs.policy(max_build_retries=1)

    async with loop.services() as services:
        dropping = replace(
            services, build=replace(services.build, factories={EnvironmentKind.SHELLSIM: DroppingFactory(2)})
        )
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, dropping) is Terminal.ABANDONED
    item_id = item_id_for(proposal)
    failures = events_of(loop, item_id, EventKind.BUILD_INFRASTRUCTURE)
    assert [e.attrs["abandon"] for e in failures] == ["false", "true"]
    assert events_of(loop, item_id, EventKind.BUILD_FAILED) == []
    (abandoned,) = events_of(loop, item_id, EventKind.TERMINAL)
    assert (abandoned.attrs["terminal"], abandoned.attrs["causes"]) == (Terminal.ABANDONED, "host_unreachable:2")
    assert derive_state(loop.entries(item_id)).phase is Phase.BUILD

    # The relaunch builds the program already authored: no second authoring call is queued.
    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services) is Terminal.ACCEPTED
    assert programs.authored(fake_glm) == 1 and len(events_of(loop, item_id, EventKind.AUTHORED)) == 1
    assert build_host_failures(loop.entries(item_id)) == Counter({InfrastructureCause.HOST_UNREACHABLE: 2})


async def test_a_build_the_host_has_no_factory_for_is_abandoned_at_once_and_built_on_a_host_with_it(
    loop, programs, fake_glm
):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)
    policy = programs.policy(max_build_retries=3)

    async with loop.services() as services:
        hostless = replace(services, build=replace(services.build, factories={}))
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, hostless) is Terminal.ABANDONED
    item_id = item_id_for(proposal)
    (failure,) = events_of(loop, item_id, EventKind.BUILD_INFRASTRUCTURE)
    assert (failure.attrs["cause"], failure.attrs["retries_used"], failure.attrs["abandon"]) == (
        InfrastructureCause.NO_FACTORY,
        "0",
        "true",
    )
    assert events_of(loop, item_id, EventKind.BUILD_FAILED) == []
    (abandoned,) = events_of(loop, item_id, EventKind.TERMINAL)
    assert (abandoned.attrs["terminal"], abandoned.attrs["causes"]) == (Terminal.ABANDONED, "no_factory:1")

    # The relaunch on a host with the factory builds the program already authored.
    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services) is Terminal.ACCEPTED
    assert programs.authored(fake_glm) == 1 and len(events_of(loop, item_id, EventKind.AUTHORED)) == 1


async def test_a_log_opened_under_another_policy_is_refused(loop, programs):
    loop.rubric.decisions[:] = [TriageDecision.REJECT]
    proposal = programs.proposal()
    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services) is Terminal.REJECTED

        with pytest.raises(ValueError, match="opened under policy"):
            await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(max_repairs=5), services)


def too_easy_rounds(programs, fake_glm, rounds: int) -> None:
    """Queue ``rounds`` distinct builds, each followed by an adversary that reports no shortcut."""
    for round in range(rounds):  # noqa: A001 - a validation round
        programs.submit(fake_glm, programs.source(grader_timeout=60 + round))
        programs.adversary_turns(fake_glm)


async def test_a_too_easy_task_is_revised_then_accepted_and_labelled_under_the_accept_choice(loop, programs, fake_glm):
    proposal = programs.proposal()
    too_easy_rounds(programs, fake_glm, 2)
    loop.model.solver = (CORRECT,)

    async with loop.services() as services:
        policy = programs.policy(band_rules=TOO_EASY_ACCEPTED)
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.ACCEPTED
    repair, accept = events_of(loop, item_id, EventKind.DECIDED)
    assert (repair.round, repair.attrs["decision"], repair.attrs["findings"]) == (0, "repair", "too_easy")
    assert (accept.round, accept.attrs["decision"], accept.attrs["band"]) == (1, "accept", "too_easy")
    assert (accept.attrs["solved"], accept.attrs["graded"], accept.attrs["solve_rate"]) == ("4", "4", "1.000")
    (closing,) = events_of(loop, item_id, EventKind.TERMINAL)
    assert closing.attrs["reason"] == "accepted outside the band: too_easy"
    decision = load_decision(evidence_dir(loop, item_id, 1) / DECISION_FILE)
    assert isinstance(decision, Accept) and decision.band is BandOutcome.TOO_EASY
    assert decision.summary.solve_rate == 1.0


async def test_a_too_easy_task_is_rejected_after_its_repair_under_the_reject_choice(loop, programs, fake_glm):
    proposal = programs.proposal()
    too_easy_rounds(programs, fake_glm, 2)
    loop.model.solver = (CORRECT,)

    async with loop.services() as services:
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.REJECTED
    decided = events_of(loop, item_id, EventKind.DECIDED)
    assert [e.attrs["decision"] for e in decided] == ["repair", "reject"]
    rejected = load_decision(evidence_dir(loop, item_id, 1) / DECISION_FILE)
    assert isinstance(rejected, Reject) and rejected.kind is RejectKind.TASK
    assert events_of(loop, item_id, EventKind.TERMINAL)[0].attrs["kind"] == RejectKind.TASK


async def test_an_accept_choice_outranks_a_spent_repair_budget(loop, programs, fake_glm):
    proposal = programs.proposal()
    too_easy_rounds(programs, fake_glm, 1)
    loop.model.solver = (CORRECT,)

    async with loop.services() as services:
        policy = programs.policy(max_repairs=0, band_rules=TOO_EASY_ACCEPTED)
        terminal = await run_item(proposal, ProposalOrigin.SUPPLIED, policy, services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.ACCEPTED
    (decided,) = events_of(loop, item_id, EventKind.DECIDED)
    assert (decided.round, decided.attrs["decision"], decided.attrs["band"]) == (0, "accept", "too_easy")


async def test_the_adversary_context_reaches_the_brief_and_the_event(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    programs.adversary_turns(fake_glm)
    loop.context = lambda p: f"context for {p.header.id}"

    async with loop.services() as services:
        assert await run_item(proposal, ProposalOrigin.SUPPLIED, programs.policy(), services) is Terminal.ACCEPTED
    item_id = item_id_for(proposal)

    context = f"context for {proposal.header.id}"
    trial = trial_files(evidence_dir(loop, item_id, 0), TrialKind.ADVERSARY)["shortcut/0"]
    assert trial.last_path is not None
    system = load_adversary_attempt(trial.last_path).system
    assert system.index(CONTEXT_HEADER) < system.index(context)
    (adversary_request,) = [r for r in fake_glm.requests if SUBMIT_TOOL_NAME in tool_names(r)]
    assert adversary_request["messages"][0] == {"role": "system", "content": system}
    (adversaries,) = events_of(loop, item_id, EventKind.ADVERSARIES_RUN)
    assert adversaries.attrs["context_digest"] == sha256_hex(context.encode())
