# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The item program end to end on ShellSim: the real author against a scripted router, real builds, real
validation through RolloutEngine, and fakes only at the source, the rubric and the rollout model."""

import asyncio
from collections import Counter
from dataclasses import replace

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Machine, MachineSpec
from taskcompendium.environment import EnvironmentKind

from taskforge.build.infrastructure import InfrastructureCause
from taskforge.build.run import item_id_for
from taskforge.ledger.records import EntryKind
from taskforge.loop.events import EventKind, Phase, RevisionKind, Terminal, build_host_failures, derive_state
from taskforge.loop.program import NOOP_FAILURE, idea_item_id, run_idea, run_item
from taskforge.review.decision import DECISION_FILE, Accept, Reject, RejectKind, Repair, load_decision
from taskforge.review.rules import STAGED_BRIEF
from taskforge.triage.verdict import TriageDecision
from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.attempts import trial_files
from taskforge.validate.outcome import TrialKind

IDEA = "d00.arithmetic.products"
CORRECT = "ANSWER = 42"


def kinds(loop, item_id: str) -> list[str]:
    return [entry.step for entry in loop.entries(item_id) if entry.kind is EntryKind.EVENT]


def events_of(loop, item_id: str, kind: EventKind):
    return [e for e in loop.entries(item_id) if e.kind is EntryKind.EVENT and e.step == kind]


def evidence_dir(loop, item_id: str, round: int):  # noqa: A002 - matches LedgerEntry.round
    (built,) = [e for e in events_of(loop, item_id, EventKind.BUILT) if e.round == round]
    return loop.root / "items" / item_id / "rounds" / str(round) / f"evidence-{built.input_hash[:12]}"


def last_user_turn(fake_glm, request: int) -> str:
    return fake_glm.requests[request]["messages"][-1]["content"]


async def test_an_idea_item_is_accepted_and_a_relaunch_changes_nothing(loop, programs, fake_glm):
    first, second = programs.proposal(1), programs.proposal(2)
    loop.source.batches.append((first, "slot 1 front matter is not YAML", second))
    programs.submit(fake_glm, programs.source())
    policy = programs.policy()

    async with loop.services() as services:
        proposals = await run_idea(IDEA, "idea", policy, services)
        terminal = await run_item(proposals[0], policy, services)
    item_id = item_id_for(first)

    assert proposals == (first, second)
    idea = idea_item_id(IDEA)
    assert kinds(loop, idea) == [EventKind.SLOT_FAILED, EventKind.PROPOSED]
    assert events_of(loop, idea, EventKind.PROPOSED)[0].attrs["items"] == f"{item_id},{item_id_for(second)}"
    assert terminal is Terminal.ACCEPTED
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
    assert (evidence / "calibration.json").exists()

    async with loop.services() as services:
        assert await run_idea(IDEA, "idea", policy, services) == (first, second)
        assert await run_item(first, policy, services) is Terminal.ACCEPTED
    assert loop.source.calls == 1 and len(fake_glm.requests) == 1 and len(loop.rubric.assessed) == 1


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


async def test_a_process_killed_mid_trials_resumes_without_repeating_finished_work(loop, programs, fake_glm, crash):
    proposal = programs.proposal()
    item_id = item_id_for(proposal)
    programs.submit(fake_glm, programs.source())
    policy = programs.policy()

    async def die_once_the_solver_is_recorded() -> None:
        while EventKind.SOLVED not in kinds(loop, item_id):
            await asyncio.sleep(0.01)
        raise crash()

    loop.model.before_role = die_once_the_solver_is_recorded
    async with loop.services() as services:
        with pytest.raises(BaseExceptionGroup) as killed:
            await run_item(proposal, policy, services)
    assert killed.group_contains(crash)
    assert derive_state(loop.entries(item_id)).phase is Phase.TRIALS
    tokenizer_calls, solver_calls = loop.tokenizer.calls, loop.model.calls

    loop.model.before_role = None
    async with loop.services() as services:
        assert await run_item(proposal, policy, services) is Terminal.ACCEPTED

    assert len(fake_glm.requests) == 1 and len(loop.rubric.assessed) == 1
    assert (loop.tokenizer.calls, loop.model.calls) == (tokenizer_calls, solver_calls)
    assert kinds(loop, item_id).count(EventKind.CONTROLS_REPLAYED) == 1
    assert kinds(loop, item_id).count(EventKind.SOLVED) == 1
    adversaries = trial_files(evidence_dir(loop, item_id, 0), TrialKind.ADVERSARY)
    assert sorted(adversaries) == [f"{role}/0" for role in sorted(AdversaryRole)]


async def test_a_repair_that_rebuilds_the_same_task_is_a_failed_revision_and_the_budget_rejects(
    loop, programs, fake_glm
):
    proposal = programs.proposal()
    first = programs.source()
    programs.submit(fake_glm, first)
    programs.submit(fake_glm, first)  # the repair returns the same program: a no-op
    programs.submit(fake_glm, programs.source(grader_timeout=90))
    loop.model.roles[AdversaryRole.SHORTCUT] = CORRECT  # the shortcut role "passes" every round
    policy = programs.policy(max_repairs=1)

    async with loop.services() as services:
        terminal = await run_item(proposal, policy, services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.REJECTED
    repair = load_decision(evidence_dir(loop, item_id, 0) / DECISION_FILE)
    assert isinstance(repair, Repair)
    assert "The shortcut adversary 0 was graded as passing" in repair.brief.failure
    assert repair.brief.failure in last_user_turn(fake_glm, 1)

    (noop,) = events_of(loop, item_id, EventKind.BUILD_FAILED)
    assert noop.round == 1 and noop.attrs["noop"] == "true"
    assert NOOP_FAILURE in last_user_turn(fake_glm, 2) and repair.brief.failure in last_user_turn(fake_glm, 2)
    revisions = [e.attrs["revision"] for e in events_of(loop, item_id, EventKind.AUTHORED)]
    assert revisions == [RevisionKind.NONE, RevisionKind.REPAIR, RevisionKind.BUILD_FAILURE]

    rejected = load_decision(evidence_dir(loop, item_id, 1) / DECISION_FILE)
    assert isinstance(rejected, Reject) and rejected.kind is RejectKind.BUDGET
    terminal_event = events_of(loop, item_id, EventKind.TERMINAL)[-1]
    assert terminal_event.attrs["kind"] == RejectKind.BUDGET


async def test_a_shortcut_that_reads_the_input_and_solves_is_noted_and_the_item_accepted(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    loop.model.roles[AdversaryRole.SHORTCUT] = CORRECT
    loop.model.reads = frozenset({AdversaryRole.SHORTCUT})

    async with loop.services() as services:
        terminal = await run_item(proposal, programs.policy(), services)
    item_id = item_id_for(proposal)

    assert terminal is Terminal.ACCEPTED
    (adversaries,) = events_of(loop, item_id, EventKind.ADVERSARIES_RUN)
    assert (adversaries.attrs["shortcut_passes"], adversaries.attrs["leak_sentinel"]) == ("1", "1")
    (decided,) = events_of(loop, item_id, EventKind.DECIDED)
    assert decided.attrs["notes"] == "shortcut_passed"
    assert len(fake_glm.requests) == 1


async def test_a_staged_draft_is_repaired_into_a_single_stage_task_without_validation(loop, programs, fake_glm):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source(staged=True))
    programs.submit(fake_glm, programs.source())

    async with loop.services() as services:
        terminal = await run_item(proposal, programs.policy(), services)
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
    policy = programs.policy(max_validation_retries=1)
    loop.model.error = unavailable

    async with loop.services() as services:
        assert await run_item(proposal, policy, services) is Terminal.ABANDONED
    item_id = item_id_for(proposal)
    decided = events_of(loop, item_id, EventKind.DECIDED)
    assert [(e.attrs["decision"], e.attrs["abandon"]) for e in decided] == [("retry", "false"), ("retry", "true")]
    assert {e.attrs["cause"] for e in decided} == {"model_unavailable"}
    assert events_of(loop, item_id, EventKind.TERMINAL)[-1].attrs["causes"] == "model_unavailable:2"

    loop.model.error = None
    async with loop.services() as services:
        assert await run_item(proposal, policy, services) is Terminal.ACCEPTED

    assert len(fake_glm.requests) == 1
    assert kinds(loop, item_id).count(EventKind.BUILT) == 1
    solver = trial_files(evidence_dir(loop, item_id, 0), TrialKind.SOLVER)
    assert all(files.attempts == 3 for files in solver.values())


async def test_triage_repairs_until_its_bound_then_rejects_without_authoring(loop, programs, fake_glm):
    loop.rubric.decisions[:] = [TriageDecision.REPAIR, TriageDecision.REPAIR]
    proposal = programs.proposal()

    async with loop.services() as services:
        terminal = await run_item(proposal, programs.policy(max_triage_repairs=1), services)
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
        terminal = await run_item(proposal, programs.policy(output_token_budget=100), services)

    assert terminal is Terminal.REJECTED
    (closing,) = events_of(loop, item_id_for(proposal), EventKind.TERMINAL)
    assert closing.attrs["kind"] == RejectKind.BUDGET and "500" in closing.attrs["reason"]
    assert fake_glm.requests == []


async def test_an_unhandled_exception_records_failed_and_a_relaunch_re_enters(loop, programs, fake_glm):
    loop.rubric.raise_on_assess = RuntimeError("rubric store unreadable")
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    policy = programs.policy()

    async with loop.services() as services:
        with pytest.raises(RuntimeError, match="rubric store unreadable"):
            await run_item(proposal, policy, services)
    item_id = item_id_for(proposal)
    (failed,) = events_of(loop, item_id, EventKind.TERMINAL)
    assert failed.attrs["terminal"] == Terminal.FAILED and "RuntimeError" in failed.attrs["reason"]

    async with loop.services() as services:
        assert await run_item(proposal, policy, services) is Terminal.ACCEPTED


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

    async with loop.services() as services:
        flaky = replace(
            services, build=replace(services.build, factories={EnvironmentKind.SHELLSIM: DroppingFactory(1)})
        )
        assert await run_item(proposal, programs.policy(max_build_retries=1), flaky) is Terminal.ACCEPTED
    item_id = item_id_for(proposal)

    (dropped,) = events_of(loop, item_id, EventKind.BUILD_INFRASTRUCTURE)
    assert (dropped.attrs["cause"], dropped.attrs["abandon"]) == (InfrastructureCause.HOST_UNREACHABLE, "false")
    assert events_of(loop, item_id, EventKind.BUILD_FAILED) == []
    assert len(fake_glm.requests) == 1 and len(events_of(loop, item_id, EventKind.AUTHORED)) == 1


async def test_a_build_the_host_keeps_failing_is_abandoned_and_a_relaunch_rebuilds_the_same_program(
    loop, programs, fake_glm
):
    proposal = programs.proposal()
    programs.submit(fake_glm, programs.source())
    policy = programs.policy(max_build_retries=1)

    async with loop.services() as services:
        hostless = replace(services, build=replace(services.build, factories={}))
        assert await run_item(proposal, policy, hostless) is Terminal.ABANDONED
    item_id = item_id_for(proposal)
    failures = events_of(loop, item_id, EventKind.BUILD_INFRASTRUCTURE)
    assert [e.attrs["abandon"] for e in failures] == ["false", "true"]
    assert events_of(loop, item_id, EventKind.BUILD_FAILED) == []
    (abandoned,) = events_of(loop, item_id, EventKind.TERMINAL)
    assert (abandoned.attrs["terminal"], abandoned.attrs["causes"]) == (Terminal.ABANDONED, "no_factory:2")
    assert derive_state(loop.entries(item_id)).phase is Phase.BUILD

    # The relaunch builds the program already authored: no second authoring call is queued.
    async with loop.services() as services:
        assert await run_item(proposal, policy, services) is Terminal.ACCEPTED
    assert len(fake_glm.requests) == 1 and len(events_of(loop, item_id, EventKind.AUTHORED)) == 1
    assert build_host_failures(loop.entries(item_id)) == Counter({InfrastructureCause.NO_FACTORY: 2})


async def test_a_log_opened_under_another_policy_is_refused(loop, programs):
    loop.rubric.decisions[:] = [TriageDecision.REJECT]
    proposal = programs.proposal()
    async with loop.services() as services:
        assert await run_item(proposal, programs.policy(), services) is Terminal.REJECTED

        with pytest.raises(ValueError, match="opened under policy"):
            await run_item(proposal, programs.policy(max_repairs=5), services)
