# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The event log: rows through the ledger, and item status as a fold that refuses a broken log."""

from dataclasses import replace

import pytest

from taskforge.builder.author import RevisionKind
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import LedgerEntry
from taskforge.loop.events import (
    DEFAULT_TRANSITIONS,
    EVENT_SCHEMA,
    EXIT,
    Event,
    EventKind,
    Exit,
    Stage,
    Terminal,
    check_transitions,
    derive_idea_state,
    derive_state,
    record_event,
)
from taskforge.review.decision import RejectKind

ITEM = "d00.x--1"

OPENED = Event(
    EventKind.OPENED, "p1", {"proposal": "d00.x/1", "idea": "d00.x", "origin": "supplied", "policy_digest": "pol"}
)
TRIAGED = Event(EventKind.TRIAGED, "p1", {EXIT: "accept"})
AUTHORED = Event(EventKind.AUTHORED, "prog1", {EXIT: "authored", "revision": "none"})
BUILT = Event(EventKind.BUILT, "task1", {EXIT: "built", "program_digest": "prog1", "steps": "3", "hits": "0"})
VALIDATED = Event(
    EventKind.VALIDATED,
    "task1",
    {EXIT: "validated", "graded": "4", "solved": "2", "timed_out": "0", "ungraded": "0", "findings": ""},
)


def decided(**attrs: str) -> Event:
    return Event(EventKind.DECIDED, "task1", {"retries_used": "0", **attrs})


def terminal(**attrs: str) -> Event:
    return Event(EventKind.TERMINAL, None, attrs)


class Log:
    """Appends events to a real JSONL ledger with contiguous ``seq`` and reads them back."""

    def __init__(self, root):
        self.ledger = JsonlLedger(root)
        self.seq = 0

    def add(self, event: Event, at_round: int = 0) -> "Log":
        self.seq += 1
        record_event(self.ledger, ITEM, at_round, self.seq, event)
        return self

    def entries(self) -> list[LedgerEntry]:
        return list(read_entries(self.ledger.path_for(ITEM)))


def validated_log(tmp_path) -> Log:
    return Log(tmp_path).add(OPENED).add(TRIAGED).add(AUTHORED).add(BUILT).add(VALIDATED)


def test_a_built_and_validated_item_folds_to_review_with_its_digests(tmp_path):
    state = derive_state(validated_log(tmp_path).entries())

    assert (state.stage, state.round, state.seq) == (Stage.REVIEW, 0, 5)
    assert (state.proposal_digest, state.program_digest, state.task_digest) == ("p1", "prog1", "task1")
    assert (state.revision, state.terminal, state.closing) == (RevisionKind.NONE, None, None)


def test_a_retry_re_enters_validation_after_its_backoff(tmp_path):
    log = validated_log(tmp_path).add(decided(exit="retry", cause="model_unavailable", count="3", not_before="1234.5"))

    state = derive_state(log.entries())

    assert (state.stage, state.retries, state.not_before, state.task_digest) == (Stage.VALIDATE, 1, 1234.5, "task1")


def test_a_retry_past_the_budget_abandons_and_the_relaunch_re_enters_validation_with_no_retries(tmp_path):
    log = validated_log(tmp_path).add(decided(exit="retry", cause="model_unavailable", count="3", not_before="0"))
    log.add(VALIDATED).add(decided(exit="abandon", cause="model_unavailable", count="4"))
    closing = derive_state(log.entries()).closing
    log.add(terminal(terminal="abandoned", reason="model_unavailable"))
    abandoned = derive_state(log.entries())
    log.add(VALIDATED)
    resumed = derive_state(log.entries())

    assert closing is not None and (closing.terminal, closing.reason) == (Terminal.ABANDONED, "model_unavailable")
    assert (abandoned.terminal, abandoned.stage, abandoned.retries) == (Terminal.ABANDONED, Stage.VALIDATE, 0)
    assert (resumed.terminal, resumed.stage) == (None, Stage.REVIEW)


@pytest.mark.parametrize(
    ("way", "stage", "closing"),
    [("revise", Stage.AUTHOR, None), ("reject", Stage.DONE, (Terminal.REJECTED, RejectKind.BUDGET))],
)
def test_a_build_failure_revises_or_rejects_as_its_event_says(tmp_path, way, stage, closing):
    failed = Event(
        EventKind.BUILD_FAILED,
        "prog1",
        {EXIT: way, "step": "grader", "failure": "boom", "failure_file": "rounds/0/build-failures/1.txt"},
    )
    log = Log(tmp_path).add(OPENED).add(TRIAGED).add(AUTHORED).add(failed)

    state = derive_state(log.entries())

    assert (state.stage, state.build_revisions, state.revision) == (stage, 1, RevisionKind.BUILD_FAILURE)
    assert (None if state.closing is None else (state.closing.terminal, state.closing.kind)) == closing


def test_a_modify_decision_starts_the_next_round_and_its_revision_holds_until_built(tmp_path):
    log = validated_log(tmp_path).add(decided(exit="modify", findings="too_easy"))

    state = derive_state(log.entries())
    log.add(Event(EventKind.AUTHORED, "prog2", {EXIT: "authored", "revision": "modify"}), at_round=1)
    authored = derive_state(log.entries())

    assert (state.stage, state.round, state.revision, state.task_digest) == (Stage.AUTHOR, 1, RevisionKind.MODIFY, None)
    assert (authored.stage, authored.program_digest, authored.revision) == (Stage.BUILD, "prog2", RevisionKind.MODIFY)
    log.add(replace(BUILT, input_hash="task2"), at_round=1)
    assert derive_state(log.entries()).revision is RevisionKind.NONE


@pytest.mark.parametrize(
    ("band", "reason"), [("in_band", "calibrated"), ("too_easy", "accepted outside the band: too_easy")]
)
def test_an_accept_closes_accepted_with_where_the_task_fell(tmp_path, band, reason):
    accept = decided(exit="accept", band=band, solved="2", graded="4", solve_rate="0.500")
    log = validated_log(tmp_path).add(accept)
    closing = derive_state(log.entries()).closing
    log.add(terminal(terminal="accepted", reason=reason))

    assert closing is not None and (closing.terminal, closing.reason) == (Terminal.ACCEPTED, reason)
    assert derive_state(log.entries()).terminal is Terminal.ACCEPTED


def test_a_failed_item_keeps_its_stage_and_a_final_terminal_ends_the_log(tmp_path):
    log = validated_log(tmp_path).add(terminal(terminal="failed", reason="RuntimeError: boom"))
    failed = derive_state(log.entries())
    assert (failed.terminal, failed.stage) == (Terminal.FAILED, Stage.REVIEW)

    log.add(decided(exit="reject", kind="host", reasons="machine_unsupported: 4 trials ungraded"))
    log.add(terminal(terminal="rejected", kind="host", reason="machine_unsupported"))
    assert derive_state(log.entries()).terminal is Terminal.REJECTED

    log.add(VALIDATED)
    with pytest.raises(ValueError, match="after terminal rejected"):
        derive_state(log.entries())


def test_an_event_outside_its_stage_or_with_a_foreign_exit_is_refused(tmp_path):
    early = Log(tmp_path / "early").add(OPENED).add(TRIAGED).add(VALIDATED)
    with pytest.raises(ValueError, match="recorded by stage validate, but the item is in stage author"):
        derive_state(early.entries())

    foreign = Log(tmp_path / "foreign").add(OPENED).add(Event(EventKind.TRIAGED, "p1", {EXIT: "retry"}))
    with pytest.raises(ValueError, match="triaged exits retry, which stage triage has no edge for"):
        derive_state(foreign.entries())

    unclosed = validated_log(tmp_path / "unclosed").add(terminal(terminal="accepted", reason="calibrated"))
    with pytest.raises(ValueError, match="TERMINAL accepted in stage review"):
        derive_state(unclosed.entries())


def test_a_gap_or_a_repeat_in_seq_is_refused(tmp_path):
    log = validated_log(tmp_path)
    log.seq += 1
    log.add(decided(exit="modify", findings=""))
    with pytest.raises(ValueError, match="seq 7 where 6"):
        derive_state(log.entries())

    repeated = validated_log(tmp_path / "repeat")
    repeated.seq -= 1
    repeated.add(decided(exit="modify", findings=""))
    with pytest.raises(ValueError, match="seq 5 where 6"):
        derive_state(repeated.entries())


def test_an_unknown_schema_is_refused(tmp_path):
    entries = validated_log(tmp_path).entries()
    future = replace(entries[-1], attrs={**entries[-1].attrs, "schema": str(int(EVENT_SCHEMA) + 1)})
    with pytest.raises(ValueError, match="event schema"):
        derive_state([*entries[:-1], future])


def test_an_idea_log_records_its_batch_or_its_exhaustion(tmp_path):
    ledger = JsonlLedger(tmp_path)
    proposed = {"slots": "1", "failures": "1"}
    record_event(ledger, "idea--d00.x", 0, 1, Event(EventKind.SLOT_FAILED, None, {"slot": "0", "error": "bad yaml"}))
    record_event(
        ledger, "idea--d00.x", 0, 2, Event(EventKind.PROPOSED, None, {**proposed, "items": "", "reproposal": "0"})
    )
    record_event(
        ledger, "idea--d00.x", 1, 3, Event(EventKind.PROPOSED, None, {**proposed, "items": "a,b", "reproposal": "1"})
    )

    state = derive_idea_state(read_entries(ledger.path_for("idea--d00.x")))

    assert (state.items, state.reproposals, state.exhausted, state.seq) == (("a", "b"), 1, False, 3)


def test_a_caller_table_re_targets_an_edge_and_a_table_missing_an_exit_is_refused(tmp_path):
    rebuild = {**DEFAULT_TRANSITIONS, (Stage.REVIEW, Exit.RETRY): Stage.BUILD}
    check_transitions(rebuild)
    log = validated_log(tmp_path).add(decided(exit="retry", cause="model_unavailable", count="1", not_before="0"))
    assert derive_state(log.entries(), rebuild).stage is Stage.BUILD

    missing = {edge: target for edge, target in DEFAULT_TRANSITIONS.items() if edge != (Stage.REVIEW, Exit.MODIFY)}
    with pytest.raises(ValueError, match=r"missing \['review/modify'\]"):
        check_transitions(missing)
    with pytest.raises(ValueError, match="only a closing edge leads to done"):
        check_transitions({**DEFAULT_TRANSITIONS, (Stage.REVIEW, Exit.RETRY): Stage.DONE})
