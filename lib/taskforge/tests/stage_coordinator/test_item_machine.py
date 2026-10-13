# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The item state as a fold over its events, and the fold's refusal of a broken log."""

from dataclasses import replace

import pytest

from taskforge.stage30_author.author import RevisionKind
from taskforge.stage60_review.decision import RejectKind
from taskforge.stage_coordinator.item_machine import (
    DEFAULT_TRANSITIONS,
    EXIT,
    EventKind,
    Exit,
    ItemEvent,
    Stage,
    Terminal,
    check_transitions,
    derive_state,
)

OPENED = ItemEvent(EventKind.OPENED, 0, "p1", {"budget_digest": "b1"})
TRIAGED = ItemEvent(EventKind.TRIAGED, 0, "p1", {EXIT: "accept"})
AUTHORED = ItemEvent(EventKind.AUTHORED, 0, "prog1", {EXIT: "authored", "revision": "none"})
BUILT = ItemEvent(EventKind.BUILT, 0, "task1", {EXIT: "built", "program_digest": "prog1", "steps": "3", "hits": "0"})
VALIDATED = ItemEvent(EventKind.VALIDATED, 0, "task1", {EXIT: "validated", "summary_digest": "s1"})


def decided(**attrs: str) -> ItemEvent:
    return ItemEvent(EventKind.DECIDED, 0, "task1", attrs)


def terminal(**attrs: str) -> ItemEvent:
    return ItemEvent(EventKind.TERMINAL, 0, None, attrs)


def numbered(*events: ItemEvent) -> list[ItemEvent]:
    return [replace(event, seq=seq) for seq, event in enumerate(events, start=1)]


VALIDATED_LOG = (OPENED, TRIAGED, AUTHORED, BUILT, VALIDATED)


def test_a_built_and_validated_item_folds_to_review_with_its_digests():
    state = derive_state(numbered(*VALIDATED_LOG))

    assert (state.stage, state.round, state.seq) == (Stage.REVIEW, 0, 5)
    assert (state.proposal_digest, state.program_digest, state.task_digest) == ("p1", "prog1", "task1")
    assert (state.revision, state.terminal, state.closing) == (RevisionKind.NONE, None, None)


def test_a_retry_past_the_budget_abandons_and_a_relaunch_re_enters_validation_with_no_retries():
    retry = decided(exit="retry", cause="model_unavailable", count="3", retries_used="1")
    abandon = decided(exit="abandon", cause="model_unavailable", count="4", retries_used="2")
    log = [*VALIDATED_LOG, retry]
    retried = derive_state(numbered(*log))
    log += [VALIDATED, abandon]
    closing = derive_state(numbered(*log)).closing
    log.append(terminal(terminal="abandoned", reason="model_unavailable"))
    abandoned = derive_state(numbered(*log))
    resumed = derive_state(numbered(*log, VALIDATED))

    assert (retried.stage, retried.retries, retried.task_digest) == (Stage.VALIDATE, 1, "task1")
    assert closing is not None and (closing.terminal, closing.reason) == (Terminal.ABANDONED, "model_unavailable")
    assert (abandoned.terminal, abandoned.stage, abandoned.retries) == (Terminal.ABANDONED, Stage.VALIDATE, 0)
    assert (resumed.terminal, resumed.stage) == (None, Stage.REVIEW)


@pytest.mark.parametrize(
    ("way", "stage", "closing"),
    [("revise", Stage.AUTHOR, None), ("reject", Stage.DONE, (Terminal.REJECTED, RejectKind.BUDGET))],
)
def test_a_build_failure_revises_or_rejects_as_its_event_says(way, stage, closing):
    failed = ItemEvent(
        EventKind.BUILD_FAILED, 0, "prog1", {EXIT: way, "step": "grader", "failure": "boom", "failure_file": "f"}
    )

    state = derive_state(numbered(OPENED, TRIAGED, AUTHORED, failed))

    assert (state.stage, state.build_revisions, state.revision) == (stage, 1, RevisionKind.BUILD_FAILURE)
    assert (None if state.closing is None else (state.closing.terminal, state.closing.kind)) == closing


@pytest.mark.parametrize(
    ("way", "stage", "closing"),
    [("revise", Stage.AUTHOR, None), ("reject", Stage.DONE, (Terminal.REJECTED, RejectKind.BUDGET))],
)
def test_an_author_failure_revises_or_rejects_as_its_event_says(way, stage, closing):
    failed = ItemEvent(EventKind.AUTHOR_FAILED, 0, "p1", {EXIT: way, "failure": "cut", "failure_file": "f"})

    state = derive_state(numbered(OPENED, TRIAGED, failed))

    assert (state.stage, state.build_revisions, state.revision) == (stage, 1, RevisionKind.AUTHOR_FAILURE)
    assert (None if state.closing is None else (state.closing.terminal, state.closing.kind)) == closing


def test_an_author_failure_after_a_build_failure_spends_the_same_budget():
    build_failed = ItemEvent(
        EventKind.BUILD_FAILED, 0, "prog1", {EXIT: "revise", "step": "", "failure": "boom", "failure_file": "f1"}
    )
    author_failed = ItemEvent(EventKind.AUTHOR_FAILED, 0, "p1", {EXIT: "revise", "failure": "cut", "failure_file": "f2"})

    state = derive_state(numbered(OPENED, TRIAGED, AUTHORED, build_failed, author_failed, AUTHORED))

    assert (state.stage, state.build_revisions, state.revision) == (Stage.BUILD, 2, RevisionKind.AUTHOR_FAILURE)


def test_a_modify_decision_starts_the_next_round_and_its_revision_holds_until_built():
    log = [*VALIDATED_LOG, decided(exit="modify", findings="too easy")]
    modified = derive_state(numbered(*log))
    log.append(ItemEvent(EventKind.AUTHORED, 1, "prog2", {EXIT: "authored", "revision": "modify"}))
    authored = derive_state(numbered(*log))
    built = derive_state(numbered(*log, replace(BUILT, round=1, digest="task2")))

    assert (modified.stage, modified.round, modified.revision, modified.task_digest) == (
        Stage.AUTHOR,
        1,
        RevisionKind.MODIFY,
        None,
    )
    assert (authored.stage, authored.program_digest, authored.revision) == (Stage.BUILD, "prog2", RevisionKind.MODIFY)
    assert (built.task_digest, built.revision) == ("task2", RevisionKind.NONE)


@pytest.mark.parametrize(
    ("band", "reason"), [("in_band", "calibrated"), ("too_easy", "accepted outside the band: too_easy")]
)
def test_an_accept_closes_accepted_with_where_the_task_fell(band, reason):
    log = [*VALIDATED_LOG, decided(exit="accept", band=band)]
    closing = derive_state(numbered(*log)).closing
    accepted = derive_state(numbered(*log, terminal(terminal="accepted", reason=reason)))

    assert closing is not None and (closing.terminal, closing.reason) == (Terminal.ACCEPTED, reason)
    assert accepted.terminal is Terminal.ACCEPTED


def test_a_failed_item_keeps_its_stage_and_a_final_terminal_ends_the_log():
    log = [*VALIDATED_LOG, terminal(terminal="failed", reason="RuntimeError: boom")]
    failed = derive_state(numbered(*log))
    log += [decided(exit="reject", kind="host", reasons="no machine"), terminal(terminal="rejected", reason="r")]
    rejected = derive_state(numbered(*log))

    assert (failed.terminal, failed.stage) == (Terminal.FAILED, Stage.REVIEW)
    assert rejected.terminal is Terminal.REJECTED
    with pytest.raises(ValueError, match="after terminal rejected"):
        derive_state(numbered(*log, VALIDATED))


@pytest.mark.parametrize(
    ("log", "message"),
    [
        ((OPENED, TRIAGED, VALIDATED), "recorded by stage validate, but the item is in stage author"),
        ((OPENED, ItemEvent(EventKind.TRIAGED, 0, "p1", {EXIT: "retry"})), "exits retry, which stage triage has no"),
        ((*VALIDATED_LOG, terminal(terminal="accepted", reason="r")), "TERMINAL accepted in stage review"),
        ((TRIAGED,), "starts with OPENED"),
    ],
    ids=["out of stage", "foreign exit", "unclosed terminal", "no opened"],
)
def test_an_event_out_of_place_is_refused(log, message):
    with pytest.raises(ValueError, match=message):
        derive_state(numbered(*log))


@pytest.mark.parametrize(("shift", "message"), [(1, "seq 7 where 6"), (-1, "seq 5 where 6")], ids=["gap", "repeat"])
def test_a_gap_or_a_repeat_in_seq_is_refused(shift, message):
    log = numbered(*VALIDATED_LOG)
    log.append(replace(decided(exit="modify", findings="0"), seq=6 + shift))

    with pytest.raises(ValueError, match=message):
        derive_state(log)


def test_a_caller_table_re_targets_an_edge_and_a_table_missing_an_exit_is_refused():
    rebuild = {**DEFAULT_TRANSITIONS, (Stage.REVIEW, Exit.RETRY): Stage.BUILD}
    check_transitions(rebuild)
    log = numbered(*VALIDATED_LOG, decided(exit="retry", cause="model_unavailable", count="1", retries_used="1"))
    assert derive_state(log, rebuild).stage is Stage.BUILD

    missing = {edge: target for edge, target in DEFAULT_TRANSITIONS.items() if edge != (Stage.REVIEW, Exit.MODIFY)}
    with pytest.raises(ValueError, match=r"missing \['review/modify'\]"):
        check_transitions(missing)
    with pytest.raises(ValueError, match="only a closing edge leads to done"):
        check_transitions({**DEFAULT_TRANSITIONS, (Stage.REVIEW, Exit.RETRY): Stage.DONE})
