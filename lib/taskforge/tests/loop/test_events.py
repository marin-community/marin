# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The event log: rows through the ledger, and item status as a fold that refuses a broken log."""

from dataclasses import replace

import pytest

from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.ledger.records import EntryKind, LedgerEntry
from taskforge.loop.events import (
    EVENT_SCHEMA,
    EventKind,
    Phase,
    RevisionKind,
    Terminal,
    derive_idea_state,
    derive_state,
    item_tokens_out,
    record_event,
)
from taskforge.validate.calibration import FindingKind

ITEM = "d00.x--1"


class Log:
    """Appends events to a real JSONL ledger with contiguous ``seq`` and reads them back."""

    def __init__(self, root):
        self.ledger = JsonlLedger(root)
        self.seq = 0

    def add(self, kind: EventKind, /, at_round: int = 0, input_hash: str | None = None, **attrs: str) -> "Log":
        self.seq += 1
        record_event(self.ledger, ITEM, at_round, kind, self.seq, input_hash, **attrs)
        return self

    def entries(self) -> list[LedgerEntry]:
        return list(read_entries(self.ledger.path_for(ITEM)))


def built_log(tmp_path) -> Log:
    return (
        Log(tmp_path)
        .add(EventKind.OPENED, input_hash="p1", proposal="d00.x/1", idea="d00.x", origin="supplied", policy_digest="pol")
        .add(EventKind.TRIAGED, input_hash="p1", decision="accept", tally="1 accept", repairs="0")
        .add(EventKind.AUTHORED, input_hash="prog1", revises="", revision="none")
        .add(EventKind.BUILT, input_hash="task1", program_digest="prog1", steps="5", hits="0", staged="false")
    )


def test_a_recorded_log_folds_through_repair_and_retry(tmp_path):
    log = built_log(tmp_path).add(
        EventKind.CONTROLS_REPLAYED, input_hash="task1", met="4", violated="0", ungraded="0", passed="true"
    )
    assert derive_state(log.entries()).phase is Phase.TRIALS
    log.add(EventKind.SOLVED, graded="8", solved="8", timed_out="0", ungraded="0")
    log.add(EventKind.ADVERSARIES_RUN)
    assert derive_state(log.entries()).phase is Phase.DECIDE
    log.add(
        EventKind.DECIDED,
        input_hash="task1",
        decision="repair",
        findings="too_easy",
        invalidate="grader,controls",
        repairs_used="1",
        retries_used="0",
    )

    state = derive_state(log.entries())

    assert (state.phase, state.round, state.revision) == (Phase.BUILD, 1, RevisionKind.REPAIR)
    assert state.program_digest is None and state.repaired_task_digest == "task1" and state.repair_round == 0
    assert state.invalidate == ("grader", "controls") and state.repairs_used == 1
    assert state.band_repairs == {FindingKind.TOO_EASY: 1}
    assert not state.solved and not state.adversaries_run

    log.add(EventKind.AUTHORED, at_round=1, input_hash="prog2", revises="r", revision="repair")
    log.add(EventKind.BUILT, at_round=1, input_hash="task2", program_digest="prog2", steps="5", hits="3", staged="false")
    log.add(
        EventKind.CONTROLS_REPLAYED, at_round=1, input_hash="task2", met="5", violated="0", ungraded="0", passed="true"
    )
    log.add(EventKind.SOLVED, at_round=1)
    log.add(EventKind.ADVERSARIES_RUN, at_round=1)
    log.add(
        EventKind.DECIDED,
        at_round=1,
        input_hash="task2",
        decision="retry",
        cause="model_unavailable",
        count="3",
        abandon="false",
        not_before="1234.5",
        repairs_used="1",
        retries_used="1",
    )

    state = derive_state(log.entries())

    assert (state.phase, state.not_before, state.validation_retries) == (Phase.CONTROLS, 1234.5, 1)
    assert state.invalidate == ("grader", "controls") and state.task_digest == "task2"


def accepted_log(tmp_path, band: str) -> Log:
    log = built_log(tmp_path).add(
        EventKind.CONTROLS_REPLAYED, input_hash="task1", met="4", violated="0", ungraded="0", passed="true"
    )
    log.add(EventKind.SOLVED).add(EventKind.ADVERSARIES_RUN)
    return log.add(
        EventKind.DECIDED,
        input_hash="task1",
        decision="accept",
        band=band,
        solved="8",
        graded="8",
        solve_rate="1.000",
        repairs_used="0",
        retries_used="0",
        notes="",
    )


@pytest.mark.parametrize(
    ("band", "reason"), [("in_band", "calibrated"), ("too_easy", "accepted outside the band: too_easy")]
)
def test_an_accept_closes_accepted_with_where_the_task_fell(tmp_path, band, reason):
    closing = derive_state(accepted_log(tmp_path, band).entries()).closing

    assert closing is not None and (closing.terminal, closing.reason) == (Terminal.ACCEPTED, reason)


def test_an_abandoned_item_re_enters_validation_with_a_fresh_retry_count(tmp_path):
    log = built_log(tmp_path).add(
        EventKind.CONTROLS_REPLAYED, input_hash="task1", met="4", violated="0", ungraded="0", passed="true"
    )
    log.add(EventKind.SOLVED).add(EventKind.ADVERSARIES_RUN)
    log.add(EventKind.DECIDED, decision="retry", cause="model_unavailable", count="8", abandon="true", not_before="0")
    assert derive_state(log.entries()).phase is Phase.DONE
    log.add(EventKind.TERMINAL, terminal="abandoned", reason="model_unavailable")

    abandoned = derive_state(log.entries())
    log.add(EventKind.CONTROLS_REPLAYED, input_hash="task1", met="4", violated="0", ungraded="0", passed="true")
    resumed = derive_state(log.entries())

    assert (abandoned.terminal, abandoned.phase, abandoned.validation_retries) == (Terminal.ABANDONED, Phase.CONTROLS, 0)
    assert resumed.terminal is None and resumed.phase is Phase.TRIALS


def test_host_failures_hold_the_build_until_abandoned_and_the_relaunch_rebuilds_the_same_program(tmp_path):
    log = (
        Log(tmp_path)
        .add(EventKind.OPENED, input_hash="p1", proposal="d00.x/1", idea="d00.x", origin="supplied", policy_digest="pol")
        .add(EventKind.TRIAGED, input_hash="p1", decision="accept", tally="1 accept", repairs="0")
        .add(EventKind.AUTHORED, input_hash="prog1", revises="", revision="none")
    )
    failure = {"message": "connection reset", "retries_used": "1"}
    log.add(
        EventKind.BUILD_INFRASTRUCTURE,
        input_hash="prog1",
        cause="host_unreachable",
        abandon="false",
        not_before="99.5",
        **failure,
    )
    waiting = derive_state(log.entries())
    log.add(
        EventKind.BUILD_INFRASTRUCTURE, input_hash="prog1", cause="no_factory", abandon="true", not_before="0", **failure
    )
    closing = derive_state(log.entries()).closing
    log.add(
        EventKind.TERMINAL, terminal="abandoned", reason="build host failures", causes="host_unreachable:1,no_factory:1"
    )
    abandoned = derive_state(log.entries())

    assert (waiting.phase, waiting.program_digest, waiting.not_before) == (Phase.BUILD, "prog1", 99.5)
    assert waiting.build_revisions == 0
    assert closing is not None and closing.terminal is Terminal.ABANDONED
    assert closing.causes == ("host_unreachable", "no_factory")
    assert (abandoned.phase, abandoned.program_digest, abandoned.build_host_failures) == (Phase.BUILD, "prog1", ())


def test_a_failed_item_keeps_its_phase_and_a_final_terminal_ends_the_log(tmp_path):
    log = built_log(tmp_path).add(EventKind.TERMINAL, terminal="failed", reason="RuntimeError: boom")
    failed = derive_state(log.entries())
    assert (failed.terminal, failed.phase) == (Terminal.FAILED, Phase.CONTROLS)

    log.add(EventKind.TERMINAL, terminal="rejected", kind="host", reason="machine_unsupported")
    log.add(EventKind.CONTROLS_REPLAYED, input_hash="task1", met="4", violated="0", ungraded="0", passed="true")

    with pytest.raises(ValueError, match="after terminal rejected"):
        derive_state(log.entries())


def test_a_gap_or_a_repeat_in_seq_is_refused(tmp_path):
    log = built_log(tmp_path)
    log.seq += 1
    log.add(EventKind.CONTROLS_REPLAYED, input_hash="task1", met="4", violated="0", ungraded="0", passed="true")
    with pytest.raises(ValueError, match="seq 6 where 5"):
        derive_state(log.entries())

    repeated = built_log(tmp_path / "repeat")
    repeated.seq -= 1
    repeated.add(EventKind.CONTROLS_REPLAYED, input_hash="task1", met="4", violated="0", ungraded="0", passed="true")
    with pytest.raises(ValueError, match="seq 4 where 5"):
        derive_state(repeated.entries())


def test_an_unknown_schema_is_refused(tmp_path):
    entries = built_log(tmp_path).entries()
    future = replace(entries[-1], attrs={**entries[-1].attrs, "schema": str(int(EVENT_SCHEMA) + 1)})
    with pytest.raises(ValueError, match="event schema"):
        derive_state([*entries[:-1], future])


def test_the_budget_counts_only_llm_call_output_tokens_outside_validation_trials(tmp_path):
    log = built_log(tmp_path)
    spans = [
        LedgerEntry(ITEM, 0, "author", EntryKind.LLM_CALL, 0.0, 1.0, tokens_out=700),
        LedgerEntry(ITEM, 0, "grader", EntryKind.LLM_CALL, 0.0, 1.0, tokens_out=None),
        LedgerEntry(ITEM, 0, "solver/0/0", EntryKind.TRIAL, 0.0, 1.0, tokens_out=9000),
        LedgerEntry(ITEM, 0, "solver/0", EntryKind.LLM_CALL, 0.0, 1.0, tokens_out=9000),
        LedgerEntry(ITEM, 0, "adversary/shortcut/1", EntryKind.LLM_CALL, 0.0, 1.0, tokens_out=9000),
        # Build steps record calls under the bare step name; one named like a trial kind still counts.
        LedgerEntry(ITEM, 0, "solver", EntryKind.LLM_CALL, 0.0, 1.0, tokens_out=40),
        LedgerEntry(ITEM, 0, "adversary", EntryKind.LLM_CALL, 0.0, 1.0, tokens_out=2),
    ]
    assert item_tokens_out([*log.entries(), *spans]) == 742
    assert derive_state([*log.entries(), *spans]).phase is Phase.CONTROLS


def test_an_idea_log_records_its_batch_or_its_exhaustion(tmp_path):
    ledger = JsonlLedger(tmp_path)
    record_event(ledger, "idea--d00.x", 0, EventKind.SLOT_FAILED, 1, None, slot="0", error="bad yaml")
    record_event(
        ledger, "idea--d00.x", 0, EventKind.PROPOSED, 2, None, items="", slots="1", failures="1", reproposal="0"
    )
    record_event(
        ledger, "idea--d00.x", 1, EventKind.PROPOSED, 3, None, items="a,b", slots="2", failures="0", reproposal="1"
    )

    state = derive_idea_state(read_entries(ledger.path_for("idea--d00.x")))

    assert (state.items, state.reproposals, state.exhausted, state.seq) == (("a", "b"), 1, False, 3)
