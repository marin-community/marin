# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invariants: none (pins the ledger span's fields, cause and cancellation)."""

import asyncio

import pytest

from taskforge.ledger.records import EntryKind, LedgerEntry, span


def test_span_records_caller_fields_and_timing(ledger):
    with span(ledger, EntryKind.LLM_CALL, workpiece_id="i1", round=2, step="author", input_hash="in") as s:
        s.model = "glm-5.3"
        s.tokens_in, s.tokens_out, s.tokens_reasoning = 10, 20, 5
        s.finish_reason = "stop"
        s.attrs["tier"] = "high"

    (entry,) = ledger.entries
    assert (entry.workpiece_id, entry.round, entry.step, entry.kind) == ("i1", 2, "author", EntryKind.LLM_CALL)
    assert (entry.model, entry.tokens_in, entry.tokens_out, entry.tokens_reasoning) == ("glm-5.3", 10, 20, 5)
    assert entry.input_hash == "in" and entry.cause is None and entry.attrs == {"tier": "high"}
    assert entry.ended >= entry.started


@pytest.mark.parametrize("classified", [None, "sandbox_timeout"])
def test_span_records_the_raised_exception_class_or_the_callers_cause_and_reraises(ledger, classified):
    with pytest.raises(TimeoutError):
        with span(ledger, EntryKind.STEP, workpiece_id="i1", round=0, step="run") as s:
            s.attrs["cmd"] = "pytest"
            s.cause = classified
            raise TimeoutError("stalled")

    (entry,) = ledger.entries
    assert entry.cause == (classified or "TimeoutError")
    assert entry.attrs == {"cmd": "pytest"}


def test_span_rejects_bad_workpiece_id_before_running_the_block(ledger):
    ran = False
    with pytest.raises(ValueError):
        with span(ledger, EntryKind.STEP, workpiece_id="../escape", round=0, step="build"):
            ran = True
    assert not ran


class FailingLedger:
    def record(self, entry: LedgerEntry) -> None:
        raise OSError("disk full")


def cancel_in_flight(ledger) -> asyncio.CancelledError:
    """Cancel a task while it is inside a ``TRIAL`` span and return the ``CancelledError`` it raised."""

    async def work():
        with span(ledger, EntryKind.TRIAL, workpiece_id="i1", round=0, step="trial-0"):
            await asyncio.sleep(10)

    async def main():
        task = asyncio.create_task(work())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as info:
            await task
        return info.value

    return asyncio.run(main())


def test_span_records_task_cancellation_and_a_record_failure_does_not_replace_it(ledger):
    cancel_in_flight(ledger)
    assert [e.cause for e in ledger.entries] == ["CancelledError"]

    assert any("disk full" in note for note in cancel_in_flight(FailingLedger()).__notes__)
