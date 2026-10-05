# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from taskforge.ledger.records import EntryKind, LedgerEntry, span


class ListLedger:
    def __init__(self):
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


def test_span_records_caller_fields_and_timing():
    ledger = ListLedger()
    with span(ledger, EntryKind.LLM_CALL, item_id="i1", round=2, step="author", input_hash="in") as s:
        s.model = "glm-5.3"
        s.tokens_in, s.tokens_out, s.tokens_reasoning = 10, 20, 5
        s.finish_reason = "stop"
        s.attrs["tier"] = "high"

    (entry,) = ledger.entries
    assert (entry.item_id, entry.round, entry.step, entry.kind) == ("i1", 2, "author", EntryKind.LLM_CALL)
    assert (entry.model, entry.tokens_in, entry.tokens_out, entry.tokens_reasoning) == ("glm-5.3", 10, 20, 5)
    assert entry.input_hash == "in" and entry.cause is None and entry.attrs == {"tier": "high"}
    assert entry.ended >= entry.started


def test_span_records_exception_class_as_cause_and_reraises():
    ledger = ListLedger()
    with pytest.raises(TimeoutError):
        with span(ledger, EntryKind.SANDBOX_OP, item_id="i1", round=0, step="run") as s:
            s.attrs["cmd"] = "pytest"
            raise TimeoutError("stalled")

    (entry,) = ledger.entries
    assert entry.cause == "TimeoutError"
    assert entry.attrs == {"cmd": "pytest"}


def test_span_records_task_cancellation():
    ledger = ListLedger()

    async def work():
        with span(ledger, EntryKind.TRIAL, item_id="i1", round=0, step="trial-0"):
            await asyncio.sleep(10)

    async def main():
        task = asyncio.create_task(work())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert [e.cause for e in ledger.entries] == ["CancelledError"]


def test_span_rejects_bad_item_id_before_running_the_block():
    ran = False
    with pytest.raises(ValueError):
        with span(ListLedger(), EntryKind.STEP, item_id="../escape", round=0, step="build"):
            ran = True
    assert not ran


class FailingLedger:
    def record(self, entry: LedgerEntry) -> None:
        raise OSError("disk full")


def test_record_failure_does_not_replace_in_flight_cancellation():
    async def work():
        with span(FailingLedger(), EntryKind.TRIAL, item_id="i1", round=0, step="trial-0"):
            await asyncio.sleep(10)

    async def main():
        task = asyncio.create_task(work())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as info:
            await task
        return info.value

    cancelled = asyncio.run(main())
    assert any("disk full" in note for note in cancelled.__notes__)


def test_caller_classified_cause_wins_over_exception_class():
    ledger = ListLedger()
    with pytest.raises(TimeoutError):
        with span(ledger, EntryKind.SANDBOX_OP, item_id="i1", round=0, step="run") as s:
            s.cause = "sandbox_timeout"
            raise TimeoutError("stalled")
    assert [e.cause for e in ledger.entries] == ["sandbox_timeout"]
