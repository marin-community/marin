# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from lib.taskforge.tests.effect_fakes import Answer, Choices, FlakySink, PausedSink, Query, Timed, local_records

from taskforge.effects.context import Context
from taskforge.effects.events import (
    Event,
    Fingerprint,
    Invalidated,
    Note,
    ReplayPolicy,
    Resolved,
    Retrying,
    StageCompleted,
    event_digest,
)
from taskforge.effects.issues import LogFailure, Restart
from taskforge.effects.log import BufferSink, EventLog, Missing, Record, Status, decode, fold
from taskforge.llm.policy import LLMPolicy
from taskforge.model.llm import Completion, Message, Role
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal
from taskforge.resources.inference import InferenceRequest

pytestmark = pytest.mark.asyncio


async def test_jsonl_reload_in_new_process_preserves_typed_results_and_replay_keys(tmp_path: Path) -> None:
    path, sink = tmp_path / "events.jsonl", BufferSink()
    log = EventLog(sink=sink)
    event = Query({"z": "last", "a": "first"})
    fingerprint = Fingerprint("fake", "digest", ReplayPolicy.CACHE)
    answer = Answer(b"\x00\xff", {"__forge_bytes__": "this is text"})
    await log.record(event)
    await log.record(Retrying(event, 1, "busy"))
    await log.record(Resolved(event, (fingerprint,), answer))
    path.write_bytes(sink.data)
    loaded = local_records(sink)
    assert loaded == log.records
    resumed = EventLog(replay_from=loaded)
    assert resumed.lookup(Query({"a": "first", "z": "last"}), (fingerprint,)) == answer
    script = """
import asyncio, importlib.util, sys
from taskforge.effects.events import Fingerprint, ReplayPolicy
from taskforge.effects.log import EventLog, decode
spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[3])
assert spec is not None and spec.loader is not None
fake = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fake
spec.loader.exec_module(fake)
async def main():
    log = EventLog(replay_from=decode(open(sys.argv[1], 'rb').read()))
    value = log.lookup(fake.Query({"a":"first", "z":"last"}), (Fingerprint("fake", "digest", ReplayPolicy.CACHE),))
    assert isinstance(value, fake.Answer)
    assert value.data == bytes([0, 255])
    assert value.labels == {"__forge_bytes__": "this is text"}
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), Query.__module__, str(Path(__file__).with_name("effect_fakes.py"))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    await resumed.record(StageCompleted(Stage.REVIEW, "cfg", "accept", 2, Terminal.ACCEPTED))
    assert resumed.records[-1].sequence == 4
    assert fold(tuple(record.event for record in resumed.records)) == Status(Stage.REVIEW, 2, Terminal.ACCEPTED)


async def test_replay_preserves_float_keys_after_numeric_coercion() -> None:
    sink = BufferSink()
    log = EventLog(sink=sink)
    await log.record(Resolved(Timed(5), (), 3))
    reloaded = EventLog(replay_from=local_records(sink))
    assert reloaded.lookup(Timed(5.0), ()) == 3.0


async def test_non_replayable_result_is_recorded_but_not_served_from_cache() -> None:
    sink = BufferSink()
    log = EventLog(sink=sink, replayable=lambda _: False)
    resolved = Resolved(Timed(5), (), 3.0)
    await log.record(resolved)
    assert isinstance(log.lookup(Timed(5), ()), Missing)
    assert [record.event for record in local_records(sink)] == [resolved]


async def test_frozen_set_payloads_roundtrip_with_stable_replay_keys() -> None:
    sink = BufferSink()
    log = EventLog(sink=sink)
    event = Choices(frozenset({"second", "first"}))
    await log.record(Resolved(event, (), frozenset({"chosen"})))
    resumed = EventLog(replay_from=local_records(sink))
    assert resumed.lookup(Choices(frozenset({"first", "second"})), ()) == frozenset({"chosen"})


async def test_log_retries_only_unwritten_records_and_retains_failed_batch() -> None:
    sink = FlakySink(1)
    log = EventLog(sink=sink, attempts=1)
    with pytest.raises(LogFailure):
        await log.record(Note("retained"))
    await log.record(Note("next"))
    await log.record(Note("last"))
    assert tuple(record.event for record in sink.records) == (Note("retained"), Note("next"), Note("last"))


async def test_unknown_recorded_kind_fails_closed_naming_it() -> None:
    sink = BufferSink()
    await sink.write((Record(1, 1.0, Note("known")),))
    with pytest.raises(LogFailure, match=r"Unknown event kind 'taskforge\.effects\.events\.Note'"):
        decode(bytes(sink.data).replace(b'"effect.note"', b'"taskforge.effects.events.Note"'))


async def test_records_queued_during_upload_coalesce_into_one_incremental_batch() -> None:
    sink = PausedSink()
    log = EventLog(sink=sink)
    first = asyncio.create_task(log.record(Note("first")))
    await sink.started.wait()
    second = asyncio.create_task(log.record(Note("second")))
    third = asyncio.create_task(log.record(Note("third")))
    # The queued records enter before the callback releases the blocked sink.
    asyncio.get_running_loop().call_soon(sink.release.set)
    await asyncio.gather(first, second, third)
    assert tuple(tuple(record.event for record in batch) for batch in sink.batches) == (
        (Note("first"),),
        (Note("second"), Note("third")),
    )


async def test_unencodable_event_does_not_poison_later_persistence() -> None:
    @dataclass(frozen=True, slots=True)
    class Invalid(Event[None]):
        kind = "test.invalid"

        opaque: object

    sink = BufferSink()
    log = EventLog(sink=sink)
    await log.record(Note("before"))
    with pytest.raises(LogFailure):
        await log.record(Invalid(object()))
    await log.record(Note("after"))
    assert tuple(record.event for record in local_records(sink)) == (Note("before"), Note("after"))
    assert tuple(record.sequence for record in log.records) == (1, 2)


async def test_conversation_logs_each_turn_once_and_replays_chained_requests() -> None:
    sink = BufferSink()
    log = EventLog(sink=sink)
    prefix = "long-system-prompt-" * 1000
    first = InferenceRequest((Message(Role.SYSTEM, prefix),), (), LLMPolicy())
    second = InferenceRequest((Message(Role.USER, "follow-up"),), (), LLMPolicy(), first)
    for request in (first, second):
        await log.record(request)
        await log.record(Resolved(request, (), Completion("reply")))
    content = sink.data.decode()
    assert content.count(prefix) == 1
    assert content.count("follow-up") == 1
    resumed = EventLog(replay_from=local_records(sink))
    assert resumed.lookup(second, ()) == Completion("reply")
    assert [r.event for r in resumed.records if isinstance(r.event, Resolved)][-1].request == second


@pytest.mark.parametrize("number", [float("inf"), float("-inf"), float("nan")])
async def test_nonfinite_event_values_roundtrip_and_replay(number: float) -> None:
    sink = BufferSink()
    log = EventLog(sink=sink)
    request = Timed(number)
    await log.record(request)
    await log.record(Resolved(request, (), number))
    loaded = local_records(sink)
    resumed = EventLog(replay_from=loaded)
    value = resumed.lookup(Timed(number), ())
    assert isinstance(value, float)
    assert math.isnan(value) if math.isnan(number) else value == number
    assert event_digest(loaded[0].event) == event_digest(request)
    # Retrying an append batch must preserve the wire representation; a gap is refused.
    before = bytes(sink.data)
    await sink.write(loaded[-1:])
    assert sink.data == before
    with pytest.raises(LogFailure, match="gap"):
        await sink.write((Record(4, 4.0, Note("after a gap")),))


HELD_JOURNAL = Path(__file__).parent / "fixtures" / "n7-single-adversary-held.jsonl.gz"
COLD_READER = """
import gzip, sys
import taskforge.driver
from taskforge.effects.log import decode
print(len(decode(gzip.decompress(open(sys.argv[1], "rb").read()))))
"""


async def test_driver_import_decodes_held_journal_in_cold_process() -> None:
    result = subprocess.run(
        [sys.executable, "-c", COLD_READER, str(HELD_JOURNAL)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["57"]


class Paid:
    """A cached resource bought under ``profile``; ``calls`` collects the profile of each paid execution."""

    def __init__(self, profile: str, calls: list[str]) -> None:
        self.profile, self.calls = profile, calls

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("paid", self.profile, ReplayPolicy.CACHE)


async def pay(event: Timed, resource: Paid) -> float:
    resource.calls.append(resource.profile)
    return event.timeout * 2


async def launch(path: Path, profile: str, calls: list[str]) -> None:
    """One cold process: the journal at ``path`` alone, then one request of the paid effect, appended to it."""
    sink = BufferSink()
    ctx = Context(log=EventLog(sink=sink, replay_from=sink.start(path.read_bytes()) if path.exists() else ()))
    ctx.handle(Timed, pay, Paid(profile, calls))
    await ctx.emit(Timed(5))
    await ctx.drain()
    with path.open("ab") as journal:
        journal.write(sink.data)


# Regression: a bare request row discarded the cached result for its request digest under
# every fingerprint. Cold runs under profiles A, then B, then A paid three times; the third replays A's result.
async def test_a_result_under_one_fingerprint_survives_a_run_under_another(tmp_path: Path) -> None:
    calls: list[str] = []
    for profile in ("A", "B", "A"):
        await launch(tmp_path / "journal.jsonl", profile, calls)
    assert calls == ["A", "B"]


# Regression: a replay's durable prefix ending between its request row and its replayed
# resolution discarded the paid result, and the next process paid again. No cut of a normal replay costs a call.
@pytest.mark.parametrize("rows", [2, 3, 4], ids=["before-request", "after-request", "after-resolution"])
async def test_no_durable_prefix_of_a_replay_costs_a_paid_call(tmp_path: Path, rows: int) -> None:
    journal, calls = tmp_path / "journal.jsonl", list[str]()
    for _ in range(2):  # a paid run, then its replay
        await launch(journal, "A", calls)
    cut = tmp_path / "cut.jsonl"
    cut.write_bytes(b"".join(journal.read_bytes().splitlines(keepends=True)[:rows]))
    await launch(cut, "A", calls)
    assert calls == ["A"]


# Regression: an invalidation names one result exactly, resource fingerprint included.
async def test_an_invalidation_for_one_fingerprint_leaves_another_fingerprints_result() -> None:
    sink, request = BufferSink(), Timed(5)
    x, y = Paid("X", []).fingerprint(), Paid("Y", []).fingerprint()
    log = EventLog(sink=sink)
    for event in (Resolved(request, (x,), 1.0), Resolved(request, (y,), 2.0), Invalidated(request, (x,))):
        await log.record(event)
    reloaded = EventLog(replay_from=local_records(sink))
    assert (reloaded.lookup(request, (y,)), type(reloaded.lookup(request, (x,)))) == (2.0, Missing)


REFUSED = 2.0
"""The stand-in refusal: a result the operator's explicit resume selects for replacement."""


async def resumed(path: Path, *, explicit: bool, outcome: float | None, paid: list[float]) -> EventLog:
    """One cold process over ``path`` asking ``Timed(5)``: explicit resumes replace a recorded refusal, and the paid
    call answers ``outcome`` or, with None, meets an outage."""
    sink = BufferSink()
    history = sink.start(path.read_bytes()) if path.exists() else ()
    stale = {id(record.event) for record in history if explicit}
    log = EventLog(sink=sink, replay_from=history, replaces=lambda event: id(event) in stale and event.value == REFUSED)
    ctx = Context(log=log)

    async def answer(_event: Timed, _resource: Paid) -> float:
        if outcome is None:
            raise Restart("the decision model is down")
        paid.append(outcome)
        return outcome

    ctx.handle(Timed, answer, Paid("A", []))
    with contextlib.suppress(Restart):
        await ctx.emit(Timed(5))
    await ctx.drain()
    with path.open("ab") as journal:
        journal.write(sink.data)
    return log


def markers(log: EventLog) -> int:
    return sum(isinstance(record.event, Invalidated) for record in log.records)


# Regression: one key recorded a success and then a refusal (two concurrent emits).
# An ordinary reload replays the latest, the refusal; an explicit one replayed the older success with no call and no
# marker. The latest result per key is what the operator's selection applies to.
async def test_an_explicit_resume_replaces_the_latest_result_and_never_replays_an_older_one(tmp_path: Path) -> None:
    journal, request, fingerprint = tmp_path / "journal.jsonl", Timed(5), Paid("A", []).fingerprint()
    log = EventLog(sink=(sink := BufferSink()))
    for value in (1.0, REFUSED):
        await log.record(Resolved(request, (fingerprint,), value))
    journal.write_bytes(sink.data)
    assert EventLog(replay_from=decode(journal.read_bytes())).lookup(request, (fingerprint,)) == REFUSED
    paid: list[float] = []
    log = await resumed(journal, explicit=True, outcome=3.0, paid=paid)
    assert (paid, markers(log), log.lookup(request, (fingerprint,))) == ([3.0], 1, 3.0)


# Regression: explicit resume, outage, explicit resume, outage wrote two markers
# for one refusal. A refusal already invalidated is not invalidated again, and the eventual replacement pays once.
async def test_one_refusal_gets_one_marker_across_explicit_resumes_cut_short_by_outages(tmp_path: Path) -> None:
    journal, paid = tmp_path / "journal.jsonl", list[float]()
    await resumed(journal, explicit=False, outcome=REFUSED, paid=paid)
    for _ in range(2):
        await resumed(journal, explicit=True, outcome=None, paid=paid)
    for explicit in (False, True):  # paid once, then replayed even by another explicit resume
        log = await resumed(journal, explicit=explicit, outcome=3.0, paid=paid)
    assert (paid, markers(log)) == ([REFUSED, 3.0], 1)


# Regression: a marker recorded live left its result readable in the live log and
# removed it on cold reload. One fold serves both: the live log and a cold rebuild of its journal agree.
@pytest.mark.parametrize("tail", ["marker", "marker-then-result"])
async def test_the_live_log_and_a_cold_rebuild_of_its_journal_agree_after_a_marker(tail: str) -> None:
    sink, request, fingerprint = BufferSink(), Timed(5), Paid("A", []).fingerprint()
    log = EventLog(sink=sink)
    events: list[Event[object]] = [Resolved(request, (fingerprint,), REFUSED), Invalidated(request, (fingerprint,))]
    for event in events + ([Resolved(request, (fingerprint,), 3.0)] if tail == "marker-then-result" else []):
        await log.record(event)
    cold = EventLog(replay_from=local_records(sink))
    found = [log.lookup(request, (fingerprint,)), cold.lookup(request, (fingerprint,))]
    shown = [None if isinstance(value, Missing) else value for value in found]
    assert shown == ([None, None] if tail == "marker" else [3.0, 3.0])
