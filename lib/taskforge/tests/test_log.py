# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from lib.taskforge.tests.effect_fakes import Answer, Choices, FlakySink, PausedSink, Query, Timed, local_records

from taskforge.effects.events import (
    Event,
    Fingerprint,
    Note,
    ReplayPolicy,
    Resolved,
    Retrying,
    StageCompleted,
    event_digest,
)
from taskforge.effects.issues import LogFailure
from taskforge.effects.log import EventLog, JsonlSink, Missing, Record, Status, decode, fold
from taskforge.llm.policy import LLMPolicy
from taskforge.model.llm import Completion, Message, Role
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal
from taskforge.resources.inference import InferenceRequest

pytestmark = pytest.mark.asyncio


async def test_jsonl_reload_in_new_process_preserves_typed_results_and_replay_keys(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonlSink(str(path))
    log = EventLog(sink=sink)
    event = Query({"z": "last", "a": "first"})
    fingerprint = Fingerprint("fake", "digest", ReplayPolicy.CACHE)
    answer = Answer(b"\x00\xff", {"__forge_bytes__": "this is text"})
    await log.record(event)
    await log.record(Retrying(event, 1, "busy"))
    await log.record(Resolved(event, (fingerprint,), answer))
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


async def test_replay_preserves_float_keys_after_numeric_coercion(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "numbers.jsonl"))
    log = EventLog(sink=sink)
    await log.record(Resolved(Timed(5), (), 3))
    reloaded = EventLog(replay_from=local_records(sink))
    assert reloaded.lookup(Timed(5.0), ()) == 3.0


async def test_non_replayable_result_is_recorded_but_not_served_from_cache(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    log = EventLog(sink=sink, replayable=lambda _: False)
    resolved = Resolved(Timed(5), (), 3.0)
    await log.record(resolved)
    assert isinstance(log.lookup(Timed(5), ()), Missing)
    assert [record.event for record in local_records(sink)] == [resolved]


async def test_frozen_set_payloads_roundtrip_with_stable_replay_keys(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "sets.jsonl"))
    log = EventLog(sink=sink)
    event = Choices(frozenset({"second", "first"}))
    await log.record(Resolved(event, (), frozenset({"chosen"})))
    resumed = EventLog(replay_from=local_records(sink))
    assert resumed.lookup(Choices(frozenset({"first", "second"})), ()) == frozenset({"chosen"})


async def test_local_sink_appends_without_rewriting_prefix_and_retries_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "append.jsonl"
    sink = JsonlSink(str(path))
    first = Record(1, 1.0, Note("first"))
    second = Record(2, 2.0, Note("second"))
    third = Record(3, 3.0, Note("third"))
    await sink.write((first,))
    prefix = b"  " + path.read_bytes()
    path.write_bytes(prefix)
    inode = path.stat().st_ino
    await sink.write((second,))
    assert path.stat().st_ino == inode
    assert path.read_bytes().startswith(prefix)
    with path.open("ab") as output:
        output.write(b'{"sequence":3,"wall_time":')
    assert local_records(sink) == (first, second)
    await sink.write((second, third))
    assert local_records(sink) == (first, second, third)


async def test_log_retries_only_unwritten_records_and_retains_failed_batch() -> None:
    sink = FlakySink(1)
    log = EventLog(sink=sink, attempts=1)
    with pytest.raises(LogFailure):
        await log.record(Note("retained"))
    await log.record(Note("next"))
    await log.record(Note("last"))
    assert tuple(record.event for record in sink.records) == (Note("retained"), Note("next"), Note("last"))


async def test_unknown_recorded_kind_fails_closed_naming_it(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    await JsonlSink(str(path)).write((Record(1, 1.0, Note("known")),))
    path.write_text(path.read_text().replace('"effect.note"', '"taskforge.effects.events.Note"'))
    with pytest.raises(LogFailure, match=r"Unknown event kind 'taskforge\.effects\.events\.Note'"):
        decode(path.read_bytes())


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


async def test_unencodable_event_does_not_poison_later_persistence(tmp_path: Path) -> None:
    @dataclass(frozen=True, slots=True)
    class Invalid(Event[None]):
        kind = "test.invalid"

        opaque: object

    sink = JsonlSink(str(tmp_path / "valid.jsonl"))
    log = EventLog(sink=sink)
    await log.record(Note("before"))
    with pytest.raises(LogFailure):
        await log.record(Invalid(object()))
    await log.record(Note("after"))
    assert tuple(record.event for record in local_records(sink)) == (Note("before"), Note("after"))
    assert tuple(record.sequence for record in log.records) == (1, 2)


async def test_conversation_logs_each_turn_once_and_replays_chained_requests(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "conversation.jsonl"))
    log = EventLog(sink=sink)
    prefix = "long-system-prompt-" * 1000
    first = InferenceRequest((Message(Role.SYSTEM, prefix),), (), LLMPolicy())
    second = InferenceRequest((Message(Role.USER, "follow-up"),), (), LLMPolicy(), first)
    for request in (first, second):
        await log.record(request)
        await log.record(Resolved(request, (), Completion("reply")))
    content = Path(sink.path).read_text()
    assert content.count(prefix) == 1
    assert content.count("follow-up") == 1
    resumed = EventLog(replay_from=local_records(sink))
    assert resumed.lookup(second, ()) == Completion("reply")
    assert [r.event for r in resumed.records if isinstance(r.event, Resolved)][-1].request == second


@pytest.mark.parametrize("number", [float("inf"), float("-inf"), float("nan")])
async def test_nonfinite_event_values_roundtrip_and_replay(tmp_path: Path, number: float) -> None:
    sink = JsonlSink(str(tmp_path / "nonfinite.jsonl"))
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
    # Rewriting the same append batch must preserve the wire representation.
    before = Path(sink.path).read_bytes()
    await sink.write(loaded[-1:])
    assert Path(sink.path).read_bytes() == before
