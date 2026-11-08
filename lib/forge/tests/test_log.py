# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from taskforge.llm.policy import LLMPolicy

from forge.effects.events import Event, Fingerprint, Note, ReplayPolicy, Resolved, Retrying, StageCompleted, event_digest
from forge.effects.issues import LogFailure
from forge.effects.log import EventLog, JsonlSink, Record, Status, fold
from forge.model.llm import Completion, Message, Role
from forge.model.stage import Stage
from forge.model.workpiece import Terminal
from forge.resources.inference import InferenceRequest

from .effect_fakes import Answer, Choices, FlakySink, PausedSink, Query, Timed

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
    loaded = await sink.read()
    assert loaded == log.records
    resumed = EventLog(replay_from=loaded)
    assert resumed.lookup(Query({"a": "first", "z": "last"}), (fingerprint,)) == answer
    script = """
import asyncio, importlib.util, sys
from forge.effects.events import Fingerprint, ReplayPolicy
from forge.effects.log import EventLog, JsonlSink
spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[3])
assert spec is not None and spec.loader is not None
fake = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fake
spec.loader.exec_module(fake)
async def main():
    log = EventLog(replay_from=await JsonlSink(sys.argv[1]).read())
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
    reloaded = EventLog(replay_from=await sink.read())
    assert reloaded.lookup(Timed(5.0), ()) == 3.0


async def test_frozen_set_payloads_roundtrip_with_stable_replay_keys(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "sets.jsonl"))
    log = EventLog(sink=sink)
    event = Choices(frozenset({"second", "first"}))
    await log.record(Resolved(event, (), frozenset({"chosen"})))
    resumed = EventLog(replay_from=await sink.read())
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
    assert await sink.read() == (first, second)
    await sink.write((second, third))
    assert await sink.read() == (first, second, third)


async def test_log_retries_only_unwritten_records_and_retains_failed_batch() -> None:
    sink = FlakySink(1)
    log = EventLog(sink=sink, attempts=1)
    with pytest.raises(LogFailure):
        await log.record(Note("retained"))
    await log.record(Note("next"))
    await log.record(Note("last"))
    assert tuple(record.event for record in sink.records) == (Note("retained"), Note("next"), Note("last"))


async def test_object_store_sink_appends_batches_and_retries_idempotently() -> None:
    sink = JsonlSink("memory://forge-test/append.jsonl")
    first = Record(1, 1.0, Note("first"))
    second = Record(2, 2.0, Note("second"))
    await sink.write((first,))
    await sink.write((second,))
    await sink.write((second,))
    assert await sink.read() == (first, second)


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
        opaque: object

    sink = JsonlSink(str(tmp_path / "valid.jsonl"))
    log = EventLog(sink=sink)
    await log.record(Note("before"))
    with pytest.raises(LogFailure):
        await log.record(Invalid(object()))
    await log.record(Note("after"))
    assert tuple(record.event for record in await sink.read()) == (Note("before"), Note("after"))
    assert tuple(record.sequence for record in log.records) == (1, 2)


async def test_legacy_progress_resumes_nonterminal_stages_and_distinguishes_held(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        '{"sequence":1,"wall_time":1,"event":{"type":"forge.effects.events.StageCompleted",'
        '"payload":{"stage":"author","config_digest":"cfg","exit":"draft",'
        '"refs":{"round":"2","terminal":""}}}}\n'
    )
    sink = JsonlSink(str(path))
    log = EventLog(sink=sink, replay_from=await sink.read())
    assert fold(tuple(record.event for record in log.records)) == Status(Stage.AUTHOR, 2, None)
    await log.record(StageCompleted(Stage.VALIDATE, "cfg", "Held", 2, Terminal.HELD))
    status = fold(tuple(record.event for record in await sink.read()))
    assert status.terminal is Terminal.HELD and not status.finished
    await log.record(StageCompleted(Stage.REVIEW, "cfg", "Accept", 2, Terminal.ACCEPTED))
    assert fold(tuple(record.event for record in await sink.read())).finished


async def test_conversation_logs_each_turn_once_and_replays_legacy_full_requests(tmp_path: Path) -> None:

    sink = JsonlSink(str(tmp_path / "conversation.jsonl"))
    log = EventLog(sink=sink)
    prefix = "long-system-prompt-" * 1000
    first = InferenceRequest((Message(Role.SYSTEM, prefix),), (), LLMPolicy())
    second = InferenceRequest((Message(Role.USER, "follow-up"),), (), LLMPolicy(), first)
    for request in (first, second):
        await log.record(request)
        await log.record(Resolved(request, (), Completion("reply")))
    content = sink.read_bytes().decode()
    assert content.count(prefix) == 1
    assert content.count("follow-up") == 1
    resumed = EventLog(replay_from=await JsonlSink(sink.path).read())
    legacy = InferenceRequest(second.conversation, (), LLMPolicy())
    assert event_digest(legacy) == event_digest(second)
    assert resumed.lookup(legacy, ()) == Completion("reply")
    assert resumed.resolved()[-1].request == second


@pytest.mark.parametrize("number", [float("inf"), float("-inf"), float("nan")])
async def test_nonfinite_event_values_roundtrip_and_replay(tmp_path: Path, number: float) -> None:
    sink = JsonlSink(str(tmp_path / "nonfinite.jsonl"))
    log = EventLog(sink=sink)
    request = Timed(number)
    await log.record(request)
    await log.record(Resolved(request, (), number))
    loaded = await sink.read()
    resumed = EventLog(replay_from=loaded)
    value = resumed.lookup(Timed(number), ())
    assert isinstance(value, float)
    assert math.isnan(value) if math.isnan(number) else value == number
    assert event_digest(loaded[0].event) == event_digest(request)
    # Rewriting the same append batch must preserve the wire representation.
    before = Path(sink.path).read_bytes()
    await sink.write(loaded[-1:])
    assert Path(sink.path).read_bytes() == before
