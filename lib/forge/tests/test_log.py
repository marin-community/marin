# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

from forge.effects.events import Fingerprint, ReplayPolicy, Resolved, Retrying, StageCompleted
from forge.effects.log import EventLog, JsonlSink, Status, fold

from .effect_fakes import Answer, Query, Timed


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
import asyncio, sys
from tests.effect_fakes import Answer, Query
from forge.effects.events import Fingerprint, ReplayPolicy
from forge.effects.log import EventLog, JsonlSink
async def main():
    log = EventLog(replay_from=await JsonlSink(sys.argv[1]).read())
    value = log.lookup(Query({"a":"first", "z":"last"}), (Fingerprint("fake", "digest", ReplayPolicy.CACHE),))
    assert isinstance(value, Answer)
    assert value.data == bytes([0, 255])
    assert value.labels == {"__forge_bytes__": "this is text"}
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    await resumed.record(StageCompleted("review", "cfg", "accept", {"round": "2", "terminal": "ACCEPTED"}))
    assert resumed.records[-1].sequence == 4
    assert fold(tuple(record.event for record in resumed.records)) == Status("review", 2, "ACCEPTED")


async def test_replay_preserves_float_keys_after_numeric_coercion(tmp_path: Path) -> None:
    sink = JsonlSink(str(tmp_path / "numbers.jsonl"))
    log = EventLog(sink=sink)
    await log.record(Resolved(Timed(5), (), 3))
    reloaded = EventLog(replay_from=await sink.read())
    assert reloaded.lookup(Timed(5.0), ()) == 3.0
