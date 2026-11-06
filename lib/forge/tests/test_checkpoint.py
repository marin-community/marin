# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from pathlib import Path

import pytest

from forge import checkpoint
from forge.checkpoint import CheckpointSink
from forge.effects.events import StageCompleted
from forge.effects.log import EventLog, JsonlSink
from forge.model.stage import Stage


@pytest.mark.parametrize("by_count", [True, False])
async def test_partial_checkpoint_on_event_count_or_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_count: bool
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 2 if by_count else 100)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 60 if by_count else 0.01)
    monkeypatch.setattr(checkpoint.time, "time", lambda: 123.0)
    destination = tmp_path / "published"
    local = JsonlSink(str(tmp_path / "events.jsonl"))
    sink = CheckpointSink(local, str(destination), b"{}")
    uploaded = threading.Event()
    publish = sink._publish

    def record_upload(events: bytes) -> None:
        publish(events)
        uploaded.set()

    monkeypatch.setattr(sink, "_publish", record_upload)
    log = EventLog(sink=sink)
    async with sink.running():
        await log.record(StageCompleted(Stage.AUTHOR, "", "draft"))
        if by_count:
            assert not (destination / "status.json").exists()
            await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
        assert await asyncio.to_thread(uploaded.wait, 2)
    status = json.loads((destination / "status.json").read_bytes())
    assert status == {"terminal": None, "upload_started_at": 123.0}
    assert JsonlSink(str(destination / "events.jsonl")).read_bytes() == Path(local.path).read_bytes()
    assert not list(destination.glob("*.tmp.*"))


async def test_cancelled_checkpoint_drains_upload_before_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    local = JsonlSink(str(tmp_path / "events.jsonl"))
    destination = tmp_path / "published"
    sink = CheckpointSink(local, str(destination), b"{}")
    entered, release = threading.Event(), threading.Event()
    publish = sink._publish

    def held_upload(events: bytes) -> None:
        entered.set()
        assert release.wait(2)
        publish(events)

    monkeypatch.setattr(sink, "_publish", held_upload)
    async with sink.running():
        record = asyncio.create_task(EventLog(sink=sink).record(StageCompleted(Stage.AUTHOR, "", "draft")))
        assert await asyncio.to_thread(entered.wait, 2)
        record.cancel()
        await asyncio.sleep(0)
        assert not record.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await record
    assert (destination / "status.json").exists()


async def test_checkpoints_upload_only_new_bytes_and_recover_without_final_snapshot(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    local = JsonlSink(str(tmp_path / "events.jsonl"))
    sink = CheckpointSink(local, str(destination), b"{}")
    log = EventLog(sink=sink)
    await log.record(StageCompleted(Stage.AUTHOR, "", "draft"))
    await sink._upload()
    first = next(destination.glob("events.jsonl.parts/*.jsonl"))
    original = first.read_bytes()
    await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
    await sink._upload()
    parts = sorted(destination.glob("events.jsonl.parts/*.jsonl"))
    assert len(parts) == 2 and first.read_bytes() == original
    assert sum(part.stat().st_size for part in parts) == Path(local.path).stat().st_size
    assert await JsonlSink(str(destination / "events.jsonl")).read() == log.records
