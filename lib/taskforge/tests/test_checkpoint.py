# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from pathlib import Path

import pytest
from rigging.filesystem.conditional_object import ConditionalObject, VersionedBytes

from taskforge import checkpoint
from taskforge.checkpoint import HEAD, CheckpointSink, Published, read_published
from taskforge.effects.events import Note, StageCompleted
from taskforge.effects.issues import LogFailure, Superseded
from taskforge.effects.log import EventLog, JsonlSink, Status, decode
from taskforge.model.proposal import TaskProposal
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal, Workpiece


def journal(destination: Path) -> Published:
    published = read_published(str(destination))
    assert published is not None
    return published


async def claimed(tmp_path: Path, name: str = "events.jsonl") -> CheckpointSink:
    sink = CheckpointSink(JsonlSink(str(tmp_path / name)), str(tmp_path / "published"))
    await sink.claim(read_published(sink.destination))
    return sink


@pytest.mark.parametrize("by_count", [True, False])
async def test_partial_checkpoint_on_event_count_or_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_count: bool
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 2 if by_count else 100)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 60 if by_count else 0.01)
    monkeypatch.setattr(checkpoint.time, "time", lambda: 123.0)
    sink = await claimed(tmp_path)
    destination = Path(sink.destination)
    uploaded = threading.Event()
    publish = sink._publish

    def record_upload(status: Status, workpiece: Workpiece | None) -> None:
        publish(status, workpiece)
        uploaded.set()

    monkeypatch.setattr(sink, "_publish", record_upload)
    log = EventLog(sink=sink)
    async with sink.running():
        await log.record(StageCompleted(Stage.AUTHOR, "", "draft"))
        if by_count:
            assert not journal(destination).events
            await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
        assert await asyncio.to_thread(uploaded.wait, 2)
    published = journal(destination)
    assert (published.head.status, published.head.workpiece, published.head.published_at) == (Status(), None, 123.0)
    assert published.events == Path(sink.local.path).read_bytes()


async def test_cancelled_checkpoint_drains_upload_before_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    sink = await claimed(tmp_path)
    entered, release = threading.Event(), threading.Event()
    publish = sink._publish

    def held_upload(status: Status, workpiece: Workpiece | None) -> None:
        entered.set()
        assert release.wait(2)
        publish(status, workpiece)

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
    assert journal(Path(sink.destination)).events


async def test_checkpoints_publish_immutable_chunks_of_only_new_bytes(tmp_path: Path) -> None:
    sink = await claimed(tmp_path)
    log = EventLog(sink=sink)
    await log.record(StageCompleted(Stage.AUTHOR, "", "draft"))
    await sink._upload()
    chunks = Path(sink.destination) / "chunks"
    first = next(chunks.iterdir())
    original = first.read_bytes()
    await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
    await sink._upload()
    assert len(list(chunks.iterdir())) == 2 and first.read_bytes() == original
    assert decode(journal(Path(sink.destination)).events) == log.records


async def test_failed_timer_upload_keeps_logging_and_republishes_from_the_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 100)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    sink = await claimed(tmp_path)
    failed, published = threading.Event(), threading.Event()
    publish = sink._publish

    def reset_once(status: Status, workpiece: Workpiece | None) -> None:
        if not failed.is_set():
            failed.set()
            raise ConnectionResetError("connection reset by peer")
        publish(status, workpiece)
        published.set()

    monkeypatch.setattr(sink, "_publish", reset_once)
    log = EventLog(sink=sink)
    async with sink.running():
        await log.record(StageCompleted(Stage.AUTHOR, "", "draft"))
        assert await asyncio.to_thread(failed.wait, 2)
        await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
        assert await asyncio.to_thread(published.wait, 2)
        await log.record(StageCompleted(Stage.REVIEW, "", "accept"))
    assert await sink.local.read() == log.records
    await sink._upload()
    assert decode(journal(Path(sink.destination)).events) == log.records


async def test_two_writers_claiming_the_same_head_cannot_both_proceed(tmp_path: Path) -> None:
    first = await claimed(tmp_path, "first.jsonl")
    await EventLog(sink=first).record(Note("first"))
    await first._upload()
    seen = journal(Path(first.destination))
    contenders = [CheckpointSink(JsonlSink(str(tmp_path / name)), first.destination) for name in ("a.jsonl", "b.jsonl")]
    await contenders[0].claim(seen)
    with pytest.raises(Superseded):
        await contenders[1].claim(seen)
    assert journal(Path(first.destination)).head.writer == contenders[0].writer


async def test_a_superseded_writer_stops_and_cannot_publish_status_or_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proposal: TaskProposal
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    stale = await claimed(tmp_path, "stale.jsonl")
    log = EventLog(sink=stale)
    await log.record(Note("before takeover"))
    await stale._upload()
    successor = CheckpointSink(JsonlSink(str(tmp_path / "successor.jsonl")), stale.destination)
    published = journal(Path(stale.destination))
    Path(successor.local.path).write_bytes(published.events)
    await successor.claim(published)
    await log.record(Note("after takeover"))
    # The timer finds the head taken, cancels the running block and raises from it.
    with pytest.raises(Superseded):
        async with stale.running():
            await asyncio.sleep(5)
    with pytest.raises(Superseded):
        await log.record(Note("later"))
    with pytest.raises(Superseded):
        await stale.finish(Status(terminal=Terminal.HELD), Workpiece(proposal.header.id, proposal))
    after = journal(Path(stale.destination))
    assert after.head.writer == successor.writer and after.head.status == Status()
    assert [record.event for record in decode(after.events)] == [Note("before takeover")]


class AckLost:
    """Apply the write, then report a transport failure, as when the response is lost."""

    def __init__(self, target: ConditionalObject) -> None:
        self.target, self.lost = target, 1

    @property
    def path(self) -> str:
        return self.target.path

    def version(self) -> str | None:
        return self.target.version()

    def read(self) -> VersionedBytes | None:
        return self.target.read()

    def write(self, data: bytes, *, expected_version: str | None) -> str:
        version = self.target.write(data, expected_version=expected_version)
        if self.lost:
            self.lost -= 1
            raise ConnectionResetError("response lost after the write landed")
        return version


async def test_ambiguous_head_acknowledgement_is_reconciled_without_overlap(tmp_path: Path) -> None:
    sink = await claimed(tmp_path)
    sink.head = AckLost(sink.head)
    log = EventLog(sink=sink)
    await log.record(Note("first"))
    with pytest.raises(ConnectionResetError):
        await sink._upload()
    await log.record(Note("second"))
    await sink._upload()
    published = journal(Path(sink.destination))
    assert published.head.writer == sink.writer and len(published.head.chunks) == 2
    assert decode(published.events) == log.records


async def test_torn_local_tail_is_not_published(tmp_path: Path) -> None:
    sink = await claimed(tmp_path)
    log = EventLog(sink=sink)
    await log.record(Note("complete"))
    with Path(sink.local.path).open("ab") as output:
        output.write(b'{"sequence": 2, "wall')
    await sink._upload()
    assert [record.event for record in decode(journal(Path(sink.destination)).events)] == [Note("complete")]
    await log.record(Note("next"))
    await sink._upload()
    assert decode(journal(Path(sink.destination)).events) == log.records


async def test_reader_fails_closed_on_chunks_short_of_the_head(tmp_path: Path) -> None:
    sink = await claimed(tmp_path)
    await EventLog(sink=sink).record(Note("complete"))
    await sink._upload()
    head = json.loads((Path(sink.destination) / HEAD).read_text())
    chunk = Path(sink.destination) / "chunks" / head["chunks"][0]
    chunk.write_bytes(chunk.read_bytes()[:-5])
    with pytest.raises(LogFailure, match="chunks hold"):
        read_published(sink.destination)
