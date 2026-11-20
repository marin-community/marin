# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, ConnectionClosedError, EndpointConnectionError
from lib.taskforge.tests.effect_fakes import local_records
from rigging.filesystem.conditional_object import ConditionalObject, VersionedBytes

from taskforge import checkpoint
from taskforge.checkpoint import HEAD, CheckpointSink, Journal, read_journal
from taskforge.effects.events import Note, StageCompleted
from taskforge.effects.issues import CorruptJournal, PublicationFailed, Superseded
from taskforge.effects.log import EventLog, Status, decode
from taskforge.model.proposal import TaskProposal
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal, Workpiece


def journal(destination: Path) -> Journal:
    published = read_journal(str(destination))
    assert published is not None
    return published


def claimed(tmp_path: Path) -> CheckpointSink:
    sink = CheckpointSink(tmp_path / "work", str(tmp_path / "published"))
    sink.claim(read_journal(sink.destination))
    return sink


@pytest.mark.parametrize("by_count", [True, False])
async def test_partial_checkpoint_on_event_count_or_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_count: bool
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 2 if by_count else 100)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 60 if by_count else 0.01)
    monkeypatch.setattr(checkpoint.time, "time", lambda: 123.0)
    sink = claimed(tmp_path)
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
            assert not journal(destination).events()
            await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
        assert await asyncio.to_thread(uploaded.wait, 2)
    published = journal(destination)
    assert (published.head.status, published.head.workpiece, published.head.published_at) == (Status(), None, 123.0)
    assert published.events() == Path(sink.local.path).read_bytes()


async def test_cancelled_checkpoint_drains_upload_before_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    sink = claimed(tmp_path)
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
    assert journal(Path(sink.destination)).events()


async def test_checkpoints_publish_immutable_chunks_of_only_new_bytes(tmp_path: Path) -> None:
    sink = claimed(tmp_path)
    log = EventLog(sink=sink)
    await log.record(StageCompleted(Stage.AUTHOR, "", "draft"))
    await sink._upload()
    chunks = Path(sink.destination) / "chunks"
    first = next(chunks.iterdir())
    original = first.read_bytes()
    await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
    await sink._upload()
    assert len(list(chunks.iterdir())) == 2 and first.read_bytes() == original
    assert decode(journal(Path(sink.destination)).events()) == log.records


async def test_transient_timer_upload_failure_keeps_the_timer_and_republishes_from_the_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 100)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    sink = claimed(tmp_path)
    failed, published = threading.Event(), threading.Event()
    publish = sink._publish

    def reset_once(status: Status, workpiece: Workpiece | None) -> None:
        if not failed.is_set():
            failed.set()
            raise EndpointConnectionError(endpoint_url="https://cwobject.com")
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
    assert local_records(sink.local) == log.records
    await sink._upload()
    assert decode(journal(Path(sink.destination)).events()) == log.records


@pytest.mark.parametrize("by_count", [True, False])
async def test_a_storage_refusal_stops_publication_on_either_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_count: bool
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1 if by_count else 100)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 60 if by_count else 0.01)
    sink = claimed(tmp_path)
    calls = []

    def refused(status: Status, workpiece: Workpiece | None) -> None:
        calls.append(status)
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "PutObject")

    monkeypatch.setattr(sink, "_publish", refused)
    log = EventLog(sink=sink)
    with pytest.raises(PublicationFailed, match="AccessDenied"):
        async with sink.running():
            await log.record(Note("first"))
            await asyncio.sleep(5)
    with pytest.raises(PublicationFailed):
        await log.record(Note("later"))
    assert len(calls) == 1


async def test_two_writers_claiming_the_same_head_cannot_both_proceed(tmp_path: Path) -> None:
    first = claimed(tmp_path)
    await EventLog(sink=first).record(Note("first"))
    await first._upload()
    seen = journal(Path(first.destination))
    contenders = [CheckpointSink(tmp_path / "work", first.destination) for _ in range(2)]
    contenders[0].claim(seen)
    with pytest.raises(Superseded):
        contenders[1].claim(seen)
    assert journal(Path(first.destination)).head.writer == contenders[0].writer


async def test_a_superseded_writer_stops_and_cannot_publish_status_or_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proposal: TaskProposal
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 0.01)
    stale = claimed(tmp_path)
    log = EventLog(sink=stale)
    await log.record(Note("before takeover"))
    await stale._upload()
    successor = CheckpointSink(tmp_path / "work", stale.destination)
    successor.claim(journal(Path(stale.destination)))
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
    assert [record.event for record in decode(after.events())] == [Note("before takeover")]


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
            raise ConnectionClosedError(endpoint_url="https://cwobject.com")
        return version


async def test_ambiguous_head_acknowledgement_is_reconciled_without_overlap(tmp_path: Path) -> None:
    sink = claimed(tmp_path)
    sink.head = AckLost(sink.head)
    log = EventLog(sink=sink)
    await log.record(Note("first"))
    await sink._upload()
    await log.record(Note("second"))
    await sink._upload()
    published = journal(Path(sink.destination))
    assert published.head.writer == sink.writer and len(published.head.chunks) == 2
    assert decode(published.events()) == log.records


async def test_a_late_chunk_put_cannot_overwrite_committed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunk PUT reported as timed out lands after a retry from the same offset has been committed."""
    sink = claimed(tmp_path)
    log = EventLog(sink=sink)
    await log.record(Note("first"))
    delayed: list[tuple[str, str, bytes]] = []
    put = checkpoint._put

    def time_out_first_put(directory: str, suffix: str, data: bytes) -> str:
        if not delayed:
            delayed.append((directory, suffix, data))
            raise TimeoutError("chunk PUT timed out")
        return put(directory, suffix, data)

    monkeypatch.setattr(checkpoint, "_put", time_out_first_put)
    await sink._upload()
    await log.record(Note("second"))
    await sink._upload()
    put(*delayed[0])
    assert decode(journal(Path(sink.destination)).events()) == log.records


async def test_torn_local_tail_is_not_published(tmp_path: Path) -> None:
    sink = claimed(tmp_path)
    log = EventLog(sink=sink)
    await log.record(Note("complete"))
    with Path(sink.local.path).open("ab") as output:
        output.write(b'{"sequence": 2, "wall')
    await sink._upload()
    assert [record.event for record in decode(journal(Path(sink.destination)).events())] == [Note("complete")]
    await log.record(Note("next"))
    await sink._upload()
    assert decode(journal(Path(sink.destination)).events()) == log.records


async def test_reader_fails_closed_on_chunks_short_of_the_head(tmp_path: Path) -> None:
    sink = claimed(tmp_path)
    await EventLog(sink=sink).record(Note("complete"))
    await sink._upload()
    head = json.loads((Path(sink.destination) / HEAD).read_text())
    chunk = Path(sink.destination) / "chunks" / head["chunks"][0]
    chunk.write_bytes(chunk.read_bytes()[:-5])
    with pytest.raises(CorruptJournal, match=re.escape(str(chunk))):
        journal(Path(sink.destination)).events()


REPUT = """
import sys
from pathlib import Path

from taskforge import checkpoint

directory, data = sys.argv[1], Path(sys.argv[2]).read_bytes()
opened = False


def audit(event, args):
    global opened
    if event == "open" and isinstance(args[0], str) and args[0].startswith(directory) and set("wxa") & set(str(args[1])):
        opened = True


def pause(frame, event, arg):
    if opened and event == "c_call" and getattr(arg, "__name__", "") == "write":
        sys.setprofile(None)
        print("paused", flush=True)
        sys.stdin.readline()


sys.addaudithook(audit)
sys.setprofile(pause)
checkpoint._put(directory, ".jsonl", data)
"""


def test_re_putting_a_committed_object_never_exposes_or_leaves_a_partial_one(tmp_path: Path) -> None:
    """A retry of identical bytes pauses on its first write into the store, a reader looks, then SIGTERM lands."""
    directory, source = tmp_path.resolve() / "chunks", tmp_path / "data"
    data = b'{"sequence": 1}\n' * 7
    name = checkpoint._put(str(directory), ".jsonl", data)
    source.write_bytes(data)
    child = subprocess.Popen(
        [sys.executable, "-c", REPUT, str(directory), str(source)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    assert child.stdout is not None and child.stdout.readline() == b"paused\n"
    assert (directory / name).read_bytes() == data
    child.terminate()
    assert child.wait(10) != 0
    assert (directory / name).read_bytes() == data


async def finished(tmp_path: Path, proposal: TaskProposal) -> Path:
    sink = claimed(tmp_path)
    await EventLog(sink=sink).record(Note("first"))
    await sink.finish(Status(terminal=Terminal.HELD), Workpiece(proposal.header.id, proposal, terminal=Terminal.HELD))
    return Path(sink.destination)


def tamper(destination: Path, case: str) -> Path:
    """Damage one committed object as ``case`` names and return the path the reader must report."""
    head = json.loads((destination / HEAD).read_bytes())
    chunk, workpiece = destination / "chunks" / head["chunks"][0], destination / "workpieces" / head["workpiece"]
    match case:
        case "altered chunk":
            chunk.write_bytes(chunk.read_bytes().replace(b"first", b"wrong"))
            return chunk
        case "missing chunk":
            chunk.unlink()
            return chunk
        case "altered workpiece":
            workpiece.write_bytes(workpiece.read_bytes().replace(b'"held"', b'"abandoned"'))
            return workpiece
        case "missing workpiece":
            workpiece.unlink()
            return workpiece
        case "truncated head":
            (destination / HEAD).write_bytes((destination / HEAD).read_bytes()[:-3])
        case _:
            (destination / HEAD).write_text(json.dumps({"format": 2, "writer": "w"}))
    return destination / HEAD


@pytest.mark.parametrize(
    "case",
    ["altered chunk", "missing chunk", "altered workpiece", "missing workpiece", "truncated head", "head schema"],
)
async def test_damaged_committed_objects_read_as_corrupt_naming_the_path(
    tmp_path: Path, proposal: TaskProposal, case: str
) -> None:
    destination = await finished(tmp_path, proposal)
    damaged = tamper(destination, case)
    before = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    with pytest.raises(CorruptJournal, match=re.escape(str(damaged))):
        resumed = journal(destination)
        resumed.workpiece()
        CheckpointSink(tmp_path / "resume", str(destination)).claim(resumed)
    assert {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before


async def test_transient_read_errors_stay_retryable(
    tmp_path: Path, proposal: TaskProposal, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = await finished(tmp_path, proposal)

    def reset(url: str, mode: str = "rb") -> object:
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(checkpoint, "open_url", reset)
    with pytest.raises(ConnectionResetError):
        journal(destination).events()
