# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import hashlib
import json
import re
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable
from functools import partial
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlsplit

import botocore.config
import botocore.session
import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    IncompleteReadError,
)
from rigging.filesystem.conditional_object import ConditionalObject, S3ConditionalObject, VersionedBytes
from rigging.timing import ExponentialBackoff
from s3fs.errors import translate_boto_error

from taskforge import checkpoint
from taskforge.checkpoint import HEAD, HEAD_JSON, CheckpointSink, Journal, is_transient_storage_error, read_journal
from taskforge.effects.events import Note, StageCompleted
from taskforge.effects.issues import CorruptJournal, PublicationFailed, Superseded
from taskforge.effects.log import BufferSink, EventLog, Status, decode
from taskforge.model.proposal import TaskProposal
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal, Workpiece


def journal(destination: Path) -> Journal:
    published = read_journal(str(destination))
    assert published is not None
    return published


def s3_error(code: str, status: int) -> ClientError:
    response: Any = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}
    return ClientError(response, "PutObject")


def claimed(tmp_path: Path) -> CheckpointSink:
    sink = CheckpointSink(str(tmp_path / "published"))
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
    assert decode(published.events()) == log.records


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
    await sink._upload(Status(), None)
    chunks = Path(sink.destination) / "chunks"
    first = next(chunks.iterdir())
    original = first.read_bytes()
    await log.record(StageCompleted(Stage.VALIDATE, "", "ready"))
    await sink._upload(Status(), None)
    assert len(list(chunks.iterdir())) == 2 and first.read_bytes() == original
    assert decode(journal(Path(sink.destination)).events()) == log.records


async def test_reclaim_discards_the_unpublished_suffix_and_its_digest_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 2)
    sink = claimed(tmp_path)
    log = EventLog(sink=sink)
    await log.record(Note("published first"))
    await log.record(Note("published second"))
    discarded = Note("from the abandoned suffix")
    await log.record(discarded)

    destination = Path(sink.destination)
    resumed = EventLog(sink=sink, replay_from=sink.claim(journal(destination)))
    await resumed.record(Note("replacement"))
    await resumed.record(discarded)  # its former occurrence was never published
    await sink._upload(Status(), None)
    assert decode(journal(destination).events()) == resumed.records
    assert not sink.local.data


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
    await sink._upload(Status(), None)
    uninterrupted = BufferSink()
    await uninterrupted.write(log.records)
    assert journal(Path(sink.destination)).events() == uninterrupted.data


@pytest.mark.parametrize("by_count", [True, False])
@pytest.mark.parametrize(
    "refusal", [s3_error("AccessDenied", 403), PermissionError(errno.EACCES, "Permission denied")], ids=["s3", "local"]
)
async def test_a_storage_refusal_stops_publication_on_either_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_count: bool, refusal: Exception
) -> None:
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1 if by_count else 100)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_SECONDS", 60 if by_count else 0.01)
    sink = claimed(tmp_path)
    calls = []

    def refused(status: Status, workpiece: Workpiece | None) -> None:
        calls.append(status)
        raise refusal

    monkeypatch.setattr(sink, "_publish", refused)
    log = EventLog(sink=sink)
    with pytest.raises(PublicationFailed, match=type(refusal).__name__):
        async with sink.running():
            await log.record(Note("first"))
            await asyncio.sleep(5)
    with pytest.raises(PublicationFailed):
        await log.record(Note("later"))
    assert len(calls) == 1


async def test_a_tick_queued_behind_a_refused_checkpoint_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A count-triggered checkpoint is refused while a timer tick waits on the lock; the stop holds for the tick."""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_EVENTS", 1)
    sink = claimed(tmp_path)
    publish, calls, gate = sink._publish, [], threading.Event()

    def refuse_first(status: Status, workpiece: Workpiece | None) -> None:
        calls.append(status)
        if len(calls) == 1:
            gate.wait(5)
            raise s3_error("AccessDenied", 403)
        publish(status, workpiece)

    async def tick() -> None:
        async with sink.lock:  # As ``_periodic`` does.
            await sink._upload(Status(), None)

    monkeypatch.setattr(sink, "_publish", refuse_first)
    foreground = asyncio.create_task(EventLog(sink=sink).record(Note("first")))
    while not calls:
        await asyncio.sleep(0.01)
    queued = asyncio.create_task(tick())
    await asyncio.sleep(0.01)
    gate.set()
    for task in (foreground, queued):
        with pytest.raises(PublicationFailed):
            await task
    assert len(calls) == 1 and journal(Path(sink.destination)).head.size == 0


async def test_two_writers_claiming_the_same_head_cannot_both_proceed(tmp_path: Path) -> None:
    first = claimed(tmp_path)
    await EventLog(sink=first).record(Note("first"))
    await first._upload(Status(), None)
    seen = journal(Path(first.destination))
    contenders = [CheckpointSink(first.destination) for _ in range(2)]
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
    await stale._upload(Status(), None)
    successor = CheckpointSink(stale.destination)
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
    await sink._upload(Status(), None)
    await log.record(Note("second"))
    await sink._upload(Status(), None)
    published = journal(Path(sink.destination))
    assert published.head.writer == sink.writer and len(published.head.chunks) == 2
    assert decode(published.events()) == log.records


class S3Server:
    """S3 conditional GET and PUT behind a real botocore client; only the HTTP exchange is replaced.

    The chosen PUT either applies and loses its acknowledgement, so the SDK resends it, or answers a retryable 409.
    """

    def __init__(self, fault: str, chosen: Callable[[str, int], bool]) -> None:
        self.objects: dict[str, bytes] = {}
        self.fault, self.chosen, self.puts, self.fired = fault, chosen, Counter[str](), False

    @staticmethod
    def reply(status: int, body: bytes = b"", **headers: str) -> AWSResponse:
        raw: Any = SimpleNamespace(stream=lambda **_: iter([body]), read=BytesIO(body).read)
        sent: Any = {"Content-Length": str(len(body)), **headers}
        return AWSResponse("https://bucket.s3.test", status, sent, raw)

    def error(self, code: str, status: int) -> AWSResponse:
        return self.reply(status, f"<Error><Code>{code}</Code><Message>{code}</Message></Error>".encode())

    def __call__(self, request: AWSPreparedRequest, **_: object) -> AWSResponse:
        key = urlsplit(request.url).path
        headers = {name: cast(bytes, value).decode() for name, value in request.headers.items()}
        current = self.objects.get(key)
        etag = None if current is None else f'"{hashlib.md5(current).hexdigest()}"'
        if request.method == "GET":
            return self.error("NoSuchKey", 404) if current is None else self.reply(200, current, ETag=str(etag))
        if headers.get("If-None-Match") == "*" and current is not None:
            return self.error("PreconditionFailed", 412)
        if (expected := headers.get("If-Match")) is not None and expected != etag:
            return self.error("PreconditionFailed", 412)
        self.puts[key] += 1
        chosen = self.chosen(key, self.puts[key]) and not self.fired
        if chosen and self.fault == "conflict":
            self.fired = True
            return self.error("ConditionalRequestConflict", 409)
        data = request.body if isinstance(request.body, bytes) else cast(BytesIO, request.body).read()
        if "aws-chunked" in headers.get("Content-Encoding", ""):  # One chunk, then the checksum trailer.
            size, data = data.split(b"\r\n", 1)
            data = data[: int(size, 16)]
        self.objects[key] = data
        if chosen and self.fault == "ack-lost":
            self.fired = True
            raise ConnectionClosedError(endpoint_url=request.url)
        return self.reply(200, ETag=f'"{hashlib.md5(data).hexdigest()}"')


def is_head(key: str, n: int, nth: int) -> bool:
    return key.endswith(HEAD) and n == nth


@pytest.mark.parametrize(
    ("fault", "chosen"),
    [
        ("ack-lost", partial(is_head, nth=1)),
        ("ack-lost", partial(is_head, nth=2)),
        ("conflict", lambda key, _n: "/chunks/" in key),
        ("conflict", partial(is_head, nth=2)),
    ],
    ids=["412-after-applied-claim", "412-after-applied-checkpoint", "409-on-first-chunk", "409-on-head-swap"],
)
async def test_an_inconclusive_conditional_put_through_the_sdk_keeps_the_sole_writer(
    monkeypatch: pytest.MonkeyPatch, fault: str, chosen: Callable[[str, int], bool]
) -> None:
    server = S3Server(fault, chosen)
    client = botocore.session.get_session().create_client(
        "s3",
        region_name="us-east-1",
        endpoint_url="https://s3.test",
        config=botocore.config.Config(s3={"addressing_style": "virtual"}),
        aws_access_key_id="k",
        aws_secret_access_key="s",
    )
    client.meta.events.register("before-send.s3", cast(Any, server))
    monkeypatch.setattr(S3ConditionalObject, "_client", staticmethod(lambda _endpoint: client))
    monkeypatch.setattr(checkpoint, "FINAL_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001, jitter=0.0))
    sink = CheckpointSink("s3://bucket/root")
    sink.claim(None)
    log = EventLog(sink=sink)
    await log.record(Note("first"))
    await sink._upload(Status(), None)
    if fault == "conflict":  # The checkpoint deferred to the timer; this is its tick.
        await sink._upload(Status(), None)
    assert server.fired and sink.stopped is None
    head = HEAD_JSON.validate_json(server.objects["/root/" + HEAD])
    assert head.writer == sink.writer and head.size == sink.local.offset + len(sink.local.data)
    assert decode(b"".join(server.objects[f"/root/chunks/{name}"] for name in head.chunks)) == log.records


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
    await sink._upload(Status(), None)
    await log.record(Note("second"))
    await sink._upload(Status(), None)
    put(*delayed[0])
    assert decode(journal(Path(sink.destination)).events()) == log.records


async def test_reader_fails_closed_on_chunks_short_of_the_head(tmp_path: Path) -> None:
    sink = claimed(tmp_path)
    await EventLog(sink=sink).record(Note("complete"))
    await sink._upload(Status(), None)
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
            (destination / HEAD).write_text(json.dumps({"format": checkpoint.JOURNAL_FORMAT, "writer": "w"}))
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
        CheckpointSink(str(destination)).claim(resumed)
    assert {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"), 2),
        (IncompleteReadError(actual_bytes=3, expected_bytes=9), 2),
        (PermissionError(errno.EACCES, "Permission denied"), 1),
    ],
    ids=["reset", "short-body", "refused"],
)
async def test_a_history_read_retries_a_storage_hiccup_and_not_a_refusal(
    tmp_path: Path, proposal: TaskProposal, monkeypatch: pytest.MonkeyPatch, error: Exception, attempts: int
) -> None:
    destination = await finished(tmp_path, proposal)
    expected = journal(destination).events()
    monkeypatch.setattr(checkpoint, "FINAL_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001, jitter=0.0))
    opened, open_url = [], checkpoint.open_url

    def fail_once(url: str, mode: str = "rb") -> object:
        opened.append(url)
        if len(opened) == 1:
            raise error
        return open_url(url, mode)

    monkeypatch.setattr(checkpoint, "open_url", fail_once)
    if attempts == 1:
        with pytest.raises(PermissionError):
            journal(destination).events()
    else:
        assert journal(destination).events() == expected
    assert opened.count(opened[0]) == attempts


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"), True),
        (IncompleteReadError(actual_bytes=3, expected_bytes=9), True),
        (s3_error("SlowDown", 503), True),
        (s3_error("BadGateway", 502), True),
        (s3_error("NotImplemented", 501), True),
        (s3_error("RequestTimeout", 408), True),
        (s3_error("TooManyRequests", 429), True),
        (translate_boto_error(s3_error("BadGateway", 502)), True),
        (OSError(errno.EIO, "Input/output error"), True),
        (s3_error("AccessDenied", 403), False),
        (s3_error("NoSuchBucket", 404), False),
        (translate_boto_error(s3_error("AccessDenied", 403)), False),
        (PermissionError(errno.EACCES, "Permission denied"), False),
        (PermissionError(errno.EACCES, "Permission denied", "chunks/SlowDown.jsonl"), False),
        (translate_boto_error(s3_error("RequestTimeTooSkewed", 403)), True),
        (OSError(errno.ENOSPC, "No space left on device"), False),
        (ValueError("not a storage error"), False),
    ],
)
def test_only_an_explicit_refusal_is_a_permanent_storage_error(error: Exception, transient: bool) -> None:
    assert is_transient_storage_error(error) is transient
