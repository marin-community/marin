# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a workpiece's journal (format 2; its layout is in the README) under one fenced writer.

Chunks and workpieces are immutable objects named by the SHA-256 of their bytes, created only if absent and checked
against their names when read. Each is written before the compare-and-swap of the head that names it. A writer claims
the head on the version it observed and advances it the same way, so of two writers starting from one head at most
one proceeds, and a writer that lost the head extends nothing.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import override

from botocore.exceptions import ClientError
from pydantic import TypeAdapter
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.storage_path import prefix_join
from s3fs.core import S3_RETRYABLE_ERRORS

from taskforge.effects.issues import CorruptJournal, LogFailure, Superseded, UnsupportedFormat
from taskforge.effects.log import JsonlSink, Record, Sink, Status
from taskforge.model.workpiece import WORKPIECE, Workpiece

JOURNAL_FORMAT = 2
FORMAT_1 = frozenset({"events.jsonl", "events.jsonl.parts", "status.json", "workpiece.json"})
"""Format 1 named events by module path and wrote these objects beside its stream; any one marks its root."""
HEAD = "journal.json"
CHECKPOINT_EVENTS = 25
CHECKPOINT_SECONDS = 60
# Publication goes through rigging's botocore client, which re-raises service (ClientError) and transport
# (HTTPClientError, ...) errors untranslated once its retries run out; local I/O raises OSError.
UPLOAD_ERRORS: tuple[type[Exception], ...] = (OSError, ClientError, *S3_RETRYABLE_ERRORS)


@dataclass(frozen=True, slots=True)
class Head:
    format: int
    writer: str
    chunks: tuple[str, ...]
    size: int
    status: Status
    workpiece: str | None
    published_at: float


HEAD_JSON = TypeAdapter(Head)


def _put(directory: str, suffix: str, data: bytes) -> str:
    """Create ``data`` under its content name in ``directory`` unless it is there already; return the name."""
    name = hashlib.sha256(data).hexdigest() + suffix
    target = conditional_object(prefix_join(directory, name))
    try:
        target.write(data, expected_version=None)
    except ConditionalWriteError:
        if (existing := target.read()) is None or existing.data != data:
            raise CorruptJournal(f"{target.path} holds bytes other than those its name digests") from None
    return name


@dataclass(frozen=True, slots=True)
class Journal:
    """A committed head and the version it was read at; its history and workpiece are fetched on demand."""

    destination: str
    head: Head
    version: str

    def _fetch(self, directory: str, name: str) -> bytes:
        """Fetch an object the head names; a missing or altered one is corruption, other I/O errors propagate."""
        path = prefix_join(self.destination, f"{directory}/{name}")
        try:
            with open_url(path, "rb") as source:
                data = source.read()
        except FileNotFoundError as error:
            raise CorruptJournal(f"{path} is missing; the head names it") from error
        if hashlib.sha256(data).hexdigest() != name.split(".")[0]:
            raise CorruptJournal(f"{path} does not hold the bytes its name digests")
        return data

    def events(self) -> bytes:
        events = b"".join(self._fetch("chunks", name) for name in self.head.chunks)
        if len(events) != self.head.size:
            raise CorruptJournal(f"{self.destination} chunks hold {len(events)} bytes; its head covers {self.head.size}")
        if events and not events.endswith(b"\n"):
            raise CorruptJournal(f"{self.destination} ends in a torn line; committed history holds whole lines")
        return events

    def workpiece(self) -> Workpiece | None:
        if self.head.workpiece is None:
            return None
        try:
            return WORKPIECE.validate_json(self._fetch("workpieces", self.head.workpiece))
        except ValueError as error:
            raise CorruptJournal(f"{self.destination}/workpieces/{self.head.workpiece} is no workpiece") from error


def read_journal(destination: str) -> Journal | None:
    """Read the head alone; fail closed on a root in any other journal format."""
    current = conditional_object(prefix_join(destination, HEAD)).read()
    if current is None:
        fs, root = url_to_fs(destination)
        if found := sorted(FORMAT_1 & {PurePosixPath(path).name for path in fs.glob(prefix_join(root, "*"))}):
            raise UnsupportedFormat(f"{destination} holds journal format 1 ({', '.join(found)}); migrate it first")
        return None
    try:
        found = json.loads(current.data).get("format")
        if isinstance(found, int) and found != JOURNAL_FORMAT:
            raise UnsupportedFormat(f"{destination}/{HEAD} has journal format {found}; this reader reads 2")
        return Journal(destination, HEAD_JSON.validate_json(current.data), current.version)
    except (ValueError, AttributeError) as error:  # Not JSON, not an object, or not a format-2 head.
        raise CorruptJournal(f"{destination}/{HEAD} is no format-2 head: {error}") from error


async def _drained(work: Coroutine[object, object, None]) -> None:
    """Run ``work`` to the end even when cancelled, then re-raise: a publishing thread or an fsync cannot stop midway."""
    task = asyncio.create_task(work)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class CheckpointSink(Sink):
    """Append to this writer's private local stream; publish its new complete lines after 25 events or 60 seconds."""

    def __init__(self, workdir: Path, destination: str) -> None:
        self.writer = uuid.uuid4().hex
        self.local, self.destination = JsonlSink(str(workdir / f"{self.writer}.jsonl")), destination
        self.head = conditional_object(prefix_join(destination, HEAD))
        self.pending = 0
        self.lock = asyncio.Lock()
        self.timer: asyncio.Task[None] | None = None
        self.owner: asyncio.Task[object] | None = None
        self.superseded: Superseded | None = None

    def _swap(self, head: Head, expected: str | None) -> None:
        try:
            self.head.write(HEAD_JSON.dump_json(head), expected_version=expected)
        except ConditionalWriteError as error:
            raise Superseded(f"Another writer advanced {self.destination}/{HEAD}; writer {self.writer} stops") from error

    def claim(self, journal: Journal | None) -> tuple[Record, ...]:
        """Own the head ``journal`` was read at and return its history, decoded before anything is claimed."""
        events = journal.events() if journal else b""
        try:
            records = self.local.start(events)
        except LogFailure as error:
            raise CorruptJournal(f"{self.destination}: {error}") from error
        chunks, size = (journal.head.chunks, journal.head.size) if journal else ((), 0)
        head = Head(JOURNAL_FORMAT, self.writer, chunks, size, Status(), None, time.time())
        self._swap(head, journal.version if journal else None)
        return records

    def _publish(self, status: Status, workpiece: Workpiece | None) -> None:
        current = self.head.read()
        if current is None or (head := HEAD_JSON.validate_json(current.data)).writer != self.writer:
            raise Superseded(f"{self.destination}/{HEAD} is owned by another writer, not {self.writer}")
        with Path(self.local.path).open("rb") as source:
            source.seek(head.size)
            events = source.read()
        # A torn last line is not yet part of the stream; the local sink truncates it on the next append.
        events = events[: events.rfind(b"\n") + 1]
        if events:
            chunk = _put(prefix_join(self.destination, "chunks"), ".jsonl", events)
            head = replace(head, chunks=(*head.chunks, chunk), size=head.size + len(events))
        elif workpiece is None:
            return  # An idle checkpoint only confirms ownership.
        if workpiece is not None:
            name = _put(prefix_join(self.destination, "workpieces"), ".json", WORKPIECE.dump_json(workpiece))
            head = replace(head, workpiece=name)
        self._swap(replace(head, status=status, published_at=time.time()), current.version)

    def _supersede(self, error: Superseded) -> None:
        if self.superseded is None:
            self.superseded = error
            logging.error("%s", error)
            if self.owner is not None:
                self.owner.cancel()

    @override
    async def write(self, records: Sequence[Record]) -> None:
        if self.superseded is not None:
            raise self.superseded
        if self.timer is not None and self.timer.done():
            self.timer.result()
        async with self.lock:
            await _drained(self.local.write(records))
            self.pending += len(records)
            if self.pending >= CHECKPOINT_EVENTS:
                await self._upload()

    async def _upload(self) -> None:
        try:
            await _drained(asyncio.to_thread(self._publish, Status(), None))
        except Superseded as error:
            self._supersede(error)
            raise
        self.pending = 0

    async def finish(self, status: Status, workpiece: Workpiece) -> None:
        """Publish the rest of the stream with the final status and workpiece."""
        async with self.lock:
            await asyncio.to_thread(self._publish, status, workpiece)

    async def _periodic(self, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=CHECKPOINT_SECONDS)
                return
            except TimeoutError:
                async with self.lock:
                    try:
                        await self._upload()
                    except Superseded:
                        return
                    except UPLOAD_ERRORS as error:
                        # The local log stays complete; the next tick republishes from the head's covered size.
                        logging.warning(
                            "Checkpoint upload to %s failed; retrying next tick: %r", self.destination, error
                        )

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Run the timer; every tick reads the head, so a writer that lost it, idle or not, cancels this block.

        The block then raises ``Superseded``; remote effects already completed stay done.
        """
        stop = asyncio.Event()
        self.owner = asyncio.current_task()
        self.timer = asyncio.create_task(self._periodic(stop))
        try:
            yield
        finally:
            owner, self.owner = self.owner, None
            stop.set()
            await self.timer
            if self.superseded is not None:
                if owner is not None and owner.cancelling():
                    owner.uncancel()
                raise self.superseded
