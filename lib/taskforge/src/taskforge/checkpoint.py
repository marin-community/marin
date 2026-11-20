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
from functools import partial
from pathlib import Path, PurePosixPath
from typing import override

from pydantic import TypeAdapter
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_errors import OBJECT_STORE_ERRORS, is_transient_s3_error
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import ExponentialBackoff, retry_with_backoff

from taskforge.effects.issues import CorruptJournal, LogFailure, PublicationFailed, Superseded, UnsupportedFormat
from taskforge.effects.log import JsonlSink, Record, Sink, Status
from taskforge.model.workpiece import WORKPIECE, Workpiece

JOURNAL_FORMAT = 2
FORMAT_1 = frozenset({"events.jsonl", "events.jsonl.parts", "status.json", "workpiece.json"})
"""Format 1 named events by module path and wrote these objects beside its stream; any one marks its root."""
HEAD = "journal.json"
CHECKPOINT_EVENTS = 25
CHECKPOINT_SECONDS = 60
FINAL_ATTEMPTS = 6
FINAL_BACKOFF = ExponentialBackoff(initial=5.0, maximum=120.0, factor=2.0, jitter=0.25)
"""The final publication outlasts a storage outage of several minutes beyond the adapters' own retries."""


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
        if type(found) is not int:
            raise ValueError(f"format {found!r} is no integer")
        if found != JOURNAL_FORMAT:
            raise UnsupportedFormat(f"{destination}/{HEAD} has journal format {found}; this reader reads 2")
        head = HEAD_JSON.validate_json(current.data)
        if head.status.terminal is not None and head.workpiece is None:
            raise ValueError(f"terminal {head.status.terminal} names no workpiece")
    except (ValueError, AttributeError) as error:  # Not JSON, not an object, or not a format-2 head.
        raise CorruptJournal(f"{destination}/{HEAD} is no format-2 head: {error}") from error
    return Journal(destination, head, current.version)


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
        self.deferred = False  # After a transient failure the timer retries, not every later event.
        self.lock = asyncio.Lock()
        self.timer: asyncio.Task[None] | None = None
        self.owner: asyncio.Task[object] | None = None
        self.stopped: LogFailure | None = None
        """Why this writer publishes nothing more: supersession, corruption or a storage refusal."""

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
            if not events and (head.status, head.workpiece) == (status, name):
                return  # A retried final swap whose lost acknowledgment had in fact been applied.
            head = replace(head, workpiece=name)
        self._swap(replace(head, status=status, published_at=time.time()), current.version)

    def _stop(self, error: LogFailure) -> None:
        if self.stopped is None:
            self.stopped = error
            logging.error("%s", error)
            if self.owner is not None and self.owner is not asyncio.current_task():
                self.owner.cancel()

    @override
    async def write(self, records: Sequence[Record]) -> None:
        async with self.lock:
            if self.stopped is not None:
                raise self.stopped
            await _drained(self.local.write(records))
            self.pending += len(records)
            if self.pending >= CHECKPOINT_EVENTS and not self.deferred:
                await self._upload(Status(), None)

    async def _upload(self, status: Status, workpiece: Workpiece | None) -> None:
        """Publish the new lines (finally with status and workpiece) under the lock, fenced against the head.

        A transient error leaves a checkpoint's lines pending for the timer; the final publication retries itself,
        each attempt rereading the head, so a storage hiccup never discards finished work.
        """
        if self.stopped is not None:
            raise self.stopped
        publish = asyncio.to_thread(
            retry_with_backoff,
            partial(self._publish, status, workpiece),
            retryable=is_transient_s3_error,
            max_attempts=1 if workpiece is None else FINAL_ATTEMPTS,
            backoff=FINAL_BACKOFF,
            operation=f"final publication of {self.destination}",
        )
        try:
            await _drained(publish)
        except (Superseded, CorruptJournal) as error:
            self._stop(error)
            raise
        except OBJECT_STORE_ERRORS as error:
            if workpiece is None and is_transient_s3_error(error):
                logging.warning("Checkpoint upload to %s failed; the timer retries: %r", self.destination, error)
                self.deferred = True
                return
            failure = PublicationFailed(f"Cannot publish {self.destination}; its head stays as it was: {error!r}")
            self._stop(failure)
            raise failure from error
        self.pending, self.deferred = 0, False

    async def finish(self, status: Status, workpiece: Workpiece) -> None:
        """Publish the rest of the stream with the final status and workpiece."""
        async with self.lock:
            await self._upload(status, workpiece)

    async def _periodic(self, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=CHECKPOINT_SECONDS)
                return
            except TimeoutError:
                async with self.lock:
                    try:
                        await self._upload(Status(), None)
                    except LogFailure:  # Stopped; the owner is cancelled.
                        return

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Run the timer; every tick reads the head, so a writer that lost it, idle or not, cancels this block.

        The block then raises why the writer stopped; remote effects already completed stay done.
        """
        stop = asyncio.Event()
        self.owner = asyncio.current_task()
        self.timer = asyncio.create_task(self._periodic(stop))
        failure: Exception | None = None
        try:
            yield
        except Exception as error:  # Kept beside the stop unless it is a log failure, such as the stop itself.
            failure = error
            raise
        finally:
            owner, self.owner = self.owner, None
            stop.set()
            await self.timer
            if self.stopped is not None:
                if owner is not None and owner.cancelling():
                    owner.uncancel()
                if failure is not None and ExceptionGroup("block", [failure]).split(LogFailure)[1] is not None:
                    raise ExceptionGroup(f"{self.destination} stopped beside another failure", [self.stopped, failure])
                raise self.stopped
