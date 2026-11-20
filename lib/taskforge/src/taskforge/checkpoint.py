# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a workpiece's journal under one fenced writer.

A workpiece's output prefix holds ``journal.json``, the head, and ``chunks/<writer>-<offset>.jsonl``, immutable
slices of a writer's local event stream starting at byte ``offset``. The head records the journal format, the
owning writer, the chunks that make up the stream, the bytes they cover, the published status and, once the
workpiece ends, the workpiece. A writer claims the head by compare-and-swap on the version it resumed from and
advances it the same way after each chunk, so two writers starting from one head cannot both proceed, and a
writer that lost the head can neither extend the stream nor publish a status. Readers follow only the chunks
the head names; chunks a loser wrote are unreachable.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override

from pydantic import TypeAdapter
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.storage_path import prefix_join
from s3fs.core import S3_RETRYABLE_ERRORS

from taskforge.effects.issues import LogFailure, Superseded
from taskforge.effects.log import JsonlSink, Record, Sink, Status
from taskforge.model.workpiece import Workpiece

JOURNAL_FORMAT = 2
"""Format 1 was the module-path event types with ``events.jsonl``, its ``.parts`` and ``status.json``."""
HEAD = "journal.json"
CHECKPOINT_EVENTS = 25
CHECKPOINT_SECONDS = 60
# s3fs translates service errors into OSError but re-raises its retryable transport errors (botocore
# HTTPClientError, ResponseParserError, IncompleteRead, ...) untranslated once its own retries run out.
UPLOAD_ERRORS: tuple[type[Exception], ...] = (OSError, *S3_RETRYABLE_ERRORS)


@dataclass(frozen=True, slots=True)
class Head:
    format: int
    writer: str
    chunks: tuple[str, ...]
    size: int
    status: Status
    workpiece: Workpiece | None
    published_at: float


HEAD_JSON = TypeAdapter(Head)


@dataclass(frozen=True, slots=True)
class Published:
    head: Head
    version: str
    events: bytes


def read_published(destination: str) -> Published | None:
    """Read the stream the head names; fail closed on any other journal format."""
    current = conditional_object(prefix_join(destination, HEAD)).read()
    if current is None:
        fs, legacy = url_to_fs(prefix_join(destination, "events.jsonl"))
        if fs.exists(legacy):
            raise LogFailure(f"{destination} holds journal format 1 (events.jsonl without {HEAD}); migrate it first")
        return None
    found = json.loads(current.data).get("format")
    if found != JOURNAL_FORMAT:
        raise LogFailure(f"{destination}/{HEAD} has journal format {found!r}; this reader accepts {JOURNAL_FORMAT}")
    head = HEAD_JSON.validate_json(current.data)
    events = b""
    for name in head.chunks:
        with open_url(prefix_join(destination, f"chunks/{name}"), "rb") as source:
            events += source.read()
    if len(events) != head.size:
        raise LogFailure(f"{destination} chunks hold {len(events)} bytes; its head covers {head.size}")
    return Published(head, current.version, events)


class CheckpointSink(Sink):
    """Append locally; publish new complete lines after 25 events or 60 seconds while this writer owns the head."""

    def __init__(self, local: JsonlSink, destination: str) -> None:
        self.local, self.destination = local, destination
        self.head = conditional_object(prefix_join(destination, HEAD))
        self.writer = uuid.uuid4().hex
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

    async def claim(self, published: Published | None) -> None:
        """Own the head ``published`` was read from; the local stream must hold exactly its events."""
        chunks, size = (published.head.chunks, published.head.size) if published else ((), 0)
        head = Head(JOURNAL_FORMAT, self.writer, chunks, size, Status(), None, time.time())
        await asyncio.to_thread(self._swap, head, published.version if published else None)

    def _publish(self, status: Status, workpiece: Workpiece | None) -> None:
        current = self.head.read()
        if current is None or (head := HEAD_JSON.validate_json(current.data)).writer != self.writer:
            raise Superseded(f"{self.destination}/{HEAD} is owned by another writer, not {self.writer}")
        events = b""
        if Path(self.local.path).exists():
            with Path(self.local.path).open("rb") as source:
                source.seek(head.size)
                events = source.read()
        # A torn last line is not yet part of the stream; the local sink truncates it on the next append.
        events = events[: events.rfind(b"\n") + 1]
        if events:
            name = f"{self.writer}-{head.size:012d}.jsonl"
            # Unreferenced until the swap below; a retry after a failed swap rewrites the same unreferenced key.
            with open_url(prefix_join(self.destination, f"chunks/{name}"), "wb") as target:
                target.write(events)
            head = replace(head, chunks=(*head.chunks, name), size=head.size + len(events))
        self._swap(replace(head, status=status, workpiece=workpiece, published_at=time.time()), current.version)

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
            append = asyncio.create_task(self.local.write(records))
            try:
                await asyncio.shield(append)
            except asyncio.CancelledError:
                await append
                raise
            self.pending += len(records)
            if self.pending >= CHECKPOINT_EVENTS:
                await self._upload()

    async def _upload(self) -> None:
        if not self.pending:
            return
        upload = asyncio.create_task(asyncio.to_thread(self._publish, Status(), None))
        try:
            await asyncio.shield(upload)
        except asyncio.CancelledError:
            # A thread cannot be cancelled: drain it before final publication can begin.
            await upload
            raise
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
        """Run the timer; a writer that loses the head cancels this block and raises ``Superseded`` from it."""
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
