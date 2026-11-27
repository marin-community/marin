# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish partial replay logs while a workpiece is still running."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import override

from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join
from s3fs.core import S3_RETRYABLE_ERRORS

from taskforge.effects.log import JsonlSink, Record, Sink

CHECKPOINT_EVENTS = 25
CHECKPOINT_SECONDS = 60
# s3fs translates service errors into OSError but re-raises its retryable transport errors (botocore
# HTTPClientError, ResponseParserError, IncompleteRead, ...) untranslated once its own retries run out.
UPLOAD_ERRORS: tuple[type[Exception], ...] = (OSError, *S3_RETRYABLE_ERRORS)


class CheckpointSink(Sink):
    def __init__(self, local: JsonlSink, destination: str, workpiece: bytes) -> None:
        self.local, self.destination, self.workpiece = local, destination, workpiece
        self.pending = 0
        self.offset = Path(local.path).stat().st_size if Path(local.path).exists() else 0
        self.lock = asyncio.Lock()
        self.timer: asyncio.Task[None] | None = None

    @override
    async def write(self, records: Sequence[Record]) -> None:
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
                await self._try_upload()

    def _publish(self, events: bytes) -> None:
        """Publish the workpiece before the events, so any published stream names the input it came from."""
        self._replace("workpiece.json", self.workpiece)
        JsonlSink(prefix_join(self.destination, "events.jsonl")).append_bytes(events)
        self._replace("status.json", json.dumps({"terminal": None, "upload_started_at": time.time()}).encode())

    def _replace(self, name: str, data: bytes) -> None:
        with atomic_rename(prefix_join(self.destination, name)) as temporary:
            with open_url(temporary, "wb") as target:
                target.write(data)

    async def _upload(self) -> None:
        if not self.pending:
            return
        with Path(self.local.path).open("rb") as source:
            source.seek(self.offset)
            events = source.read()
        upload = asyncio.create_task(asyncio.to_thread(self._publish, events))
        try:
            await asyncio.shield(upload)
        except asyncio.CancelledError:
            # A thread cannot be cancelled: drain it before final publication can begin.
            await upload
            raise
        self.offset += len(events)
        self.pending = 0

    async def _try_upload(self) -> None:
        """Publish pending events; a failure is logged and the next upload republishes from the same offset.

        The local log is already complete, so an upload error must not reach ``EventLog.record``: its retry
        would append the batch locally a second time.
        """
        try:
            await self._upload()
        except UPLOAD_ERRORS as error:
            logging.warning("Checkpoint upload to %s failed; retrying on the next upload: %r", self.destination, error)

    async def _periodic(self, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=CHECKPOINT_SECONDS)
                return
            except TimeoutError:
                async with self.lock:
                    await self._try_upload()

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        stop = asyncio.Event()
        self.timer = asyncio.create_task(self._periodic(stop))
        try:
            yield
        finally:
            stop.set()
            await self.timer
