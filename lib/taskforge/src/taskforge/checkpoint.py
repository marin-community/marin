# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish partial replay logs while a workpiece is still running."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import override

from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join

from taskforge.effects.log import JsonlSink, Record, Sink

CHECKPOINT_EVENTS = 25
CHECKPOINT_SECONDS = 60


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
                await self._upload()

    def _publish(self, events: bytes) -> None:
        status = json.dumps({"terminal": None, "upload_started_at": time.time()}).encode()
        JsonlSink(prefix_join(self.destination, "events.jsonl")).append_bytes(events)
        for name, data in (("workpiece.json", self.workpiece), ("status.json", status)):
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

    async def _periodic(self, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=CHECKPOINT_SECONDS)
                return
            except TimeoutError:
                async with self.lock:
                    await self._upload()

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        stop = asyncio.Event()
        self.timer = asyncio.create_task(self._periodic(stop))
        try:
            yield
        finally:
            stop.set()
            await self.timer
