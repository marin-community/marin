"""Publish credential-free installed-agent phase journals during active attempts."""

import asyncio
import json
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from skyrl_train.io import io


class DiagnosticSpool:
    """Upload atomic journal snapshots to the attempt's co-located artifact prefix.

    Only the fixed-schema diagnostic journal is streamed; task data and Pi
    stdout remain subject to the normal final artifact upload.
    """

    def __init__(
        self,
        journal: Path,
        destination: str,
        *,
        interval: float = 2.0,
        heartbeat_interval: float = 30.0,
        publisher: Callable[[str, str], None] = io.upload_file,
    ):
        self.journal = journal
        self.destination = destination
        self.interval = interval
        self.heartbeat_interval = heartbeat_interval
        self.publisher = publisher
        self._published: bytes | None = None

    def record(self, phase: str) -> None:
        """Append a fixed diagnostic phase without task or credential content."""
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a") as stream:
            stream.write(json.dumps({"timestamp_utc": datetime.now(UTC).isoformat(), "phase": phase}) + "\n")

    def _upload_snapshot(self, content: bytes) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as snapshot:
            snapshot.write(content)
            snapshot.flush()
            self.publisher(snapshot.name, self.destination)

    async def publish(self) -> None:
        """Upload the latest full snapshot if it differs from the last successful PUT."""
        content = self.journal.read_bytes()
        if content != self._published:
            await asyncio.to_thread(self._upload_snapshot, content)
            self._published = content

    async def run(self, stop: asyncio.Event) -> None:
        """Stream while running and flush after the attempt ends."""
        self.record("attempt:started")
        last_heartbeat = asyncio.get_running_loop().time()
        while not stop.is_set():
            if asyncio.get_running_loop().time() - last_heartbeat >= self.heartbeat_interval:
                self.record("attempt:heartbeat")
                last_heartbeat = asyncio.get_running_loop().time()
            await self.publish()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval)
            except TimeoutError:
                continue
        await self.publish()
