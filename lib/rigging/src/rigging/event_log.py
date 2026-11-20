# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append-only phase boundaries, shared by training and experiment reporting."""

import json
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rigging.filesystem.storage_path import StoragePath


@dataclass(frozen=True)
class Event:
    run: str
    stage: str
    attempt: str
    span: str
    clock: str
    kind: str
    event: str
    utc: str
    monotonic: float
    parent: str = ""
    job: str = ""
    task: str = ""
    step: int | None = None
    benchmark: str = ""
    wave: str = ""
    gpus: int = 0
    status: str = ""
    schema: int = 1


class EventLog:
    """One writer per file; merge worker files only after preserving originals."""

    def __init__(self, path: Path, run: str, stage: str, attempt: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run = run
        self.stage = stage
        self.attempt = attempt
        self.clock = uuid4().hex
        self.lock = threading.Lock()

    def append(self, event: Event) -> None:
        with self.lock, self.path.open("a") as output:
            output.write(json.dumps(asdict(event), sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())

    @contextmanager
    def span(self, kind: str, *, gpus: int = 0, parent: str = "", step: int | None = None) -> Iterator[str]:
        span_id = uuid4().hex

        def emit(boundary: str, status: str = "") -> None:
            self.append(
                Event(
                    run=self.run,
                    stage=self.stage,
                    attempt=self.attempt,
                    span=span_id,
                    clock=self.clock,
                    kind=kind,
                    event=boundary,
                    utc=datetime.now(UTC).isoformat(),
                    monotonic=time.monotonic(),
                    parent=parent,
                    step=step,
                    gpus=gpus,
                    status=status,
                )
            )

        emit("start")
        try:
            yield span_id
        except BaseException:
            emit("end", "failed")
            raise
        else:
            emit("end", "completed")


@contextmanager
def durable_event_log(
    destination: str, run: str, stage: str, attempt: str, *, local_directory: str | None = None
) -> Iterator[EventLog]:
    """Keep a local worker journal and upload it on normal or exceptional exit."""
    directory = Path(tempfile.mkdtemp(prefix="native-timing-", dir=local_directory))
    log = EventLog(directory / "events.jsonl", run, stage, attempt)
    try:
        yield log
    finally:
        if log.path.exists():
            target = StoragePath(destination)
            target.parent.mkdirs(exist_ok=True)
            target.write_bytes(log.path.read_bytes())
