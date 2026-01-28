# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class BufferedLogRecord:
    timestamp: float
    level: str
    logger_name: str
    message: str


class LogBuffer(Protocol):
    def append(self, record: BufferedLogRecord) -> None: ...
    def query(self, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]: ...


class LogRingBuffer:
    """Thread-safe ring buffer that collects Python log records."""

    def __init__(self, maxlen: int = 5000):
        self._buffer: deque[BufferedLogRecord] = deque(maxlen=maxlen)
        self._lock = Lock()

    def append(self, record: BufferedLogRecord) -> None:
        with self._lock:
            self._buffer.append(record)

    def query(self, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]:
        with self._lock:
            items = list(self._buffer)
        if prefix:
            items = [r for r in items if r.logger_name.startswith(prefix)]
        return items[-limit:]


class RingBufferHandler(logging.Handler):
    def __init__(self, buffer: LogRingBuffer):
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(
            BufferedLogRecord(
                timestamp=record.created,
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
            )
        )


_global_buffer = LogRingBuffer()


def get_global_buffer() -> LogRingBuffer:
    return _global_buffer


_configured = False


def configure_logging(level: int = logging.INFO) -> LogRingBuffer:
    """Configure iris logging: stderr handler + ring buffer. Idempotent."""
    global _configured
    if _configured:
        root = logging.getLogger()
        root.setLevel(level)
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RingBufferHandler):
                h.setLevel(level)
        return _global_buffer

    _configured = True
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    ring_handler = RingBufferHandler(_global_buffer)
    ring_handler.setLevel(logging.DEBUG)
    ring_handler.setFormatter(formatter)
    root.addHandler(ring_handler)

    return _global_buffer
