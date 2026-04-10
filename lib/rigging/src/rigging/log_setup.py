# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

LOG_FORMAT = "%(levelprefix)s%(asctime)s %(thread)s %(name)s %(message)s"
LOG_DATEFMT = "%Y%m%d %H:%M:%S"

# Map log level names to single-character prefixes for compact log output.
# E.g., INFO -> "I", ERROR -> "E", WARNING -> "W".
_LEVEL_PREFIX = {
    "DEBUG": "D",
    "INFO": "I",
    "WARNING": "W",
    "ERROR": "E",
    "CRITICAL": "C",
}

# Reverse map: single-letter prefix -> canonical level name.
_PREFIX_TO_LEVEL = {v: k for k, v in _LEVEL_PREFIX.items()}

# Canonical level names, derived from _LEVEL_PREFIX keys.
LOG_LEVEL_NAMES = tuple(_LEVEL_PREFIX.keys())


def parse_log_level(line: str) -> str | None:
    """Best-effort parse a log level from a log line.

    Recognizes the single-letter prefix format produced by LevelPrefixFormatter:
        "I20260102 12:34:56 ..."  ->  "INFO"
        "E20260102 12:44:05 ..."  ->  "ERROR"

    Returns the canonical level name (e.g. "INFO") or None.
    """
    if not line or len(line) < 2:
        return None

    # Single-letter prefix format: "I20260102 12:34:56 ..."
    if line[0] in _PREFIX_TO_LEVEL and line[1:2].isdigit():
        return _PREFIX_TO_LEVEL[line[0]]

    return None


class LevelPrefixFormatter(logging.Formatter):
    """Formatter that prepends a single-letter level prefix (I/W/E/D/C).

    Produces lines like: I20260306 12:44:05 iris.worker starting up
    """

    def format(self, record: logging.LogRecord) -> str:
        record.levelprefix = _LEVEL_PREFIX.get(record.levelname, "?")
        return super().format(record)


@dataclass(frozen=True)
class BufferedLogRecord:
    seq: int
    timestamp: float
    level: str
    logger_name: str
    message: str


class LogBuffer(Protocol):
    def append(self, record: BufferedLogRecord) -> None: ...
    def query(self, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]: ...
    def query_since(self, last_seq: int, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]: ...


class LogRingBuffer:
    """Thread-safe ring buffer that collects Python log records."""

    def __init__(self, maxlen: int = 5000):
        self._buffer: deque[BufferedLogRecord] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._seq = 0

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def append(self, record: BufferedLogRecord) -> None:
        with self._lock:
            self._buffer.append(record)

    def query(self, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]:
        with self._lock:
            items = list(self._buffer)
        if prefix:
            items = [r for r in items if r.logger_name.startswith(prefix)]
        return items[-limit:]

    def query_since(self, last_seq: int, *, prefix: str | None = None, limit: int = 200) -> list[BufferedLogRecord]:
        with self._lock:
            items = list(self._buffer)
        if prefix:
            items = [r for r in items if r.logger_name.startswith(prefix)]
        newer = [r for r in items if r.seq > last_seq]
        return newer[-limit:]


class RingBufferHandler(logging.Handler):
    def __init__(self, buffer: LogRingBuffer):
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(
            BufferedLogRecord(
                seq=self._buffer.next_seq(),
                timestamp=record.created,
                level=record.levelname,
                logger_name=record.name,
                message=self.format(record),
            )
        )


@contextmanager
def slow_log(log: logging.Logger, operation: str, threshold_ms: int = 100) -> Iterator[None]:
    """Log a WARNING if the enclosed block takes longer than threshold_ms.

    Silent when the operation completes within budget.
    """
    start = time.monotonic()
    yield
    elapsed_ms = int((time.monotonic() - start) * 1000)
    if elapsed_ms >= threshold_ms:
        log.warning("Slow %s: %dms (threshold: %dms)", operation, elapsed_ms, threshold_ms)


_global_buffer = LogRingBuffer()


def get_global_buffer() -> LogRingBuffer:
    return _global_buffer


_configured = False


def configure_logging(level: int = logging.INFO) -> LogRingBuffer:
    """Configure logging: stderr handler + ring buffer. Idempotent."""
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

    formatter = LevelPrefixFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    ring_handler = RingBufferHandler(_global_buffer)
    ring_handler.setLevel(logging.DEBUG)
    ring_handler.setFormatter(formatter)
    root.addHandler(ring_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return _global_buffer
