# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor-backed JSONL chunk writer for durable telemetry streams."""

import dataclasses
import datetime as dt
import json
import logging
import posixpath
import queue
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

_DONE = object()


class BackpressurePolicy(StrEnum):
    """Queue behavior when the writer actor is full."""

    BLOCK = "block"
    DROP = "drop"


@dataclass(frozen=True)
class JsonlChunkWriterConfig:
    """Configuration for :class:`JsonlChunkWriter`.

    Args:
        output_uri: Destination directory for ``parts/part-*.jsonl`` and
            ``manifest.json``.
        records_per_chunk: Number of records per durable chunk file.
        max_queue_items: Maximum queued records before backpressure policy is
            applied.
        backpressure_policy: Whether ``write`` blocks or drops when the queue is
            full.
        log_every: Log writer progress after this many enqueued records.
    """

    output_uri: str
    records_per_chunk: int = 120
    max_queue_items: int = 10_000
    backpressure_policy: BackpressurePolicy = BackpressurePolicy.BLOCK
    log_every: int = 1_000

    def __post_init__(self) -> None:
        if not self.output_uri:
            raise ValueError("output_uri must be non-empty")
        if self.records_per_chunk <= 0:
            raise ValueError("records_per_chunk must be positive")
        if self.max_queue_items <= 0:
            raise ValueError("max_queue_items must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if isinstance(self.backpressure_policy, str):
            object.__setattr__(self, "backpressure_policy", BackpressurePolicy(self.backpressure_policy))
        if not isinstance(self.backpressure_policy, BackpressurePolicy):
            raise ValueError("backpressure_policy must be a BackpressurePolicy")


@dataclass(frozen=True)
class JsonlChunkWriterStats:
    """Snapshot of writer counters."""

    records_enqueued: int
    records_written: int
    records_dropped: int
    chunks_written: int
    bytes_written: int
    max_queue_size_observed: int


class JsonlChunkWriter:
    """JSONL writer backed by a queue and writer thread.

    ``write`` serializes the object and enqueues one JSON line. Remote I/O is
    performed only by the writer thread. Queue backpressure is controlled by
    ``JsonlChunkWriterConfig.backpressure_policy``.
    """

    def __init__(self, config: JsonlChunkWriterConfig):
        self.config = config
        self._queue: queue.Queue[str | object] = queue.Queue(maxsize=config.max_queue_items)
        self._thread: threading.Thread | None = None
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._chunks: list[dict[str, Any]] = []
        self._records_enqueued = 0
        self._records_written = 0
        self._records_dropped = 0
        self._chunks_written = 0
        self._bytes_written = 0
        self._max_queue_size_observed = 0
        self._closed = False
        self._writer_error: str | None = None

    def __enter__(self) -> "JsonlChunkWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def start(self) -> None:
        """Start the writer actor thread."""
        if self._thread is not None:
            raise RuntimeError("JsonlChunkWriter is already started")
        self._started_at = dt.datetime.now(dt.UTC).isoformat()
        self._thread = threading.Thread(target=self._run_writer, name="jsonl-chunk-writer", daemon=False)
        self._thread.start()

    def write(self, obj: Any) -> bool:
        """Serialize and enqueue one JSON object.

        Returns ``True`` when the record was accepted. Returns ``False`` when
        the object is not JSON-serializable, the writer is closed, or the queue
        is full and ``backpressure_policy`` is ``DROP``.
        """
        try:
            line = json.dumps(obj, default=_json_default, sort_keys=True, separators=(",", ":")) + "\n"
        except (TypeError, ValueError):
            self._records_dropped += 1
            if self._records_dropped == 1 or self._records_dropped % self.config.log_every == 0:
                logger.warning(
                    "jsonl-writer dropping non-json record output_uri=%s dropped=%d",
                    self.config.output_uri,
                    self._records_dropped,
                )
            return False
        if self._closed:
            return False
        self._raise_if_writer_failed()
        if self.config.backpressure_policy is BackpressurePolicy.BLOCK:
            self._put_record_with_backpressure(line)
        else:
            try:
                self._queue.put_nowait(line)
            except queue.Full:
                self._raise_if_writer_failed()
                self._records_dropped += 1
                if self._records_dropped == 1 or self._records_dropped % self.config.log_every == 0:
                    logger.warning(
                        "jsonl-writer dropping records output_uri=%s queue_size=%d dropped=%d",
                        self.config.output_uri,
                        self._queue.qsize(),
                        self._records_dropped,
                    )
                return False

        self._records_enqueued += 1
        queue_size = self._queue.qsize()
        self._max_queue_size_observed = max(self._max_queue_size_observed, queue_size)
        if self._records_enqueued % self.config.log_every == 0:
            logger.info(
                "jsonl-writer queued output_uri=%s enqueued=%d queue_size=%d dropped=%d",
                self.config.output_uri,
                self._records_enqueued,
                queue_size,
                self._records_dropped,
            )
        return True

    def close(self) -> None:
        """Signal completion, wait for final flush, and write the manifest."""
        if self._closed:
            return
        self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._put_done_signal()
            self._thread.join()
        if self._writer_error is not None:
            raise RuntimeError(f"JSONL writer failed: {self._writer_error}")

    def stats(self) -> JsonlChunkWriterStats:
        """Return a best-effort counter snapshot."""
        return JsonlChunkWriterStats(
            records_enqueued=self._records_enqueued,
            records_written=self._records_written,
            records_dropped=self._records_dropped,
            chunks_written=self._chunks_written,
            bytes_written=self._bytes_written,
            max_queue_size_observed=self._max_queue_size_observed,
        )

    def _put_record_with_backpressure(self, line: str) -> None:
        while True:
            self._raise_if_writer_failed()
            try:
                self._queue.put(line, timeout=1.0)
                return
            except queue.Full:
                continue

    def _put_done_signal(self) -> None:
        while self._thread is not None and self._thread.is_alive():
            try:
                self._queue.put(_DONE, timeout=1.0)
                return
            except queue.Full:
                continue

    def _raise_if_writer_failed(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError(f"JSONL writer failed: {self._writer_error}")
        if self._thread is not None and not self._thread.is_alive() and not self._closed:
            raise RuntimeError("JSONL writer thread exited before close")

    def _run_writer(self) -> None:
        part_index = 0
        records: list[str] = []
        try:
            while True:
                item = self._queue.get()
                if item is _DONE:
                    break
                assert isinstance(item, str)
                records.append(item)
                if len(records) >= self.config.records_per_chunk:
                    self._flush(records, part_index)
                    part_index += 1
                    records = []
            if records:
                self._flush(records, part_index)
            self._ended_at = dt.datetime.now(dt.UTC).isoformat()
            self._write_manifest(completed=True)
        except Exception as exc:
            self._writer_error = f"{type(exc).__name__}: {exc}"
            self._ended_at = dt.datetime.now(dt.UTC).isoformat()
            try:
                self._write_manifest(completed=False)
            except Exception:
                logger.exception("jsonl-writer failed to write failure manifest output_uri=%s", self.config.output_uri)

    def _flush(self, records: list[str], part_index: int) -> None:
        relative_path = f"parts/part-{part_index:06d}.jsonl"
        uri = f"{self.config.output_uri.rstrip('/')}/{relative_path}"
        body = "".join(records)
        self._write_text_file(uri, body)
        byte_count = len(body.encode("utf-8"))
        self._records_written += len(records)
        self._chunks_written += 1
        self._bytes_written += byte_count
        self._chunks.append(
            {
                "path": relative_path,
                "records": len(records),
                "bytes": byte_count,
                "written_at": dt.datetime.now(dt.UTC).isoformat(),
            }
        )
        logger.info(
            "jsonl-writer flush output_uri=%s part=%06d records=%d bytes=%d queue_size=%d dropped=%d",
            self.config.output_uri,
            part_index,
            len(records),
            byte_count,
            self._queue.qsize(),
            self._records_dropped,
        )

    def _write_manifest(self, *, completed: bool) -> None:
        manifest = {
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "completed": completed,
            "error": self._writer_error,
            "config": dataclasses.asdict(self.config),
            "records_enqueued": self._records_enqueued,
            "records_written": self._records_written,
            "records_dropped": self._records_dropped,
            "chunks_written": self._chunks_written,
            "bytes_written": self._bytes_written,
            "max_queue_size_observed": self._max_queue_size_observed,
            "chunks": list(self._chunks),
        }
        manifest_uri = f"{self.config.output_uri.rstrip('/')}/manifest.json"
        self._write_text_file(manifest_uri, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    @classmethod
    def _write_text_file(cls, uri: str, body: str) -> None:
        fs, path = fsspec.core.url_to_fs(uri)
        parent = posixpath.dirname(path)
        if parent:
            fs.mkdirs(parent, exist_ok=True)
        with fs.open(path, "wt") as f:
            f.write(body)


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


