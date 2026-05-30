# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sinks that persist each ProbeResult beyond the stdout log line:

- ``JsonlGcsSink`` — append every result to a per-UTC-day local JSONL file;
  on the first result of a new day (and for files stranded by a restart),
  gzip the finished file, upload it to GCS under ``dt=<date>/``, and delete
  the local copy.
- ``FinelogTableSink`` — write every result as a row into a dedicated finelog
  namespace.

Sinks are best-effort telemetry: the runner calls them after logging and
swallows their failures, so a sink fault never disrupts probing.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import ClassVar, Protocol, TextIO

from finelog.client.log_client import LogClient, Table
from rigging.filesystem import open_url

logger = logging.getLogger(__name__)


class ProbeResultLike(Protocol):
    """The fields a sink reads off a ProbeResult. A Protocol (not the concrete
    ProbeResult) keeps this module decoupled from the entrypoint module."""

    name: str
    is_success: bool
    started_at: datetime
    wall_time: float | None


class ProbeSink(Protocol):
    def record(self, result: ProbeResultLike) -> None: ...


def result_to_json(result: ProbeResultLike) -> str:
    """Serialize a result to a single JSON object (one JSONL line)."""
    return json.dumps(
        {
            "name": result.name,
            "is_success": result.is_success,
            "started_at": result.started_at.isoformat(timespec="milliseconds"),
            "wall_time": result.wall_time,
        }
    )


class JsonlGcsSink:
    """Append results to a daily local JSONL file, rolling finished days up to
    GCS. The file is keyed by the result's UTC date; on a date change the prior
    file (and any stranded files from a previous process) is gzipped, uploaded
    to ``<gcs_prefix>/dt=<date>/probes-<date>.jsonl.gz``, and removed locally.
    Thread-safe: probes record concurrently."""

    def __init__(self, local_dir: Path, gcs_prefix: str) -> None:
        self._dir = local_dir
        self._gcs_prefix = gcs_prefix.rstrip("/")
        self._lock = threading.Lock()
        self._day: date | None = None
        self._fh: TextIO | None = None
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, result: ProbeResultLike) -> None:
        day = result.started_at.date()
        line = result_to_json(result) + "\n"
        with self._lock:
            if self._day != day:
                self._roll_to(day)
            assert self._fh is not None
            self._fh.write(line)
            self._fh.flush()

    @staticmethod
    def _filename(day: date) -> str:
        return f"probes-{day.isoformat()}.jsonl"

    def _roll_to(self, day: date) -> None:
        """Open ``day``'s file (appending if it already exists from earlier
        today) after finalizing every other local JSONL file."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        keep = self._filename(day)
        for path in sorted(self._dir.glob("probes-*.jsonl")):
            if path.name != keep:
                self._finalize(path)
        self._day = day
        self._fh = (self._dir / keep).open("a")

    def _finalize(self, path: Path) -> None:
        """gzip ``path``, upload to GCS under its date partition, delete locally."""
        file_date = path.stem.removeprefix("probes-")
        gz = path.with_name(path.name + ".gz")
        with path.open("rb") as raw, gzip.open(gz, "wb") as compressed:
            shutil.copyfileobj(raw, compressed)
        dest = f"{self._gcs_prefix}/dt={file_date}/{gz.name}"
        with gz.open("rb") as src, open_url(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        gz.unlink()
        path.unlink()
        logger.info("rolled probe results to %s", dest)


@dataclass
class ProbeResultRow:
    """Row schema for the finelog namespace; grouped (keyed) by probe name."""

    name: str
    is_success: bool
    started_at: datetime
    wall_time: float
    key_column: ClassVar[str] = "name"


class FinelogTableSink:
    """Write each result as a row into a dedicated finelog namespace."""

    def __init__(self, finelog: LogClient, namespace: str) -> None:
        self._table: Table = finelog.get_table(namespace, ProbeResultRow)

    def record(self, result: ProbeResultLike) -> None:
        self._table.write(
            [
                ProbeResultRow(
                    name=result.name,
                    is_success=result.is_success,
                    started_at=result.started_at,
                    wall_time=result.wall_time if result.wall_time is not None else 0.0,
                )
            ]
        )
