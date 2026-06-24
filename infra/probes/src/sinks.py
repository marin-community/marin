# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sinks that persist each Sample beyond the stdout log line:

- ``JsonlGcsSink`` — append every sample to a per-UTC-day local JSONL file;
  on the first sample of a new day (and for files stranded by a restart),
  gzip the finished file, upload it to GCS under ``dt=<date>/``, and delete
  the local copy.
- ``FinelogTableSink`` — write every sample as a row into a dedicated finelog
  namespace.

Both satisfy the runner's ``MetricSink`` protocol structurally. Sinks are
best-effort telemetry: the runner calls them after logging and swallows their
failures, so a sink fault never disrupts collection.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import threading
from datetime import date
from pathlib import Path
from typing import TextIO

from finelog.client.log_client import LogClient, Table
from rigging.filesystem import open_url
from sample import Sample

logger = logging.getLogger(__name__)


def sample_to_json(sample: Sample) -> str:
    """Serialize a sample to a single JSON object (one JSONL line). Labels are
    re-parsed into a nested object so the JSONL is queryable without a second
    decode."""
    assert sample.collected_at is not None, "runner stamps collected_at before record"
    return json.dumps(
        {
            "metric": sample.metric,
            "value": sample.value,
            "labels": json.loads(sample.labels),
            "collected_at": sample.collected_at.isoformat(timespec="milliseconds"),
        }
    )


class JsonlGcsSink:
    """Append samples to a daily local JSONL file, rolling finished days up to
    GCS. The file is keyed by the sample's UTC date; on a date change the prior
    file (and any stranded files from a previous process) is gzipped, uploaded
    to ``<gcs_prefix>/dt=<date>/probes-<date>.jsonl.gz``, and removed locally.
    Thread-safe: collectors record concurrently."""

    def __init__(self, local_dir: Path, gcs_prefix: str) -> None:
        self._dir = local_dir
        self._gcs_prefix = gcs_prefix.rstrip("/")
        self._lock = threading.Lock()
        self._day: date | None = None
        self._fh: TextIO | None = None
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, sample: Sample) -> None:
        assert sample.collected_at is not None, "runner stamps collected_at before record"
        day = sample.collected_at.date()
        line = sample_to_json(sample) + "\n"
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
        logger.info("rolled probe samples to %s", dest)


class FinelogTableSink:
    """Write each sample as a row into a dedicated finelog namespace. ``Sample``
    is the table schema directly: get_table derives the columns from it and
    Table.write reads its fields."""

    def __init__(self, finelog: LogClient, namespace: str) -> None:
        self._table: Table = finelog.get_table(namespace, Sample)

    def record(self, sample: Sample) -> None:
        self._table.write([sample])
