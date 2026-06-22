# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JsonlGcsSink behavior: daily local append, and rollover to GCS (gzip +
upload + local delete). The GCS write (open_url) is the only I/O boundary and
is stubbed to capture uploads in memory."""

from __future__ import annotations

import dataclasses
import gzip
import io
import json
from datetime import UTC, datetime

import pytest
import sinks
from sample import Sample
from sinks import JsonlGcsSink

GCS_PREFIX = "gs://bucket/infra/probes"


def _sample(day: str, metric: str = "probe_up", value: float = 1.0, **labels: str) -> Sample:
    collected = datetime.fromisoformat(day).replace(tzinfo=UTC)
    return dataclasses.replace(Sample.of(metric, value, **labels), collected_at=collected)


@pytest.fixture
def captured_uploads(monkeypatch):
    """Stub sinks.open_url so finalized files are captured in memory instead of
    hitting GCS. Returns {dest_url: gzipped_bytes}."""
    uploads: dict[str, bytes] = {}

    def fake_open_url(dest: str, mode: str = "rb"):
        buf = io.BytesIO()

        class _CM:
            def __enter__(self):
                return buf

            def __exit__(self, *exc):
                uploads[dest] = buf.getvalue()
                return False

        return _CM()

    monkeypatch.setattr(sinks, "open_url", fake_open_url)
    return uploads


def _decompressed(uploads: dict[str, bytes]) -> dict[str, str]:
    return {dest: gzip.decompress(blob).decode() for dest, blob in uploads.items()}


def test_same_day_appends_without_upload(tmp_path, captured_uploads):
    sink = JsonlGcsSink(tmp_path, GCS_PREFIX)
    sink.record(_sample("2026-05-30T00:00:01", metric="probe_up", probe="controller-ping"))
    sink.record(_sample("2026-05-30T00:01:01", metric="tpu_provision_success", value=3.0, zone="us-east5-a"))

    assert captured_uploads == {}, "no rollover should have happened"
    lines = (tmp_path / "probes-2026-05-30.jsonl").read_text().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {r["metric"] for r in rows} == {"probe_up", "tpu_provision_success"}
    # labels round-trip as a nested object
    assert any(r["labels"] == {"zone": "us-east5-a"} for r in rows)


def test_day_rollover_finalizes_previous_file(tmp_path, captured_uploads):
    sink = JsonlGcsSink(tmp_path, GCS_PREFIX)
    sink.record(_sample("2026-05-30T23:59:00"))
    sink.record(_sample("2026-05-31T00:00:30"))  # new UTC day -> rolls 05-30 up

    dest = f"{GCS_PREFIX}/dt=2026-05-30/probes-2026-05-30.jsonl.gz"
    assert dest in captured_uploads
    assert "2026-05-30T23:59:00" in _decompressed(captured_uploads)[dest]
    # the finalized day's local files are gone; today's is present and live
    assert not (tmp_path / "probes-2026-05-30.jsonl").exists()
    assert not (tmp_path / "probes-2026-05-30.jsonl.gz").exists()
    assert (tmp_path / "probes-2026-05-31.jsonl").exists()


def test_startup_sweeps_stranded_file(tmp_path, captured_uploads):
    # A file left behind by a previous process that died before rollover.
    (tmp_path / "probes-2026-05-29.jsonl").write_text('{"metric":"probe_up"}\n')

    sink = JsonlGcsSink(tmp_path, GCS_PREFIX)
    sink.record(_sample("2026-05-30T00:00:01"))

    dest = f"{GCS_PREFIX}/dt=2026-05-29/probes-2026-05-29.jsonl.gz"
    assert dest in captured_uploads
    assert not (tmp_path / "probes-2026-05-29.jsonl").exists()
