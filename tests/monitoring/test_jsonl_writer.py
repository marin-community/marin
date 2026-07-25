# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime as dt
import json
import time

import pytest

from marin.monitoring.jsonl_writer import JsonlChunkWriter, JsonlChunkWriterConfig


@dataclasses.dataclass(frozen=True)
class SampleRecord:
    name: str
    created_at: dt.datetime
    values: tuple[int, int]


class FailingJsonlChunkWriter(JsonlChunkWriter):
    @classmethod
    def _write_text_file(cls, uri: str, body: str) -> None:
        raise OSError("simulated object store failure")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_jsonl_writer_persists_chunks_and_manifest(tmp_path):
    config = JsonlChunkWriterConfig(output_uri=str(tmp_path / "telemetry"), records_per_chunk=2, max_queue_items=10)

    with JsonlChunkWriter(config) as writer:
        assert writer.write({"i": 0}) is True
        assert writer.write({"i": 1}) is True
        assert writer.write({"i": 2}) is True

    telemetry_dir = tmp_path / "telemetry"
    parts = sorted((telemetry_dir / "parts").glob("part-*.jsonl"))
    assert [_read_jsonl(part) for part in parts] == [[{"i": 0}, {"i": 1}], [{"i": 2}]]

    manifest = json.loads((telemetry_dir / "manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["records_enqueued"] == 3
    assert manifest["records_written"] == 3
    assert manifest["records_dropped"] == 0
    assert [chunk["records"] for chunk in manifest["chunks"]] == [2, 1]


def test_jsonl_writer_serializes_native_containers_and_dataclasses(tmp_path):
    config = JsonlChunkWriterConfig(output_uri=str(tmp_path / "telemetry"), records_per_chunk=10, max_queue_items=10)
    created_at = dt.datetime(2026, 7, 24, 12, 34, tzinfo=dt.UTC)

    with JsonlChunkWriter(config) as writer:
        assert writer.write(
            {
                "string": "value",
                "list": [1, "two", {"three": 3}],
                "tuple": (4, 5),
                "dataclass": SampleRecord("sample", created_at, (6, 7)),
            }
        ) is True

    assert _read_jsonl(tmp_path / "telemetry" / "parts" / "part-000000.jsonl") == [
        {
            "dataclass": {"created_at": "2026-07-24T12:34:00+00:00", "name": "sample", "values": [6, 7]},
            "list": [1, "two", {"three": 3}],
            "string": "value",
            "tuple": [4, 5],
        }
    ]


def test_jsonl_writer_raises_after_chunk_write_failure(tmp_path):
    config = JsonlChunkWriterConfig(output_uri=str(tmp_path / "telemetry"), records_per_chunk=1, max_queue_items=1)
    writer = FailingJsonlChunkWriter(config)
    writer.start()

    assert writer.write({"i": 0}) is True
    deadline = time.monotonic() + 5
    while writer._writer_error is None and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="simulated object store failure"):
        writer.write({"i": 1})
    with pytest.raises(RuntimeError, match="simulated object store failure"):
        writer.close()


def test_jsonl_writer_drops_non_json_records_without_blocking(tmp_path):
    config = JsonlChunkWriterConfig(output_uri=str(tmp_path / "telemetry"), records_per_chunk=10, max_queue_items=10)

    with JsonlChunkWriter(config) as writer:
        assert writer.write({"ok": True}) is True
        assert writer.write({"bad": object()}) is False

    manifest = json.loads((tmp_path / "telemetry" / "manifest.json").read_text())
    assert manifest["records_written"] == 1
    assert manifest["records_dropped"] == 1
    assert _read_jsonl(tmp_path / "telemetry" / "parts" / "part-000000.jsonl") == [{"ok": True}]
