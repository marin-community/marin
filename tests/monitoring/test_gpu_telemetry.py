# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import sys
import time

from marin.monitoring.gpu_telemetry import NvidiaSmiTelemetryConfig, nvidia_smi_telemetry, run_nvidia_smi_telemetry


def _csv_command(row_count: int) -> tuple[str, ...]:
    script = "\n".join(
        [
            "print('timestamp, index, utilization.gpu [%], memory.used [MiB]')",
            *[f"print('2026-07-24 00:00:{i:02d}, 0, {10 + i} %, {100 + i} MiB')" for i in range(row_count)],
        ]
    )
    return (sys.executable, "-c", script)


def _read_jsonl_parts(path):
    records = []
    for part in sorted((path / "parts").glob("part-*.jsonl")):
        records.extend(json.loads(line) for line in part.read_text().splitlines())
    return records


def test_run_nvidia_smi_telemetry_writes_metadata_samples_and_manifest(tmp_path):
    config = NvidiaSmiTelemetryConfig(
        output_uri=str(tmp_path / "telemetry"),
        command=_csv_command(5),
        records_per_chunk=3,
        max_queue_items=10,
        log_every=10,
    )

    run_nvidia_smi_telemetry(config, multiprocessing.Event())

    records = _read_jsonl_parts(tmp_path / "telemetry")
    assert [record["record_type"] for record in records] == [
        "metadata",
        "gpu_sample",
        "gpu_sample",
        "gpu_sample",
        "gpu_sample",
        "gpu_sample",
        "metadata",
    ]
    assert records[1]["nvidia_smi"] == {
        "timestamp": "2026-07-24 00:00:00",
        "index": "0",
        "utilization_gpu": "10 %",
        "memory_used": "100 MiB",
    }

    manifest = json.loads((tmp_path / "telemetry" / "manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["records_written"] == 7
    assert [chunk["records"] for chunk in manifest["chunks"]] == [3, 3, 1]


def test_context_manager_stops_silent_command_before_timeout(tmp_path):
    script = "\n".join(
        [
            "import time",
            "print('timestamp, index, utilization.gpu [%]', flush=True)",
            "time.sleep(60)",
        ]
    )
    config = NvidiaSmiTelemetryConfig(
        output_uri=str(tmp_path / "telemetry"),
        command=(sys.executable, "-c", script),
        records_per_chunk=10,
        max_queue_items=10,
        log_every=10,
        stop_timeout=10,
        start_method="fork",
    )

    started_at = time.monotonic()
    with nvidia_smi_telemetry(config):
        pass

    assert time.monotonic() - started_at < 5
    manifest = json.loads((tmp_path / "telemetry" / "manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["records_written"] == 2
    assert [record["event"] for record in _read_jsonl_parts(tmp_path / "telemetry") if "event" in record] == [
        "telemetry_stop"
    ]


def test_context_manager_runs_telemetry_in_child_process(tmp_path):
    config = NvidiaSmiTelemetryConfig(
        output_uri=str(tmp_path / "telemetry"),
        command=_csv_command(3),
        records_per_chunk=10,
        max_queue_items=10,
        log_every=10,
        stop_timeout=10,
        start_method="fork",
    )

    with nvidia_smi_telemetry(config) as handle:
        assert handle.process.pid is not None
        assert handle.process.pid != multiprocessing.current_process().pid
        handle.process.join(timeout=10)
        assert handle.process.exitcode == 0

    manifest = json.loads((tmp_path / "telemetry" / "manifest.json").read_text())
    assert manifest["completed"] is True
    assert manifest["records_written"] == 5
    assert len([record for record in _read_jsonl_parts(tmp_path / "telemetry") if record["record_type"] == "gpu_sample"]) == 3
