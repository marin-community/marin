# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-local execution evidence survives concurrency and partial failures."""

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.stage_io import ZephyrWorkerError

from taskcompendium.pipeline.execution_telemetry import SourceTelemetry, execute_phase

PAYLOAD_CANARY = "candidate-payload-must-never-enter-telemetry"


def measured_value(value: int) -> dict[str, int | str]:
    metrics = counters.current_stage()
    metrics.update_counter("fixture/items", value)
    metrics.set_aggregation("fixture/peak", counters.Aggregation.MAX)
    metrics.update_counter("fixture/peak", value)
    return {"value": value, "payload": PAYLOAD_CANARY}


def record_source(context: ZephyrContext, output: Path, source: str, value: int) -> list:
    telemetry = SourceTelemetry(source, str(output))
    with telemetry.record():
        with telemetry.phase("prepare") as phase:
            first = execute_phase(context, Dataset.from_list([value, value + 1]).map(measured_value), telemetry=phase)
        with telemetry.phase("prepare") as phase:
            second = execute_phase(context, Dataset.from_list([value + 2]).map(measured_value), telemetry=phase)
        with telemetry.phase("empty") as phase:
            empty = execute_phase(context, Dataset.from_list([]), telemetry=phase)
    return [first, second, empty]


def test_concurrent_sources_preserve_distinct_final_counters_without_result_payloads(tmp_path):
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                source: executor.submit(
                    contextvars.copy_context().run, record_source, context, tmp_path / source, source, value
                )
                for source, value in [("left", 2), ("right", 20)]
            }
            results = {source: future.result() for source, future in futures.items()}
    ids = set()
    for source, value in [("left", 2), ("right", 20)]:
        serialized = (tmp_path / source / "telemetry.json").read_text()
        assert PAYLOAD_CANARY not in serialized
        report = json.loads(serialized)
        assert report["source"] == source and report["status"] == "completed"
        assert [phase["phase"] for phase in report["phases"]] == ["prepare", "prepare", "empty"]
        executions = [phase["executions"][0] for phase in report["phases"]]
        assert [result.results for result in results[source]] == [
            [{"value": value, "payload": PAYLOAD_CANARY}, {"value": value + 1, "payload": PAYLOAD_CANARY}],
            [{"value": value + 2, "payload": PAYLOAD_CANARY}],
            [],
        ]
        for execution, result in zip(executions, results[source], strict=True):
            assert execution["counters"] == result.counters
            assert execution["execution_id"] == result.execution_id
        assert executions[0]["counters"]["fixture/items"] == value * 2 + 1
        assert executions[0]["counters"]["fixture/peak"] == value + 1
        assert executions[1]["counters"]["fixture/peak"] == value + 2
        assert executions[2]["execution_id"] == "" and executions[2]["counters"] == {}
        source_ids = {entry["execution_id"] for entry in executions if entry["execution_id"]}
        assert len(source_ids) == 2 and not ids.intersection(source_ids)
        ids.update(source_ids)


def failing_value(_value: int) -> int:
    raise ValueError("Invalid source fixture")


def test_execution_failure_preserves_completed_phase_and_propagates_error(tmp_path):
    telemetry = SourceTelemetry("broken", str(tmp_path / "output"))
    with ZephyrContext(max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with pytest.raises(ZephyrWorkerError, match="Invalid source fixture"):
            with telemetry.record():
                with telemetry.phase("prepare") as phase:
                    completed = execute_phase(context, Dataset.from_list([3]).map(measured_value), telemetry=phase)
                with telemetry.phase("normalize") as phase:
                    execute_phase(context, Dataset.from_list([3]).map(failing_value), telemetry=phase)
    report = json.loads((tmp_path / "output/telemetry.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "ZephyrWorkerError"
    first, last = report["phases"]
    assert first["status"] == "completed"
    assert first["executions"][0]["counters"] == completed.counters
    assert first["executions"][0]["execution_id"] == completed.execution_id
    assert last["status"] == "failed" and last["error_type"] == "ZephyrWorkerError"
    failed = last["executions"][0]
    assert failed["status"] == "failed" and failed["error_type"] == "ZephyrWorkerError"
    assert failed["execution_id"] == "" and failed["counters"] == {}
