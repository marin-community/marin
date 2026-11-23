# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the stage-f storage recalibration inputs.

The fixture is one real frozen abstention task archive row: it carries the declared
20480 MiB defect and the exact-gate verifier the calibration-only solution feeds.
"""

import base64
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tomlkit
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.gym_storage_recalibration import (
    ABSTENTION_SOURCE,
    INJECTION_SOURCE,
    MAXIMUMS,
    abstention_calibration_solution,
    build_calibration_packet,
    build_revision,
    census_release,
    census_report,
    declared_environment_limits,
    gate_precheck,
    injection_calibration_solution,
    measured_calibration_storage,
)

try:
    from shellbox.backends.daytona.usage_cli import PER_SANDBOX_MAXIMUM
except ImportError:  # the Daytona SDK is a production-only dependency
    PER_SANDBOX_MAXIMUM = None

FIXTURE = Path(__file__).with_name("fixtures") / "gym_storage_recalibration" / "abstention_task.json"
TASK_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("path", pa.string()),
        ("task_spec", pa.string()),
        ("task_sha256", pa.string()),
        ("runtime", pa.string()),
    ]
)


def fixture_row() -> dict:
    document = json.loads(FIXTURE.read_text())
    serialized = document["task_spec"]
    assert hashlib.sha256(serialized.encode()).hexdigest() == document["task_sha256"]
    return document


def fixture_task() -> TaskSpec:
    return TaskSpec.model_validate_json(fixture_row()["task_spec"])


def release_parquet(tmp_path: Path, count: int = 3) -> Path:
    """A small release parquet of distinct tasks carrying the same declared defect."""
    document = fixture_row()
    path = tmp_path / "release.parquet"
    rows = []
    for index in range(count):
        task = TaskSpec.model_validate_json(document["task_spec"])
        task = task.model_copy(update={"id": f"{task.id.rstrip('02468acef')}{index:064d}"})
        serialized = task.model_dump_json()
        rows.append(
            {
                "id": task.id,
                "path": task.id,
                "task_spec": serialized,
                "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "runtime": "{}",
            }
        )
    pq.write_table(
        pa.table({name: [row[name] for row in rows] for name in ("id", "path", "task_spec", "task_sha256", "runtime")}),
        path,
    )
    return path


def test_daytona_maximum_matches_the_documented_per_sandbox_limits() -> None:
    if PER_SANDBOX_MAXIMUM is None:
        pytest.skip("the Daytona SDK is a production-only dependency")
    assert PER_SANDBOX_MAXIMUM.cpu == MAXIMUMS["cpus"]
    assert PER_SANDBOX_MAXIMUM.memory_gib == MAXIMUMS["memory_mb"] // 1024
    assert PER_SANDBOX_MAXIMUM.disk_gib == MAXIMUMS["storage_mb"] // 1024


def test_fixture_carries_the_declared_defect() -> None:
    task = fixture_task()
    declared = declared_environment_limits(task)
    assert declared == {"cpus": 1, "memory_mb": 4096, "storage_mb": 20480}
    assert declared["storage_mb"] > MAXIMUMS["storage_mb"]
    assert declared["cpus"] <= MAXIMUMS["cpus"] and declared["memory_mb"] <= MAXIMUMS["memory_mb"]


def test_census_reports_the_uniform_declared_defect(tmp_path) -> None:
    census = census_release(ABSTENTION_SOURCE, release_parquet(tmp_path))
    report = census_report(census)
    assert report["task_count"] == 3
    assert report["uniform"] is True
    declared = json.loads(next(iter(report["declared_distribution"])))
    assert declared == {"cpus": 1, "memory_mb": 4096, "storage_mb": 20480}
    assert report["over_daytona_maximum"] == {"storage_mb": 20480}
    assert all(len(hashes) == 1 for hashes in report["template_fingerprints"].values())


def test_calibration_packet_removes_only_the_declared_storage(tmp_path) -> None:
    task = fixture_task()
    original_toml = next(
        resource
        for group in (task.resources.all, task.resources.worker, task.resources.verifier, task.resources.oracle)
        for resource in group
        if resource.path == "task.toml"
    )
    before = tomlkit.parse(base64.b64decode(original_toml.source.content_base64).decode())
    packet = tmp_path / "packet"
    build_calibration_packet(task, packet)
    after = tomlkit.parse((packet / "task.toml").read_text())
    assert "storage_mb" not in after["environment"]
    assert after["environment"]["cpus"] == before["environment"]["cpus"] == 1
    assert after["environment"]["memory_mb"] == before["environment"]["memory_mb"] == 4096
    assert after["agent"]["timeout_sec"] == before["agent"]["timeout_sec"] == 900.0
    assert after["verifier"]["timeout_sec"] == before["verifier"]["timeout_sec"] == 600.0
    # The packet remains a native package with its environment and tests intact.
    assert (packet / "environment" / "Dockerfile").is_file()
    assert (packet / "tests" / "test.sh").is_file()
    with pytest.raises(FileExistsError):
        build_calibration_packet(task, packet)


def test_abstention_calibration_solution_writes_the_expected_answer() -> None:
    task = fixture_task()
    solution, payload = abstention_calibration_solution(task)
    data = json.loads(
        base64.b64decode(
            next(
                resource
                for group in (task.resources.all, task.resources.worker, task.resources.verifier, task.resources.oracle)
                for resource in group
                if resource.path == "tests/verifier_data.json"
            ).source.content_base64,
            validate=True,
        )
    )
    assert payload == data["expected_answer"]
    script = next(line for line in solution.splitlines() if line.startswith("printf"))
    assert script.startswith("printf '%s' '")
    assert script.endswith("' > /app/answer.txt")
    precheck = gate_precheck(ABSTENTION_SOURCE, task)
    assert precheck["exit_code"] == 0
    assert precheck["reward"] == 1.0
    assert precheck["solution_sha256"] == hashlib.sha256(solution.encode()).hexdigest()


def test_injection_calibration_solution_is_a_rewarded_refusal() -> None:
    task = fixture_task()
    _solution, payload = injection_calibration_solution(task)
    action = json.loads(payload)
    assert action["action"] == "respond"
    precheck = gate_precheck(INJECTION_SOURCE, task)
    assert precheck["reward"] == 1.0


def _calibration_receipt(storage_mb: int) -> dict:
    shape = {
        "cpus": 1,
        "memory_mb": 4096,
        "storage_mb": storage_mb,
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 600,
    }
    return {
        "calibration": {
            "attempts": [{"execution_requirements": shape}],
            "oracle_kind": "calibration_only",
        }
    }


def test_measured_storage_requires_one_agreed_value() -> None:
    measured, per_task = measured_calibration_storage({"a": _calibration_receipt(2048), "b": _calibration_receipt(2048)})
    assert measured == 2048
    assert per_task == {"a": 2048, "b": 2048}
    with pytest.raises(ValueError, match="disagreeing storage shapes"):
        measured_calibration_storage({"a": _calibration_receipt(2048), "b": _calibration_receipt(4096)})


def test_revision_replaces_storage_and_keeps_everything_else(tmp_path) -> None:
    release = release_parquet(tmp_path)
    rows = pq.read_table(release).to_pylist()
    output = tmp_path / "revision"
    summary = build_revision(ABSTENTION_SOURCE, release, 2048, output)
    assert summary["measured_storage_mb"] == 2048
    revised = pq.read_table(output / "tasks.parquet").to_pylist()
    assert len(revised) == len(rows)
    with (output / "revision-ledger.json").open() as stream:
        ledger = {entry["task_id"]: entry for entry in json.load(stream)}

    def other_resources(spec: TaskSpec) -> list[tuple[str, str]]:
        return sorted(
            (resource.path, resource.source.content_base64)
            for group in (spec.resources.all, spec.resources.worker, spec.resources.verifier, spec.resources.oracle)
            for resource in group
            if resource.path != "task.toml"
        )

    for row, original in zip(revised, rows, strict=True):
        task = TaskSpec.model_validate_json(row["task_spec"])
        declared = declared_environment_limits(task)
        assert declared["storage_mb"] == 2048
        assert declared["cpus"] == 1 and declared["memory_mb"] == 4096
        entry = ledger[task.id]
        assert entry["before_storage_mb"] == 20480 and entry["after_storage_mb"] == 2048
        assert entry["before_task_sha256"] == original["task_sha256"]
        assert entry["after_task_sha256"] == row["task_sha256"]
        assert other_resources(task) == other_resources(TaskSpec.model_validate_json(original["task_spec"]))
        assert task.execution_requirements.storage_mb == 2048


def test_revision_refuses_values_outside_the_daytona_maximum(tmp_path) -> None:
    release = release_parquet(tmp_path)
    with pytest.raises(ValueError, match="per-sandbox maximum"):
        build_revision(ABSTENTION_SOURCE, release, 20480, tmp_path / "out")
