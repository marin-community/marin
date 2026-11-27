# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SWE exception preserves contracts and remains visible through scored cell reports."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
from finestore.rl.difficulty import PROTOCOL_ID, Solver
from pydantic import TypeAdapter, ValidationError
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty import resource_deviations as resources
from experiments.post_training.task_curation.difficulty.importer import population_revision, receipt_row
from experiments.post_training.task_curation.difficulty.ledger import append_rows, read_rows, row_payload
from experiments.post_training.task_curation.difficulty.report import status_report
from experiments.post_training.task_curation.difficulty.state import CatalogSource, CoveragePolicy
from experiments.post_training.task_curation.difficulty_prepared import PreparedTask


@pytest.fixture
def native(tmp_path, monkeypatch):
    directory = tmp_path / "task"
    directory.mkdir()
    (directory / "environment").mkdir()
    (directory / "instruction.md").write_text("Fix the repository defect.\n")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    (directory / "task.toml").write_text(
        '[environment]\ndocker_image="alpine:3.22"\ncpus=6\nmemory_mb=24576\nstorage_mb=32768\n'
        "[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=720\n"
    )
    sample = {
        "source_id": resources.SWE_SOURCE,
        "release": {"repo": "fixture", "revision": "pin", "url": "https://fixture.invalid/pin"},
        "converter_version": "fixture",
        "converter_revision": "a" * 40,
        "task_specs_sha256": "b" * 64,
        "tasks": [{"task_id": "task", "task_sha256": "a" * 64}],
    }
    scope = {"registry_revision": population_revision(sample), "tasks": {"task": "a" * 64}}
    monkeypatch.setattr(resources, "authority", lambda: scope)
    return directory, sample, scope


def deviation(native):
    path, sample, scope = native
    return resources.prepare_resource_deviation(
        sample["source_id"], scope["registry_revision"], "task", "a" * 64, "c" * 64, path
    )


def test_native_daytona_adapter_applies_condition_without_rewriting_source(native, tmp_path):
    DaytonaEnvironment = pytest.importorskip("harbor.environments.daytona").DaytonaEnvironment
    TaskEnvironmentConfig = pytest.importorskip("harbor.models.task.config").EnvironmentConfig
    TrialPaths = pytest.importorskip("harbor.models.trial.paths").TrialPaths
    trial_environment_config = pytest.importorskip(
        "experiments.post_training.task_curation.difficulty_trials"
    ).trial_environment_config

    path, _, _ = native
    original = (path / "task.toml").read_bytes()
    record = deviation(native)
    task = PreparedTask(
        task_id="task",
        runtime_sha256="c" * 64,
        path=str(path),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=None,
        resource_deviation=record,
    )
    config = trial_environment_config(task, {})
    environment = DaytonaEnvironment(
        environment_dir=path / "environment",
        environment_name="fixture",
        session_id="fixture",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=TaskEnvironmentConfig(docker_image="alpine:3.22", cpus=6, memory_mb=24576, storage_mb=32768),
        override_cpus=config.override_cpus,
        override_memory_mb=config.override_memory_mb,
        override_storage_mb=config.override_storage_mb,
    )
    actual = environment.task_env_config
    assert (actual.cpus, actual.memory_mb, actual.storage_mb) == (4, 8192, 10240)
    assert (path / "task.toml").read_bytes() == original
    assert PreparedTask.model_validate_json(task.model_dump_json()).resource_deviation == record


@pytest.mark.parametrize(
    "status,reward,reason", [("scored", 0.0, None), ("infrastructure", None, "grader_infrastructure_failure")]
)
def test_receipt_ledger_and_cell_report_keep_versioned_declared_actual_condition(
    native, tmp_path, status, reward, reason
):
    _, sample, _ = native
    record = deviation(native)
    attempt = {
        "task_id": "task",
        "status": status,
        "reward": reward,
        "reason": reason,
        "resource_deviation": asdict(record),
    }
    path = tmp_path / "attempt.json"
    path.write_text(json.dumps(attempt))
    row = receipt_row(
        {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        sample,
        Solver.SMALL,
        "run",
        "2026-11-25T15:00:00+00:00",
    )
    assert row.resource_deviation == record
    ledger = str(tmp_path / "ledger")
    append_rows(ledger, [row])
    restored = read_rows(ledger)
    assert restored == [row]
    catalog = [
        CatalogSource(resources.SWE_SOURCE, population_revision(sample), (("task", "a" * 64),), "queue", "d" * 64)
    ]
    report = status_report(restored, catalog, CoveragePolicy())
    cell = next(cell for cell in report["cells"] if cell["solver"] == Solver.SMALL)
    assert cell["resource_deviations"] == [asdict(record)]
    assert cell["state"] == ("MEASURED" if status == "scored" else "NEEDS_ATTEMPTS")
    with pytest.raises(ValueError, match="another frozen task"):
        replace(row, task_id="foreign")
    legacy = replace(row, resource_deviation=None)
    assert "resource_deviation" not in row_payload(legacy)
    if status == "scored":
        with pytest.raises(ValueError, match="cannot mix scored execution-resource"):
            status_report([row, replace(legacy, attempt_id="other")], catalog, CoveragePolicy())


def test_deviation_rejects_foreign_tasks_backends_and_modified_native_shape(native):
    path, sample, scope = native
    assert (
        resources.prepare_resource_deviation("other", scope["registry_revision"], "task", "a" * 64, "c" * 64, path)
        is None
    )
    assert resources.prepare_resource_deviation(sample["source_id"], "foreign", "task", "a" * 64, "c" * 64, path) is None
    with pytest.raises(ValueError, match="frozen authorized"):
        resources.prepare_resource_deviation(
            sample["source_id"], scope["registry_revision"], "task", "f" * 64, "c" * 64, path
        )
    record = deviation(native)
    with pytest.raises(ValueError, match="only to native Daytona"):
        record.environment_overrides(path, backend="local", workload="native")
    with pytest.raises(ValueError, match="only to native Daytona"):
        record.environment_overrides(path, backend="daytona", workload="task_spec")
    original = (path / "task.toml").read_text()
    (path / "task.toml").write_text(original.replace("cpus=6", "cpus=8"))
    with pytest.raises(ValueError, match="declaration changed"):
        record.environment_overrides(path, backend="daytona", workload="native")
    payload = asdict(record)
    payload["actual"]["memory_mb"] = 4096
    with pytest.raises((ValueError, ValidationError), match="authorized SWE"):
        TypeAdapter(resources.ResourceDeviation).validate_python(payload)


def test_resource_condition_is_recorded_even_when_provider_construction_fails(native, tmp_path, monkeypatch):
    from types import SimpleNamespace  # noqa: PLC0415

    worker = pytest.importorskip("experiments.post_training.task_curation.difficulty.ledger_worker")
    capabilities = pytest.importorskip("harbor.host_capabilities").HostCapabilities()
    path, sample, _ = native
    record = deviation(native)
    task = PreparedTask(
        task_id="task",
        runtime_sha256="c" * 64,
        path=str(path),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=None,
        resource_deviation=record,
    )
    source = SimpleNamespace(sample=SimpleNamespace(source_id=sample["source_id"], model_dump=lambda **_: sample))

    def fail(*args, **kwargs):
        raise RuntimeError("Provider construction failed")

    monkeypatch.setattr(worker, "run_task", fail)
    ledger = str(tmp_path / "ledger")
    output = StoragePath(str(tmp_path / "output"))
    with pytest.raises(RuntimeError, match="Provider construction failed"):
        worker.execute_task(
            task,
            source,
            SimpleNamespace(solver=Solver.SMALL, solver_protocol_id=PROTOCOL_ID),
            "relay",
            tmp_path / "local",
            output,
            None,
            ledger,
            SimpleNamespace(start=lambda _: None),
            {},
            capabilities,
        )
    rows = read_rows(ledger)
    assert len(rows) == 1 and rows[0].outcome.value == "observation"
    assert rows[0].resource_deviation == record
    assert json.loads(StoragePath(rows[0].evidence_uri).read_text()) == asdict(record)
