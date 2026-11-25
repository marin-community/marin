# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repaired MiMo NoP controls: fail-closed rewards over the LOCAL trial path."""

import subprocess
from pathlib import Path

import pytest
from taskcompendium.harbor.adapter import declared_execution_limits
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.sampling import seeded_order
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.mimo import controls as mimo_controls
from experiments.post_training.task_curation.datasets.mimo.controls import (
    TRIAL_STORAGE_MB,
    control_request,
    original_package_checksum,
    run_control,
    seeded_control_tasks,
)
from experiments.post_training.task_curation.datasets.mimo.controls import (
    write_harbor_task as _write,
)


def _repaired_task(task_id: str, memory_mb: int) -> TaskSpec:
    toml = (
        "[agent]\ntimeout_sec = 1800\n[verifier]\ntimeout_sec = 1800\n"
        f"[environment]\ncpus = 4\nmemory_mb = {memory_mb}\n"
    ).encode()
    task = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the repository"),)),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        source=Source(dataset="fixture", revision="fixture", row=task_id, importer_revision="1"),
        environment_requirements=EnvironmentRequirements(
            command_semantics="linux_process", docker_image="actor@sha256:" + "a" * 64
        ),
        grader=ScriptGrader(
            argv=("bash", "tests/test.sh"),
            environment=EnvironmentRequirements(
                command_semantics="linux_process", docker_image="actor@sha256:" + "a" * 64
            ),
        ),
        resources=ResourceGroups(
            verifier=(
                inline_resource("source_archive/tests/test.sh", b"test"),
                inline_resource("source_archive/task.toml", toml),
                inline_resource("source_archive/instruction.md", b"repair"),
            )
        ),
        integrity_contract=RootDeliveryContract(
            dependency_state=SharedDependencyState.AGENT_MUTABLE,
            verifier_files=(DeliveredVerifierFile(name="test.sh", mode=493, sha256="b" * 64),),
        ),
        tags=("harbor", "root-delivery", "prepared-images"),
    )
    return task


@pytest.fixture
def rows(monkeypatch, tmp_path):
    tasks = [_repaired_task(f"fweindel-validated-tasks-mimo-v1-{index:064x}", 4096 + index) for index in range(6)]
    encoded = [task.model_dump_json() for task in tasks]
    monkeypatch.setattr(
        mimo_controls,
        "release_rows",
        lambda workdir: list(zip(tasks, [{"task_machine": {}}] * len(tasks), encoded, strict=True)),
    )
    return list(zip(tasks, [{"task_machine": {}}] * len(tasks), encoded, strict=True))


def test_seeded_selection_follows_the_standard_seed_zero_order(rows):
    selected = seeded_control_tasks(rows, 3)
    expected = sorted(rows, key=lambda row: seeded_order(row[0].id, mimo_controls.SAMPLE_SEED))[:3]
    assert [row[0].id for row in selected] == [row[0].id for row in expected]
    assert seeded_control_tasks(rows, 100)[0][0].id == expected[0][0].id


def test_original_package_materialization_restores_the_archived_layout(rows, tmp_path):
    task = rows[0][0]
    checksum = original_package_checksum(task, tmp_path)
    directory = tmp_path / "original-packages" / canonical_sha256(task.id)
    assert (directory / "task.toml").is_file()
    assert (directory / "instruction.md").is_file()
    assert (directory / "tests" / "test.sh").is_file()
    assert checksum and checksum != trial_layout_checksum(task, tmp_path)


def trial_layout_checksum(task: TaskSpec, tmp_path: Path) -> str:
    package = tmp_path / "packages" / canonical_sha256(task.id)
    requirements = ExecutionRequirements.model_validate(
        {**declared_execution_limits(task), "storage_mb": TRIAL_STORAGE_MB}
    )
    _write(task, package, requirements)
    return mimo_controls.trial_package_checksum(package)


def test_control_request_carries_the_nop_agent_and_the_local_path(rows, tmp_path):
    task, runtime, _encoded = rows[0]
    requirements = ExecutionRequirements.model_validate(
        {**declared_execution_limits(task), "storage_mb": TRIAL_STORAGE_MB}
    )
    package = tmp_path / "packages" / canonical_sha256(task.id)
    _write(task, package, requirements)
    request = control_request(
        task,
        runtime,
        package,
        requirements,
        task_checksum="c" * 64,
        source_task_checksum="d" * 64,
        output_root=tmp_path / "control",
        environment_kwargs={"disk_root": str(tmp_path / "disk")},
    )
    assert request["trial"]["agent"] == {"name": "nop"}
    assert request["trial"]["environment"] == {"type": "local", "kwargs": {"disk_root": str(tmp_path / "disk")}}
    binding = request["trial"]["task"]["source_runtime"]
    assert binding["runtime"] == runtime
    assert binding["runtime_sha256"] == canonical_sha256(runtime)
    assert binding["task_sha256"] == task_spec_sha256(task)
    assert request["source"]["task_checksum"] == "c" * 64
    assert request["source"]["source_task_checksum"] == "d" * 64
    assert request["source"]["execution_requirements"] == requirements.model_dump(mode="json")
    assert request["source"]["verifier_execution_requirements"] == mimo_controls.verifier_shape(
        requirements, runtime
    ).model_dump(mode="json")
    assert request["native_setup"] is None


def test_run_control_refuses_a_reward_that_does_not_fail_closed(rows, tmp_path, monkeypatch):
    task, runtime, _encoded = rows[0]
    requirements = ExecutionRequirements.model_validate(
        {**declared_execution_limits(task), "storage_mb": TRIAL_STORAGE_MB}
    )
    package = tmp_path / "packages" / canonical_sha256(task.id)
    _write(task, package, requirements)
    request = control_request(
        task,
        runtime,
        package,
        requirements,
        "c" * 64,
        "d" * 64,
        tmp_path / "control",
        {"disk_root": str(tmp_path / "disk")},
    )

    class FakeTrial:
        verdict = None

        def __init__(self, result):
            self.result = result

    class FakeReceipt:
        score = 1.0
        annotation = None
        verifier_exit_code = None
        trial = FakeTrial({})

        def model_dump(self, mode=None):
            return {"score": 1.0}

    monkeypatch.setattr(
        mimo_controls.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr=""),
    )
    monkeypatch.setattr(mimo_controls, "negative_control_from_archive", lambda *args, **kwargs: FakeReceipt())
    control_root = tmp_path / "control"
    control_root.mkdir()
    (control_root / "negative-control.json").write_text("{}")
    with pytest.raises(ValueError, match="did not fail closed"):
        run_control(request, control_root, str(tmp_path / "archives"))


def test_run_controls_refuses_any_failed_task(rows, tmp_path, monkeypatch):
    def broken_drive(*args, **kwargs):
        raise RuntimeError("sandbox startup failed")

    monkeypatch.setattr(mimo_controls, "release_rows", lambda workdir: rows)
    monkeypatch.setattr(mimo_controls, "seeded_control_tasks", lambda all_rows, sample: all_rows[:2])
    monkeypatch.setattr(mimo_controls, "run_control", broken_drive)
    monkeypatch.setattr(
        mimo_controls,
        "original_package_checksum",
        lambda task, workdir: "d" * 64,
    )
    monkeypatch.setattr(
        mimo_controls,
        "declared_execution_limits",
        lambda task: {"cpus": 4, "memory_mb": 4096, "agent_timeout_sec": 1800, "verifier_timeout_sec": 1800},
    )
    package_root = tmp_path / "packages"
    package_root.mkdir()
    for task, _runtime, _encoded in rows[:2]:
        (package_root / canonical_sha256(task.id)).mkdir()
    monkeypatch.setattr(mimo_controls, "write_harbor_task", lambda task, package, requirements: None)
    monkeypatch.setattr(mimo_controls, "trial_package_checksum", lambda package: "c" * 64)

    with pytest.raises(ValueError, match="NoP controls failed for 2 tasks"):
        mimo_controls.run_controls(str(tmp_path / "controls"), tmp_path, 2, 2)


def test_run_control_retries_infrastructure_failures_but_not_defects(rows, tmp_path, monkeypatch):
    """A trial that never reached its verifier is retryable; a credited nop is a refusal."""
    task, runtime, _encoded = rows[0]
    requirements = ExecutionRequirements.model_validate(
        {**declared_execution_limits(task), "storage_mb": TRIAL_STORAGE_MB}
    )
    package = tmp_path / "packages" / canonical_sha256(task.id)
    _write(task, package, requirements)
    request = control_request(
        task,
        runtime,
        package,
        requirements,
        "c" * 64,
        "d" * 64,
        tmp_path / "control",
        {"disk_root": str(tmp_path / "disk")},
    )

    class FakeTrial:
        verdict = None

        def __init__(self, result):
            self.result = result

    def receipt_with(score, exception=None):
        class FakeReceipt:
            annotation = None
            verifier_exit_code = None
            trial = FakeTrial(
                {} if exception is None else {"exception_info": {"exception_type": exception, "exception_message": "x"}}
            )

            def model_dump(self, mode=None):
                return {"score": score}

        receipt = FakeReceipt()
        receipt.score = score
        return receipt

    calls = []

    def run(*args, **kwargs):
        calls.append(1)
        return subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr="")

    monkeypatch.setattr(mimo_controls.subprocess, "run", run)
    monkeypatch.setattr(
        mimo_controls,
        "negative_control_from_archive",
        lambda *args, **kwargs: receipt_with(None, "EnvironmentStartTimeoutError"),
    )
    control_root = tmp_path / "control-retry"
    control_root.mkdir()
    (control_root / "negative-control.json").write_text("{}")
    with pytest.raises(mimo_controls.ControlInfrastructureError, match="infrastructure failure"):
        run_control(request, control_root, str(tmp_path / "archives"))

    credited = tmp_path / "control-credited"
    credited.mkdir()
    (credited / "negative-control.json").write_text("{}")
    monkeypatch.setattr(mimo_controls, "negative_control_from_archive", lambda *args, **kwargs: receipt_with(1.0))
    with pytest.raises(ValueError, match="did not fail closed"):
        run_control(request, credited, str(tmp_path / "archives"))

    def drive_returns_after_retry(*args, **kwargs):
        assert mimo_controls.RETRIES >= 2
        return {"task_id": request["source"]["task_id"], "receipt": {}, "score": 0.0, "attempts": 2}

    monkeypatch.setattr(mimo_controls, "run_control", drive_returns_after_retry)
    monkeypatch.setattr(mimo_controls, "release_rows", lambda workdir: rows)
    monkeypatch.setattr(mimo_controls, "seeded_control_tasks", lambda all_rows, sample: all_rows[:1])
    summary = mimo_controls.run_controls(str(tmp_path / "controls-out"), tmp_path, 1, 1)
    # The first attempt succeeded, so the drive records one attempt for the task.
    assert summary["attempts"][request["source"]["task_id"]] == 1
