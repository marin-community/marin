# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Approved fixed-backend phases preserve prior attempts and exact task inputs."""

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import BackendFixIdentity, OracleExclusionReason
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation import oracle_filter
from experiments.post_training.task_curation.oracle_filter import (
    backend_code_files,
    calibrate_cohort,
    filter_cohort,
    harbor_inspection,
    merge_retry_maps,
    native_requirements,
    retry_cohort,
    write_retry_plan,
    write_task_map,
)

from .prebuilt_fixtures import prebuilt_task


@dataclass
class RetryCase:
    cohort: Path
    parent_map: Path
    approvals: Path
    backend_fix: Path
    backend_file: Path
    native: Path
    calls: list[str]


@pytest.fixture
def retry_case(tmp_path, monkeypatch, request):
    row = prebuilt_task("task")
    envelope = json.loads(row["lowered_task_spec"])
    task = TaskSpec.model_validate(envelope["task"])
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(update={"oracle": (inline_resource("solution/solve.sh", b"true\n"),)})
        }
    )
    envelope["task"] = task.model_dump(mode="json")
    row["lowered_task_spec"] = json.dumps(envelope)
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist([row]), cohort)
    native = tmp_path / "native/task"
    for name in ("environment", "tests", "solution"):
        (native / name).mkdir(parents=True)
    (native / "instruction.md").write_text("Solve the task.")
    (native / "tests/test.sh").write_text("true\n")
    (native / "solution/solve.sh").write_text("true\n")
    (native / "task.toml").write_text(
        "[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n"
        "[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\n"
    )
    mapping = tmp_path / "original-map.json"
    write_task_map(cohort, native.parent, mapping)
    production_run = subprocess.run
    calls = []
    outcome = getattr(request, "param", "pass")

    def execute(args, **kwargs):
        if "harbor" not in args:
            return production_run(args, **kwargs)
        config = json.loads(Path(args[-1]).read_text())
        package = Path(config["task"]["path"])
        calls.append(str(package))
        shape = native_requirements(package)
        config["resolved_timeouts"] = {
            "agent": shape.agent_timeout_sec,
            "verifier": shape.verifier_timeout_sec,
            "agent_setup": 360,
            "environment_build": 600,
        }
        if "retry-run" in package.parts:
            current = "pass" if outcome == "legacy-unknown" else outcome
        else:
            current = "unknown" if outcome == "legacy-unknown" else "infra"
        root = Path(config["trials_dir"]) / "oracle"
        (root / "agent").mkdir(parents=True)
        (root / "verifier").mkdir()
        checksum = harbor_inspection("tasks", [str(package)])[str(package)]["task_checksum"]
        result = {
            "id": str(uuid4()),
            "task_name": "task",
            "trial_name": "oracle",
            "trial_uri": root.as_uri(),
            "task_id": {"path": str(package)},
            "task_checksum": checksum,
            "config": config,
            "agent_info": {"name": "oracle", "version": "1"},
            "verifier_result": {"rewards": {"reward": int(current == "pass")}},
            "finished_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
            "completion": {"trajectory_present": False},
            "exception_info": (
                {
                    "exception_type": "ConnectionError" if current == "infra" else "RuntimeError",
                    "exception_message": "Provider connection failed",
                    "exception_traceback": "",
                    "occurred_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
                }
                if current in {"infra", "unknown"}
                else None
            ),
        }
        (root / "result.json").write_text(json.dumps(result))
        (root / "config.json").write_text(json.dumps(result["config"]))
        if current != "unknown":
            (root / "verifier/verdict.json").write_text(
                json.dumps(
                    {
                        "status": "infra_error" if current == "infra" else "scored",
                        "reward": int(current == "pass"),
                    }
                )
            )
        if current not in {"infra", "unknown"}:
            (root / "agent/exit-code.txt").write_text("0")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    initial = tmp_path / "initial"
    calibrate_cohort(cohort, mapping, initial, "local", {}, 1)
    parent_map = initial / "task-map.json"
    document = json.loads(parent_map.read_text())
    for attempt in document["task"]["calibration"]["attempts"]:
        attempt.pop("infrastructure_retry", None)
    parent_map.write_text(json.dumps(document))
    parent = document["task"]
    approvals = tmp_path / "approvals.json"
    approvals.write_text(
        json.dumps(
            {
                "task": {
                    "classification": "our_infrastructure",
                    "parent_result_sha256": parent["calibration"]["attempts"][-1]["result_sha256"],
                    "failure_signature": "provider-connection",
                    "decision_ref": "coordinator:test-task-partition",
                }
            }
        )
    )
    code_root = tmp_path / "code"
    backend_file = code_root / "lib/shellbox/src/fixed_backend.py"
    backend_file.parent.mkdir(parents=True)
    backend_file.write_text("FIXED = True\n")
    (code_root / "pyproject.toml").write_text("")
    monkeypatch.setattr(oracle_filter, "PROJECT_ROOT", code_root)
    backend_fix = tmp_path / "backend-fix.json"
    backend_fix.write_text(
        BackendFixIdentity(revision="a" * 40, files_sha256=backend_code_files(code_root)).model_dump_json()
    )
    return RetryCase(cohort, parent_map, approvals, backend_fix, backend_file, native, calls)


@pytest.mark.parametrize(
    ("retry_case", "expected_passed", "expected_attempts"),
    [
        ("pass", True, 1),
        ("infra", False, 3),
        ("zero", False, 1),
        ("unknown", False, 3),
        ("legacy-unknown", True, 1),
    ],
    indirect=["retry_case"],
)
def test_fixed_backend_phase_preserves_parent_attempts_and_never_turns_failed_retry_into_source_negative(
    tmp_path, retry_case, expected_passed, expected_attempts
):
    case = retry_case
    parent_bytes = case.parent_map.read_bytes()
    parent_attempt_count = len(case.calls)
    native_bytes = (case.native / "task.toml").read_bytes()
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen)
    destination = tmp_path / "retry-run"
    retry_cohort(frozen / "plan.json", destination, 1)
    combined = tmp_path / "combined-map.json"
    merge_retry_maps(case.parent_map, [destination / "task-map.json"], combined)
    receipt = filter_cohort(case.cohort, combined, tmp_path, tmp_path / "filtered/tasks.parquet")
    entry = receipt.tasks["task"]
    assert len(entry.infrastructure_retries) == 1
    transition = entry.infrastructure_retries[0]
    assert len(transition.parent_calibration.attempts) == parent_attempt_count
    assert transition.backend_fix.revision == "a" * 40
    assert case.parent_map.read_bytes() == parent_bytes
    assert (case.native / "task.toml").read_bytes() == native_bytes
    assert all(Path(attempt.path).exists() for attempt in transition.parent_calibration.attempts)
    assert entry.passed is expected_passed
    assert len(entry.calibration.attempts) == expected_attempts
    assert entry.exclusion_reason is (None if expected_passed else OracleExclusionReason.INFRASTRUCTURE)
    assert pq.read_table(tmp_path / "filtered/tasks.parquet").num_rows == int(expected_passed)
    assert len(case.calls) == parent_attempt_count + len(entry.calibration.attempts)


@pytest.mark.parametrize("changed", ["task", "native", "code", "parent"])
def test_changed_retry_inputs_do_not_dispatch_any_new_provider_attempt(tmp_path, retry_case, changed):
    case = retry_case
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen)
    plan = json.loads((frozen / "plan.json").read_text())
    if changed == "task":
        path = Path(plan["cohort_path"])
        path.write_bytes(path.read_bytes() + b"changed")
    elif changed == "native":
        (case.native / "tests/test.sh").write_text("echo changed\n")
    elif changed == "code":
        case.backend_file.write_text("FIXED = False\n")
    else:
        case.parent_map.write_text(case.parent_map.read_text() + "\n")
    with pytest.raises(ValueError, match=r"changed|differ"):
        retry_cohort(frozen / "plan.json", tmp_path / "retry-run", 1)
    assert len(case.calls) == 3


def test_retry_requires_approval_for_the_actual_parent_result(tmp_path, retry_case):
    case = retry_case
    approvals = json.loads(case.approvals.read_text())
    approvals["task"]["parent_result_sha256"] = "another-result"
    case.approvals.write_text(json.dumps(approvals))
    with pytest.raises(ValueError, match="another parent attempt"):
        write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, tmp_path / "plan")
    assert len(case.calls) == 3


@pytest.mark.parametrize("changed", ["parent-file", "reused-attempts", "backend", "phase-context"])
def test_recollection_checks_parent_artifacts_and_distinct_phase_attempts(tmp_path, retry_case, changed):
    case = retry_case
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen)
    destination = tmp_path / "retry-run"
    retry_cohort(frozen / "plan.json", destination, 1)
    if changed == "parent-file":
        case.parent_map.write_text(case.parent_map.read_text() + "\n")
    elif changed == "reused-attempts":
        current = json.loads((destination / "task-map.json").read_text())
        current["task"]["calibration"] = json.loads(case.parent_map.read_text())["task"]["calibration"]
        (destination / "task-map.json").write_text(json.dumps(current))
    elif changed == "backend":
        current = json.loads((destination / "task-map.json").read_text())
        attempt = current["task"]["calibration"]["attempts"][0]
        attempt["result"]["config"]["environment"]["type"] = "daytona"
        attempt["result_sha256"] = canonical_sha256(attempt["result"])
        Path(attempt["path"]).write_text(json.dumps(attempt["result"]))
        (destination / "task-map.json").write_text(json.dumps(current))
    else:
        current = json.loads((destination / "task-map.json").read_text())
        current["task"]["calibration"]["attempts"][0]["infrastructure_retry"]["backend_fix_revision"] = "b" * 40
        (destination / "task-map.json").write_text(json.dumps(current))
    combined = tmp_path / "combined-map.json"
    merge_retry_maps(case.parent_map, [destination / "task-map.json"], combined)
    with pytest.raises(
        ValueError, match=r"retry input changed|repeats a prior attempt|backend configuration|attempt context differs"
    ):
        filter_cohort(case.cohort, combined, tmp_path, tmp_path / "filtered/tasks.parquet")
