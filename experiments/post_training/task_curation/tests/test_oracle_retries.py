# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Approved fixed-backend phases preserve prior attempts and exact task inputs."""

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tomlkit
from finestore.rl.curation import ContentEvidenceReference
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath
from shellbox.backends.local import cgroup
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.oracle_filter import (
    BackendFixIdentity,
    LocalExecutionHostTransition,
    OracleExclusionReason,
    OracleFilterReceipt,
    ResourceCalibrationReceipt,
)
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation import oracle_filter
from experiments.post_training.task_curation.local_host_admission import (
    HostSlotRecord,
    LocalHostAdmission,
    LocalHostAdmissionConfig,
    SlotStatus,
)
from experiments.post_training.task_curation.oracle_filter import (
    backend_code_files,
    calibrate_cohort,
    collect_trials,
    declared_native_requirements,
    filter_cohort,
    harbor_inspection,
    merge_retry_maps,
    native_requirements,
    native_verifier_profile,
    retry_cohort,
    row_runtime,
    row_task,
    source_runtime_binding,
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
    outcome = getattr(request, "param", "pass")
    row = prebuilt_task("task")
    envelope = json.loads(row["lowered_task_spec"])
    task = TaskSpec.model_validate(envelope["task"])
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={"oracle": (inline_resource("solution/solve.sh", b"true\n"),)}
            ),
            **(
                {"execution_requirements": None}
                if outcome in {"grown", "retry-grown", "role-resources", "role-verifier-oom", "role-recovery"}
                else {}
            ),
        }
    )
    if outcome in {"role-resources", "role-verifier-oom", "role-recovery"}:
        envelope["runtime"] = (
            {"task_machine": {"cpus": 2}, "verifier_machine": {"memory_mb": 4096}}
            if outcome == "role-verifier-oom"
            else {
                role: {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}
                for role in ("task_machine", "verifier_machine")
            }
        )
    if outcome == "role-recovery":
        task = task.model_copy(
            update={
                "resources": task.resources.model_copy(
                    update={
                        "verifier": (
                            *task.resources.verifier,
                            inline_resource("task.toml", b"[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=720\n"),
                        )
                    }
                )
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
        "[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=720\n"
        if outcome in {"grown", "retry-grown", "role-resources", "role-verifier-oom", "role-recovery"}
        else "[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n"
        "[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\n"
    )
    if outcome in {"role-resources", "role-verifier-oom", "role-recovery"}:
        (native / "task-spec.json").write_text(task.model_dump_json())
        document = tomlkit.parse((native / "task.toml").read_text())
        document["verifier"]["environment_mode"] = "separate"
        if outcome == "role-recovery":
            document["environment"] = {"cpus": 1, "memory_mb": 2048, "storage_mb": 2048}
        (native / "task.toml").write_text(tomlkit.dumps(document))
    mapping = tmp_path / "original-map.json"
    write_task_map(cohort, native.parent, mapping)
    production_run = subprocess.run
    calls = []

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
            current = (
                "pass"
                if outcome in {"legacy-unknown", "grown", "retry-grown", "host", "role-resources", "role-recovery"}
                else outcome
            )
            if outcome == "retry-grown" and len(calls) == 5:
                current = "oom"
            if outcome == "role-verifier-oom":
                current = "oom"
        else:
            current = "unknown" if outcome == "legacy-unknown" else "infra"
            if outcome in {"grown", "retry-grown"} and len(calls) == 1:
                current = "oom"
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
                    "exception_type": (
                        {"infra": "ConnectionError", "oom": "OutOfMemoryError", "timeout": "AgentTimeoutError"}.get(
                            current, "RuntimeError"
                        )
                    ),
                    "exception_message": "Provider connection failed",
                    "exception_traceback": "",
                    "occurred_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
                }
                if current in {"infra", "unknown", "oom", "timeout"}
                else None
            ),
        }
        (root / "result.json").write_text(json.dumps(result))
        (root / "config.json").write_text(json.dumps(result["config"]))
        if outcome == "role-verifier-oom" and "retry-run" in package.parts:
            (root / "verifier/resource-usage.json").write_text(json.dumps({"oom_kills": 1}))
        if current not in {"unknown", "timeout"}:
            (root / "verifier/verdict.json").write_text(
                json.dumps(
                    {
                        "status": "infra_error" if current in {"infra", "oom"} else "scored",
                        "reward": int(current == "pass"),
                    }
                )
            )
        if current not in {"infra", "unknown", "oom", "timeout"}:
            (root / "agent/exit-code.txt").write_text("0")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    initial = tmp_path / "initial"
    environment = (
        {
            "cgroup_parent": "/dead/worker.scope",
            "disk_root": "/dead/disks",
            "registry_authfile": "/dead/auth.json",
            "image_cache": "/dead/cache",
            "signature_policy": "/immutable/policy.json",
        }
        if outcome == "host"
        else {}
    )
    if outcome == "role-recovery":
        for index in range(3):
            attempt = initial / f"attempt-{index:02d}"
            attempt.mkdir(parents=True)
            config_path = attempt / "trial.json"
            config_path.write_text(
                json.dumps(
                    {
                        "task": {"path": str(native)},
                        "agent": {"name": "oracle"},
                        "environment": {"type": "local", "kwargs": {}},
                        "trials_dir": str(attempt / "jobs"),
                        "trial_name": "oracle",
                    }
                )
            )
            execute(["harbor", str(config_path)])
        attempts = tuple(item for group in collect_trials(initial).values() for item in group)
        old = json.loads(mapping.read_text())
        old["task"]["source_task_checksum"] = old["task"]["task_checksum"]
        old["task"]["calibration"] = ResourceCalibrationReceipt(
            declared_requirements=None,
            preserved_requirements=declared_native_requirements(native),
            verifier=native_verifier_profile(native),
            oracle_kind="original",
            attempts=attempts,
        ).model_dump(mode="json")
        (initial / "task-map.json").write_text(json.dumps(old))
    else:
        calibrate_cohort(cohort, mapping, initial, "local", environment, 1)
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
    ("retry_case", "expected_passed", "expected_attempts", "expected_reason"),
    [
        ("pass", True, 1, None),
        ("infra", False, 3, OracleExclusionReason.INFRASTRUCTURE),
        ("zero", False, 1, OracleExclusionReason.ORACLE_FAILED),
        ("timeout", False, 1, OracleExclusionReason.AGENT_TIMEOUT),
        ("unknown", False, 3, OracleExclusionReason.INFRASTRUCTURE),
        ("legacy-unknown", True, 1, None),
    ],
    indirect=["retry_case"],
)
def test_fixed_backend_phase_retains_parent_and_classifies_its_actual_final_source_outcome(
    tmp_path, retry_case, expected_passed, expected_attempts, expected_reason
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
    assert entry.exclusion_reason is expected_reason
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


def test_retry_capacity_rejection_preserves_parent_without_dispatching_trial(tmp_path, retry_case):
    case = retry_case
    parent_bytes = case.parent_map.read_bytes()
    native_bytes = (case.native / "task.toml").read_bytes()
    parent_attempt_count = len(case.calls)
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen)

    def reject_capacity(package: Path, shape: ExecutionRequirements) -> None:
        assert native_requirements(package) == shape
        raise RuntimeError("snapshot capacity unavailable")

    destination = tmp_path / "retry-run"
    with pytest.raises(RuntimeError, match="snapshot capacity unavailable"):
        retry_cohort(frozen / "plan.json", destination, 1, prepare_attempt=reject_capacity)
    assert case.parent_map.read_bytes() == parent_bytes
    assert (case.native / "task.toml").read_bytes() == native_bytes
    assert len(case.calls) == parent_attempt_count
    assert not list(destination.rglob("result.json"))
    assert not (destination / "task-map.json").exists()


def test_retry_requires_approval_for_the_actual_parent_result(tmp_path, retry_case):
    case = retry_case
    approvals = json.loads(case.approvals.read_text())
    approvals["task"]["parent_result_sha256"] = "another-result"
    case.approvals.write_text(json.dumps(approvals))
    with pytest.raises(ValueError, match="another parent attempt"):
        write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, tmp_path / "plan")
    assert len(case.calls) == 3


def test_retry_holds_shared_host_slot_during_the_actual_harbor_subprocess(tmp_path, monkeypatch, retry_case):
    case = retry_case

    def storage(path):
        return conditional_object(str(tmp_path / "controls" / path.removeprefix("s3://controls/")))

    admission = LocalHostAdmission(
        LocalHostAdmissionConfig(control_root="s3://controls/admission", max_trials_per_host=1),
        UUID(int=1),
        "node-1",
        "worker-1",
        storage,
    )
    runner = subprocess.run

    def process_boundary(args, **kwargs):
        if "harbor" in args:
            control = storage(admission.control_path("slot-0000.json")).read()
            assert control is not None
            record = HostSlotRecord.model_validate_json(control.data)
            assert record.status is SlotStatus.HELD
            assert (record.worker_id, record.task_id) == ("worker-1", "task")
        return runner(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", process_boundary)
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen)
    destination = tmp_path / "retry-run"
    retry_cohort(frozen / "plan.json", destination, 1, host_admission=admission)
    observations = list(destination.rglob("host-admission.json"))
    assert len(observations) == 1
    assert json.loads(observations[0].read_text())["node_id"] == "node-1"
    next_claim = admission.try_claim("next-task")
    assert next_claim is not None
    next_claim.release()


@pytest.mark.parametrize(
    ("retry_case", "expected_retry_memory"), [("grown", [4096]), ("retry-grown", [4096, 8192])], indirect=["retry_case"]
)
def test_retry_resumes_grown_allocation_and_preserves_partial_source_limits(tmp_path, retry_case, expected_retry_memory):
    case = retry_case
    original = case.parent_map.read_bytes()
    native_bytes = (case.native / "task.toml").read_bytes()
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen)
    destination = tmp_path / "retry-run"
    retry_cohort(frozen / "plan.json", destination, 1)
    combined = tmp_path / "combined-map.json"
    merge_retry_maps(case.parent_map, [destination / "task-map.json"], combined)
    receipt = filter_cohort(case.cohort, combined, tmp_path, tmp_path / "filtered/tasks.parquet")
    entry = receipt.tasks["task"]
    parent = entry.infrastructure_retries[0].parent_calibration
    assert [attempt.execution_requirements.memory_mb for attempt in parent.attempts] == [2048, 4096, 4096, 4096]
    assert entry.calibration.initial_requirements == parent.execution_requirements
    assert [attempt.execution_requirements.memory_mb for attempt in entry.calibration.attempts] == expected_retry_memory
    assert entry.calibration.declared_requirements is None
    assert entry.calibration.preserved_requirements == {"agent_timeout_sec": 900, "verifier_timeout_sec": 720}
    assert entry.passed
    assert pq.read_table(tmp_path / "filtered/tasks.parquet").num_rows == 1
    assert case.parent_map.read_bytes() == original
    assert (case.native / "task.toml").read_bytes() == native_bytes
    changed = json.loads((destination / "task-map.json").read_text())
    changed["task"]["source_task_checksum"] = "changed-source"
    changed_path = tmp_path / "changed-source-map.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source task/environment checksum"):
        merge_retry_maps(case.parent_map, [changed_path], tmp_path / "invalid-combined.json")


@pytest.mark.parametrize("retry_case", ["host"], indirect=True)
def test_retry_replaces_dead_host_paths_only_inside_the_current_worker_scope(tmp_path, monkeypatch, retry_case):
    case = retry_case
    hierarchy = tmp_path / "cgroups"
    own = hierarchy / "cri-containerd-new.scope"
    driver = own / "atlas-driver"
    driver.mkdir(parents=True)
    (hierarchy / "cgroup.controllers").write_text("cpu memory")
    for scope in (own, driver):
        for name, value in {"cgroup.controllers": "cpu memory", "cgroup.procs": "", "cgroup.kill": ""}.items():
            (scope / name).write_text(value)
    real_read_text = Path.read_text

    def read_kernel(path, *args, **kwargs):
        if path == Path("/proc/self/cgroup"):
            return "0::/cri-containerd-new.scope/atlas-driver\n"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_kernel)
    monkeypatch.setattr(cgroup, "CGROUP_ROOT", hierarchy)
    proof = tmp_path / "worker-scope.json"
    proof.write_text(json.dumps({"parent": str(own), "driver": str(driver)}))
    host = LocalExecutionHostTransition(
        parent_kwargs={
            "cgroup_parent": "/dead/worker.scope",
            "disk_root": "/dead/disks",
            "registry_authfile": "/dead/auth.json",
            "image_cache": "/dead/cache",
        },
        current_kwargs={
            "cgroup_parent": str(own),
            "disk_root": str(tmp_path / "new-disks"),
            "registry_authfile": str(tmp_path / "existing-auth.json"),
            "image_cache": str(tmp_path / "new-cache"),
        },
        scope_evidence=(ContentEvidenceReference(uri=str(proof), sha256=file_sha256(StoragePath(str(proof)))),),
    )
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen, host)
    destination = tmp_path / "retry-run"
    retry_cohort(frozen / "plan.json", destination, 1)
    combined = tmp_path / "combined-map.json"
    merge_retry_maps(case.parent_map, [destination / "task-map.json"], combined)
    receipt = filter_cohort(case.cohort, combined, tmp_path, tmp_path / "filtered/tasks.parquet")
    entry = receipt.tasks["task"]
    assert entry.passed
    assert entry.infrastructure_retries[0].local_host == host
    assert entry.calibration.attempts[0].result["config"]["environment"]["kwargs"] == {
        **host.current_kwargs,
        "signature_policy": "/immutable/policy.json",
    }
    count = len(case.calls)
    foreign = host.model_copy(update={"current_kwargs": {**host.current_kwargs, "cgroup_parent": "/dead/worker.scope"}})
    foreign_plan = tmp_path / "foreign-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, foreign_plan, foreign)
    with pytest.raises(ValueError, match="outside the current worker"):
        retry_cohort(foreign_plan / "plan.json", tmp_path / "foreign-retry-run", 1)
    assert len(case.calls) == count
    proof.write_text("changed scope evidence")
    with pytest.raises(ValueError, match="scope evidence changed"):
        retry_cohort(frozen / "plan.json", tmp_path / "another-retry-run", 1)
    assert len(case.calls) == count


@pytest.mark.parametrize("retry_case", ["infra"], indirect=True)
def test_retry_history_from_archived_field_set_round_trips_across_another_phase(tmp_path, retry_case):
    case = retry_case
    first_plan = tmp_path / "first-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, first_plan)
    first = tmp_path / "first-retry-run"
    retry_cohort(first_plan / "plan.json", first, 1)
    first_map = first / "task-map.json"
    document = json.loads(first_map.read_text())
    document["task"]["calibration"].pop("initial_requirements")
    first_map.write_text(json.dumps(document))
    approvals = json.loads(case.approvals.read_text())
    approvals["task"]["parent_result_sha256"] = document["task"]["calibration"]["attempts"][-1]["result_sha256"]
    second_approval = tmp_path / "second-approval.json"
    second_approval.write_text(json.dumps(approvals))
    second_plan = tmp_path / "second-plan"
    write_retry_plan(case.cohort, first_map, second_approval, case.backend_fix, second_plan)
    second = tmp_path / "second-retry-run"
    retry_cohort(second_plan / "plan.json", second, 1)
    combined = tmp_path / "combined-map.json"
    merge_retry_maps(case.parent_map, [first_map, second / "task-map.json"], combined)
    receipt = filter_cohort(case.cohort, combined, tmp_path, tmp_path / "filtered/tasks.parquet")
    reloaded = OracleFilterReceipt.model_validate_json((tmp_path / "filtered/oracle-filter.json").read_text())
    assert reloaded == receipt
    assert len(reloaded.tasks["task"].infrastructure_retries) == 2
    assert not reloaded.tasks["task"].passed


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


@pytest.mark.parametrize(
    ("retry_case", "expected_passed"), [("role-resources", True), ("role-verifier-oom", False)], indirect=["retry_case"]
)
def test_source_roles_fix_actual_calibration_and_admitted_actor_budget(tmp_path, retry_case, expected_passed):
    case = retry_case
    frozen = tmp_path / "retry-plan"
    write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen)
    destination = tmp_path / "retry-run"
    retry_cohort(frozen / "plan.json", destination, 1)
    combined = tmp_path / "combined-map.json"
    merge_retry_maps(case.parent_map, [destination / "task-map.json"], combined)
    receipt = filter_cohort(case.cohort, combined, tmp_path, tmp_path / "filtered/tasks.parquet")
    entry = receipt.tasks["task"]
    initial = entry.infrastructure_retries[0].parent_calibration
    assert len(entry.calibration.attempts) == 1
    assert entry.passed is expected_passed
    if expected_passed:
        assert entry.exclusion_reason is None
        assert all(attempt.execution_requirements.memory_mb == 4096 for attempt in initial.attempts)
        admitted = row_task(pq.read_table(tmp_path / "filtered/tasks.parquet").to_pylist()[0])
        assert admitted is not None
        assert admitted.execution_requirements == ExecutionRequirements(
            cpus=2, memory_mb=4096, storage_mb=10240, agent_timeout_sec=300, verifier_timeout_sec=60
        )
        source_runtime_binding(admitted, row_runtime(pq.read_table(case.cohort).to_pylist()[0]))
    else:
        assert entry.exclusion_reason is OracleExclusionReason.OOM
        assert not entry.passed
        assert entry.calibration.execution_requirements.memory_mb == 2048
        assert entry.calibration.verifier.fixed == {"memory_mb": 4096}
        assert pq.read_table(tmp_path / "filtered/tasks.parquet").num_rows == 0


@pytest.mark.parametrize("retry_case", ["role-recovery"], indirect=True)
def test_approved_runtime_recovery_retains_unqualified_old_allocations_and_runs_exact_source_budget(
    tmp_path, retry_case
):
    case = retry_case
    original = case.parent_map.read_bytes()
    row = pq.read_table(case.cohort).to_pylist()[0]
    task = row_task(row)
    assert task is not None
    binding = source_runtime_binding(task, row_runtime(row))
    assert binding is not None
    count = len(case.calls)
    with pytest.raises(ValueError, match="approved recovery transition"):
        write_retry_plan(case.cohort, case.parent_map, case.approvals, case.backend_fix, tmp_path / "unapproved-plan")
    assert len(case.calls) == count
    frozen = tmp_path / "retry-plan"
    write_retry_plan(
        case.cohort, case.parent_map, case.approvals, case.backend_fix, frozen, source_runtime_resource_recovery=binding
    )
    destination = tmp_path / "retry-run"
    retry_cohort(frozen / "plan.json", destination, 1)
    combined = tmp_path / "combined-map.json"
    merge_retry_maps(case.parent_map, [destination / "task-map.json"], combined)
    receipt = filter_cohort(case.cohort, combined, tmp_path, tmp_path / "filtered/tasks.parquet")
    entry = receipt.tasks["task"]
    prior = entry.infrastructure_retries[0]
    assert prior.source_runtime_resource_recovery == binding
    assert all(attempt.execution_requirements.cpus == 1 for attempt in prior.parent_calibration.attempts)
    assert all(attempt.execution_requirements.memory_mb == 2048 for attempt in prior.parent_calibration.attempts)
    assert entry.calibration.execution_requirements == ExecutionRequirements(
        cpus=2, memory_mb=4096, storage_mb=10240, agent_timeout_sec=900, verifier_timeout_sec=720
    )
    assert entry.calibration.verifier.fixed == {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}
    assert entry.passed and pq.read_table(tmp_path / "filtered/tasks.parquet").num_rows == 1
    assert case.parent_map.read_bytes() == original
    assert OracleFilterReceipt.model_validate_json((tmp_path / "filtered/oracle-filter.json").read_text()) == receipt
