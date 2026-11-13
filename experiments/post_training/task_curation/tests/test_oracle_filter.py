# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full Harbor trial receipts preserve oracle exits and filter exact converted bytes."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.oracle_filter import NoOracleLivenessReceipt, OracleExclusionReason, require_oracle_filter
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.oracle_filter import (
    calibrate_cohort,
    collect_trials,
    filter_cohort,
    harbor_inspection,
    native_requirements,
    write_task_map,
)
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_cohort

from .prebuilt_fixtures import prebuilt_task


def test_full_oracle_filter_excludes_failed_solutions_even_with_reward_one(tmp_path):
    rows = []
    mappings = {}
    for task_id in (
        "pass",
        "failed-solution",
        "missing-exit",
        "infra",
        "source-env",
        "unmatched-env",
        "observed-env",
        "newer-env",
        "invalid-reward",
        "calibration-only",
        "live",
    ):
        row = prebuilt_task(task_id)
        envelope = json.loads(row["lowered_task_spec"])
        task = TaskSpec.model_validate(envelope["task"])
        if task_id not in {"calibration-only", "live"}:
            task = task.model_copy(
                update={
                    "resources": task.resources.model_copy(
                        update={"oracle": (inline_resource("solution/solve.sh", b"true\n"),)}
                    )
                }
            )
        envelope["task"] = task.model_dump(mode="json")
        row["lowered_task_spec"] = json.dumps(envelope)
        rows.append(row)
        native = tmp_path / "native" / task_id
        (native / "environment").mkdir(parents=True)
        (native / "tests").mkdir(parents=True)
        (native / "solution").mkdir()
        (native / "instruction.md").write_text("Run the task.")
        (native / "tests/test.sh").write_text("true\n")
        (native / "solution/solve.sh").write_text("true\n")
        (native / "task.toml").write_text(
            "[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n"
            + (
                '[environment.setup]\ncommand="pip install -q pytest==6.2.5"\ntimeout_sec=300\n'
                if task_id == "source-env"
                else ""
            )
            + "[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\n"
        )
        checksum = harbor_inspection("tasks", [str(native)])[str(native)]["task_checksum"]
        mappings[task_id] = {
            "task_path": str(native),
            "task_name": task_id,
            "task_checksum": checksum,
            "task_sha256": canonical_sha256(task.model_dump(mode="json")),
            "runtime_sha256": canonical_sha256(envelope["runtime"]),
        }
        for attempt in range(
            3 if task_id in {"infra", "source-env", "unmatched-env", "observed-env", "newer-env"} else 1
        ):
            root = tmp_path / "jobs" / f"{task_id}-{attempt}"
            (root / "verifier").mkdir(parents=True)
            (root / "agent").mkdir()
            task_path = str(tmp_path / "native" / task_id)
            result = {
                "id": str(uuid4()),
                "task_name": task_id,
                "trial_name": root.name,
                "trial_uri": root.as_uri(),
                "task_id": {"path": task_path},
                "task_checksum": checksum,
                "config": {
                    "task": {"path": task_path},
                    "agent": {"name": "oracle"},
                    "environment": {"type": "local"},
                    "resolved_timeouts": {"agent": 300, "verifier": 60, "agent_setup": 360, "environment_build": 600},
                    "trials_dir": str(tmp_path / "jobs"),
                    "trial_name": root.name,
                },
                "agent_info": {"name": "oracle", "version": "1"},
                "verifier_result": {"rewards": {"reward": 1}},
                "finished_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
                "completion": {"trajectory_present": False},
                "exception_info": (
                    {
                        "exception_type": "RuntimeError",
                        "exception_message": "backend unavailable",
                        "exception_traceback": "",
                        "occurred_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
                    }
                    if task_id == "infra"
                    else None
                ),
            }
            (root / "result.json").write_text(json.dumps(result))
            verdict = (
                {
                    "status": "infra_error",
                    "reward": 0,
                    "detail": {
                        "error": (
                            "RuntimeError: pytest wrote no json report (exit 1):\n"
                            'File "/opt/taskcompendium-shared/runner/pytest_metadata/plugin.py", line 36\n'
                            "AttributeError: module 'pytest' has no attribute 'StashKey'"
                        )
                    },
                }
                if task_id in {"source-env", "unmatched-env", "observed-env", "newer-env"}
                else {"status": "scored", "reward": 1}
            )
            if task_id == "invalid-reward":
                verdict["reward"] = True
            (root / "verifier/verdict.json").write_text(json.dumps(verdict))
            if task_id in {"observed-env", "newer-env"}:
                (root / "verifier/source-environment.json").write_text(
                    json.dumps(
                        {
                            "pytest_version": "6.2.4" if task_id == "observed-env" else "7.4.0",
                            "pytest_metadata_version": "3.1.1",
                        }
                    )
                )
            if task_id != "missing-exit":
                (root / "agent/exit-code.txt").write_text("7" if task_id == "failed-solution" else "0")
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    task_map = tmp_path / "task-map.json"
    task_map.write_text(json.dumps(mappings))
    live = TaskSpec.model_validate(json.loads(rows[-1]["lowered_task_spec"])["task"])
    proof = NoOracleLivenessReceipt(
        task_id=live.id,
        task_sha256=canonical_sha256(live.model_dump(mode="json")),
        runtime_sha256=canonical_sha256(json.loads(rows[-1]["lowered_task_spec"])["runtime"]),
        submission_kind="empty",
        outcome="negative_verdict",
        exit_code=0,
        reward=0,
        tests_executed=3,
    )
    liveness = tmp_path / "liveness.jsonl"
    liveness.write_text(proof.model_dump_json() + "\n")
    output = tmp_path / "filtered/tasks.parquet"
    conversion_manifest = tmp_path / "conversion.json"
    conversion_manifest.write_text(
        json.dumps(
            {
                "input_rows": 13,
                "converted_rows": 11,
                "within_source_dedup": {"raw_count": 13, "unique_count": 11, "within_source_duplicates": 2},
                "oracle_recipe_identity": "fixture-recipe",
            }
        )
    )
    receipt = filter_cohort(
        cohort, task_map, tmp_path / "jobs", output, conversion_manifest=conversion_manifest, no_oracle_liveness=liveness
    )
    assert [row["path"] for row in pq.read_table(output).to_pylist()] == ["pass", "live"]
    assert receipt.input_rows == 11
    assert receipt.conversion_rejections == 0
    assert receipt.tasks["invalid-reward"].exclusion_reason is OracleExclusionReason.INFRASTRUCTURE
    incompatible = receipt.tasks["source-env"]
    assert incompatible.exclusion_reason is OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER
    assert (
        incompatible.source_environment_incompatibility.source_setup_sha256
        == hashlib.sha256(b"pip install -q pytest==6.2.5").hexdigest()
    )
    assert incompatible.calibration.attempts[-1].verdict["status"] == "infra_error"
    assert not incompatible.passed
    unmatched = receipt.tasks["unmatched-env"]
    assert unmatched.exclusion_reason is OracleExclusionReason.INFRASTRUCTURE
    assert unmatched.source_environment_incompatibility is None
    observed = receipt.tasks["observed-env"]
    assert observed.exclusion_reason is OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER
    assert observed.source_environment_incompatibility.signature == "pytest-before-7/pytest-metadata-stashkey"
    assert observed.source_environment_incompatibility.source_environment_sha256 == canonical_sha256(
        {"pytest_version": "6.2.4", "pytest_metadata_version": "3.1.1"}
    )
    assert receipt.tasks["newer-env"].exclusion_reason is OracleExclusionReason.INFRASTRUCTURE
    calibration_only = receipt.tasks["calibration-only"]
    assert calibration_only.calibration.passed and not calibration_only.passed
    assert calibration_only.exclusion_reason == "no_oracle_liveness_missing_or_failed"
    assert receipt.tasks["live"].trials == ()
    assert require_oracle_filter(live, receipt, json.loads(rows[-1]["lowered_task_spec"])["runtime"])
    assert receipt.tasks["failed-solution"].trials[0].oracle_exit_code == 7
    assert receipt.tasks["failed-solution"].exclusion_reason is OracleExclusionReason.ORACLE_FAILED
    assert not receipt.tasks["failed-solution"].passed
    assert not receipt.tasks["infra"].passed
    assert receipt.tasks["missing-exit"].trials[0].oracle_exit_code is None
    assert receipt.tasks["missing-exit"].exclusion_reason is OracleExclusionReason.INFRASTRUCTURE
    assert not receipt.tasks["missing-exit"].passed

    envelope = json.loads(rows[0]["lowered_task_spec"])
    envelope["task"]["labels"] = {"changed": "after Harbor export"}
    rows[0]["lowered_task_spec"] = json.dumps(envelope)
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    with pytest.raises(ValueError, match="changed after Harbor export"):
        filter_cohort(cohort, task_map, tmp_path / "jobs", tmp_path / "stale/tasks.parquet")

    (tmp_path / "native/pass/instruction.md").write_text("Changed after the trial.")
    with pytest.raises(ValueError):
        collect_trials(tmp_path / "jobs")


@pytest.mark.parametrize(("declared", "native_declared"), [(True, True), (False, False), (False, True)])
@pytest.mark.parametrize("canonical", [False, True])
def test_population_calibration_doubles_missing_limits_but_preserves_swe_declarations(
    tmp_path, monkeypatch, declared, native_declared, canonical
):
    row = prebuilt_task("task")
    envelope = json.loads(row["lowered_task_spec"])
    task = TaskSpec.model_validate(envelope["task"])
    swe = ExecutionRequirements(
        cpus=2, memory_mb=4096, storage_mb=10240, agent_timeout_sec=900, verifier_timeout_sec=600
    )
    task = task.model_copy(
        update={
            "execution_requirements": swe if declared else None,
            "resources": task.resources.model_copy(
                update={"oracle": (inline_resource("solution/solve.sh", b"true\n"),)}
            ),
        }
    )
    envelope["task"] = task.model_dump(mode="json")
    row["lowered_task_spec"] = json.dumps(envelope)
    if canonical:
        serialized = task.model_dump_json()
        row = {"path": task.id, "task_spec": serialized, "task_sha256": hashlib.sha256(serialized.encode()).hexdigest()}
    cohort = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([row]), cohort)
    native = tmp_path / "native" / "task"
    for folder in ("environment", "tests", "solution"):
        (native / folder).mkdir(parents=True)
    (native / "instruction.md").write_text("Solve the task.")
    (native / "tests/test.sh").write_text("true\n")
    (native / "solution/solve.sh").write_text("true\n")
    (native / "task.toml").write_text(
        "[environment]\ncpus=2\nmemory_mb=4096\nstorage_mb=10240\n"
        "[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=600\n"
        if native_declared
        else "[environment]\n"
    )
    native_document = (native / "task.toml").read_bytes()
    mapping = tmp_path / "map.json"
    write_task_map(cohort, native.parent, mapping)
    production_run = subprocess.run
    shapes = []

    def execute(args, **kwargs):
        if "harbor" not in args:
            return production_run(args, **kwargs)
        config = json.loads(Path(args[-1]).read_text())
        package = Path(config["task"]["path"])
        shape = native_requirements(package)
        shapes.append(shape)
        config["resolved_timeouts"] = {
            "agent": shape.agent_timeout_sec,
            "verifier": shape.verifier_timeout_sec,
            "agent_setup": 360,
            "environment_build": 600,
        }
        root = Path(config["trials_dir"]) / "oracle"
        (root / "agent").mkdir(parents=True)
        (root / "verifier").mkdir()
        oom = len(shapes) == 1
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
            "verifier_result": {"rewards": {"reward": 1}},
            "finished_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
            "completion": {"trajectory_present": False},
            "exception_info": None,
        }
        (root / "result.json").write_text(json.dumps(result))
        (root / "agent/exit-code.txt").write_text("137" if oom else "0")
        (root / "agent/resource-usage.json").write_text(json.dumps({"oom_kills": int(oom)}))
        (root / "verifier/verdict.json").write_text(json.dumps({"status": "scored", "reward": 1}))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    run = tmp_path / "calibrated"
    calibrate_cohort(cohort, mapping, run, "local", {}, 1)
    output = tmp_path / "filtered/tasks.parquet"
    receipt = filter_cohort(cohort, run / "task-map.json", run, output)
    entry = receipt.tasks[task.id]
    if native_declared:
        final_native = Path(entry.calibration.attempts[-1].result["config"]["task"]["path"]) / "task.toml"
        assert final_native.read_bytes() == native_document
        assert shapes == [swe]
        assert not entry.passed
        assert entry.exclusion_reason == "resource_calibration:oom"
        assert pq.read_table(output).num_rows == 0
        assert receipt.uniform_requirements is None
    else:
        assert [(shape.cpus, shape.memory_mb, shape.storage_mb) for shape in shapes] == [
            (1, 2048, 2048),
            (1, 4096, 2048),
        ]
        retained_row = pq.read_table(output).to_pylist()[0]
        if canonical:
            retained = TaskSpec.model_validate_json(retained_row["task_spec"])
            monkeypatch.setattr(
                "experiments.post_training.task_curation.prebuilt.hf_hub_download", lambda **kwargs: str(output)
            )
            assert import_taskcompendium_cohort(
                PrebuiltCohort(
                    intended_use=IntendedUse.TRAIN,
                    name="calibrated",
                    version="1",
                    repo="open-athena/calibrated",
                    revision="a" * 40,
                    expected_rows=1,
                )
            ) == str(output)
        else:
            retained = TaskSpec.model_validate(json.loads(retained_row["lowered_task_spec"])["task"])
        assert retained.execution_requirements.memory_mb == 4096
        assert require_oracle_filter(retained, receipt, None if canonical else envelope["runtime"])
        assert len(entry.calibration.attempts) == 2
