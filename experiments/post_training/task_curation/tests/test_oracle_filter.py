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
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.harbor.adapter import write_harbor_task
from taskcompendium.harbor.protected_runtime import ExpectedRuntime, RuntimeRole
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.models import Controls, IntendedUse
from taskcompendium.pipeline.oracle_filter import (
    ExecutabilityRunReceipt,
    OracleExclusionReason,
    load_oracle_filter,
    require_oracle_filter,
)
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.tasktrove.calendar import calendar_golden
from experiments.post_training.task_curation.datasets.tasktrove.math import MATH_CONTROLS
from experiments.post_training.task_curation.oracle_filter import (
    HARBOR_PROJECT,
    calibrate_cohort,
    calibrated_task,
    calibration_package_identity,
    collect_trials,
    filter_cohort,
    harbor_inspection,
    native_requirements,
    write_task_map,
)
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_cohort
from experiments.post_training.task_curation.task_spec_harbor import prepare

from .prebuilt_fixtures import prebuilt_task


@pytest.mark.parametrize("kind", ["native_text", "native_json", "task_spec_missing", "task_spec_infra"])
def test_production_native_reward_formats_qualify_without_bypassing_taskspec_verdicts(tmp_path, kind):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("native-grade")["lowered_task_spec"])["task"])
    native = tmp_path / "native"
    for name in ("environment", "tests"):
        (native / name).mkdir(parents=True)
    (native / "environment/Dockerfile").write_text("FROM fixture@sha256:" + "a" * 64 + "\n")
    (native / "instruction.md").write_text("Answer the original task.")
    (native / "tests/test.sh").write_text("true\n")
    (native / "task.toml").write_text(
        "[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n"
        "[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\n"
    )
    if kind.startswith("task_spec"):
        (native / "task-spec.json").write_text(task.model_dump_json())
        (native / "tests/taskcompendium-verifier.toml").write_text("")
    checksum = harbor_inspection("tasks", [str(native)])[str(native)]["task_checksum"]
    trial = tmp_path / "jobs/trial"
    (trial / "verifier").mkdir(parents=True)
    artifact = trial / "verifier" / ("reward.json" if kind == "native_json" else "reward.txt")
    artifact.write_text('{"reward":0.25}' if kind == "native_json" else "0.25")
    result = {
        "id": str(uuid4()),
        "task_name": "native",
        "trial_name": "trial",
        "trial_uri": trial.as_uri(),
        "task_id": {"path": str(native)},
        "task_checksum": checksum,
        "config": {
            "task": {"path": str(native)},
            "agent": {
                "name": "terminus-2",
                "model_name": "openai/glm-5.3",
                "kwargs": {"max_turns": 3, "temperature": 0.7},
            },
            "environment": {"type": "local"},
            "resolved_timeouts": {"agent": 300, "verifier": 60, "agent_setup": 360, "environment_build": 600},
            "trials_dir": str(trial.parent),
            "trial_name": "trial",
        },
        "agent_info": {"name": "terminus-2", "version": "1"},
        "verifier_result": {"rewards": {"reward": 0.25}},
        "finished_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
        "completion": {"trajectory_present": True},
        "exception_info": None,
    }
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "config.json").write_text(json.dumps(result["config"]))
    if kind == "task_spec_infra":
        (trial / "verifier/verdict.json").write_text(
            json.dumps({"status": "infra_error", "reward": 0.25, "detail": {"error": "provider unavailable"}})
        )
    receipt = collect_trials(trial.parent)[("native", checksum)][0]
    control = ExecutabilityRunReceipt(
        actor_turns=1,
        trial=receipt,
        model_evidence=(ContentEvidenceReference(uri="s3://fixtures/model.json", sha256="b" * 64),),
    )
    assert control.passed == kind.startswith("native")
    if kind.startswith("task_spec"):
        assert receipt.native_grade is None
        return
    assert receipt.native_grade.artifact_sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert receipt.native_grade.rewards == {"reward": 0.25}
    artifact.write_text("0.75" if kind == "native_text" else '{"reward":0.75}')
    with pytest.raises(subprocess.CalledProcessError):
        collect_trials(trial.parent)


@pytest.mark.parametrize("binding", ["oracle_command", "calendar_original", "protected_runtime", "source_runtime"])
def test_changed_original_command_runtime_stops_before_trial(tmp_path, binding):
    envelope = json.loads(prebuilt_task("bound-original")["lowered_task_spec"])
    task = TaskSpec.model_validate(envelope["task"])
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={"oracle": (inline_resource("solution/solve.sh", b"exit 0\n"),)}
            )
        }
    )
    envelope["task"] = task.model_dump(mode="json")
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist([{"lowered_task_spec": json.dumps(envelope)}]), cohort)
    output = tmp_path / "exported"
    expectations = (
        {
            RuntimeRole.SOURCE_GRADER: ExpectedRuntime(
                python_version=(3, 12),
                verifyit_root="/frozen/verifyit",
                files={},
                lock_sha256="c" * 64,
            )
        }
        if binding == "protected_runtime"
        else None
    )
    controls = MATH_CONTROLS if binding == "oracle_command" else Controls(golden=calendar_golden)
    prepare(
        cohort,
        output,
        source_revision="a" * 40,
        controls=controls if binding in {"oracle_command", "calendar_original"} else None,
        runtime_expectations=expectations,
    )
    directory = output / "tasks" / task.id
    if binding in {"oracle_command", "calendar_original"}:
        config = tmp_path / "accepted-trial.json"
        config.write_text(
            json.dumps(
                {
                    "task": {"path": str(directory)},
                    "agent": {"name": "oracle"},
                    "trials_dir": str(tmp_path / "accepted"),
                    "trial_name": "exported-math",
                    "environment": {"type": "local", "kwargs": {"disk_root": str(tmp_path / "disks")}},
                }
            )
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--isolated",
                "--project",
                str(HARBOR_PROJECT),
                "--frozen",
                "--extra",
                "runtime",
                "python",
                "-c",
                "import asyncio,sys; from pathlib import Path; from harbor.models.trial.config import TrialConfig; "
                "from harbor.trial.trial import Trial; "
                "trial=asyncio.run(Trial.create(TrialConfig.model_validate_json(Path(sys.argv[1]).read_text()))); "
                "assert trial.agent.command is not None; trial.agent.command.require_task(trial.specification)",
                str(config),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    destination = tmp_path / "trials"
    with pytest.raises(ValueError, match="runtime changed"):
        calibrated_task(
            task, {"task_path": str(directory), "runtime_sha256": canonical_sha256({})}, destination, "local", {}, None
        )
    assert not destination.exists()


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
        "no-oracle-env",
        "no-oracle-oom",
        "live",
    ):
        row = prebuilt_task(task_id)
        envelope = json.loads(row["lowered_task_spec"])
        task = TaskSpec.model_validate(envelope["task"])
        if task_id not in {"calibration-only", "live", "no-oracle-env", "no-oracle-oom"}:
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
                if task_id in {"source-env", "no-oracle-env"}
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
            3 if task_id in {"infra", "source-env", "unmatched-env", "observed-env", "newer-env", "no-oracle-env"} else 1
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
                        "exception_type": "OutOfMemoryError" if task_id == "no-oracle-oom" else "RuntimeError",
                        "exception_message": "backend unavailable",
                        "exception_traceback": "",
                        "occurred_at": datetime(2026, 11, 12, tzinfo=UTC).isoformat(),
                    }
                    if task_id in {"infra", "no-oracle-oom"}
                    else None
                ),
            }
            if task_id == "live":
                result["agent_info"]["name"] = "task-spec-chat"
                result["config"]["agent"] = {
                    "name": "task-spec-chat",
                    "model_name": "openai/glm-5.3",
                    "kwargs": {"max_turns": 3, "request_parameters": {"temperature": 0.7}},
                }
            (root / "result.json").write_text(json.dumps(result))
            (root / "config.json").write_text(json.dumps(result["config"]))
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "result.json").write_text('{"task_output":1}\n{"task_output":2}\n')
            (artifacts / "config.json").write_text('{"task":"source-controlled","agent":"source-controlled"}')
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
                if task_id in {"source-env", "unmatched-env", "observed-env", "newer-env", "no-oracle-env"}
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
    proof = ExecutabilityRunReceipt(
        actor_turns=1,
        model_evidence=(ContentEvidenceReference(uri="s3://fixtures/model-request.json", sha256="b" * 64),),
        trial=collect_trials(tmp_path / "jobs")[(live.id, mappings[live.id]["task_checksum"])][0],
    )
    controls = tmp_path / "executability.jsonl"
    controls.write_text(json.dumps({"task_id": live.id, "receipt": proof.model_dump(mode="json")}) + "\n")
    output = tmp_path / "filtered/tasks.parquet"
    conversion_manifest = tmp_path / "conversion.json"
    conversion_manifest.write_text(
        json.dumps(
            {
                "input_rows": 15,
                "converted_rows": 13,
                "within_source_dedup": {"raw_count": 15, "unique_count": 13, "within_source_duplicates": 2},
                "oracle_recipe_identity": "fixture-recipe",
            }
        )
    )
    receipt = filter_cohort(
        cohort, task_map, tmp_path / "jobs", output, conversion_manifest=conversion_manifest, executability_runs=controls
    )
    assert [row["path"] for row in pq.read_table(output).to_pylist()] == ["pass", "live"]
    assert receipt.input_rows == 13
    assert load_oracle_filter(str(output.parent / "oracle-filter.json")) == receipt
    assert receipt.tasks["no-oracle-env"].exclusion_reason is OracleExclusionReason.SOURCE_ENV_INCOMPATIBLE_WITH_GRADER
    assert receipt.tasks["no-oracle-oom"].exclusion_reason is OracleExclusionReason.OOM
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
    assert calibration_only.exclusion_reason is OracleExclusionReason.EXECUTABILITY_PROOF_MISSING
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


def test_missing_production_proof_filters_and_archives_the_exact_original_row(tmp_path):
    row = prebuilt_task("unexecuted")
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist([row]), cohort)
    mapping = tmp_path / "task-map.json"
    mapping.write_text("{}")
    output = tmp_path / "filtered/tasks.parquet"
    evidence = ContentEvidenceReference(
        uri="s3://fixtures/cohort.parquet", sha256=hashlib.sha256(cohort.read_bytes()).hexdigest()
    )
    receipt = filter_cohort(cohort, mapping, tmp_path / "jobs", output, input_evidence=evidence)
    entry = receipt.tasks["unexecuted"]
    assert entry.exclusion_reason is OracleExclusionReason.EXECUTABILITY_PROOF_MISSING
    assert not entry.passed
    assert pq.read_table(output).num_rows == 0
    assert pq.read_table(output.parent / "excluded-tasks.parquet").to_pylist() == [row]
    assert json.loads((output.parent / "oracle-filter-summary.json").read_text())["exclusions_by_reason"] == {
        "executability_proof_missing": 1
    }


@pytest.mark.parametrize(
    ("declared", "native_declared", "canonical", "task_spec_package"),
    [
        (True, True, False, False),
        (True, True, True, False),
        (False, False, False, False),
        (False, False, True, False),
        (False, True, False, False),
        (False, True, True, False),
        (False, False, True, True),
    ],
)
def test_population_calibration_doubles_missing_limits_but_preserves_swe_declarations(
    tmp_path, monkeypatch, declared, native_declared, canonical, task_spec_package
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
        if task_spec_package:
            row["runtime"] = json.dumps(envelope["runtime"])
    cohort = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([row]), cohort)
    native = tmp_path / "native" / "task"
    if task_spec_package:
        write_harbor_task(task, native, INITIAL_REQUIREMENTS)
    for folder in ("environment", "tests", "solution"):
        (native / folder).mkdir(parents=True, exist_ok=True)
    (native / "instruction.md").write_text("Solve the task.")
    (native / "tests/test.sh").write_text("true\n")
    (native / "solution/solve.sh").write_text("true\n")
    (native / "task.toml").write_text(
        "[environment]\ncpus=2\nmemory_mb=4096\nstorage_mb=10240\n"
        "[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=600\n"
        if native_declared
        else "[environment]\n"
    )
    if task_spec_package:
        (native / "task.toml").write_text(
            '[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\nenvironment_mode="separate"\n'
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
        assert json.loads((package.parent / "prepared.json").read_text()) == shape.model_dump()
        shapes.append(shape)
        if task_spec_package:
            assert TaskSpec.model_validate_json((package / "task-spec.json").read_text()) == task
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
        (root / "config.json").write_text(json.dumps(result["config"]))
        (root / "agent/exit-code.txt").write_text("137" if oom else "0")
        (root / "agent/resource-usage.json").write_text(json.dumps({"oom_kills": int(oom)}))
        (root / "verifier/verdict.json").write_text(json.dumps({"status": "scored", "reward": 1}))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", execute)

    def prepare_attempt(package: Path, shape: ExecutionRequirements) -> None:
        assert native_requirements(package) == shape
        (package.parent / "prepared.json").write_text(shape.model_dump_json())

    if task_spec_package:

        def reject_capacity(package: Path, shape: ExecutionRequirements) -> None:
            assert native_requirements(package) == shape
            raise RuntimeError("snapshot capacity unavailable")

        blocked = tmp_path / "blocked"
        with pytest.raises(RuntimeError, match="snapshot capacity unavailable"):
            calibrate_cohort(cohort, mapping, blocked, "local", {}, 1, prepare_attempt=reject_capacity)
        assert not list(blocked.rglob("result.json"))
        assert not (blocked / "task-map.json").exists()
        assert shapes == []

    run = tmp_path / "calibrated"
    calibrate_cohort(cohort, mapping, run, "local", {}, 1, prepare_attempt=prepare_attempt)
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
        assert require_oracle_filter(
            retained, receipt, None if canonical and not task_spec_package else envelope["runtime"]
        )
        assert len(entry.calibration.attempts) == 2


@pytest.mark.parametrize("role", ["actor", "verifier"])
def test_capacity_only_environment_tables_preserve_original_package_identity(tmp_path, role):
    original = tmp_path / "original"
    staged = tmp_path / "staged"
    for package in (original, staged):
        package.mkdir()
        (package / "instruction.md").write_text("Run the original source task.")
        (package / "task.toml").write_text(
            '[metadata]\nsource="smartie"\n'
            + ('[verifier]\nenvironment_mode="separate"\n' if role == "verifier" else "")
        )
    expected = calibration_package_identity(original, "original")
    section = "environment" if role == "actor" else "verifier.environment"
    with (staged / "task.toml").open("a") as output:
        output.write(f"[{section}]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n")
    assert calibration_package_identity(staged, "original") == expected
    with (staged / "task.toml").open("a") as output:
        output.write('docker_image="changed-source-image"\n')
    assert calibration_package_identity(staged, "original") != expected


def test_capacity_only_verifier_table_cannot_hide_a_shared_to_separate_grader_change(tmp_path):
    original = tmp_path / "original"
    staged = tmp_path / "staged"
    for package in (original, staged):
        package.mkdir()
        (package / "task.toml").write_text('[metadata]\nsource="smartie"\n')
    with (staged / "task.toml").open("a") as output:
        output.write("[verifier.environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=2048\n")
    assert calibration_package_identity(original, "original") != calibration_package_identity(staged, "original")
