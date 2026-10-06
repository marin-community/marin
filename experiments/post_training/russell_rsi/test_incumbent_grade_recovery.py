# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext, artifact_identity
from marin.execution.step_status import STATUS_FAILED, STATUS_SUCCESS
from marin.experiment.cli import graph_handles

from experiments.post_training.russell_rsi import test_teacher_coverage_study as fixtures
from experiments.post_training.russell_rsi.incumbent_grade_recovery import (
    GradeRecoverySealConfig,
    grade_recovery_step,
    recovered_calibration_summary,
    seal_grade_recovery,
)
from experiments.post_training.russell_rsi.launch_incumbent_bank_trial import incumbent_bank_workflow
from experiments.post_training.russell_rsi.teacher_coverage_study import require_coverage_condition

coverage_condition_inputs = fixtures.coverage_condition_inputs
continuation_inputs = fixtures.continuation_inputs
incumbent_inputs = fixtures.incumbent_inputs
pin = fixtures.pin


def status_pin(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def recovery_rewards():
    return [0, 1] * 4


@pytest.fixture
def recovery_inputs(incumbent_inputs, tmp_path, recovery_rewards):
    original, tasks, _ = incumbent_inputs
    stages = incumbent_bank_workflow(original, "calibrate")
    prefix = str(tmp_path / "original-artifacts")
    calibration_root = Path(stages["calibration"].path(prefix))
    config = json.loads(
        canonical_json(
            stages["calibration"].build_config(
                StepContext.for_run(str(calibration_root), prefix, deps=stages["calibration"].deps)
            )
        )
    )
    producer = {
        "name": stages["calibration"].name,
        "version": stages["calibration"].version,
        "fingerprint": stages["calibration"].fingerprint(),
        "output_path": str(calibration_root),
        "config": config,
    }
    decision_root = Path(stages["decision"].path(prefix))
    keys = [f"{task.task_id}/{index}" for task in tasks for index in range(8)]
    missing = keys[:3]
    records = [
        {
            "key": key,
            "result_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "result_size": 123,
            "grade": {
                "status": "unavailable" if key in missing else "graded",
                "reward": None if key in missing else recovery_rewards[int(key.rsplit("/", 1)[1])],
                "passed": None if key in missing else bool(recovery_rewards[int(key.rsplit("/", 1)[1])]),
            },
            "startup_counts": {"simulator": 1},
            "interrupted_operation": "grade" if key in missing else "model",
            "execution_error": {"type": "ModelResponseRejected", "message": "Preserved rollout error"},
            "original_category": (
                "execution_grade"
                if key in missing
                else ("passed" if recovery_rewards[int(key.rsplit("/", 1)[1])] else "incorrect")
            ),
        }
        for key in keys
    ]
    rewards = {
        task.task_id: [
            entry["grade"]["reward"]
            for entry in records
            if entry["key"].rsplit("/", 1)[0] == task.task_id and entry["grade"]["status"] == "graded"
        ]
        for task in tasks
    }
    categories = Counter(
        "execution_grade" if entry["key"] in missing else "passed" if entry["grade"]["reward"] > 0 else "incorrect"
        for entry in records
    )
    summary = {
        **{
            key: config[key]
            for key in ("model_identity", "tasks_path", "tasks_identity", "samples_per_task", "startup_attempts")
        },
        "count": 32,
        "startup_counts": {"simulator": 256},
        "informative_groups": sum(len(set(values)) > 1 for values in rewards.values()),
        "task_rewards": rewards,
        "categories": dict(categories),
        "failed_task_ids": sorted(rewards),
    }
    binding = {
        "protocol": "russell-calibration-journal-v1",
        "config": config,
        "attempts": {"task": {key: "a" * 64 for key in keys}},
        "parquet_sha256": original["bank"]["identity_config"]["train_sha256"],
    }
    evidence = {
        "original_config": pin(tmp_path / "original-config.json", original),
        "original_calibration": {
            "producer": pin(calibration_root / ".artifact.json", producer),
            "status": status_pin(calibration_root / ".executor_status", STATUS_SUCCESS),
        },
        "failed_decision": {
            "info": pin(
                decision_root / ".executor_info",
                {
                    "name": stages["decision"].name,
                    "output_path": str(decision_root),
                    "config": stages["decision"].lower().hash_attrs,
                },
            ),
            "status": status_pin(decision_root / ".executor_status", STATUS_FAILED),
        },
        "binding": pin(calibration_root / "journal/binding.json", binding),
        "original_summary": pin(calibration_root / "failure_summary.json", summary),
    }
    output = tmp_path / "cpu-recovery"
    inputs = {
        name: {**data, "size": Path(data["uri"]).stat().st_size, "path": name}
        for name, data in {
            "config": evidence["original_config"],
            "producer": evidence["original_calibration"]["producer"],
            "status": evidence["original_calibration"]["status"],
            "binding": evidence["binding"],
            "summary": evidence["original_summary"],
        }.items()
    }
    inputs["parquet"] = {"uri": "fixture-parquet", "sha256": binding["parquet_sha256"], "size": 123, "path": "parquet"}
    slots, deltas = [], []
    for index, key in enumerate(missing):
        slot = {
            "key": key,
            "task_id": key.rsplit("/", 1)[0],
            "result_input": f"result-{index}",
            "reservation_input": f"reservation-{index}",
            "submission_input": f"submission-{index}",
            "decoded_artifact_sha256": hashlib.sha256(f"patch-{index}".encode()).hexdigest(),
            "decoded_artifact_size": index,
            "verifier_sha256": "b" * 64,
            "private_grader_timeout": 120,
        }
        for kind in ("result", "reservation", "submission"):
            inputs[f"{kind}-{index}"] = {
                "uri": str(calibration_root / f"journal/task/{key}/{kind}.json"),
                "sha256": (
                    records[index]["result_sha256"]
                    if kind == "result"
                    else hashlib.sha256(f"{kind}-{index}".encode()).hexdigest()
                ),
                "size": 123,
                "path": f"{kind}-{index}",
            }
        slots.append(slot)
        deltas.append(
            {
                "key": key,
                "grade": {
                    "status": "graded",
                    "reward": recovery_rewards[index],
                    "passed": bool(recovery_rewards[index]),
                },
                "grade_calls": 1,
                "execution_error": None,
                "submission_sha256": inputs[slot["submission_input"]]["sha256"],
                "original_result_sha256": inputs[slot["result_input"]]["sha256"],
                "decoded_artifact_sha256": slot["decoded_artifact_sha256"],
                "model_requests": 0,
                "tokenizer_requests": 0,
                "original_journal_modified": False,
                "accepted_grade": True,
            }
        )
    modules = {
        "rolloutengine.grading": "lib/rolloutengine/src/rolloutengine/grading.py",
        "rolloutengine.machines": "lib/rolloutengine/src/rolloutengine/machines.py",
        "taskcompendium.models": "lib/taskcompendium/src/taskcompendium/models.py",
        "shellbox.backends.qemu.machine": "lib/shellbox/src/shellbox/backends/qemu/machine.py",
    }
    manifest = {
        "inputs": inputs,
        "journal_prefix": str(calibration_root / "journal"),
        "maximum_grade_calls": 3,
        "model_requests": 0,
        "grade_retries": 0,
        "output_prefix": str(output),
        "model_identity": config["model_identity"],
        "tasks_identity": config["tasks_identity"],
        "runtime_bundle": original["runtime_bundle"],
        "runtime_source": original["runtime_commit"],
        "slots": slots,
        "source_commit": "f" * 40,
        "source_module_hashes": {path: "c" * 64 for path in modules.values()},
        "protocol": "fixture-cpu-grade-recovery-v1",
    }
    evidence["input_manifest"] = pin(tmp_path / "input-manifest.json", manifest)
    worker = tmp_path / "worker.py"
    worker.write_text("# Synthetic metadata boundary; this worker is not executed.\n")
    evidence["worker"] = {"uri": str(worker), "sha256": hashlib.sha256(worker.read_bytes()).hexdigest()}
    request = {
        "input_manifest_sha256": evidence["input_manifest"]["sha256"],
        "output_prefix": str(output),
        "worker_sha256": evidence["worker"]["sha256"],
        "source_head": manifest["source_commit"],
        "resources": {"gpu": 0, "cpu": 8, "target_cluster": "cw-us-east-02a"},
        "priority": "batch",
        "retries": {"max_retries_failure": 0, "max_retries_preemption": 0, "max_task_failures": 0},
        "job_name": "fixture-cpu-recovery",
    }
    evidence["request"] = pin(tmp_path / "request.json", request)
    evidence["source_review"] = pin(
        tmp_path / "source-review.json",
        {
            "status": "approved",
            "source_head": manifest["source_commit"],
            "worker_sha256": request["worker_sha256"],
            "manifest_sha256": evidence["input_manifest"]["sha256"],
            "request_sha256": evidence["request"]["sha256"],
            "maximum_grade_calls": 3,
            "automatic_retries": 0,
            "original_journal_writes": 0,
            "expected_original_grades": 253,
        },
    )
    evidence["completed_job"] = pin(
        tmp_path / "job.json",
        {
            "job_id": "/fixture/fixture-cpu-recovery",
            "cluster": "cw-us-east-02a",
            "state": "JOB_STATE_SUCCEEDED",
            "exit_code": 0,
            "failure_count": 0,
            "preemption_count": 0,
            "task_count": 1,
        },
    )
    evidence["launch_record"] = pin(
        tmp_path / "launch.json",
        {
            "job": "/fixture/fixture-cpu-recovery",
            "job_name": request["job_name"],
            "request_sha256": evidence["request"]["sha256"],
            "input_manifest_sha256": evidence["input_manifest"]["sha256"],
            "worker_sha256": request["worker_sha256"],
            "output_prefix": str(output),
            "resources": request["resources"],
            "retries": request["retries"],
            "typed_job_request_validated": True,
            "submissions": 1,
            "model_requests": 0,
        },
    )
    evidence["issuance"] = pin(
        output / "issuance.json",
        {
            "manifest_sha256": evidence["input_manifest"]["sha256"],
            "maximum_grade_calls": 3,
            "model_requests": 0,
            "tokenizer_requests": 0,
            "source_provenance": [
                {"module": name, "path": "/app/" + path, "sha256": "c" * 64} for name, path in modules.items()
            ],
        },
    )
    audit = {"records": records, "original_summary": summary, "missing_keys": missing}
    evidence["original_record_audit"] = pin(output / "original-record-audit.json", audit)
    evidence["deltas"] = []
    for delta in deltas:
        directory = output / "grade-delta" / delta["key"]
        issued = {
            "key": delta["key"],
            "submission_sha256": delta["submission_sha256"],
            "decoded_artifact_sha256": delta["decoded_artifact_sha256"],
            "maximum_grade_calls": 1,
            "model_requests": 0,
        }
        evidence["deltas"].append(
            {"issued": pin(directory / "issued.json", issued), "delta": pin(directory / "delta.json", delta)}
        )
    evidence["cpu_summary"] = pin(
        output / "summary.json",
        {
            "protocol": manifest["protocol"],
            "original_records_audited": 256,
            "grade_calls": 3,
            "model_requests": 0,
            "tokenizer_requests": 0,
            "original_journal_modified": False,
            "slots": [
                {key: delta[key] for key in ("key", "grade_calls", "execution_error", "accepted_grade")}
                | {"status": "graded"}
                for delta in deltas
            ],
        },
    )
    return evidence, summary, audit, manifest, deltas


def test_local_recovery_seal_preserves_253_grades_and_excludes_failed_step(recovery_inputs, tmp_path):
    evidence, original, _, _, _ = recovery_inputs
    original_bytes = Path(evidence["original_summary"]["uri"]).read_bytes()
    step = grade_recovery_step(evidence, "2026.10.06.23")
    assert graph_handles([step]) == [step]
    ctx = StepContext.for_run(str(tmp_path / "sealed"), str(tmp_path / "artifacts"))
    seal_grade_recovery(step.build_config(ctx))
    summary_path = Path(ctx.output_path) / "failure_summary.json"
    summary = json.loads(summary_path.read_bytes())
    assert sum(len(values) for values in summary["task_rewards"].values()) == 256
    for task, values in original["task_rewards"].items():
        assert summary["task_rewards"][task][: len(values)] == values
    decision = json.loads((Path(ctx.output_path) / "calibration-decision.json").read_bytes())
    assert decision["summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert decision["signal_gate_passed"] is True
    assert Path(evidence["original_summary"]["uri"]).read_bytes() == original_bytes
    assert Path(evidence["failed_decision"]["status"]["uri"]).read_text() == STATUS_FAILED


def test_recovery_rejects_changed_submission_and_original_grade(recovery_inputs):
    _, original, audit, manifest, deltas = recovery_inputs
    binding = json.loads(Path(recovery_inputs[0]["binding"]["uri"]).read_bytes())
    deltas[0]["submission_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="saved submission"):
        recovered_calibration_summary(binding=binding, original=original, audit=audit, manifest=manifest, deltas=deltas)
    deltas[0]["submission_sha256"] = manifest["inputs"][manifest["slots"][0]["submission_input"]]["sha256"]
    audit["records"][3]["grade"]["reward"] = 1 - audit["records"][3]["grade"]["reward"]
    audit["records"][3]["original_category"] = "passed" if audit["records"][3]["grade"]["reward"] else "incorrect"
    with pytest.raises(ValueError, match="original-record audit"):
        recovered_calibration_summary(binding=binding, original=original, audit=audit, manifest=manifest, deltas=deltas)


@pytest.mark.parametrize("recovery_rewards", [[0] * 8, [0, 1] * 4])
def test_recovered_condition_uses_original_signal_gate(
    recovery_rewards, recovery_inputs, coverage_condition_inputs, tmp_path
):
    config = coverage_condition_inputs
    evidence, _, _, _, _ = recovery_inputs
    step = grade_recovery_step(evidence, "2026.10.06.25")
    root = Path(step.path(str(tmp_path / "sealed")))
    seal_grade_recovery(GradeRecoverySealConfig(evidence, str(root)))
    condition = config["condition"]
    condition["v21_config"] = evidence["original_config"]
    condition["producers"] = {
        "calibration": evidence["original_calibration"],
        "recovery": fixtures.producer_pins(step, root),
    }
    condition["summary"] = {
        "uri": str(root / "failure_summary.json"),
        "sha256": hashlib.sha256((root / "failure_summary.json").read_bytes()).hexdigest(),
    }
    condition["decision"] = pin(
        root / "calibration-decision.json", json.loads((root / "calibration-decision.json").read_bytes())
    )
    condition["grade_recovery"] = {
        "config": pin(tmp_path / "seal-config.json", {"version": step.version, "evidence": evidence}),
        "provenance": pin(
            root / "grade-recovery-provenance.json", json.loads((root / "grade-recovery-provenance.json").read_bytes())
        ),
    }
    if len(set(recovery_rewards)) == 1:
        assert require_coverage_condition(config)["calibration_decision_sha256"] == condition["decision"]["sha256"]
    else:
        with pytest.raises(ValueError, match="complete failed calibration gate"):
            require_coverage_condition(config)


@pytest.mark.parametrize("reward", [0.5, float("nan")])
def test_recovered_grade_keeps_original_binary_reward_domain(recovery_inputs, reward):
    evidence, original, audit, manifest, deltas = recovery_inputs
    binding = json.loads(Path(evidence["binding"]["uri"]).read_bytes())
    deltas[0]["grade"]["reward"] = reward
    with pytest.raises(ValueError, match="bounded private grader"):
        recovered_calibration_summary(binding=binding, original=original, audit=audit, manifest=manifest, deltas=deltas)


def test_recovered_trial_graph_excludes_original_failed_decision(recovery_inputs, tmp_path):
    evidence, _, _, _, _ = recovery_inputs
    step = grade_recovery_step(evidence, "2026.10.06.25")
    root = Path(step.path(str(tmp_path / "recovered-trial")))
    seal_grade_recovery(GradeRecoverySealConfig(evidence, str(root)))
    original = json.loads(Path(evidence["original_config"]["uri"]).read_bytes())
    failed_decision = incumbent_bank_workflow(original, "calibrate")["decision"]
    bound = {
        **original,
        "calibration_summary_uri": str(root / "failure_summary.json"),
        "calibration_decision_uri": str(root / "calibration-decision.json"),
        "calibration_decision_sha256": hashlib.sha256((root / "calibration-decision.json").read_bytes()).hexdigest(),
    }
    train = incumbent_bank_workflow(bound, "train")
    identities = {artifact_identity(handle) for handle in graph_handles([train["terminal"]])}
    assert artifact_identity(failed_decision) not in identities
    assert train["rl"] in graph_handles([train["terminal"]])
