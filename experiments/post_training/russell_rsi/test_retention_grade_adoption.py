# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover one saved grade while preserving the original journal and valid scores."""

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from taskcompendium.environment import ArtifactKind, ShellVerifierSpec, VerifierArtifact
from taskcompendium.models import VerifierKind
from taskcompendium.parquet import write_tasks

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.completed_sft_selection import (
    CompletedProducer,
    CompletedRetention,
    retention_result_summary,
)
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.retention_grade_adoption import (
    AMENDMENT_PROTOCOL,
    GRADE_PROTOCOL,
    GRADER_SOURCE_PATHS,
    RecoveredRetentionConfig,
    recovered_retention_summary,
    write_recovered_retention,
)
from experiments.post_training.russell_rsi.token_preflight import preflight_task


@pytest.fixture
def recovered_inputs(tmp_path):
    def pin(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        path.write_bytes(raw)
        return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    original_path, output = tmp_path / "original", tmp_path / "grade-only"
    tasks = [preflight_task(index, "Return the saved value.", 23) for index in range(3)]
    artifact = VerifierArtifact(source="/workspace/patch", target="/workspace/patch", kind=ArtifactKind.FILE)
    verifier = ShellVerifierSpec(argv=("true",), timeout=120, environment=tasks[2].environment, artifacts=(artifact,))
    tasks[2] = tasks[2].model_copy(
        update={
            "verifier": (
                tasks[2].verifier.model_copy(
                    update={"kind": VerifierKind.SHELL, "parameters_json": verifier.model_dump_json()}
                )
            )
        }
    )
    tasks_path = tmp_path / "tasks.parquet"
    write_tasks(str(tasks_path), tasks)
    task_ids = tuple(task.id for task in tasks)
    evaluation = {
        "model_identity": "candidate",
        "tasks_identity": "retention",
        "tasks_path": str(tasks_path),
        "runtime_bundle": {"sha256": "runtime"},
    }
    binding = {"evaluation": evaluation}
    binding_pin = pin(original_path / "journal/binding.json", binding)
    records, attempts, results, audit = {}, [], [], []
    for index, task_id in enumerate(task_ids):
        key = task_id + "/0"
        grade = {"status": "unavailable" if index == 2 else "graded", "reward": None if index == 2 else index}
        value = {
            "record": {
                "task_id": task_id,
                "grade": grade,
                "interrupted_operation": "grade" if index == 2 else None,
                "execution_error": {"type": "MachineStartupError"} if index == 2 else None,
            },
            "startup_counts": {"retries": 0},
        }
        records[("task", key)] = value
        directory = original_path / "journal/task" / key
        reservation = pin(directory / "reservation.json", {"evaluation": binding, "key": key})
        result = pin(directory / "result.json", {"binding": {"evaluation": binding, "key": key}, "result": value})
        attempts.append({"kind": "task", "key": key, "reservation": reservation, "result": result})
        results.append({"key": key, "reservation": reservation, "result": result, "expected_grade": grade})
        audit.append(
            {"key": key, "result_sha256": result["sha256"], "reservation_sha256": reservation["sha256"], "grade": grade}
        )
    key = task_ids[2] + "/0"
    patch = b"saved patch bytes"
    patch_hash = hashlib.sha256(patch).hexdigest()
    submission = pin(
        original_path / "journal/task" / key / "submission.json",
        {
            "binding": {"evaluation": binding, "key": key},
            "artifact": artifact.model_dump(mode="json"),
            "body_base64": base64.b64encode(patch).decode(),
            "sha256": patch_hash,
        },
    )
    original_summary = {
        **{k: evaluation[k] for k in ("model_identity", "tasks_path", "tasks_identity")},
        "count": 3,
        "samples_per_task": 1,
        "startup_attempts": 1,
        "startup_counts": {"retries": 0},
        "informative_groups": 0,
        "task_rewards": {task_ids[0]: [0], task_ids[1]: [1]},
        "categories": {"incorrect": 1, "passed": 1, "execution_grade": 1},
        "failed_task_ids": sorted([task_ids[0], task_ids[2]]),
    }
    retained = CompletedProducer(
        {
            "journal_binding": binding_pin,
            "attempts": attempts,
            "summary": pin(original_path / "failure_summary.json", original_summary),
        },
        {"producer_identity": "original-retention", "runtime_commit": "runtime-commit", "source_head": "source-commit"},
        {"output_path": str(original_path)},
        {},
    )
    amendment = pin(
        tmp_path / "amendment.json",
        {
            "protocol": AMENDMENT_PROTOCOL,
            "grade_only_recovery": {"scope": key},
            "original_binding": binding_pin,
            "original_producer": "original-retention",
            "source_module_hashes": {name: "source-hash" for name in GRADER_SOURCE_PATHS},
        },
    )
    manifest = {
        "protocol": GRADE_PROTOCOL,
        "amendment": amendment,
        "grade_only_recovery": {"scope": key},
        "original_binding": binding_pin,
        **{k: evaluation[k] for k in ("model_identity", "tasks_identity", "runtime_bundle")},
        "runtime_source": "runtime-commit",
        "source_commit": "source-commit",
        "source_module_hashes": {name: "source-hash" for name in GRADER_SOURCE_PATHS},
        "maximum_grade_calls": 1,
        "output_prefix": str(output),
        "results": results,
        "submission": {
            "key": key,
            "task_id": task_ids[2],
            "submission": submission,
            "verifier_sha256": digest(verifier.model_dump(mode="json")),
            "private_grader_timeout": 120,
            "decoded_artifact_sha256": patch_hash,
            "decoded_artifact_size": len(patch),
        },
    }
    pins = {
        "amendment": amendment,
        "manifest": pin(tmp_path / "manifest.json", manifest),
        "worker": pin(tmp_path / "worker.py", b"frozen worker"),
    }
    request = {
        "amendment": amendment,
        "output_prefix": str(output),
        "input_manifest_sha256": pins["manifest"]["sha256"],
        "worker_sha256": pins["worker"]["sha256"],
        "source_head": "source-commit",
        "model_http_requests": 0,
        "resources": {"gpu": 0},
        "retries": {"max_retries_failure": 0, "max_retries_preemption": 0, "max_task_failures": 0},
        "private_grader_timeout": 120,
    }
    pins["request"] = pin(tmp_path / "request.json", request)
    launch = {
        "job": "/test/regrade",
        "request_sha256": pins["request"]["sha256"],
        **{k: request[k] for k in ("worker_sha256", "input_manifest_sha256", "source_head")},
        "status": "submitted",
        "submissions": 1,
        "generation_requests": 0,
    }
    pins["launch"] = pin(tmp_path / "launch.json", launch)
    pins["terminal"] = pin(
        tmp_path / "terminal.json",
        {
            "job": launch["job"],
            "request_sha256": launch["request_sha256"],
            "state": "succeeded",
            "exit_code": 0,
            "failure_count": 0,
            "preemption_count": 0,
            "task_count": 1,
            "completed_count": 1,
            "tasks": [{"state": "succeeded", "exit_code": 0}],
        },
    )
    pins["issuance"] = pin(
        output / "issuance.json",
        {
            "manifest_sha256": pins["manifest"]["sha256"],
            "amendment_sha256": amendment["sha256"],
            "maximum_grade_calls": 1,
            "model_requests": 0,
            "tokenizer_requests": 0,
            "source_provenance": [{"path": "/app/" + name, "sha256": "source-hash"} for name in GRADER_SOURCE_PATHS],
        },
    )
    pins["audit"] = pin(
        output / "completed-record-audit.json",
        {
            "records": audit,
            "graded": 2,
            "unavailable": 1,
            "original_journal_modified": False,
            "capability_labels": False,
        },
    )
    pins["issued"] = pin(
        output / "grade-delta/issued.json",
        {
            "key": key,
            "submission_sha256": submission["sha256"],
            "decoded_artifact_sha256": patch_hash,
            "model_requests": 0,
        },
    )
    pins["delta"] = pin(
        output / "grade-delta/delta.json",
        {
            "protocol": GRADE_PROTOCOL,
            "manifest_sha256": pins["manifest"]["sha256"],
            "key": key,
            "submission_sha256": submission["sha256"],
            "decoded_artifact_sha256": patch_hash,
            "grade": {"status": "graded", "reward": 0, "error": None},
            "grade_calls": 1,
            "execution_error": None,
            "model_requests": 0,
            "tokenizer_requests": 0,
            "original_result_written": False,
            "capability_label": False,
        },
    )
    pins["summary"] = pin(
        output / "summary.json",
        {
            "protocol": GRADE_PROTOCOL,
            "completed_records_validated": 2,
            "unavailable_result_validated": 1,
            "grade_delta_calls": 1,
            "grade_status": "graded",
            "model_requests": 0,
            "tokenizer_requests": 0,
            "original_journal_modified": False,
            "capability_labels": False,
        },
    )
    return retained, CompletedRetention(evaluation, binding, records), task_ids, pins, pin


def test_saved_grade_produces_separate_complete_score_without_changing_original(recovered_inputs, tmp_path):
    producer, completed, task_ids, pins, pin = recovered_inputs
    before = {path: path.read_bytes() for path in Path(producer.output_path).rglob("*") if path.is_file()}
    original_records = copy.deepcopy(completed.records)
    with pytest.raises(ValueError, match="valid canonical grade"):
        retention_result_summary(completed, task_ids)
    summary = recovered_retention_summary(producer, completed, task_ids, pins)
    assert summary["task_rewards"] == {task_ids[0]: [0], task_ids[1]: [1], task_ids[2]: [0]}
    assert summary["categories"] == {"incorrect": 2, "passed": 1}
    assert summary["startup_counts"] == {"retries": 0}
    input_pin = PinnedFile(**pin(tmp_path / "input.json", {"retention": producer.pins, "grade_recovery": pins}))
    output = tmp_path / "derived"
    write_recovered_retention(RecoveredRetentionConfig(input_pin, "original-retention", summary, str(output)))
    assert json.loads((output / "failure_summary.json").read_bytes()) == summary
    assert json.loads((output / "grade-recovery.json").read_bytes())["grade_recovery"] == pins
    assert completed.records == original_records
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("defect", ["slot", "patch", "runtime", "valid_grade", "missing_source"])
def test_recovery_rejects_grade_transplant_or_changed_original(recovered_inputs, defect):
    producer, completed, task_ids, pins, pin = recovered_inputs
    target = pins["delta"]
    value = PinnedFile(**target).read_json()
    if defect == "slot":
        value["key"] = task_ids[0] + "/0"
    elif defect == "patch":
        value["decoded_artifact_sha256"] = hashlib.sha256(b"another patch").hexdigest()
    elif defect == "runtime":
        producer.launch["runtime_commit"] = "another-runtime"
    elif defect == "missing_source":
        target = pins["issuance"]
        value = PinnedFile(**target).read_json()
        value["source_provenance"] = []
    else:
        completed.records[("task", task_ids[0] + "/0")]["record"]["grade"]["reward"] = 1
    target.update(pin(target["uri"], value))
    message = {
        "slot": "exactly the original unavailable grade",
        "patch": "exact saved patch",
        "runtime": "original inputs",
        "valid_grade": "changed an original grade",
        "missing_source": "private grading runtime",
    }[defect]
    with pytest.raises(ValueError, match=message):
        recovered_retention_summary(producer, completed, task_ids, pins)
