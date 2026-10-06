# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adopt one saved-submission grade without changing the original retention journal."""

import base64
import hashlib
from dataclasses import asdict, dataclass, replace

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.environment import ShellVerifierSpec
from taskcompendium.parquet import read_tasks

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.completed_sft_selection import (
    CompletedProducer,
    CompletedRetention,
    completed_coding_evidence,
    completed_producer,
    completed_retention_records,
    pinned_at,
    pinned_json,
    retention_result_summary,
)
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.interrupted_calibration import (
    InterruptedSelectionConfig,
    seal_interrupted_selection,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import StudyBaseline, post_sft_selection_stages

PROTOCOL = "russell-rsi-completed-retention-grade-selection-v1"
VERSION = "2026.10.06.13"
GRADE_PROTOCOL = "russell-rsi-retention-v10-grade-delta-v1"
AMENDMENT_PROTOCOL = "russell-rsi-retention-v10-grade-recovery-amendment-v1"
GRADER_SOURCE_PATHS = (
    "lib/rolloutengine/src/rolloutengine/grading.py",
    "lib/rolloutengine/src/rolloutengine/machines.py",
    "lib/taskcompendium/src/taskcompendium/models.py",
    "lib/shellbox/src/shellbox/backends/qemu/machine.py",
)


def recovered_retention_summary(
    producer: CompletedProducer, completed: CompletedRetention, task_ids: tuple[str, ...], pins: dict
) -> dict:
    """Verify the exact saved patch, successful CPU worker, and one replacement grade."""
    manifest = pinned_json(pins["manifest"])
    amendment = pinned_json(pins["amendment"])
    request = pinned_json(pins["request"])
    launch = pinned_json(pins["launch"])
    terminal = pinned_json(pins["terminal"])
    PinnedFile(**pins["worker"]).read_bytes()
    if (
        manifest["protocol"] != GRADE_PROTOCOL
        or amendment["protocol"] != AMENDMENT_PROTOCOL
        or manifest["amendment"] != pins["amendment"]
        or request["amendment"] != pins["amendment"]
        or manifest["grade_only_recovery"] != amendment["grade_only_recovery"]
        or manifest["original_binding"] != amendment["original_binding"]
        or amendment["original_producer"] != producer.launch["producer_identity"]
        or any(
            manifest["source_module_hashes"][name] != value for name, value in amendment["source_module_hashes"].items()
        )
        or manifest["original_binding"]["sha256"] != producer.pins["journal_binding"]["sha256"]
        or manifest["original_binding"]["uri"] != producer.pins["journal_binding"]["uri"]
        or manifest["model_identity"] != completed.evaluation["model_identity"]
        or manifest["tasks_identity"] != completed.evaluation["tasks_identity"]
        or manifest["runtime_bundle"] != completed.evaluation["runtime_bundle"]
        or manifest["runtime_source"] != producer.launch["runtime_commit"]
        or manifest["source_commit"] != producer.launch["source_head"]
        or manifest["maximum_grade_calls"] != 1
        or manifest["output_prefix"] != request["output_prefix"]
    ):
        raise ValueError("Retention recovery changed the original inputs")
    if (
        request["input_manifest_sha256"] != pins["manifest"]["sha256"]
        or request["worker_sha256"] != pins["worker"]["sha256"]
        or request["source_head"] != manifest["source_commit"]
        or request["model_http_requests"] != 0
        or request["resources"]["gpu"] != 0
        or request["retries"] != {"max_retries_failure": 0, "max_retries_preemption": 0, "max_task_failures": 0}
    ):
        raise ValueError("Retention recovery changed the completed CPU request")
    if (
        launch["request_sha256"] != pins["request"]["sha256"]
        or launch["worker_sha256"] != request["worker_sha256"]
        or launch["input_manifest_sha256"] != pins["manifest"]["sha256"]
        or launch["source_head"] != request["source_head"]
        or launch["status"] != "submitted"
        or launch["submissions"] != 1
        or launch["generation_requests"] != 0
    ):
        raise ValueError("Retention recovery launch differs from its pinned CPU request")
    if (
        terminal["job"] != launch["job"]
        or terminal["request_sha256"] != pins["request"]["sha256"]
        or terminal["state"] != "succeeded"
        or terminal["exit_code"] != 0
        or terminal["failure_count"] != 0
        or terminal["preemption_count"] != 0
        or terminal["task_count"] != 1
        or terminal["completed_count"] != 1
        or len(terminal["tasks"]) != 1
        or terminal["tasks"][0]["state"] != "succeeded"
        or terminal["tasks"][0]["exit_code"] != 0
    ):
        raise ValueError("Retention recovery has no successful one-task CPU terminal record")
    output = StoragePath(manifest["output_prefix"])
    issuance = pinned_at(pins["issuance"], str(output / "issuance.json"))
    audit = pinned_at(pins["audit"], str(output / "completed-record-audit.json"))
    issued = pinned_at(pins["issued"], str(output / "grade-delta/issued.json"))
    delta = pinned_at(pins["delta"], str(output / "grade-delta/delta.json"))
    recovery_summary = pinned_at(pins["summary"], str(output / "summary.json"))
    if (
        issuance["manifest_sha256"] != pins["manifest"]["sha256"]
        or issuance["amendment_sha256"] != pins["amendment"]["sha256"]
        or issuance["maximum_grade_calls"] != 1
        or issuance["model_requests"] != 0
        or issuance["tokenizer_requests"] != 0
        or delta["protocol"] != GRADE_PROTOCOL
        or delta["manifest_sha256"] != pins["manifest"]["sha256"]
        or delta["grade_calls"] != 1
        or delta["execution_error"] is not None
        or delta["grade"]["error"] is not None
        or delta["model_requests"] != 0
        or delta["tokenizer_requests"] != 0
        or delta["original_result_written"] is not False
        or delta["capability_label"] is not False
        or recovery_summary
        != {
            "protocol": GRADE_PROTOCOL,
            "completed_records_validated": 2,
            "unavailable_result_validated": 1,
            "grade_delta_calls": 1,
            "grade_status": "graded",
            "model_requests": 0,
            "tokenizer_requests": 0,
            "original_journal_modified": False,
            "capability_labels": False,
        }
    ):
        raise ValueError("Retention recovery is incomplete or issued additional work")
    expected_sources = {name: manifest["source_module_hashes"][name] for name in GRADER_SOURCE_PATHS}
    actual_sources = {entry["path"].removeprefix("/app/"): entry["sha256"] for entry in issuance["source_provenance"]}
    if actual_sources != expected_sources or len(issuance["source_provenance"]) != len(expected_sources):
        raise ValueError("Recovery imported a different private grading runtime")
    expected_audit = []
    original_tasks = {item["key"]: item for item in producer.pins["attempts"] if item["kind"] == "task"}
    for item in manifest["results"]:
        original = original_tasks[item["key"]]
        record = completed.records[("task", item["key"])]["record"]
        for field in ("reservation", "result"):
            if {key: item[field][key] for key in ("uri", "sha256")} != original[field]:
                raise ValueError("Recovery audited a different original retention record")
        grade = {key: record["grade"][key] for key in ("status", "reward")}
        if item["expected_grade"] != grade:
            raise ValueError("Recovery changed an original grade")
        expected_audit.append(
            {
                "key": item["key"],
                "result_sha256": item["result"]["sha256"],
                "reservation_sha256": item["reservation"]["sha256"],
                "grade": grade,
            }
        )
    if (
        {item["key"] for item in expected_audit} != set(original_tasks)
        or len(expected_audit) != 3
        or audit
        != {
            "records": expected_audit,
            "graded": 2,
            "unavailable": 1,
            "original_journal_modified": False,
            "capability_labels": False,
        }
    ):
        raise ValueError("Recovery did not audit the exact three-task cohort")
    missing = [
        key for key in original_tasks if completed.records[("task", key)]["record"]["grade"]["status"] != "graded"
    ]
    saved = manifest["submission"]
    key = saved["key"]
    if missing != [key] or saved["task_id"] + "/0" != key or delta["key"] != key:
        raise ValueError("Recovery must replace exactly the original unavailable grade")
    original = completed.records[("task", key)]
    record = original["record"]
    if (
        record["grade"]["status"] != "unavailable"
        or record["grade"]["reward"] is not None
        or record["interrupted_operation"] != "grade"
        or record["execution_error"]["type"] != "MachineStartupError"
    ):
        raise ValueError("Recovery is not the saved private-grader startup interruption")
    submission_pin = {field: saved["submission"][field] for field in ("uri", "sha256")}
    submission = pinned_at(
        submission_pin, str(StoragePath(producer.output_path) / "journal/task" / key / "submission.json")
    )
    reservation = pinned_json(original_tasks[key]["reservation"])
    task = next(task for task in read_tasks(completed.evaluation["tasks_path"]) if task.id == saved["task_id"])
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    patch = base64.b64decode(submission["body_base64"], validate=True)
    patch_hash = hashlib.sha256(patch).hexdigest()
    if (
        submission["binding"] != reservation
        or len(verifier.artifacts) != 1
        or submission["artifact"] != verifier.artifacts[0].model_dump(mode="json")
        or saved["verifier_sha256"] != digest(verifier.model_dump(mode="json"))
        or saved["private_grader_timeout"] != verifier.timeout
        or request["private_grader_timeout"] != verifier.timeout
        or patch_hash != submission["sha256"]
        or patch_hash != saved["decoded_artifact_sha256"]
        or len(patch) != saved["decoded_artifact_size"]
        or delta["submission_sha256"] != submission_pin["sha256"]
        or delta["decoded_artifact_sha256"] != patch_hash
        or issued
        != {
            "key": key,
            "submission_sha256": submission_pin["sha256"],
            "decoded_artifact_sha256": patch_hash,
            "model_requests": 0,
        }
    ):
        raise ValueError("Recovered grade does not belong to the exact saved patch and private verifier")
    records = {**completed.records, ("task", key): {**original, "record": {**record, "grade": delta["grade"]}}}
    summary = retention_result_summary(replace(completed, records=records), task_ids)
    expected_original = {
        **summary,
        "task_rewards": dict(summary["task_rewards"]),
        "categories": dict(summary["categories"]),
    }
    del expected_original["task_rewards"][saved["task_id"]]
    category = "passed" if delta["grade"]["reward"] == 1 else "incorrect"
    expected_original["categories"][category] -= 1
    if expected_original["categories"][category] == 0:
        del expected_original["categories"][category]
    expected_original["categories"]["execution_grade"] = 1
    expected_original["failed_task_ids"] = sorted(set(summary["failed_task_ids"]) | {saved["task_id"]})
    if (
        pinned_at(producer.pins["summary"], str(StoragePath(producer.output_path) / "failure_summary.json"))
        != expected_original
    ):
        raise ValueError("Original incomplete retention summary differs from its saved records")
    return summary


@dataclass(frozen=True)
class RecoveredRetentionConfig:
    input_pin: PinnedFile
    original_producer: str
    summary: dict
    output_path: str


def write_recovered_retention(config: RecoveredRetentionConfig) -> None:
    """Write the derived score and its inputs under a new artifact path."""
    source = config.input_pin.read_json()
    output = StoragePath(config.output_path)
    write_once(
        output / "grade-recovery.json",
        {
            "protocol": PROTOCOL,
            "input": asdict(config.input_pin),
            "original_producer": config.original_producer,
            "original_retention": source["retention"],
            "grade_recovery": source["grade_recovery"],
        },
    )
    write_once(output / "failure_summary.json", config.summary)


@dataclass(frozen=True)
class RecoveredSelectionConfig:
    selection: InterruptedSelectionConfig
    input_pin: PinnedFile
    coding_producer: str
    original_retention_producer: str
    recovered_retention_producer: str


def seal_recovered_selection(config: RecoveredSelectionConfig) -> None:
    write_once(
        StoragePath(config.selection.selection.record.output_path) / "completed-producers.json",
        {
            "protocol": PROTOCOL,
            "input": asdict(config.input_pin),
            "coding_producer": config.coding_producer,
            "original_retention_producer": config.original_retention_producer,
            "recovered_retention_producer": config.recovered_retention_producer,
            "calibration_status": "incomplete_infrastructure",
            "signal_gate_passed": None,
            "rl_authorized": False,
        },
    )
    seal_interrupted_selection(config.selection)


def recovered_retention_selection_stages(
    config: dict, input_pin: PinnedFile, original: dict[str, ArtifactStep]
) -> dict[str, ArtifactStep]:
    """Select with completed coding and one verified saved-patch grade recovery."""
    if (
        config != input_pin.read_json()
        or set(config) != {"protocol", "version", "source_config", "coding", "retention", "grade_recovery"}
        or config["protocol"] != PROTOCOL
        or config["version"] != VERSION
        or "evidence" not in config["coding"]
        or set(config["retention"])
        != {"config", "launch_proof", "producer_record", "journal_binding", "summary", "preflight_summary", "attempts"}
        or set(config["grade_recovery"])
        != {
            "manifest",
            "amendment",
            "request",
            "launch",
            "terminal",
            "worker",
            "issuance",
            "audit",
            "issued",
            "delta",
            "summary",
        }
    ):
        raise ValueError("Recovered selection requires its frozen schema and completed coding evidence")
    source = pinned_json(config["source_config"])
    coding = completed_coding_evidence(config, original)
    retained = completed_producer(config["retention"], "2026.10.06.10")
    old_retention = original["retention-sft"]
    evaluation = old_retention.build_config(
        StepContext.for_run(retained.output_path, source["recovery_artifact_prefix"], deps=old_retention.deps)
    )
    baseline = coding.baseline
    task_ids = baseline.record.retention_task_ids
    completed = completed_retention_records(retained, asdict(evaluation), task_ids)
    summary = recovered_retention_summary(retained, completed, task_ids, config["grade_recovery"])
    recovered = ArtifactStep(
        name=f"evals/{PROTOCOL}-retention",
        version=VERSION,
        artifact_type=Artifact,
        deps=(),
        build_config=lambda ctx: RecoveredRetentionConfig(
            input_pin, retained.launch["producer_identity"], summary, ctx.output_path
        ),
        run=write_recovered_retention,
    )
    outputs = post_sft_selection_stages(
        version=VERSION,
        checkpoints=[("sft", coding.model)],
        outputs={"coding-sft": coding.evidence, "retention-sft": recovered},
        panel_sha256=baseline.record.panel_sha256,
        retention=old_retention.deps[0],
        retention_task_ids=task_ids,
        parent_score=baseline.record.parent,
        study=StudyBaseline(PROTOCOL, baseline.record.parent, baseline.original_parent, task_ids),
    )
    selection = outputs["selection"]

    def build_selection(ctx: StepContext) -> RecoveredSelectionConfig:
        return RecoveredSelectionConfig(
            InterruptedSelectionConfig(
                selection.build_config(ctx),
                source["calibration_interruption_uri"],
                source["calibration_interruption_sha256"],
            ),
            input_pin,
            coding.producer.launch["producer_identity"],
            retained.launch["producer_identity"],
            artifact_identity(recovered),
        )

    final = replace(selection, build_config=build_selection, run=seal_recovered_selection)
    return {**outputs, "selection": final, "terminal": final}
