# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seal exact saved-submission grades without another calibration rollout."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

from marin.execution.artifact import Artifact, artifact_record_identity
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import STATUS_FAILED, STATUS_SUCCESS
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.launch_incumbent_bank_trial import (
    calibration_decision,
    incumbent_bank_workflow,
)

PROTOCOL = "incumbent-calibration-grade-recovery-seal-v1"


@dataclass(frozen=True)
class GradeRecoverySealConfig:
    evidence: dict
    output_path: str


def require_failed_calibration_decision(pins: dict, expected: ArtifactStep) -> None:
    """Check failed step metadata without treating it as a completed producer."""
    info_pin = PinnedFile(**pins["info"])
    info = info_pin.read_json()
    status = PinnedFile(**pins["status"])
    root = info["output_path"]
    if (
        info_pin.uri != prefix_join(root, ".executor_info")
        or status.uri != prefix_join(root, ".executor_status")
        or status.read_bytes().decode().strip() != STATUS_FAILED
        or info["name"] != expected.name
        or canonical_json(info["config"]) != canonical_json(expected.lower().hash_attrs)
    ):
        raise ValueError("Grade recovery requires the exact failed original decision")


def recovered_calibration_summary(
    *, binding: dict, original: dict, audit: dict, manifest: dict, deltas: list[dict]
) -> dict:
    """Preserve every original grade and append only the three recovered grades."""
    keys = set(binding["attempts"]["task"])
    records = {entry["key"]: entry for entry in audit["records"]}
    slots = {entry["key"]: entry for entry in manifest["slots"]}
    recovered = {entry["key"]: entry for entry in deltas}
    tasks = {key.rsplit("/", 1)[0] for key in keys}
    if (
        len(keys) != 256
        or len(tasks) != 32
        or keys != {f"{task}/{index}" for task in tasks for index in range(8)}
        or len(audit["records"]) != len(records)
        or set(records) != keys
        or len(slots) != 3
        or len(manifest["slots"]) != 3
        or len(deltas) != 3
        or set(recovered) != set(slots)
        or set(audit["missing_keys"]) != set(slots)
        or binding["config"]["samples_per_task"] != 8
    ):
        raise ValueError("Grade recovery changed its exact task and missing-slot census")
    original_rewards = {task: [] for task in tasks}
    categories, startup = Counter(), Counter()
    original_failed = set()
    for key in sorted(keys):
        entry = records[key]
        grade = entry["grade"]
        reward = grade["reward"]
        task = key.rsplit("/", 1)[0]
        if len(entry["result_sha256"]) != 64 or entry["result_size"] <= 0:
            raise ValueError("Original calibration record has no exact byte witness")
        startup.update(entry["startup_counts"])
        category = (
            f"execution_{entry['interrupted_operation'] or 'ungraded'}"
            if grade["status"] != "graded"
            else "passed" if reward > 0 else "incorrect"
        )
        if entry["original_category"] != category:
            raise ValueError("Original record category differs from its preserved interruption and grade")
        categories[category] += 1
        if key in slots:
            if grade["status"] != "unavailable" or reward is not None or grade["passed"] is not None:
                raise ValueError("Recovery slot was not an original missing grade")
            original_failed.add(task)
            continue
        if grade["status"] != "graded" or type(reward) not in (int, float) or reward not in (0, 1):
            raise ValueError("Recovery would replace an invalid original grade")
        original_rewards[task].append(reward)
        if reward <= 0:
            original_failed.add(task)
    original_expected = {
        **{
            key: binding["config"][key]
            for key in ("model_identity", "tasks_path", "tasks_identity", "samples_per_task", "startup_attempts")
        },
        "count": len(tasks),
        "startup_counts": dict(startup),
        "informative_groups": sum(len(set(values)) > 1 for values in original_rewards.values()),
        "task_rewards": original_rewards,
        "categories": dict(categories),
        "failed_task_ids": sorted(original_failed),
    }
    expected = {
        **original_expected,
        "task_rewards": {task: sorted(values) for task, values in original_rewards.items()},
    }
    for summary in (original, audit["original_summary"]):
        comparable = {
            **summary,
            "task_rewards": {task: sorted(values) for task, values in summary["task_rewards"].items()},
        }
        if comparable != expected:
            raise ValueError("Full original-record audit differs from the pinned calibration summary")
    rewards = {task: list(values) for task, values in original["task_rewards"].items()}
    grades = {key: entry["grade"] for key, entry in records.items()}
    for key in sorted(slots):
        slot, delta = slots[key], recovered[key]
        grade = delta["grade"]
        reward = None if grade is None else grade["reward"]
        if (
            slot["task_id"] != key.rsplit("/", 1)[0]
            or key not in binding["attempts"]["task"]
            or records[key]["result_sha256"] != manifest["inputs"][slot["result_input"]]["sha256"]
            or records[key]["result_size"] != manifest["inputs"][slot["result_input"]]["size"]
            or delta["original_result_sha256"] != records[key]["result_sha256"]
            or delta["submission_sha256"] != manifest["inputs"][slot["submission_input"]]["sha256"]
            or delta["decoded_artifact_sha256"] != slot["decoded_artifact_sha256"]
            or len(slot["verifier_sha256"]) != 64
            or slot["private_grader_timeout"] <= 0
            or grade is None
            or grade["status"] != "graded"
            or type(reward) not in (int, float)
            or not math.isfinite(reward)
            or reward not in (0, 1)
            or delta["grade_calls"] != 1
            or delta["execution_error"] is not None
            or delta["accepted_grade"] is not True
            or delta["model_requests"] != 0
            or delta["tokenizer_requests"] != 0
            or delta["original_journal_modified"] is not False
        ):
            raise ValueError("Recovered grade differs from its exact saved submission or bounded private grader")
        grades[key] = grade
        rewards[slot["task_id"]].append(reward)
    categories = Counter("passed" if grade["reward"] > 0 else "incorrect" for grade in grades.values())
    return {
        **original,
        "task_rewards": rewards,
        "categories": dict(categories),
        "informative_groups": sum(len(set(values)) > 1 for values in rewards.values()),
        "failed_task_ids": sorted({key.rsplit("/", 1)[0] for key, grade in grades.items() if grade["reward"] <= 0}),
    }


def validated_grade_recovery(evidence: dict) -> tuple[dict, dict, dict]:
    """Check the exact completed CPU recovery and its immutable input lineage."""
    original_pin = PinnedFile(**evidence["original_config"])
    original = original_pin.read_json()
    stages = incumbent_bank_workflow(original, "calibrate")
    calibration_pin = PinnedFile(**evidence["original_calibration"]["producer"])
    calibration = calibration_pin.read_json()
    status = PinnedFile(**evidence["original_calibration"]["status"])
    root = calibration["output_path"]
    if (
        artifact_record_identity(calibration) != artifact_identity(stages["calibration"])
        or calibration_pin.uri != prefix_join(root, ".artifact.json")
        or status.uri != prefix_join(root, ".executor_status")
        or status.read_bytes().decode().strip() != STATUS_SUCCESS
        or evidence["binding"]["uri"] != prefix_join(root, "journal/binding.json")
        or evidence["original_summary"]["uri"] != prefix_join(root, "failure_summary.json")
    ):
        raise ValueError("Recovery requires the exact successful original calibration")
    require_failed_calibration_decision(evidence["failed_decision"], stages["decision"])
    manifest_pin = PinnedFile(**evidence["input_manifest"])
    manifest = manifest_pin.read_json()
    binding = PinnedFile(**evidence["binding"]).read_json()
    summary = PinnedFile(**evidence["original_summary"]).read_json()
    for key, pin in (
        ("config", evidence["original_config"]),
        ("producer", evidence["original_calibration"]["producer"]),
        ("status", evidence["original_calibration"]["status"]),
        ("binding", evidence["binding"]),
        ("summary", evidence["original_summary"]),
    ):
        if {name: manifest["inputs"][key][name] for name in ("uri", "sha256")} != pin:
            raise ValueError("Recovery manifest cites different original input bytes")
    if (
        manifest["journal_prefix"] != prefix_join(root, "journal")
        or manifest["inputs"]["parquet"]["sha256"] != binding["parquet_sha256"]
        or manifest["inputs"]["parquet"]["sha256"] != original["bank"]["identity_config"]["train_sha256"]
    ):
        raise ValueError("Recovery task and journal pins differ from the original calibration")
    for slot in manifest["slots"]:
        task, sample = slot["key"].rsplit("/", 1)
        if task != slot["task_id"] or sample not in {str(index) for index in range(8)}:
            raise ValueError("Recovery slot differs from its original task attempt")
        for key, name in (
            ("reservation_input", "reservation.json"),
            ("result_input", "result.json"),
            ("submission_input", "submission.json"),
        ):
            pin = manifest["inputs"][slot[key]]
            if pin["uri"] != prefix_join(manifest["journal_prefix"], f"task/{slot['key']}/{name}") or pin["size"] <= 0:
                raise ValueError("Recovery input differs from its exact saved task attempt")
    request_pin = PinnedFile(**evidence["request"])
    request = request_pin.read_json()
    PinnedFile(**evidence["worker"]).read_bytes()
    source = PinnedFile(**evidence["source_review"]).read_json()
    job = PinnedFile(**evidence["completed_job"]).read_json()
    launch = PinnedFile(**evidence["launch_record"]).read_json()
    output = manifest["output_prefix"]
    if (
        manifest["maximum_grade_calls"] != 3
        or manifest["model_requests"] != 0
        or manifest["grade_retries"] != 0
        or binding["protocol"] != "russell-calibration-journal-v1"
        or binding["config"] != calibration["config"]
        or manifest["model_identity"] != binding["config"]["model_identity"]
        or manifest["tasks_identity"] != binding["config"]["tasks_identity"]
        or manifest["runtime_bundle"] != original["runtime_bundle"]
        or manifest["runtime_source"] != original["runtime_commit"]
    ):
        raise ValueError("Recovery manifest differs from the original journal or runtime")
    if (
        request["input_manifest_sha256"] != manifest_pin.sha256
        or request["output_prefix"] != output
        or request["worker_sha256"] != evidence["worker"]["sha256"]
        or request["source_head"] != manifest["source_commit"]
        or request["resources"]["gpu"] != 0
        or request["priority"] != "batch"
        or any(
            request["retries"][key] != 0
            for key in ("max_retries_failure", "max_retries_preemption", "max_task_failures")
        )
    ):
        raise ValueError("Recovery request differs from the bounded CPU package")
    if (
        source["status"] != "approved"
        or source["source_head"] != manifest["source_commit"]
        or source["worker_sha256"] != request["worker_sha256"]
        or source["manifest_sha256"] != manifest_pin.sha256
        or source["request_sha256"] != request_pin.sha256
        or source["maximum_grade_calls"] != 3
        or source["automatic_retries"] != 0
        or source["original_journal_writes"] != 0
        or source["expected_original_grades"] != 253
    ):
        raise ValueError("Recovery source review differs from the exact frozen package")
    if (
        job["state"] != "JOB_STATE_SUCCEEDED"
        or job["exit_code"] != 0
        or job["failure_count"] != 0
        or job["preemption_count"] != 0
        or job["task_count"] != 1
        or job["cluster"] != request["resources"]["target_cluster"]
    ):
        raise ValueError("Recovery job did not complete exactly once without failures")
    if (
        launch["job"] != job["job_id"]
        or launch["job_name"] != request["job_name"]
        or job["job_id"].rsplit("/", 1)[1] != request["job_name"]
        or launch["request_sha256"] != request_pin.sha256
        or launch["input_manifest_sha256"] != manifest_pin.sha256
        or launch["worker_sha256"] != request["worker_sha256"]
        or launch["output_prefix"] != output
        or launch["resources"] != request["resources"]
        or launch["retries"] != request["retries"]
        or launch["typed_job_request_validated"] is not True
        or launch["submissions"] != 1
        or launch["model_requests"] != 0
    ):
        raise ValueError("Grade recovery lacks its exact completed CPU job and no-retry source binding")
    output_files = (
        ("issuance", "issuance.json"),
        ("original_record_audit", "original-record-audit.json"),
        ("cpu_summary", "summary.json"),
    )
    for key, name in output_files:
        if evidence[key]["uri"] != prefix_join(output, name):
            raise ValueError("Recovery output is outside its exact CPU job")
    issuance = PinnedFile(**evidence["issuance"]).read_json()
    if (
        issuance["manifest_sha256"] != manifest_pin.sha256
        or issuance["maximum_grade_calls"] != 3
        or issuance["model_requests"] != 0
        or issuance["tokenizer_requests"] != 0
    ):
        raise ValueError("Recovery issuance changed its input manifest or request budget")
    modules = {
        "rolloutengine.grading": "lib/rolloutengine/src/rolloutengine/grading.py",
        "rolloutengine.machines": "lib/rolloutengine/src/rolloutengine/machines.py",
        "taskcompendium.models": "lib/taskcompendium/src/taskcompendium/models.py",
        "shellbox.backends.qemu.machine": "lib/shellbox/src/shellbox/backends/qemu/machine.py",
    }
    provenance = {entry["module"]: entry for entry in issuance["source_provenance"]}
    if set(provenance) != set(modules) or any(
        provenance[name]["sha256"] != manifest["source_module_hashes"][path]
        or not provenance[name]["path"].endswith("/" + path)
        for name, path in modules.items()
    ):
        raise ValueError("Recovery ran a different grading or task-verifier source")
    audit = PinnedFile(**evidence["original_record_audit"]).read_json()
    deltas = []
    slots = {slot["key"]: slot for slot in manifest["slots"]}
    for delta_pins in evidence["deltas"]:
        delta = PinnedFile(**delta_pins["delta"]).read_json()
        issued = PinnedFile(**delta_pins["issued"]).read_json()
        key = delta["key"]
        directory = prefix_join(output, f"grade-delta/{key}")
        if (
            key not in slots
            or delta_pins["delta"]["uri"] != prefix_join(directory, "delta.json")
            or delta_pins["issued"]["uri"] != prefix_join(directory, "issued.json")
            or issued
            != {
                "key": key,
                "submission_sha256": manifest["inputs"][slots[key]["submission_input"]]["sha256"],
                "decoded_artifact_sha256": slots[key]["decoded_artifact_sha256"],
                "maximum_grade_calls": 1,
                "model_requests": 0,
            }
        ):
            raise ValueError("Recovered grade lacks its exact one-call issuance")
        deltas.append(delta)
    cpu = PinnedFile(**evidence["cpu_summary"]).read_json()
    expected_cpu = {
        "protocol": manifest["protocol"],
        "original_records_audited": 256,
        "grade_calls": 3,
        "model_requests": 0,
        "tokenizer_requests": 0,
        "original_journal_modified": False,
        "slots": [
            {
                "key": delta["key"],
                "status": delta["grade"]["status"] if delta["grade"] else "execution_error",
                "grade_calls": delta["grade_calls"],
                "execution_error": delta["execution_error"],
                "accepted_grade": delta["accepted_grade"],
            }
            for delta in deltas
        ],
    }
    if cpu != expected_cpu:
        raise ValueError("Recovery CPU summary differs from its complete three grade deltas")
    recovered = recovered_calibration_summary(
        binding=binding, original=summary, audit=audit, manifest=manifest, deltas=deltas
    )
    decision_step = stages["decision"]
    calibration_suffix = f"/{stages['calibration'].name}/{stages['calibration'].version}"
    if not root.endswith(calibration_suffix):
        raise ValueError("Original calibration is outside its canonical artifact path")
    original_prefix = root.removesuffix(calibration_suffix)
    bound = decision_step.build_config(
        StepContext.for_run(decision_step.path(original_prefix), original_prefix, deps=decision_step.deps)
    )
    # Bind the durable JSON bytes written by write_once, including its final newline.
    raw = (json.dumps(recovered, sort_keys=True, indent=2) + "\n").encode()
    decision = calibration_decision(
        summary=recovered,
        summary_sha256=hashlib.sha256(raw).hexdigest(),
        plan=bound.plan,
        families=bound.families,
        targeted_tasks=bound.targeted_tasks,
        qualification_sha256=bound.qualification_sha256,
        lineage_review_sha256=bound.lineage_review_sha256,
    )
    return (
        recovered,
        decision,
        {
            "protocol": PROTOCOL,
            "evidence": evidence,
            "preserved_grade_count": 253,
            "recovered_grade_count": 3,
            "original_record_sha256": {entry["key"]: entry["result_sha256"] for entry in audit["records"]},
            "model_requests": 0,
            "grade_retries": 0,
            "original_journal_modified": False,
        },
    )


def seal_grade_recovery(config: GradeRecoverySealConfig) -> None:
    summary, decision, provenance = validated_grade_recovery(config.evidence)
    output = StoragePath(config.output_path)
    if config.output_path == str(StoragePath(config.evidence["original_summary"]["uri"]).parent):
        raise ValueError("Grade recovery cannot replace the original calibration files")
    write_once(output / "failure_summary.json", summary)
    if hashlib.sha256((output / "failure_summary.json").read_bytes()).hexdigest() != decision["summary_sha256"]:
        raise ValueError("Recovery summary serialization differs from the recomputed decision")
    write_once(output / "calibration-decision.json", decision)
    write_once(output / "grade-recovery-provenance.json", provenance)


def grade_recovery_step(evidence: dict, version: str) -> ArtifactStep:
    """Build a local CPU seal with no dependency on the failed original decision."""
    return ArtifactStep(
        name="documents/russell-rsi-incumbent-current-bank-grade-recovery-seal",
        version=version,
        artifact_type=Artifact,
        build_config=lambda ctx: GradeRecoverySealConfig(evidence, ctx.output_path),
        run=seal_grade_recovery,
    )
