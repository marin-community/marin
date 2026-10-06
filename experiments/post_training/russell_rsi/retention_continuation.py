# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue never-issued retention after a transport failure, reusing completed coding."""

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest
from marin.evaluation.records import read_record as read_evaluation_record
from marin.evaluation.records import record_path
from marin.execution.artifact import read_record
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.pipeline import EvaluationResult
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingEvidenceConfig,
    CodingPanel,
    PanelItem,
    coding_evidence_payload,
)
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal
from experiments.post_training.russell_rsi.interrupted_calibration import (
    InterruptedSelectionConfig,
    coding_attempt,
    retention_journal,
    retention_request,
    seal_interrupted_selection,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    StudyBaseline,
    StudySelectionConfig,
    post_sft_selection_stages,
)
from experiments.post_training.russell_rsi.rollout_eval import DevelopmentEvaluationConfig, run_development_evaluation
from experiments.post_training.russell_rsi.sources import compact_json_sha256

PROTOCOL = "russell-rsi-interrupted-retention-continuation-v1"
FAILURE_PROTOCOL = "russell-rsi-interrupted-retention-launch-failure-v1"
VERSION = "2026.10.06.8"
HASH_REPAIR_PROTOCOL = "russell-rsi-retention-preflight-hash-repair-v1"
HASH_REPAIR_VERSION = "2026.10.06.10"
HASH_FAILURE_SOURCE = "be82779e7e39765054eaf020fe8f1bd247aa34e8"
HASH_FAILURE_SHA256 = "beca078b0dca2480ce0378b0f85cfc0971529e08bdf24669d4c4d8e6a0aecded"
FAILED_SOURCE_COMMIT = "249d57f88b2845e1d82282b6219c8d5bb2f25c4a"
NEVER_ADMITTED_WITNESSES = {"summary", "rejection", "not_found", "empty_controller_prefix", "absent_journal"}


def _pin(config: dict, name: str) -> PinnedFile:
    return PinnedFile(config[f"{name}_uri"], config[f"{name}_sha256"])


@dataclass(frozen=True)
class ContinuationRetentionConfig:
    evaluation: DevelopmentEvaluationConfig
    retention_config: PinnedFile
    failure: PinnedFile


def run_continuation_retention(config: ContinuationRetentionConfig) -> None:
    # Validate both frozen metadata hashes before reservation or model startup.
    retention_config = config.retention_config.read_json()
    config.failure.read_bytes()
    binding = {
        "protocol": retention_config["protocol"],
        "retention_config": asdict(config.retention_config),
        "launch_failure": asdict(config.failure),
        "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    journal = retention_journal(config.evaluation, continuation_binding=binding)
    run_development_evaluation(config.evaluation, journal=journal)


def continuation_retention_request(config: ContinuationRetentionConfig) -> JobRequest:
    """Return the bounded request with its frozen transport amendment."""
    request = retention_request(config.evaluation)
    return replace(request, entrypoint=Entrypoint.from_callable(lambda: run_continuation_retention(config)))


def submit_continuation_retention(config: ContinuationRetentionConfig) -> None:
    current_client().submit(continuation_retention_request(config), adopt_existing=True).wait(raise_on_failure=True)


@dataclass(frozen=True)
class PreparedRetentionContinuation:
    source: dict
    model: ArtifactStep
    coding: ArtifactStep
    coding_evidence: ArtifactStep
    tasks: ArtifactStep
    selection: StudySelectionConfig
    failure: PinnedFile
    retention_config: PinnedFile
    step: ArtifactStep


@dataclass(frozen=True)
class RetentionContinuationSelection:
    selection: InterruptedSelectionConfig
    failure_pin: PinnedFile
    coding_result_pin: PinnedFile
    coding_journal_result_pin: PinnedFile


def seal_retention_continuation(config: RetentionContinuationSelection) -> None:
    write_once(
        StoragePath(config.selection.selection.record.output_path) / "retention-continuation.json",
        {
            "protocol": PROTOCOL,
            "launch_failure": asdict(config.failure_pin),
            "coding_result": asdict(config.coding_result_pin),
            "coding_journal_result": asdict(config.coding_journal_result_pin),
            "coding_reused": True,
            "signal_gate_passed": None,
            "rl_authorized": False,
        },
    )
    seal_interrupted_selection(config.selection)


def prepare_retention_continuation(
    config: dict, source: dict, original: dict[str, ArtifactStep], config_pin: PinnedFile
) -> PreparedRetentionContinuation:
    """Bind only never-issued retention, independently of the coding result."""
    if config != config_pin.read_json():
        raise ValueError("Retention configuration differs from its frozen pin")
    if set(config) != {
        "protocol",
        "version",
        "source_config_uri",
        "source_config_sha256",
        "launch_failure_uri",
        "launch_failure_sha256",
    }:
        raise ValueError("Frozen retention configuration must exclude completed coding pins")
    if config["protocol"] != PROTOCOL or config["version"] != VERSION:
        raise ValueError("Retention continuation requires its separate protocol and version")
    source_pin = _pin(config, "source_config")
    if source != source_pin.read_json():
        raise ValueError("Retention continuation changed the original configuration")
    failure_pin = _pin(config, "launch_failure")
    failure = failure_pin.read_json()
    old_evidence = original["coding-sft"]
    old_coding, model = old_evidence.deps
    old_retention = original["retention-sft"]
    terminal = original["terminal"]
    old_selection = terminal.build_config(
        StepContext.for_run("/metadata-only-selection", source["recovery_artifact_prefix"], deps=terminal.deps)
    ).selection
    if (
        failure["protocol"] != FAILURE_PROTOCOL
        or failure["calibration_status"] != "incomplete_infrastructure"
        or failure["signal_gate_passed"] is not None
        or failure["rl_authorized"] is not False
        or failure["source"]["config"] != asdict(source_pin)
        or failure["source"]["model_identity"] != artifact_identity(model)
        or failure["source"]["coding_identity"] != artifact_identity(old_coding)
        or failure["source"]["retention_identity"] != artifact_identity(old_retention)
        or failure["source"]["panel_sha256"] != old_selection.record.panel_sha256
        or failure["source"]["source_commit"] != FAILED_SOURCE_COMMIT
        or failure["source"]["runtime_commit"] != MARIN_SKYRL.commit
        or source["runtime_commit"] != MARIN_SKYRL.commit
        or failure["retention"] != {"status": "never_issued", "reservations": 0, "submissions": 0, "model_requests": 0}
        or failure["evaluation"]
        != {"version": VERSION, "conditions": ["sft"], "coding_reused": True, "retention_limit": 3}
        or not NEVER_ADMITTED_WITNESSES.issubset(failure["evidence"])
    ):
        raise ValueError("Retention continuation does not bind the original transport failure")
    for pin in failure["evidence"].values():
        PinnedFile(**pin).read_bytes()
    admission = PinnedFile(**failure["evidence"]["summary"]).read_json()
    if (
        admission["protocol"] != "russell-retention-never-admitted-launch-evidence-v1"
        or admission["source"]["config_sha256"] != source_pin.sha256
        or admission["retention_artifact_prefix"] != old_retention.path(source["recovery_artifact_prefix"])
        or admission["conclusions"]["worker_admitted"] is not False
        or admission["conclusions"]["retention_journal_present"] is not False
        or admission["conclusions"]["retention_scientific_slots_issued"] != 0
        or admission["conclusions"]["coding_replay_authorized"] is not False
    ):
        raise ValueError("Retention admission evidence differs from the original rejected worker")
    if failure["coding"] != {"status": "original_v7_pending", "replay_authorized": False}:
        raise ValueError("Retention amendment must preserve the original coding execution")

    def retention_config(ctx: StepContext) -> ContinuationRetentionConfig:
        return ContinuationRetentionConfig(old_retention.build_config(ctx), config_pin, failure_pin)

    retained = replace(
        old_retention,
        name=f"evals/russell-rsi-{PROTOCOL}-sft-retention-development",
        version=VERSION,
        build_config=retention_config,
        run=submit_continuation_retention,
    )
    return PreparedRetentionContinuation(
        source, model, old_coding, old_evidence, old_retention.deps[0], old_selection, failure_pin, config_pin, retained
    )


def prepare_retention_hash_repair(
    config: dict, config_pin: PinnedFile, predecessor: PreparedRetentionContinuation
) -> ArtifactStep:
    """Create fresh retention after the pinned preflight hash failure.

    Selection requires a separate amendment for completed v9 coding and v10 retention.
    """
    if config != config_pin.read_json() or set(config) != {
        "protocol",
        "version",
        "predecessor_config_uri",
        "predecessor_config_sha256",
        "repair_amendment_uri",
        "repair_amendment_sha256",
    }:
        raise ValueError("Hash repair configuration differs from its frozen pin or schema")
    if config["protocol"] != HASH_REPAIR_PROTOCOL or config["version"] != HASH_REPAIR_VERSION:
        raise ValueError("Hash repair requires its separate protocol and version")
    if _pin(config, "predecessor_config") != predecessor.retention_config:
        raise ValueError("Hash repair substituted the predecessor configuration")
    amendment_pin = _pin(config, "repair_amendment")
    amendment = amendment_pin.read_json()
    prefix = predecessor.source["recovery_artifact_prefix"]
    expected_predecessor = {
        "config": asdict(predecessor.retention_config),
        "retention_identity": artifact_identity(predecessor.step),
        "retention_uri": predecessor.step.path(prefix),
        "model_identity": artifact_identity(predecessor.model),
        "tasks_identity": artifact_identity(predecessor.tasks),
        "source_commit": HASH_FAILURE_SOURCE,
        "runtime_commit": MARIN_SKYRL.commit,
    }
    if (
        set(amendment)
        != {
            "protocol",
            "version",
            "calibration_status",
            "signal_gate_passed",
            "rl_authorized",
            "predecessor",
            "failure",
            "repair",
        }
        or amendment["protocol"] != HASH_REPAIR_PROTOCOL
        or amendment["version"] != HASH_REPAIR_VERSION
        or amendment["calibration_status"] != "incomplete_infrastructure"
        or amendment["signal_gate_passed"] is not None
        or amendment["rl_authorized"] is not False
        or amendment["predecessor"] != expected_predecessor
        or amendment["repair"]
        != {
            "preflight_hash": "compact_json_sha256",
            "task_hash": "contract_tasks.digest",
            "scored_tasks_unchanged": True,
            "new_coding_attempts": 0,
        }
        or amendment["failure"]["sha256"] != HASH_FAILURE_SHA256
    ):
        raise ValueError("Hash repair changed the frozen failure, model, tasks, or science")
    failure = PinnedFile(**amendment["failure"]).read_json()
    if (
        failure["status"] != "root-reviewed-failed-before-first-journal-attempt"
        or failure["source_commit"] != HASH_FAILURE_SOURCE
        or failure["artifact_prefix"] != expected_predecessor["retention_uri"]
        or failure["calibration_status"] != "incomplete_infrastructure"
        or failure["signal_gate_passed"] is not None
        or failure["rl_authorized"] is not False
        or any(
            failure[name] != 0
            for name in (
                "preflight_attempt_reservations",
                "preflight_request_issuances",
                "scored_task_attempt_reservations",
                "scored_task_request_issuances",
            )
        )
        or failure["provider_startup_generations"] != "not established; not counted as scored samples"
    ):
        raise ValueError("Hash repair requires failure before any journal attempt or request")
    files = failure["files"]
    expected_files = {
        ".executor_info",
        ".executor_status",
        "journal/binding.json",
        "token-preflight-suite.json",
        "worker-import-provenance.json",
    }
    if len(files) != len(expected_files) or {item["relative_path"] for item in files} != expected_files:
        raise ValueError("Hash repair failure inventory contains an attempt or lacks evidence")
    root = StoragePath(expected_predecessor["retention_uri"])
    actual_files = set()
    for directory, _, names in root.walk():
        for name in names:
            relative = str((directory / name).relative_to(root))
            if relative not in expected_files:
                raise ValueError("Hash repair predecessor output contains an unrecorded attempt or file")
            actual_files.add(relative)
    if actual_files != expected_files:
        raise ValueError("Hash repair predecessor output lacks frozen failure evidence")
    for item in files:
        if item["uri"] != str(StoragePath(expected_predecessor["retention_uri"]) / item["relative_path"]):
            raise ValueError("Hash repair failure file is outside the predecessor output")
        raw = PinnedFile(item["uri"], item["sha256"]).read_bytes()
        if len(raw) != item["size"] or (item["relative_path"] == ".executor_status" and raw != b"FAILED"):
            raise ValueError("Hash repair failure file size or terminal status differs")

    def repair_config(ctx: StepContext) -> ContinuationRetentionConfig:
        previous = predecessor.step.build_config(ctx)
        return ContinuationRetentionConfig(previous.evaluation, config_pin, amendment_pin)

    return replace(predecessor.step, version=HASH_REPAIR_VERSION, build_config=repair_config)


def completed_coding_evidence(
    config: dict,
    *,
    source: dict,
    old_coding: ArtifactStep,
    old_evidence: ArtifactStep,
    model: ArtifactStep,
    old_selection: StudySelectionConfig,
    version: str,
    protocol: str,
    journal_factory: Callable[[Any], AttemptJournal] = coding_attempt,
) -> tuple[ArtifactStep, PinnedFile, PinnedFile]:
    """Validate a completed producer and reuse or extract its canonical coding evidence."""
    result_pin = _pin(config, "coding_result")
    journal_pin = _pin(config, "coding_journal_result")
    record = result_pin.read_json()
    if "path" in record["result"]:
        raise ValueError("Completed coding producer record has an unexpected embedded path")
    result = {"path": record["output_path"], **record["result"]}
    if StatusFile(result["path"], worker_id="continuation-preflight").status != STATUS_SUCCESS:
        raise ValueError("Selection requires completed coding")
    expected_config = old_coding.build_config(
        StepContext.for_run(
            record["output_path"],
            source["recovery_artifact_prefix"],
            deps=old_coding.deps,
            runtime_args=old_coding.runtime_args,
        )
    )
    journal = journal_factory(expected_config)
    if (
        record["name"] != old_coding.name
        or record["version"] != old_coding.version
        or record["fingerprint"] != old_coding.fingerprint()
        or canonical_json(record["config"]) != canonical_json(expected_config)
        or str(StoragePath(result_pin.uri).parent) != result["path"]
        or len(result["run_ids"]) != 2
        or len(result["results_paths"]) != 2
        or str(StoragePath(journal_pin.uri)) != str(StoragePath(result["path"]) / "journal/coding/result.json")
        or journal_pin.read_json() != {"binding": journal.binding, "result": result}
        or journal.saved_result() != result
    ):
        raise ValueError("Completed coding producer, model, panels, or journal differ")
    evidence_pair = ("coding_evidence_uri" in config, "coding_evidence_sha256" in config)
    if evidence_pair[0] != evidence_pair[1]:
        raise ValueError("Coding evidence requires both URI and hash")
    panel_value = _pin(source, "panel").read_json()
    panel = CodingPanel(tuple(PanelItem(**item) for item in panel_value["items"]), panel_value["protocols"])
    evidence_config = CodingEvidenceConfig(
        result["records_prefix"],
        tuple(result["run_ids"]),
        tuple(result["results_paths"]),
        artifact_identity(model),
        panel,
        "",
    )
    if evidence_pair[0]:
        evidence_pin = _pin(config, "coding_evidence")
        evidence = evidence_pin.read_json()
        evidence_path = str(StoragePath(evidence_pin.uri).parent)
        producer = read_record(evidence_path)
        if (
            StatusFile(evidence_path, worker_id="continuation-preflight").status != STATUS_SUCCESS
            or producer is None
            or producer.fingerprint != old_evidence.fingerprint()
        ):
            raise ValueError("Completed coding evidence does not match its original producer")
        records = tuple(
            read_evaluation_record(record_path(result["records_prefix"], run_id)).model_dump(mode="json", by_alias=True)
            for run_id in result["run_ids"]
        )
        if (
            evidence["model_identity"] != artifact_identity(model)
            or evidence["panel_sha256"] != old_selection.record.panel_sha256
            or set(evidence["scores"]) != {"humanevalplus", "mbppplus"}
            or tuple(record["results_path"] for record in records) != tuple(result["results_paths"])
            or any(record["group_id"] != result["group_id"] for record in records)
            or evidence["records_sha256"] != [compact_json_sha256(record) for record in records]
            or evidence != coding_evidence_payload(evidence_config)
        ):
            raise ValueError("Completed coding evidence changed its model or panel")
        coding_evidence = ArtifactStep.adopt(
            f"documents/{protocol}-completed-coding",
            version,
            str(StoragePath(evidence_pin.uri).parent),
            config={"producer_identity": artifact_identity(old_evidence), **asdict(evidence_pin)},
        )
    else:
        saved_result = ArtifactStep.adopt(
            f"evals/{protocol}-completed-coding",
            version,
            result["path"],
            kind=EvaluationResult,
            config={"producer_identity": artifact_identity(old_coding), **asdict(result_pin)},
        )

        def bound_evidence(ctx: StepContext):
            return replace(evidence_config, output_path=ctx.output_path)

        coding_evidence = replace(old_evidence, version=version, deps=(saved_result, model), build_config=bound_evidence)
    return coding_evidence, result_pin, journal_pin


def retention_continuation_stages(config: dict, prepared: PreparedRetentionContinuation) -> dict[str, ArtifactStep]:
    """Select only after original coding and the same frozen retention complete."""
    if config["protocol"] != PROTOCOL or config["version"] != VERSION:
        raise ValueError("Retention selection changed the continuation protocol or version")
    if _pin(config, "retention_config") != prepared.retention_config:
        raise ValueError("Selection substituted a different retention configuration")
    source, old_selection = prepared.source, prepared.selection
    old_evidence, old_coding, model, retention = (
        prepared.coding_evidence,
        prepared.coding,
        prepared.model,
        prepared.tasks,
    )
    failure_pin = prepared.failure
    retention_path = prepared.step.path(source["recovery_artifact_prefix"])
    if StatusFile(retention_path, worker_id="continuation-preflight").status != STATUS_SUCCESS:
        raise ValueError("Selection requires completed frozen retention")
    retention_record = read_record(retention_path)
    if retention_record is None or retention_record.fingerprint != prepared.step.fingerprint():
        raise ValueError("Completed retention does not match its frozen continuation producer")
    coding_evidence, result_pin, journal_pin = completed_coding_evidence(
        config,
        source=source,
        old_coding=old_coding,
        old_evidence=old_evidence,
        model=model,
        old_selection=old_selection,
        version=VERSION,
        protocol=PROTOCOL,
    )
    study = StudyBaseline(
        PROTOCOL, old_selection.record.parent, old_selection.original_parent, old_selection.record.retention_task_ids
    )
    outputs = post_sft_selection_stages(
        version=VERSION,
        checkpoints=[("sft", model)],
        outputs={"coding-sft": coding_evidence, "retention-sft": prepared.step},
        panel_sha256=old_selection.record.panel_sha256,
        retention=retention,
        retention_task_ids=old_selection.record.retention_task_ids,
        parent_score=old_selection.record.parent,
        study=study,
    )
    selection = outputs["selection"]

    def selection_config(ctx: StepContext) -> RetentionContinuationSelection:
        return RetentionContinuationSelection(
            InterruptedSelectionConfig(
                selection.build_config(ctx),
                source["calibration_interruption_uri"],
                source["calibration_interruption_sha256"],
            ),
            failure_pin,
            result_pin,
            journal_pin,
        )

    final = replace(selection, build_config=selection_config, run=seal_retention_continuation)
    return {**outputs, "selection": final, "terminal": final}
