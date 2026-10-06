# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace a coding transport failure without repeating completed retention."""

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.external_dependencies import EVALCHEMY, MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.pipeline import EvalStepConfig, EvaluationResult
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingEvidenceConfig, CodingPanel, PanelItem
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal
from experiments.post_training.russell_rsi.interrupted_calibration import (
    CODING_TRANSPORT_RETRY_BUDGET,
    InterruptedSelectionConfig,
    coding_attempt,
    run_foreground_coding,
    seal_interrupted_selection,
)
from experiments.post_training.russell_rsi.launch_post_teacher_sft import StudyBaseline, post_sft_selection_stages
from experiments.post_training.russell_rsi.retention_continuation import (
    FAILED_SOURCE_COMMIT,
    PreparedRetentionContinuation,
    completed_coding_evidence,
)

PROTOCOL = "russell-rsi-coding-transport-replacement-v1"
AMENDMENT_PROTOCOL = "russell-rsi-coding-transport-amendment-v1"
SELECTION_PROTOCOL = "russell-rsi-coding-replacement-selection-v1"
VERSION = "2026.10.06.9"
REPLACEMENT_SETTINGS = {
    "version": VERSION,
    "conditions": ["sft"],
    "coding_limit": 32,
    "endpoint_route": "capability",
    "worker_preflight": "models_exact_id",
    "max_retries": 1,
    "transport_retry_budget": CODING_TRANSPORT_RETRY_BUDGET,
    "scored_samples_per_case": 1,
    "transport_retries": "unchanged",
}


@dataclass(frozen=True)
class ReplacementCodingConfig:
    evaluation: EvalStepConfig
    coding_config: PinnedFile
    amendment: PinnedFile


def replacement_transport_binding(config: ReplacementCodingConfig) -> dict:
    return {
        "protocol": PROTOCOL,
        "coding_config": asdict(config.coding_config),
        "amendment": asdict(config.amendment),
        "settings": REPLACEMENT_SETTINGS,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def replacement_coding_attempt(config: ReplacementCodingConfig) -> AttemptJournal:
    return coding_attempt(config.evaluation, transport_binding=replacement_transport_binding(config))


def run_replacement_coding(config: ReplacementCodingConfig) -> EvaluationResult:
    config.coding_config.read_bytes()
    config.amendment.read_bytes()
    return run_foreground_coding(config.evaluation, transport_binding=replacement_transport_binding(config))


@dataclass(frozen=True)
class PreparedCodingReplacement:
    retention: PreparedRetentionContinuation
    config_pin: PinnedFile
    amendment: PinnedFile
    coding: ArtifactStep
    evidence: ArtifactStep


def prepare_coding_replacement(
    config: dict, config_pin: PinnedFile, retention: PreparedRetentionContinuation
) -> PreparedCodingReplacement:
    """Require reviewed zero benchmark issuance and preserve the original model and panels."""
    if config != config_pin.read_json() or set(config) != {
        "protocol",
        "version",
        "source_config_uri",
        "source_config_sha256",
        "transport_amendment_uri",
        "transport_amendment_sha256",
    }:
        raise ValueError("Coding replacement differs from its frozen configuration")
    if config["protocol"] != PROTOCOL or config["version"] != VERSION:
        raise ValueError("Coding replacement requires its separate protocol and version")
    source_pin = PinnedFile(config["source_config_uri"], config["source_config_sha256"])
    amendment_pin = PinnedFile(config["transport_amendment_uri"], config["transport_amendment_sha256"])
    amendment = amendment_pin.read_json()
    if source_pin.read_json() != retention.source:
        raise ValueError("Coding replacement changed the original study")
    if (
        amendment["protocol"] != AMENDMENT_PROTOCOL
        or amendment["calibration_status"] != "incomplete_infrastructure"
        or amendment["signal_gate_passed"] is not None
        or amendment["rl_authorized"] is not False
        or amendment["source"]
        != {
            "config": asdict(source_pin),
            "source_commit": FAILED_SOURCE_COMMIT,
            "runtime_commit": MARIN_SKYRL.commit,
            "evalchemy_commit": EVALCHEMY.commit,
            "model_identity": artifact_identity(retention.model),
            "coding_identity": artifact_identity(retention.coding),
            "coding_panel_sha256": retention.selection.record.panel_sha256,
        }
        or amendment["original"]
        != {
            "status": "failed_infrastructure",
            "startup_probe_requests": 1,
            "benchmark_generation_requests": 0,
        }
        or amendment["replacement"] != REPLACEMENT_SETTINGS
        or amendment["retention"]
        != {
            "config": asdict(retention.retention_config),
            "fingerprint": retention.step.fingerprint(),
        }
        or set(amendment["evidence"]) != {"terminal_inventory", "child_config", "server_metrics", "issuance_attribution"}
    ):
        raise ValueError("Coding transport amendment changed the frozen study or issuance")
    for pin in amendment["evidence"].values():
        PinnedFile(**pin).read_bytes()
    attribution = PinnedFile(**amendment["evidence"]["issuance_attribution"]).read_json()
    if (
        attribution["protocol"] != "russell-rsi-coding-issuance-attribution-v1"
        or attribution["status"] != "root-reviewed"
        or attribution["benchmark_generation_requests"] != 0
        or attribution["startup_probe_requests"] != 1
        or attribution["scored_samples"] != 0
        or attribution["calibration_status"] != "incomplete_infrastructure"
        or attribution["signal_gate_passed"] is not None
        or attribution["rl_authorized"] is not False
    ):
        raise ValueError("Coding issuance evidence does not authorize a transport replacement")

    def coding_config(ctx: StepContext) -> ReplacementCodingConfig:
        original = retention.coding.build_config(ctx)
        return ReplacementCodingConfig(replace(original, version=VERSION), config_pin, amendment_pin)

    coding = replace(
        retention.coding,
        name=f"evals/{PROTOCOL}-sft-coding",
        version=VERSION,
        build_config=coding_config,
        run=run_replacement_coding,
    )
    panel_value = PinnedFile(retention.source["panel_uri"], retention.source["panel_sha256"]).read_json()
    panel = CodingPanel(tuple(PanelItem(**item) for item in panel_value["items"]), panel_value["protocols"])

    def evidence_config(ctx: StepContext):
        if ctx.is_fingerprint:
            return {"evaluation": artifact_identity(coding), "model": artifact_identity(retention.model), "panel": panel}
        result = ctx.resolved(coding)
        return CodingEvidenceConfig(
            result.records_prefix,
            result.run_ids,
            result.results_paths,
            artifact_identity(retention.model),
            panel,
            ctx.output_path,
        )

    evidence = replace(
        retention.coding_evidence,
        version=VERSION,
        deps=(coding, retention.model),
        build_config=evidence_config,
    )
    return PreparedCodingReplacement(retention, config_pin, amendment_pin, coding, evidence)


@dataclass(frozen=True)
class ReplacementSelectionConfig:
    selection: InterruptedSelectionConfig
    coding_config: PinnedFile
    amendment: PinnedFile
    coding_result: PinnedFile
    coding_journal_result: PinnedFile
    retention_config: PinnedFile


def seal_replacement_selection(config: ReplacementSelectionConfig) -> None:
    write_once(
        StoragePath(config.selection.selection.record.output_path) / "coding-transport-replacement.json",
        {
            "protocol": SELECTION_PROTOCOL,
            "coding_source": "replacement_v9",
            "retention_source": "frozen_v8",
            "coding_config": asdict(config.coding_config),
            "amendment": asdict(config.amendment),
            "coding_result": asdict(config.coding_result),
            "coding_journal_result": asdict(config.coding_journal_result),
            "retention_config": asdict(config.retention_config),
            "calibration_status": "incomplete_infrastructure",
            "signal_gate_passed": None,
            "rl_authorized": False,
        },
    )
    seal_interrupted_selection(config.selection)


def replacement_selection_stages(config: dict, prepared: PreparedCodingReplacement) -> dict[str, ArtifactStep]:
    """Join completed replacement coding with the exact completed retention producer."""
    retention = prepared.retention
    required = {
        "protocol",
        "version",
        "coding_config_uri",
        "coding_config_sha256",
        "retention_config_uri",
        "retention_config_sha256",
        "coding_result_uri",
        "coding_result_sha256",
        "coding_journal_result_uri",
        "coding_journal_result_sha256",
    }
    if set(config) not in (required, required | {"coding_evidence_uri", "coding_evidence_sha256"}):
        raise ValueError("Replacement selection configuration has unexpected keys")
    if (
        config["protocol"] != SELECTION_PROTOCOL
        or config["version"] != VERSION
        or PinnedFile(config["coding_config_uri"], config["coding_config_sha256"]) != prepared.config_pin
        or PinnedFile(config["retention_config_uri"], config["retention_config_sha256"]) != retention.retention_config
    ):
        raise ValueError("Replacement selection changed a frozen producer")
    retention_path = retention.step.path(retention.source["recovery_artifact_prefix"])
    record = read_record(retention_path)
    if (
        StatusFile(retention_path, worker_id="replacement-preflight").status != STATUS_SUCCESS
        or record is None
        or record.fingerprint != retention.step.fingerprint()
    ):
        raise ValueError("Selection requires the completed frozen retention")
    evidence, result_pin, journal_pin = completed_coding_evidence(
        config,
        source=retention.source,
        old_coding=prepared.coding,
        old_evidence=prepared.evidence,
        model=retention.model,
        old_selection=retention.selection,
        version=VERSION,
        protocol=PROTOCOL,
        journal_factory=replacement_coding_attempt,
    )
    baseline = retention.selection
    completed_retention = ArtifactStep.adopt(
        f"evals/{PROTOCOL}-completed-retention",
        VERSION,
        retention_path,
        kind=retention.step.artifact_type,
        config={"producer_identity": artifact_identity(retention.step), **asdict(retention.retention_config)},
    )
    outputs = post_sft_selection_stages(
        version=VERSION,
        checkpoints=[("sft", retention.model)],
        outputs={"coding-sft": evidence, "retention-sft": completed_retention},
        panel_sha256=baseline.record.panel_sha256,
        retention=retention.tasks,
        retention_task_ids=baseline.record.retention_task_ids,
        parent_score=baseline.record.parent,
        study=StudyBaseline(
            PROTOCOL, baseline.record.parent, baseline.original_parent, baseline.record.retention_task_ids
        ),
    )
    selection = outputs["selection"]

    def selection_config(ctx: StepContext) -> ReplacementSelectionConfig:
        return ReplacementSelectionConfig(
            InterruptedSelectionConfig(
                selection.build_config(ctx),
                retention.source["calibration_interruption_uri"],
                retention.source["calibration_interruption_sha256"],
            ),
            prepared.config_pin,
            prepared.amendment,
            result_pin,
            journal_pin,
            retention.retention_config,
        )

    final = replace(selection, build_config=selection_config, run=seal_replacement_selection)
    return {**outputs, "selection": final, "terminal": final}
