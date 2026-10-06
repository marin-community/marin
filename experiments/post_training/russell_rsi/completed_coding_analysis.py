# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an analysis graph from completed development evidence."""

from dataclasses import asdict

from marin.execution.artifact import Artifact, artifact_record_identity
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_worker import (
    WORKER_SETTINGS,
    RegionalCodingAnalysisConfig,
    submit_regional_coding_analysis,
    worker_source_files,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingAnalysisConfig,
    coding_analysis_request,
    coding_evidence_payload,
)
from experiments.post_training.russell_rsi.completed_sft_selection import (
    EXTRACTION_PROTOCOL,
    completed_coding_extraction_stages,
    pinned_at,
)
from experiments.post_training.russell_rsi.completed_sft_selection import (
    VERSION as EXTRACTION_VERSION,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256

PROTOCOL = "russell-rsi-completed-coding-analysis-v1"
VERSION = "2026.10.06.12"
EVIDENCE_SOURCE_HEAD = "cf51b56e426fb7fea8b53a46dde36550fc3d7f73"
RELAY_JOB = "/muchanem/glm53-relay"
MAXIMUM_FAILED_ROWS = 64
MAXIMUM_EVIDENCE_BYTES = 589824
DECISION_PROTOCOL = "russell-rsi-coding-analysis-budget-amendment-v1"


def completed_evidence_analysis_stages(
    config: dict, input_pin: PinnedFile, expected_evidence: ArtifactStep
) -> dict[str, ArtifactStep]:
    """Adopt a verified completed extractor and schedule only the canonical analyst."""
    if (
        config != input_pin.read_json()
        or set(config)
        != {"protocol", "version", "extraction_config", "evidence_producer", "evidence", "relay_job", "decision"}
        or config["protocol"] != PROTOCOL
        or config["version"] != VERSION
        or config["relay_job"] != RELAY_JOB
    ):
        raise ValueError("Completed coding analysis changed its frozen protocol or relay")
    extraction = PinnedFile(**config["extraction_config"]).read_json()
    if extraction["protocol"] != EXTRACTION_PROTOCOL or extraction["version"] != EXTRACTION_VERSION:
        raise ValueError("Analysis requires the completed v11 extractor")
    source = PinnedFile(**extraction["source_config"]).read_json()
    prefix = source["recovery_artifact_prefix"]
    path = expected_evidence.path(prefix)
    producer = pinned_at(config["evidence_producer"], str(StoragePath(path) / ".artifact.json"))
    identity = artifact_identity(expected_evidence)
    expected_config = expected_evidence.build_config(StepContext.for_run(path, prefix, deps=expected_evidence.deps))
    provenance = producer["provenance"]
    if (
        artifact_record_identity(producer) != identity
        or producer["output_path"] != path
        or canonical_json(producer["config"]) != canonical_json(asdict(expected_config))
        or len(provenance["base_commit"]) < 9
        or not EVIDENCE_SOURCE_HEAD.startswith(provenance["base_commit"])
        or provenance["dirty"] is not False
        or StatusFile(path, worker_id="completed-analysis").status != STATUS_SUCCESS
    ):
        raise ValueError("Analysis requires the exact completed evidence producer")
    evidence = pinned_at(config["evidence"], str(StoragePath(path) / "coding-evidence.json"))
    expected_payload = coding_evidence_payload(expected_config)
    if evidence != expected_payload:
        raise ValueError("Completed evidence differs from its canonical coding archives")
    decision_pin = PinnedFile(**config["decision"])
    decision = decision_pin.read_json()
    original_input = PinnedFile(**decision["original_input"]).read_json()
    request = coding_analysis_request(evidence, MAXIMUM_FAILED_ROWS, MAXIMUM_EVIDENCE_BYTES)
    expected_analysis = {
        "maximum_failed_rows": MAXIMUM_FAILED_ROWS,
        "maximum_evidence_bytes": MAXIMUM_EVIDENCE_BYTES,
        "maximum_output_tokens": request["max_tokens"],
        "reasoning_effort": request["extra_body"]["chat_template_kwargs"]["reasoning_effort"],
        "model_retries": 0,
        "maximum_generation_requests": 1,
        "context_limit": 262144,
        "failed_rows": sum(row["pass_rate"] == 0 for row in evidence["rows"]),
        "evidence_rows": len(evidence["rows"]),
    }
    if (
        set(decision)
        != {
            "protocol",
            "original_input",
            "evidence",
            "evidence_identity",
            "request_sha256",
            "analysis",
            "worker",
            "calibration_status",
            "signal_gate_passed",
            "rl_authorized",
        }
        or decision["protocol"] != DECISION_PROTOCOL
        or original_input != {key: value for key, value in config.items() if key != "decision"}
        or decision["evidence"] != config["evidence"]
        or decision["evidence_identity"] != identity
        or decision["request_sha256"] != compact_json_sha256(request)
        or decision["analysis"] != expected_analysis
        or decision["worker"] != WORKER_SETTINGS
        or decision["calibration_status"] != "incomplete_infrastructure"
        or decision["signal_gate_passed"] is not None
        or decision["rl_authorized"] is not False
    ):
        raise ValueError("Analyst decision changes the frozen request, bounds, or worker")
    adopted = ArtifactStep.adopt(
        f"documents/{PROTOCOL}-completed-evidence",
        VERSION,
        path,
        config={
            "producer_identity": identity,
            "extraction_config": config["extraction_config"],
            "producer": config["evidence_producer"],
            "evidence": config["evidence"],
        },
    )
    analysis = ArtifactStep(
        name=f"documents/{PROTOCOL}",
        version=VERSION,
        artifact_type=Artifact,
        deps=(adopted,),
        build_config=lambda ctx: RegionalCodingAnalysisConfig(
            analysis=CodingAnalysisConfig(
                evidence_path=ctx.artifact_path(adopted),
                evidence_identity=identity,
                relay_job=config["relay_job"],
                output_path=ctx.output_path,
                maximum_failed_rows=MAXIMUM_FAILED_ROWS,
                maximum_evidence_bytes=MAXIMUM_EVIDENCE_BYTES,
            ),
            input_pin=input_pin,
            decision=decision_pin,
            evidence=PinnedFile(**config["evidence"]),
            source_files=worker_source_files(),
        ),
        run=submit_regional_coding_analysis,
    )
    return {"evidence": adopted, "analysis": analysis, "terminal": analysis}


def completed_coding_analysis_stages(
    config: dict, input_pin: PinnedFile, original: dict[str, ArtifactStep]
) -> dict[str, ArtifactStep]:
    """Validate the original coding records before adopting their completed extraction."""
    extraction_pin = PinnedFile(**config["extraction_config"])
    extraction = extraction_pin.read_json()
    expected = completed_coding_extraction_stages(extraction, extraction_pin, original)["terminal"]
    return completed_evidence_analysis_stages(config, input_pin, expected)
