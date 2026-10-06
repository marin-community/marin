# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use two complete preflighted partitions after a failed oversized analysis."""

from dataclasses import replace

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_recovery import (
    PartitionedCodingAnalysisConfig,
    partition_analysis_requests,
)
from experiments.post_training.russell_rsi.coding_analysis_worker import (
    RegionalPartitionedCodingAnalysisConfig,
    submit_regional_partitioned_coding_analysis,
    worker_source_files,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import CodingAnalysisConfig, coding_analysis_request
from experiments.post_training.russell_rsi.completed_coding_analysis import (
    MAXIMUM_EVIDENCE_BYTES,
    MAXIMUM_FAILED_ROWS,
    completed_coding_analysis_stages,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256

PROTOCOL = "russell-rsi-completed-partitioned-coding-analysis-v1"
VERSION = "2026.10.06.14"
CONTEXT_LIMIT = 262144
PREDECESSOR_SOURCE_HEAD = "5489f732a9cd28e46543040db68f17588ea79603"
FAILURE_PROTOCOL = "russell-rsi-coding-analysis-preflight-failure-v1"


def completed_partitioned_analysis_stages(
    config: dict, input_pin: PinnedFile, original: dict[str, ArtifactStep]
) -> dict[str, ArtifactStep]:
    """Validate completed sources without scheduling their old analyst or evaluator."""
    if (
        config != input_pin.read_json()
        or set(config)
        != {
            "protocol",
            "version",
            "original_config",
            "failure",
            "oversize",
            "partition_manifest",
            "maximum_analysis_calls",
        }
        or config["protocol"] != PROTOCOL
        or config["version"] != VERSION
        or config["maximum_analysis_calls"] != 2
    ):
        raise ValueError("Partitioned analysis changed its frozen protocol or two-call limit")
    original_pin = PinnedFile(**config["original_config"])
    previous = original_pin.read_json()
    validated = completed_coding_analysis_stages(previous, original_pin, original)
    evidence = validated["evidence"]
    evidence_identity = PinnedFile(**previous["decision"]).read_json()["evidence_identity"]
    payload = PinnedFile(**previous["evidence"]).read_json()
    request = coding_analysis_request(payload, MAXIMUM_FAILED_ROWS, MAXIMUM_EVIDENCE_BYTES)
    request_sha256 = compact_json_sha256(request)
    failure = PinnedFile(**config["failure"]).read_json()
    if (
        failure["original_config"] != config["original_config"]
        or failure["evidence_sha256"] != previous["evidence"]["sha256"]
        or failure["request_sha256"] != request_sha256
        or failure["status"] != "failed"
        or failure["generation_requests"] != 0
        or failure["issuance_markers"] != 0
        or failure["response_records"] != 0
    ):
        raise ValueError("Partitioned analysis lacks the exact failed unissued predecessor")
    terminal = PinnedFile(**failure["terminal"]).read_json()
    inventory = PinnedFile(**failure["inventory"]).read_json()
    foreground = PinnedFile(**failure["foreground_exit"]).read_json()
    extraction = PinnedFile(**previous["extraction_config"]).read_json()
    source = PinnedFile(**extraction["source_config"]).read_json()
    predecessor_path = validated["terminal"].path(source["recovery_artifact_prefix"])
    names = {
        ".executor_info",
        ".executor_status",
        "analysis-worker-provenance.json",
        "context-preflight/models.json",
        "worker-submission/reservation.json",
    }
    if (
        failure["protocol"] != FAILURE_PROTOCOL
        or failure["error"] != {"type": "KeyError", "field": "max_model_len"}
        or failure["failed_before"] != "private-analysis-issued.json"
        or failure["source_head"] != PREDECESSOR_SOURCE_HEAD
        or terminal["source_head"] != failure["source_head"]
        or foreground["exit_code"] != 1
        or terminal["state"] != "failed"
        or terminal["failure_count"] != 1
        or terminal["preemption_count"] != 0
        or terminal["task_count"] != 1
        or terminal["completed_count"] != 0
        or terminal["tasks"] != [{"id": terminal["job"] + "/0", "state": "failed", "exit_code": 1}]
        or inventory["prefix"] != predecessor_path
        or inventory["complete_recursive_listing"] is not True
        or set(inventory["files"]) != names
        or len(inventory["files"]) != len(names)
        or inventory["issuance_markers"] != 0
        or inventory["response_records"] != 0
        or inventory["artifacts"] != failure["artifacts"]
        or set(failure["artifacts"]) != names
    ):
        raise ValueError("Original analyst terminal or complete inventory contradicts no issuance")
    for name, pin in failure["artifacts"].items():
        raw = PinnedFile(pin["uri"], pin["sha256"]).read_bytes()
        if (
            pin["uri"] != predecessor_path + "/" + name
            or len(raw) != pin["bytes"]
            or (name == ".executor_status" and raw != b"FAILED")
        ):
            raise ValueError("Original analyst metadata differs from its terminal pins")
    oversize = PinnedFile(**config["oversize"]).read_json()
    if (
        oversize["verified"] is not False
        or oversize["analysis_requests_issued"] != 0
        or oversize["evidence_identity"] != evidence_identity
        or oversize["evidence_sha256"] != previous["evidence"]["sha256"]
        or oversize["request_sha256"] != request_sha256
        or oversize["served_model"] != request["model"]
        or oversize["max_output_tokens"] != request["max_tokens"]
        or oversize["context_limit"] != CONTEXT_LIMIT
        or oversize["prompt_tokens"] + request["max_tokens"] <= CONTEXT_LIMIT
        or oversize["tokenizer_evidence"]["method"] != "served_vllm_tokenize_and_chat_render"
        or oversize["tokenizer_evidence"]["direct_server_evidence"]["render_error"]["status"] != 400
    ):
        raise ValueError("Partitioned analysis lacks its exact served full-request overflow")
    manifest = PinnedFile(**config["partition_manifest"])
    settings = CodingAnalysisConfig(
        evidence_path=str(evidence.adopt_source),
        evidence_identity=evidence_identity,
        relay_job=previous["relay_job"],
        output_path="",
        maximum_failed_rows=MAXIMUM_FAILED_ROWS,
        maximum_evidence_bytes=MAXIMUM_EVIDENCE_BYTES,
    )
    partitions = partition_analysis_requests(PartitionedCodingAnalysisConfig(settings, manifest.uri, manifest.sha256))
    if len(partitions) != config["maximum_analysis_calls"]:
        raise ValueError("Partition count differs from the frozen maximum analysis calls")
    analysis = ArtifactStep(
        name=f"documents/{PROTOCOL}",
        version=VERSION,
        artifact_type=Artifact,
        deps=(evidence,),
        build_config=lambda ctx: RegionalPartitionedCodingAnalysisConfig(
            analysis=replace(settings, evidence_path=ctx.artifact_path(evidence), output_path=ctx.output_path),
            input_pin=input_pin,
            failure=PinnedFile(**config["failure"]),
            evidence=PinnedFile(**previous["evidence"]),
            source_files=worker_source_files(),
            manifest=manifest,
        ),
        run=submit_regional_partitioned_coding_analysis,
    )
    return {"evidence": evidence, "analysis": analysis, "terminal": analysis}
