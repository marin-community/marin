# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover token-context overflow with two complete byte-bounded requests."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CodingAnalysisConfig,
    analyze_coding_failures,
    coding_analysis_request,
)
from experiments.post_training.russell_rsi.feedback import PrivateFeedbackAnalysis, generation_feedback
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.sources import compact_json_sha256

PARTITION_PROTOCOL = "complete-failure-partitions-v1"
PARTITION_ALGORITHM = "two_bins_largest_pair_bytes_first_tie_original_index_then_smallest_load_then_bin_index"
MERGE_RULE = "confidence-0.7-max-per-label-top4-confidence-then-label-v1"


@dataclass(frozen=True)
class CodingAnalysisAmendment:
    maximum_evidence_bytes: int
    artifact_name_suffix: str
    preflight_uri: str
    preflight_sha256: str


@dataclass(frozen=True)
class PartitionedCodingAnalysisConfig:
    analysis: CodingAnalysisConfig
    manifest_path: str
    manifest_sha256: str


@dataclass(frozen=True)
class PartitionRequest:
    part: int
    evidence: dict
    request: dict
    binding: dict


def failure_key(row: dict) -> dict:
    return {key: row[key] for key in ("suite", "benchmark_id", "source_sha256")}


def complete_failure_partitions(evidence: dict, full_request: dict) -> tuple[dict, dict]:
    """Balance complete pairs by bytes, then retain their original order."""
    private = json.loads(full_request["messages"][1]["content"])
    pairs = list(zip(private["failures"], private["static_test_evidence"]["items"], strict=True))
    if len(pairs) < 2 or len({tuple(failure_key(row).values()) for row, _item in pairs}) != len(pairs):
        raise ValueError("Two partitions require unique complete failure pairs")
    sizes = [len(json.dumps({"row": row, "static": item}).encode()) for row, item in pairs]
    bins: list[list[int]] = [[], []]
    loads = [0, 0]
    for index in sorted(range(len(pairs)), key=lambda index: (-sizes[index], index)):
        selected = min(range(2), key=lambda part: (loads[part], part))
        bins[selected].append(index)
        loads[selected] += sizes[index]
    partitions = []
    for indices in bins:
        indices.sort()
        partitions.append(
            {
                **evidence,
                "rows": [pairs[index][0] for index in indices],
                "static_test_evidence": {
                    **evidence["static_test_evidence"],
                    "items": [pairs[index][1] for index in indices],
                },
            }
        )
    return partitions[0], partitions[1]


def partition_analysis_requests(config: PartitionedCodingAnalysisConfig) -> tuple[PartitionRequest, PartitionRequest]:
    """Validate both complete requests and both pinned preflights before issuance."""
    manifest = json.loads(pinned_bytes(config.manifest_path, config.manifest_sha256))
    evidence_bytes = StoragePath(prefix_join(config.analysis.evidence_path, "coding-evidence.json")).read_bytes()
    evidence = json.loads(evidence_bytes)
    full_request = coding_analysis_request(
        evidence, config.analysis.maximum_failed_rows, config.analysis.maximum_evidence_bytes
    )
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    if (
        manifest["protocol"] != PARTITION_PROTOCOL
        or manifest["algorithm"] != PARTITION_ALGORITHM
        or manifest["merge_rule"] != MERGE_RULE
        or manifest["evidence_identity"] != config.analysis.evidence_identity
        or manifest["original_evidence_sha256"] != evidence_sha256
        or manifest["original_request_sha256"] != compact_json_sha256(full_request)
        or manifest["maximum_evidence_bytes"] != config.analysis.maximum_evidence_bytes
        or manifest["maximum_failed_rows"] != config.analysis.maximum_failed_rows
        or [part["part"] for part in manifest["partitions"]] != [1, 2]
    ):
        raise ValueError("Coding analysis amendment does not identify the complete original evidence")
    results = []
    for partition, supplied in zip(
        complete_failure_partitions(evidence, full_request), manifest["partitions"], strict=True
    ):
        request = coding_analysis_request(
            partition, config.analysis.maximum_failed_rows, config.analysis.maximum_evidence_bytes
        )
        partition_sha256 = compact_json_sha256(partition)
        request_sha256 = compact_json_sha256(request)
        if (
            supplied["failure_keys"] != [failure_key(row) for row in partition["rows"]]
            or supplied["partition_evidence_sha256"] != partition_sha256
            or supplied["request_sha256"] != request_sha256
        ):
            raise ValueError("Coding analysis partition changes a complete failure pair or its request")
        preflight = json.loads(pinned_bytes(supplied["preflight_uri"], supplied["preflight_sha256"]))
        if (
            preflight["verified"] is not True
            or preflight["request_sha256"] != request_sha256
            or preflight["partition_evidence_sha256"] != partition_sha256
            or preflight["evidence_identity"] != config.analysis.evidence_identity
            or preflight["evidence_sha256"] != evidence_sha256
            or preflight["served_model"] != request["model"]
            or preflight["max_output_tokens"] != request["max_tokens"]
        ):
            raise ValueError("Coding analysis preflight does not identify its complete partition")
        tokenizer = preflight["tokenizer_evidence"]
        direct = tokenizer["direct_server_evidence"]
        if (
            tokenizer["method"] != "served_vllm_tokenize_and_chat_render"
            or direct["render_error"] is not None
            or not direct["render_request_sha256"]
            or not direct["render_response_sha256"]
            or direct["model_max_model_len"] != preflight["context_limit"]
            or direct["tokenizer_max_model_len"] != preflight["context_limit"]
            or not any(
                vector["equals_tokenize"] is True
                and vector["count"] == preflight["prompt_tokens"]
                and vector["sha256"] == tokenizer["token_ids_sha256"]
                for vector in direct["token_vectors"]
            )
        ):
            raise ValueError("Coding analysis preflight lacks matching tokenizer and chat-render evidence")
        if (
            preflight["prompt_tokens"] < 0
            or preflight["prompt_tokens"] + request["max_tokens"] > preflight["context_limit"]
        ):
            raise ValueError("Complete coding analysis partition exceeds the verified token context")
        binding = {
            "evidence_identity": config.analysis.evidence_identity,
            "evidence_sha256": evidence_sha256,
            "partition_manifest_sha256": config.manifest_sha256,
            "partition_sha256": partition_sha256,
            "request_sha256": request_sha256,
            "model": request["model"],
            "part": supplied["part"],
        }
        results.append(PartitionRequest(supplied["part"], partition, request, binding))
    return results[0], results[1]


def partition_directory(config: PartitionedCodingAnalysisConfig, part: int) -> StoragePath:
    return StoragePath(prefix_join(config.analysis.output_path, f"part-{part}"))


def read_bound_record(path: StoragePath, binding: dict) -> dict:
    record = json.loads(path.read_text())
    if record["binding"] != binding:
        raise ValueError("Stored partition analysis has different evidence or request identity")
    return record


def partition_evidence_identity(partition: PartitionRequest) -> str:
    return f"{partition.binding['evidence_identity']}/part-{partition.part}:{compact_json_sha256(partition.binding)}"


async def partition_response(config: PartitionedCodingAnalysisConfig, partition: PartitionRequest) -> dict:
    directory = partition_directory(config, partition.part)
    directory.mkdirs()
    request_path = directory / "private-request.json"
    if request_path.exists():
        if read_bound_record(request_path, partition.binding)["request"] != partition.request:
            raise ValueError("Stored partition request differs from the frozen request")
    else:
        request_path.write_text(json.dumps({"binding": partition.binding, "request": partition.request}) + "\n")
    evidence_path = directory / "coding-evidence.json"
    if evidence_path.exists():
        if compact_json_sha256(json.loads(evidence_path.read_text())) != partition.binding["partition_sha256"]:
            raise ValueError("Stored partition evidence differs from its frozen complete pairs")
    else:
        evidence_path.write_text(json.dumps(partition.evidence) + "\n")
    analysis = CodingAnalysisConfig(
        evidence_path=str(directory),
        evidence_identity=partition_evidence_identity(partition),
        relay_job=config.analysis.relay_job,
        output_path=str(directory),
        maximum_failed_rows=config.analysis.maximum_failed_rows,
        maximum_evidence_bytes=config.analysis.maximum_evidence_bytes,
    )
    await analyze_coding_failures(analysis)
    return json.loads((directory / "private-analysis.json").read_text())["response"]


def merged_partition_feedback(responses: tuple[dict, dict]) -> tuple[PrivateFeedbackAnalysis, dict]:
    """Rank raw evidence without adding confidence or bypassing private review."""
    entries: list[dict[str, Any]] = []
    for part, response in enumerate(responses, 1):
        analysis = PrivateFeedbackAnalysis.model_validate_json(response["choices"][0]["message"]["content"])
        response_sha256 = compact_json_sha256(response)
        for index, entry in enumerate(analysis.skills):
            entries.append(
                {"part": part, "entry_index": index, "response_sha256": response_sha256, **entry.model_dump(mode="json")}
            )
    representatives = {}
    for entry in sorted(entries, key=lambda entry: (-entry["confidence"], entry["part"], entry["entry_index"])):
        if entry["confidence"] >= 0.7:
            representatives.setdefault(entry["skill"], entry)
    selected = sorted(representatives.values(), key=lambda entry: (-entry["confidence"], entry["skill"]))[:4]
    for entry in entries:
        if entry["confidence"] < 0.7:
            entry["disposition"] = "below-confidence-threshold"
        elif representatives[entry["skill"]] is not entry:
            entry["disposition"] = "duplicate-label"
        elif any(chosen is entry for chosen in selected):
            entry["disposition"] = "selected"
        else:
            entry["disposition"] = "four-label-limit"
    merged = PrivateFeedbackAnalysis.model_validate(
        {"skills": [{key: entry[key] for key in ("skill", "confidence", "evidence")} for entry in selected]}
    )
    return merged, {
        "review_status": "raw-unreviewed",
        "confidence_scope": "Uncalibrated model confidence from separate complete partitions",
        "merge_rule": MERGE_RULE,
        "entries": entries,
        "ranked_selected_labels": [entry["skill"] for entry in selected],
    }


async def analyze_partitioned_coding_failures(config: PartitionedCodingAnalysisConfig) -> None:
    partitions = partition_analysis_requests(config)
    # Inspect every issued marker before either new provider request.
    for partition in partitions:
        directory = partition_directory(config, partition.part)
        issued = directory / "private-analysis-issued.json"
        if issued.exists():
            marker = json.loads(issued.read_text())
            if marker != {
                "request_sha256": partition.binding["request_sha256"],
                "evidence_identity": partition_evidence_identity(partition),
            }:
                raise ValueError("Issued partition analysis has different evidence or request identity")
            if not (directory / "private-analysis.json").exists():
                raise ValueError("Partition analysis request outcome is ambiguous. Refuse to issue it again.")
    responses = []
    for partition in partitions:
        response = await partition_response(config, partition)
        responses.append(response)
    analysis, provenance = merged_partition_feedback((responses[0], responses[1]))
    directory = StoragePath(config.analysis.output_path)
    (directory / "private-merged-analysis.json").write_text(
        json.dumps({"partition_manifest_sha256": config.manifest_sha256, **provenance}, sort_keys=True) + "\n"
    )
    (directory / "capabilities.json").write_text(generation_feedback(analysis) + "\n")


def analyze_partitioned_coding_eval_failures(config: PartitionedCodingAnalysisConfig) -> None:
    asyncio.run(analyze_partitioned_coding_failures(config))
