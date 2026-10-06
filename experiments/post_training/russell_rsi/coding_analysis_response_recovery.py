# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse one saved partition after a private response-parser length failure."""

import json
from dataclasses import asdict, dataclass, replace
from functools import partial

from fray.types import Entrypoint, JobRequest
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_analysis_recovery import (
    MERGE_RULE,
    analyze_partitioned_coding_eval_failures,
    partition_analysis_requests,
    partition_evidence_identity,
)
from experiments.post_training.russell_rsi.coding_analysis_worker import (
    WORKER_SETTINGS,
    RegionalPartitionedCodingAnalysisConfig,
    regional_worker_request,
    require_regional_worker_source,
    submit_regional_worker,
    worker_source_files,
)
from experiments.post_training.russell_rsi.completed_partitioned_coding_analysis import (
    completed_partitioned_analysis_stages,
)
from experiments.post_training.russell_rsi.feedback import (
    FeedbackAnalysis,
    PrivateFeedbackAnalysis,
    PrivateSkillEvidence,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256

PROTOCOL = "russell-rsi-coding-analysis-response-recovery-v1"
VERSION = "2026.10.06.16"
AMENDMENT_PROTOCOL = "russell-rsi-analysis-response-length-amendment-v1"
PREDECESSOR_SOURCE = "857f8ac68091f629bcf1e6c3f3c3b751f013e756"
PART_FILES = {"coding-evidence.json", "private-request.json", "private-analysis-issued.json", "private-analysis.json"}


@dataclass(frozen=True)
class ResponseRecoveryConfig:
    worker: RegionalPartitionedCodingAnalysisConfig
    amendment: PinnedFile


def saved_partition_records(config: ResponseRecoveryConfig) -> tuple[dict, dict[str, bytes]]:
    """Bind the complete saved response to the unchanged first partition request."""
    amendment = config.amendment.read_json()
    first, _second = partition_analysis_requests(config.worker.partitioned())
    records = {}
    for name, pin in amendment["part1_files"].items():
        raw = PinnedFile(pin["uri"], pin["sha256"]).read_bytes()
        if pin["uri"] != amendment["predecessor_output_uri"] + "/part-1/" + name or len(raw) != pin["bytes"]:
            raise ValueError("Saved partition file differs from its frozen source")
        records[name] = raw
    evidence = json.loads(records["coding-evidence.json"])
    request = json.loads(records["private-request.json"])
    issued = json.loads(records["private-analysis-issued.json"])
    saved = json.loads(records["private-analysis.json"])
    identity = partition_evidence_identity(first)
    if (
        compact_json_sha256(evidence) != first.binding["partition_sha256"]
        or request != {"binding": first.binding, "request": first.request}
        or issued != {"request_sha256": first.binding["request_sha256"], "evidence_identity": identity}
        or saved["request"] != first.request
        or saved["request_sha256"] != first.binding["request_sha256"]
        or saved["evidence_identity"] != identity
        or saved["response"]["choices"][0]["finish_reason"] != "stop"
        or not 0 < saved["response"]["usage"]["completion_tokens"] <= first.request["max_tokens"]
    ):
        raise ValueError("Saved response differs from its issued unchanged partition request")
    PrivateFeedbackAnalysis.model_validate_json(saved["response"]["choices"][0]["message"]["content"])
    return amendment, records


def run_response_recovery(config: ResponseRecoveryConfig) -> None:
    amendment, records = saved_partition_records(config)
    require_regional_worker_source(
        config.worker.analysis.output_path,
        config.worker.source_files,
        (
            config.worker.input_pin,
            config.worker.evidence,
            config.worker.failure,
            config.worker.manifest,
            config.amendment,
        ),
    )
    directory = StoragePath(config.worker.analysis.output_path)
    part = directory / "part-1"
    part.mkdirs()
    write_once(
        directory / "response-recovery-ancestry.json",
        {
            "amendment": asdict(config.amendment),
            "predecessor_identity": amendment["predecessor_identity"],
            "predecessor_source_head": amendment["predecessor_source_head"],
            "predecessor_output_uri": amendment["predecessor_output_uri"],
            "part1_files": amendment["part1_files"],
            "bounds": amendment["bounds"],
        },
    )
    for name, raw in records.items():
        target = part / name
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("Recovery partition differs from its copied predecessor bytes")
        else:
            target.write_bytes(raw)
    analyze_partitioned_coding_eval_failures(config.worker.partitioned())


def response_recovery_request(config: ResponseRecoveryConfig) -> JobRequest:
    return regional_worker_request(
        config.worker.analysis.output_path, Entrypoint.from_callable(run_response_recovery, args=(config,))
    )


def submit_response_recovery(config: ResponseRecoveryConfig) -> None:
    submit_regional_worker(
        config.worker.analysis.output_path, asdict(config), partial(response_recovery_request, config)
    )


def response_recovery_stages(
    config: dict, input_pin: PinnedFile, original: dict[str, ArtifactStep]
) -> dict[str, ArtifactStep]:
    if (
        config != input_pin.read_json()
        or set(config) != {"protocol", "version", "predecessor_config", "parser_amendment"}
        or config["protocol"] != PROTOCOL
        or config["version"] != VERSION
    ):
        raise ValueError("Response recovery changed its frozen configuration")
    previous_pin = PinnedFile(**config["predecessor_config"])
    previous = previous_pin.read_json()
    validated = completed_partitioned_analysis_stages(previous, previous_pin, original)
    current_predecessor = validated["terminal"]
    amendment_pin = PinnedFile(**config["parser_amendment"])
    amendment = amendment_pin.read_json()
    original_config = PinnedFile(**previous["original_config"]).read_json()
    extraction = PinnedFile(**original_config["extraction_config"]).read_json()
    source = PinnedFile(**extraction["source_config"]).read_json()
    prefix = source["recovery_artifact_prefix"]
    inventory = PinnedFile(**amendment["inventory"]).read_json()
    provenance_pin = inventory["artifacts"]["analysis-worker-provenance.json"]
    provenance = PinnedFile(provenance_pin["uri"], provenance_pin["sha256"]).read_json()
    reservation_pin = inventory["artifacts"]["worker-submission/reservation.json"]
    reservation = PinnedFile(reservation_pin["uri"], reservation_pin["sha256"]).read_json()
    frozen_files = provenance["source_files"]
    if (
        reservation["source_files"] != frozen_files
        or provenance["worker"] != WORKER_SETTINGS
        or provenance["skyrl"]["direct_url"]["vcs_info"]["commit_id"] != MARIN_SKYRL.commit
    ):
        raise ValueError("Response recovery predecessor has different source or runtime provenance")
    predecessor = replace(
        current_predecessor,
        build_config=lambda ctx: replace(current_predecessor.build_config(ctx), source_files=frozen_files),
    )
    old_bound = predecessor.build_config(StepContext.for_run(predecessor.path(prefix), prefix, deps=predecessor.deps))
    if canonical_json(reservation) != canonical_json(asdict(old_bound)):
        raise ValueError("Response recovery reservation differs from the exact historical worker")
    evidence_schema = PrivateSkillEvidence.model_json_schema()["properties"]["evidence"]
    expected_parser = {
        "request_schema_sha256": compact_json_sha256(FeedbackAnalysis.model_json_schema()),
        "response_evidence_min_length": evidence_schema["minLength"],
        "response_evidence_max_length": evidence_schema.get("maxLength"),
        "other_constraints": "unchanged",
    }
    expected_bounds = {
        "prior_generation_requests": 1,
        "maximum_new_generation_requests": 1,
        "maximum_total_generation_requests": 2,
        "max_tokens": 2048,
        "reasoning_effort": "low",
        "retries": 0,
        "merge_rule": MERGE_RULE,
    }
    if (
        amendment["protocol"] != AMENDMENT_PROTOCOL
        or amendment["predecessor_config"] != config["predecessor_config"]
        or amendment["predecessor_source_head"] != PREDECESSOR_SOURCE
        or amendment["predecessor_identity"] != artifact_identity(predecessor)
        or amendment["predecessor_output_uri"] != predecessor.path(prefix)
        or amendment["parser"] != expected_parser
        or amendment["bounds"] != expected_bounds
        or set(amendment["part1_files"]) != PART_FILES
    ):
        raise ValueError("Response recovery changed its parser, request or lifetime generation bounds")
    terminal = PinnedFile(**amendment["terminal"]).read_json()
    foreground = PinnedFile(**amendment["foreground_exit"]).read_json()
    names = {
        ".executor_info",
        ".executor_status",
        "analysis-worker-provenance.json",
        "worker-submission/reservation.json",
    }
    names.update("part-1/" + name for name in PART_FILES)
    if (
        terminal["state"] != "failed"
        or terminal["failure_count"] != 1
        or terminal["preemption_count"] != 0
        or terminal["completed_count"] != 0
        or terminal["task_count"] != 1
        or terminal["tasks"] != [{"id": terminal["job"] + "/0", "state": "failed", "exit_code": 1}]
        or terminal["source_head"] != PREDECESSOR_SOURCE
        or foreground["exit_code"] != 1
        or inventory["complete_recursive_listing"] is not True
        or set(inventory["files"]) != names
        or len(inventory["files"]) != len(names)
        or set(inventory["artifacts"]) != names
        or inventory["prefix"] != amendment["predecessor_output_uri"]
        or inventory["issuance_per_part"] != {"1": 1, "2": 0}
        or inventory["response_per_part"] != {"1": 1, "2": 0}
    ):
        raise ValueError("Response recovery requires exact one-response terminal ancestry and no part2 issuance")
    for name, pin in inventory["artifacts"].items():
        if name.startswith("part-1/"):
            if pin != amendment["part1_files"][name.removeprefix("part-1/")]:
                raise ValueError("Response recovery copy does not bind its terminal inventory")
            continue
        raw = PinnedFile(pin["uri"], pin["sha256"]).read_bytes()
        if pin["uri"] != inventory["prefix"] + "/" + name or len(raw) != pin["bytes"]:
            raise ValueError("Response recovery inventory differs from pinned predecessor bytes")
        if name == ".executor_status" and raw != b"FAILED":
            raise ValueError("Response recovery predecessor is not failed")

    def bound(ctx: StepContext) -> ResponseRecoveryConfig:
        base = predecessor.build_config(ctx)
        return ResponseRecoveryConfig(
            replace(
                base,
                analysis=replace(base.analysis, output_path=ctx.output_path),
                input_pin=input_pin,
                source_files=worker_source_files(),
            ),
            amendment_pin,
        )

    repaired = replace(
        predecessor, name=f"documents/{PROTOCOL}", version=VERSION, build_config=bound, run=submit_response_recovery
    )
    probe = bound(StepContext.for_run("/unused-response-recovery-preflight", prefix, deps=repaired.deps))
    saved_partition_records(probe)
    return {"evidence": validated["evidence"], "analysis": repaired, "terminal": repaired}
