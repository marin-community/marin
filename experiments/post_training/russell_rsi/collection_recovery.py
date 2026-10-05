# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate one counted teacher collection amendment without provider requests."""

import hashlib
import json
from dataclasses import asdict, dataclass

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.sources import compact_json_sha256

REASONING_MAPPING_VERSION = "assistant-reasoning-to-reasoning-content-v1"


@dataclass(frozen=True)
class CollectionRecovery:
    predecessor_uri: str
    predecessor_identity: str
    executor_info_sha256: str
    executor_status_sha256: str
    plan_sha256: str
    fatal_marker_sha256: str
    slot_reservation_sha256: str
    preflight_identity: str
    preflight_reservation_sha256: str
    preflight_result_sha256: str
    token_proof_uri: str
    token_proof_sha256: str
    mapping_version: str
    consumed_slot: str


@dataclass(frozen=True)
class StudentContextAmendment:
    record_uri: str
    record_sha256: str
    protocol: str
    previous_context_tokens: int
    context_tokens: int


def student_context_amendment_record(
    amendment: StudentContextAmendment, recovery: CollectionRecovery, plan: dict
) -> dict:
    """Bind the prospective 16K study to its unchanged tasks and counted budget."""
    if (
        amendment.protocol != "champion-rsi-teacher-sixteen-k-context-amendment-v1"
        or amendment.previous_context_tokens != 4096
        or amendment.context_tokens != 16384
    ):
        raise ValueError("Teacher study requires the declared 4096-to-16384 context amendment")
    record = json.loads(pinned_bytes(amendment.record_uri, amendment.record_sha256))
    expected = {
        "protocol": amendment.protocol,
        "previous_context_tokens": amendment.previous_context_tokens,
        "context_tokens": amendment.context_tokens,
        "predecessor_identity": recovery.predecessor_identity,
        "predecessor_plan_sha256": recovery.plan_sha256,
        "fatal_marker_sha256": recovery.fatal_marker_sha256,
        "plan_sha256": compact_json_sha256(plan),
        "consumed_slot": "00-0",
        "consumed_trajectories": 1,
        "remaining_trajectories": 19,
        "student_rows": 8,
        "sft_updates": 4,
        "assistant_only_loss": True,
        "full_untruncated_rows": True,
    }
    if record != expected:
        raise ValueError("Teacher context amendment differs from its exact prospective binding")
    return record


@dataclass(frozen=True)
class CollectionRecoveryEvidence:
    preflight: dict
    consumed_attempt: dict
    lineage: dict


def collection_recovery_evidence(recovery: CollectionRecovery, plan: dict) -> CollectionRecoveryEvidence:
    """Require an exact failed predecessor and a positive native token proof."""
    source = StoragePath(recovery.predecessor_uri)
    if recovery.mapping_version != REASONING_MAPPING_VERSION or recovery.consumed_slot != "00-0":
        raise ValueError("Teacher recovery requires the reviewed mapping and first consumed slot")
    if len(plan["selected"]) != 10:
        raise ValueError("Teacher recovery requires the frozen ten-family, twenty-attempt plan")
    info = json.loads(pinned_bytes(str(source / ".executor_info"), recovery.executor_info_sha256))
    config = info["config"]
    identity = f"{info['name']}@{config['version']}:{config['fingerprint']}"
    if identity != recovery.predecessor_identity or StoragePath(info["output_path"]) != source:
        raise ValueError("Teacher recovery predecessor identity differs")
    status = pinned_bytes(str(source / ".executor_status"), recovery.executor_status_sha256).decode().strip()
    if status != "FAILED":
        raise ValueError("Teacher recovery predecessor is not failed")
    prior_plan = json.loads(pinned_bytes(str(source / "plan.json"), recovery.plan_sha256))
    if prior_plan != plan:
        raise ValueError("Teacher recovery differs from the exact predecessor plan")
    marker = json.loads(pinned_bytes(str(source / "contract-failure.json"), recovery.fatal_marker_sha256))
    consumed = {"task": plan["selected"][0], "attempt": 0}
    reservation = json.loads(
        pinned_bytes(str(source / "trajectories/00-0/trajectory.json"), recovery.slot_reservation_sha256)
    )
    if (
        reservation != consumed
        or marker["task"] != consumed["task"]
        or marker["attempt"] != 0
        or marker["slot"] != "00-0"
        or marker["exception_type"] != "RolloutContractError"
        or marker["exception_message"] != "Native GLM chat changed the served token prefix"
        or marker["plan_sha256"] != compact_json_sha256(plan)
    ):
        raise ValueError("Teacher recovery does not bind the exact consumed contract failure")
    if (source / "recovery-lineage.json").exists():
        raise ValueError("Teacher recovery does not permit a second amendment")
    if (source / "collection.json").exists():
        raise ValueError("Teacher recovery predecessor already has a collection result")
    reservations = (source / "trajectories/*/trajectory.json").glob()
    if len(reservations) != 1 or reservations[0] != source / "trajectories/00-0/trajectory.json":
        raise ValueError("Teacher recovery predecessor has other consumed attempts")
    for name in ("rollout.json", "student-row.json", "qualification.json"):
        if (source / "trajectories" / "*" / name).glob():
            raise ValueError("Teacher recovery predecessor has completed rollout or row evidence")
    expected_preflight = {**plan["model"], "session_identity": f"{plan['model']['session_identity']}-preflight"}
    preflight_reservation = json.loads(
        pinned_bytes(str(source / "preflight/reservation.json"), recovery.preflight_reservation_sha256)
    )
    if (
        preflight_reservation != expected_preflight
        or recovery.preflight_identity != expected_preflight["session_identity"]
    ):
        raise ValueError("Teacher recovery preflight request identity differs")
    preflight = json.loads(
        pinned_bytes(str(source / "preflight/token-preflight.json"), recovery.preflight_result_sha256)
    )
    if (
        preflight["status"] != "passed"
        or len(preflight["attempts"]) != 2
        or any(
            attempt["status"] != "passed"
            or len(attempt["requests"]) != 2
            or attempt["rollout"]["grade"]["status"] != "graded"
            or attempt["rollout"]["grade"]["reward"] != 1
            for attempt in preflight["attempts"]
        )
    ):
        raise ValueError("Teacher recovery requires a successful saved preflight")
    for attempt in preflight["attempts"]:
        for request in attempt["requests"]:
            body = request["request"]
            if (
                body["model"] != GLM_MODEL
                or body["prompt_cache_key"] != recovery.preflight_identity
                or body["max_tokens"] != plan["model"]["max_tokens"]
                or body["temperature"] != plan["model"]["temperature"]
                or body["chat_template_kwargs"] != {"reasoning_effort": plan["model"]["reasoning_effort"]}
                or body["return_token_ids"] is not True
            ):
                raise ValueError("Teacher recovery saved preflight request differs from the plan")
    proof = json.loads(pinned_bytes(recovery.token_proof_uri, recovery.token_proof_sha256))
    if (
        proof["status"] != "passed"
        or proof["model"] != GLM_MODEL
        or proof["mapping_version"] != recovery.mapping_version
        or proof["proof_method"] != "bounded-real-relay-transport"
        or proof["generation_requests"] != 2
        or proof["diagnostic_generation_requests"] != 2
        or proof["diagnostic_output_tokens"] != 2
        or proof["teacher_generation_requests"] != 0
        or proof["tokenizer_requests"] != 7
        or proof["tokenizer_successful_renders"] != 4
        or proof["http_retries"] != 0
        or proof["outputs_reused"] is not False
        or proof["scientific_trajectory_budget_remaining"] != 19
        or proof["transport_diagnostic_declared_and_issued_before_send"] is not True
        or not all(
            proof[field] is True
            for field in (
                "original_render_matches_saved_prompt",
                "mapped_render_preserves_expected_prefix",
                "reasoning_insertion_only",
                "scientific_plan_unchanged",
                "scientific_trajectory_budget_unchanged",
                "failed_teacher_slot_still_consumed",
                "request_pair_difference_only_alias_mapping",
            )
        )
    ):
        raise ValueError("Teacher recovery lacks the exact positive native token proof")
    for name, field, filename in (
        ("source_request", "source_request_sha256", "request.json"),
        ("source_issued", "source_issued_sha256", "issued.json"),
        ("source_raw_response", "source_raw_response_sha256", "response.json"),
    ):
        raw = (source / "trajectories/00-0/turns/003" / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != proof[field] or proof["artifacts"][name]["sha256"] != proof[field]:
            raise ValueError("Teacher recovery token proof differs from the saved failed request")
    required_artifacts = {
        "source_request",
        "source_issued",
        "source_raw_response",
        "declaration",
        "original_tokenizer_request",
        "original_tokenizer_response",
        "mapped_tokenizer_request",
        "mapped_tokenizer_response",
    }
    if not required_artifacts.issubset(proof["artifacts"]):
        raise ValueError("Teacher recovery token proof has incomplete saved tokenizer evidence")
    for evidence in proof["artifacts"].values():
        pinned_bytes(evidence["path"], evidence["sha256"])
    declaration_artifact = proof["artifacts"]["declaration"]
    declaration = json.loads(pinned_bytes(declaration_artifact["path"], declaration_artifact["sha256"]))
    if (
        declaration["request_limit"] != 2
        or declaration["maximum_total_output_tokens"] != 2
        or declaration["http_retries"] != 0
        or declaration["failed_teacher_slot"] != "00-0"
        or declaration["scientific_trajectory_budget_remaining"] != 19
        or declaration["settings"]["model"] != GLM_MODEL
        or declaration["settings"]["max_tokens"] != 1
        or declaration["settings"]["temperature"] != 0
        or declaration["settings"]["return_token_ids"] is not True
        or not proof["endpoint_identity"]["url"].endswith("/v1/chat/completions")
    ):
        raise ValueError("Teacher recovery lacks its separate bounded relay diagnostic declaration")
    issued = json.loads((source / "trajectories/00-0/turns/003/issued.json").read_text())
    if proof["expected_prefix_token_count"] != len(issued["prefix_token_ids"]):
        raise ValueError("Teacher recovery token proof prefix length differs")
    lineage = {"recovery": asdict(recovery), "plan_sha256": compact_json_sha256(plan), "cumulative_attempt_limit": 20}
    return CollectionRecoveryEvidence(
        preflight,
        {**consumed, "status": "contract_failure_predecessor", "predecessor_identity": identity},
        lineage,
    )
