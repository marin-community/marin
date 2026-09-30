"""Credential-free contract and aggregation for machine-check plus native judge."""

from __future__ import annotations

import math
import re
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = "taskcompendium-composite-verifier-v1"


def composite_evidence_detail(
    detail: dict[str, Any],
    machine_results: list[dict[str, Any]],
    *,
    adapter_sha256: str,
    policy_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Bind completed private checks to every later composite outcome."""
    return {
        **detail,
        "machine_results": machine_results,
        "composite_adapter_sha256": adapter_sha256,
        "composite_policy_sha256": policy_sha256,
        "composite_config_sha256": config_sha256,
    }


def _number(value: Any, *, positive: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else 0 <= value <= 1)
    )


def validate_composite_config(
    document: Any,
    *,
    specification_sha256: str,
    adapter_sha256: str,
    policy_sha256: str,
    step_count: int,
) -> dict[int, dict[str, Any]]:
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("composite verifier has the wrong schema version")
    provenance = document.get("implementation")
    from .composite_extension import native_judge_protocol_sha256

    if (
        provenance
        != {
            "taskcompendium_revision": "dc6b501c8604bcd2e3c20c1e9947679845fdfef8",
            "adapter_sha256": adapter_sha256,
            "policy_sha256": policy_sha256,
            "native_judge_protocol_sha256": native_judge_protocol_sha256(),
        }
        or document.get("specification_sha256") != specification_sha256
    ):
        raise ValueError(
            "composite verifier is not bound to its implementation and specification"
        )
    steps = document.get("steps")
    if not isinstance(steps, list) or len(steps) != step_count:
        raise ValueError(
            "composite verifier must configure every semantic step exactly"
        )
    validated = {}
    for expected_index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("step_index") != expected_index:
            raise ValueError("composite verifier step indices are not contiguous")
        checks = step.get("machine_checks")
        judge = step.get("judge")
        if (
            not isinstance(checks, list)
            or not checks
            or any(not isinstance(check, dict) for check in checks)
            or not isinstance(judge, dict)
        ):
            raise ValueError(
                "composite step needs machine checks and judge aggregation"
            )
        identifiers = [check.get("id") for check in checks]
        if any(not isinstance(value, str) or not value for value in identifiers) or len(
            set(identifiers)
        ) != len(identifiers):
            raise ValueError(
                "composite machine check IDs must be unique nonempty strings"
            )
        for check in checks:
            role = check.get("role")
            weight = check.get("weight")
            if role not in {"gate", "criterion", "penalty"}:
                raise ValueError("composite machine check has an unknown role")
            if role == "gate" and weight is not None:
                raise ValueError("machine gates do not take weights")
            if role != "gate" and not _number(weight, positive=True):
                raise ValueError(
                    "machine criterion and penalty weights must be positive"
                )
            script_path = check.get("script_path")
            image = check.get("image")
            supervisor_python = check.get("supervisor_python", "python3")
            if (
                not isinstance(script_path, str)
                or not script_path
                or PurePosixPath(script_path).is_absolute()
                or ".." in PurePosixPath(script_path).parts
                or not isinstance(check.get("args", []), list)
                or any(not isinstance(value, str) for value in check.get("args", []))
                or not isinstance(image, str)
                or re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image) is None
                or type(check.get("timeout")) is not int
                or check["timeout"] < 1
                or not isinstance(supervisor_python, str)
                or not supervisor_python
                or "\n" in supervisor_python
                or "\r" in supervisor_python
            ):
                raise ValueError("machine check lacks a pinned executable contract")
        weights = judge.get("criterion_weights")
        critical = judge.get("critical_indices")
        if (
            not isinstance(weights, list)
            or not weights
            or any(not _number(weight, positive=True) for weight in weights)
            or not isinstance(critical, list)
            or any(
                type(index) is not int or not 0 <= index < len(weights)
                for index in critical
            )
            or len(set(critical)) != len(critical)
            or not _number(judge.get("critical_min", 1.0))
        ):
            raise ValueError("judge weights or critical criterion indices are invalid")
        consensus = judge.get("consensus")
        if consensus is not None and (
            not isinstance(consensus, dict)
            or consensus.get("mode") != "two_then_third"
            or consensus.get("initial_samples") != 2
            or not _number(consensus.get("disagreement_tolerance"))
            or consensus.get("resolution") != "median"
        ):
            raise ValueError("judge consensus policy is invalid")
        anchor_groups = judge.get("anchor_groups", [])
        if not isinstance(anchor_groups, list) or any(
            not isinstance(group, dict) for group in anchor_groups
        ):
            raise ValueError("judge anchor groups must be a list of objects")
        if anchor_groups and consensus is None:
            raise ValueError("judge anchor groups require explicit consensus")
        anchor_ids = [group.get("id") for group in anchor_groups]
        if any(not isinstance(value, str) or not value for value in anchor_ids) or len(
            set(anchor_ids)
        ) != len(anchor_ids):
            raise ValueError("judge anchor group IDs must be unique strings")
        grouped_indices: list[int] = []
        for group in anchor_groups:
            indices = group.get("criterion_indices")
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    type(index) is not int or not 0 <= index < len(weights)
                    for index in indices
                )
                or indices != sorted(set(indices))
                or len({weights[index] for index in indices}) != 1
            ):
                raise ValueError("judge anchor group criterion mapping is invalid")
            grouped_indices.extend(indices)
        if len(grouped_indices) != len(set(grouped_indices)):
            raise ValueError("judge anchor group criteria must not overlap")
        if not any(check["role"] == "gate" for check in checks):
            raise ValueError("composite verifier needs at least one deterministic gate")
        caps = judge.get("conditional_caps", [])
        if not isinstance(caps, list) or any(not isinstance(cap, dict) for cap in caps):
            raise ValueError("judge conditional caps must be a list of objects")
        cap_ids = [cap.get("id") for cap in caps]
        if any(not isinstance(value, str) or not value for value in cap_ids) or len(
            set(cap_ids)
        ) != len(cap_ids):
            raise ValueError("judge conditional cap IDs must be unique strings")
        check_ids = set(identifiers)
        for cap in caps:
            targets = cap.get("target_judge_indices")
            if (
                cap.get("trigger_check_id") not in check_ids
                or not _number(cap.get("trigger_min"), positive=True)
                or not isinstance(targets, list)
                or not targets
                or any(
                    type(index) is not int or not 0 <= index < len(weights)
                    for index in targets
                )
                or len(set(targets)) != len(targets)
                or not _number(cap.get("max_fraction"))
            ):
                raise ValueError("judge conditional cap is invalid")
        validated[expected_index] = step
    return validated


def _sample_scores(
    judgments: Any, *, samples: int, criterion_count: int, label: str
) -> list[list[float]]:
    if not isinstance(judgments, list):
        raise TypeError(f"{label} judge judgments are absent")
    by_sample: list[list[float | None]] = [
        [None] * criterion_count for _ in range(samples)
    ]
    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise TypeError(f"{label} judge judgment is malformed")
        sample = judgment.get("sample")
        criterion = judgment.get("criterion")
        score = judgment.get("score")
        model = judgment.get("model")
        revision = judgment.get("revision")
        if (
            type(sample) is not int
            or not 0 <= sample < samples
            or type(criterion) is not int
            or not 0 <= criterion < criterion_count
            or type(score) not in (int, float)
            or score not in (0, 1)
            or not isinstance(model, str)
            or not model
            or revision is not None
            and not isinstance(revision, str)
            or by_sample[sample][criterion] is not None
        ):
            raise ValueError(f"{label} judge criterion coverage is invalid")
        by_sample[sample][criterion] = float(score)
    if any(score is None for row in by_sample for score in row):
        raise ValueError(f"{label} judge criterion coverage is incomplete")
    return [[float(score) for score in row] for row in by_sample]


def _anchor_scores(
    scores: list[float], anchor_groups: list[dict[str, Any]], *, label: str
) -> list[dict[str, Any]]:
    resolved = []
    for group in anchor_groups:
        indices = group["criterion_indices"]
        bits = [scores[index] for index in indices]
        if any(bit not in (0.0, 1.0) for bit in bits) or any(
            lower < higher for lower, higher in pairwise(bits)
        ):
            raise ValueError(
                f"{label} judge anchor group is not a monotone threshold sequence: {group['id']}"
            )
        resolved.append(
            {
                "id": group["id"],
                "criterion_indices": indices,
                "score": int(sum(bits)),
                "max_score": len(indices),
            }
        )
    return resolved


def consensus_disagreements(
    judge_result: dict[str, Any], policy: dict[str, Any]
) -> list[int]:
    """Validate the initial pair and return criteria requiring adjudication."""
    consensus = policy["judge"].get("consensus")
    if consensus is None:
        return []
    weights = policy["judge"]["criterion_weights"]
    rows = _sample_scores(
        judge_result.get("detail", {}).get("judgments"),
        samples=consensus["initial_samples"],
        criterion_count=len(weights),
        label="initial",
    )
    for sample, scores in enumerate(rows):
        _anchor_scores(
            scores,
            policy["judge"].get("anchor_groups", []),
            label=f"initial sample {sample}",
        )
    tolerance = consensus["disagreement_tolerance"]
    return [
        index
        for index in range(len(weights))
        if max(row[index] for row in rows) - min(row[index] for row in rows) > tolerance
    ]


def resolve_judge_scores(
    judge_result: dict[str, Any],
    adjudicator_result: dict[str, Any] | None,
    policy: dict[str, Any],
) -> tuple[list[float], dict[str, Any] | None]:
    """Resolve native per-criterion evidence under an explicit consensus policy."""
    weights = policy["judge"]["criterion_weights"]
    consensus = policy["judge"].get("consensus")
    if consensus is None:
        judgments = judge_result.get("detail", {}).get("judgments")
        if not isinstance(judgments, list) or not judgments:
            raise ValueError("native judge omitted per-criterion evidence")
        scores = []
        for index in range(len(weights)):
            values = [
                judgment.get("score")
                for judgment in judgments
                if isinstance(judgment, dict) and judgment.get("criterion") == index
            ]
            if not values or any(not _number(value) for value in values):
                raise ValueError("native judge criterion coverage is incomplete")
            scores.append(sum(values) / len(values))
        return scores, None

    initial_judgments = judge_result.get("detail", {}).get("judgments")
    initial_rows = _sample_scores(
        initial_judgments,
        samples=consensus["initial_samples"],
        criterion_count=len(weights),
        label="initial",
    )
    for sample, scores in enumerate(initial_rows):
        _anchor_scores(
            scores,
            policy["judge"].get("anchor_groups", []),
            label=f"initial sample {sample}",
        )
    disagreements = consensus_disagreements(judge_result, policy)
    adjudicator_judgments: list[dict[str, Any]] = []
    adjudicator_rows: list[list[float]] = []
    if disagreements:
        if adjudicator_result is None:
            raise ValueError("judge disagreement requires an adjudicator result")
        if adjudicator_result.get("status") != "graded" or not _number(
            adjudicator_result.get("reward")
        ):
            raise ValueError("judge adjudicator did not produce a finite graded reward")
        adjudicator_judgments = adjudicator_result.get("detail", {}).get("judgments")
        adjudicator_rows = _sample_scores(
            adjudicator_judgments,
            samples=1,
            criterion_count=len(weights),
            label="adjudicator",
        )
        _anchor_scores(
            adjudicator_rows[0],
            policy["judge"].get("anchor_groups", []),
            label="adjudicator",
        )
    elif adjudicator_result is not None:
        raise ValueError("judge adjudicator ran without a declared disagreement")

    scores = []
    for index in range(len(weights)):
        values = [row[index] for row in initial_rows]
        if disagreements:
            values.append(adjudicator_rows[0][index])
        ordered = sorted(values)
        scores.append(ordered[len(ordered) // 2] if len(ordered) == 3 else values[0])
    anchors = _anchor_scores(
        scores,
        policy["judge"].get("anchor_groups", []),
        label="resolved",
    )
    receipt = {
        "mode": consensus["mode"],
        "resolution": consensus["resolution"],
        "disagreement_tolerance": consensus["disagreement_tolerance"],
        "initial_judgments": initial_judgments,
        "disagreed_indices": disagreements,
        "adjudicator_judgments": adjudicator_judgments,
        "initial_judge_provenance": {
            key: judge_result.get("detail", {}).get(key)
            for key in (
                "provider",
                "requested_model",
                "temperature",
                "judge_protocol",
            )
            if key in judge_result.get("detail", {})
        },
        "adjudicator_judge_provenance": (
            None
            if adjudicator_result is None
            else {
                key: adjudicator_result.get("detail", {}).get(key)
                for key in (
                    "provider",
                    "requested_model",
                    "temperature",
                    "judge_protocol",
                )
                if key in adjudicator_result.get("detail", {})
            }
        ),
        "resolved_criterion_scores": scores,
        "anchor_scores": anchors,
        "native_grade_attempt_count": 1 + bool(disagreements),
        "judge_vector_count": len(initial_rows) + len(adjudicator_rows),
        "judge_call_count": len(initial_judgments) + len(adjudicator_judgments),
    }
    protocol_records = [judge_result.get("detail", {}).get("judge_protocol")]
    if adjudicator_result is not None:
        protocol_records.append(
            adjudicator_result.get("detail", {}).get("judge_protocol")
        )
    if any(record is not None for record in protocol_records):
        if any(
            not isinstance(record, dict)
            or type(record.get("actual_request_count")) is not int
            or record["actual_request_count"] < 0
            for record in protocol_records
        ):
            raise ValueError("judge protocol request accounting is incomplete")
        receipt["actual_judge_request_count"] = sum(
            record["actual_request_count"] for record in protocol_records
        )
    return scores, receipt


def aggregate_composite(
    machine_results: list[dict[str, Any]],
    judge_result: dict[str, Any],
    policy: dict[str, Any],
    adjudicator_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = policy["machine_checks"]
    if [result.get("id") for result in machine_results] != [
        check["id"] for check in checks
    ]:
        raise ValueError("machine results do not match the declared checks")
    by_id = dict(zip((check["id"] for check in checks), machine_results, strict=True))
    for check in checks:
        result = by_id[check["id"]]
        if result.get("status") != "graded" or not _number(result.get("reward")):
            raise ValueError("machine check did not produce a finite graded reward")
    failed_gates = [
        check["id"]
        for check in checks
        if check["role"] == "gate" and by_id[check["id"]]["reward"] < 1.0
    ]
    if failed_gates:
        return {
            "status": "graded",
            "reward": 0.0,
            "detail": {
                "aggregation": SCHEMA_VERSION,
                "failed_machine_gates": failed_gates,
                "failed_judge_critical_indices": [],
                "judge_path": "skipped_machine_gate",
                "judge_criterion_scores": [],
                "effective_judge_criterion_scores": [],
                "applied_conditional_caps": [],
                "machine_results": machine_results,
                "judge": None,
            },
        }
    if judge_result.get("status") != "graded" or not _number(
        judge_result.get("reward")
    ):
        raise ValueError("native judge did not produce a finite graded reward")
    weights = policy["judge"]["criterion_weights"]
    criterion_scores, consensus_receipt = resolve_judge_scores(
        judge_result, adjudicator_result, policy
    )
    failed_critical = [
        index
        for index in policy["judge"]["critical_indices"]
        if criterion_scores[index] < policy["judge"].get("critical_min", 1.0)
    ]
    if failed_critical:
        return {
            "status": "graded",
            "reward": 0.0,
            "detail": {
                "aggregation": SCHEMA_VERSION,
                "failed_machine_gates": failed_gates,
                "failed_judge_critical_indices": failed_critical,
                "judge_criterion_scores": criterion_scores,
                "effective_judge_criterion_scores": criterion_scores,
                "applied_conditional_caps": [],
                "machine_results": machine_results,
                "judge": judge_result["detail"],
                "judge_consensus": consensus_receipt,
                "judge_path": "model",
            },
        }
    effective_scores = list(criterion_scores)
    applied_caps = []
    for cap in policy["judge"].get("conditional_caps", []):
        trigger_reward = by_id[cap["trigger_check_id"]]["reward"]
        if trigger_reward < cap["trigger_min"]:
            continue
        targets = cap["target_judge_indices"]
        raw_weighted = sum(
            effective_scores[index] * weights[index] for index in targets
        )
        maximum = cap["max_fraction"] * sum(weights[index] for index in targets)
        scale = 1.0 if raw_weighted <= maximum else maximum / raw_weighted
        for index in targets:
            effective_scores[index] *= scale
        applied_caps.append(
            {
                "id": cap["id"],
                "trigger_check_id": cap["trigger_check_id"],
                "trigger_reward": trigger_reward,
                "target_judge_indices": targets,
                "max_fraction": cap["max_fraction"],
            }
        )
    positive = sum(
        by_id[check["id"]]["reward"] * check["weight"]
        for check in checks
        if check["role"] == "criterion"
    ) + sum(
        score * weight for score, weight in zip(effective_scores, weights, strict=True)
    )
    denominator = sum(
        check["weight"] for check in checks if check["role"] == "criterion"
    ) + sum(weights)
    penalty = sum(
        by_id[check["id"]]["reward"] * check["weight"]
        for check in checks
        if check["role"] == "penalty"
    )
    reward = max(0.0, min(1.0, (positive - penalty) / denominator))
    return {
        "status": "graded",
        "reward": reward,
        "detail": {
            "aggregation": SCHEMA_VERSION,
            "positive_weighted_sum": positive,
            "penalty_weighted_sum": penalty,
            "positive_weight_denominator": denominator,
            "failed_machine_gates": [],
            "failed_judge_critical_indices": [],
            "judge_criterion_scores": criterion_scores,
            "effective_judge_criterion_scores": effective_scores,
            "applied_conditional_caps": applied_caps,
            "machine_results": machine_results,
            "judge": judge_result["detail"],
            "judge_consensus": consensus_receipt,
            "judge_path": "model",
        },
    }
