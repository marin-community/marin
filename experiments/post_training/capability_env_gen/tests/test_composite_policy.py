import hashlib
from pathlib import Path

import pytest

from capability_pipeline.composite_policy import (
    SCHEMA_VERSION,
    aggregate_composite,
    consensus_disagreements,
    resolve_judge_scores,
    validate_composite_config,
)


def policy():
    return {
        "step_index": 0,
        "machine_checks": [
            {
                "id": "fatal",
                "role": "gate",
                "script_path": "fatal.py",
                "args": [],
                "image": "python@sha256:" + "a" * 64,
                "timeout": 60,
            },
            {
                "id": "measured",
                "role": "criterion",
                "weight": 2.0,
                "script_path": "measured.py",
                "args": [],
                "image": "python@sha256:" + "a" * 64,
                "timeout": 60,
            },
            {
                "id": "false-positive",
                "role": "penalty",
                "weight": 1.0,
                "script_path": "penalty.py",
                "args": [],
                "image": "python@sha256:" + "a" * 64,
                "timeout": 60,
            },
        ],
        "judge": {
            "criterion_weights": [1.0, 3.0],
            "critical_indices": [0],
            "critical_min": 1.0,
            "conditional_caps": [],
        },
    }


def machine(gate=1.0, measured=0.5, penalty=0.0):
    return [
        {"id": "fatal", "status": "graded", "reward": gate, "detail": {}},
        {
            "id": "measured",
            "status": "graded",
            "reward": measured,
            "detail": {},
        },
        {
            "id": "false-positive",
            "status": "graded",
            "reward": penalty,
            "detail": {},
        },
    ]


def judge(first=1.0, second=1.0):
    return {
        "status": "graded",
        "reward": (first + second) / 2,
        "detail": {
            "judgments": [
                {"criterion": 0, "score": first},
                {"criterion": 1, "score": second},
            ],
        },
    }


def consensus_policy():
    configured = policy()
    configured["judge"] = {
        "criterion_weights": [1.0, 1.0],
        "critical_indices": [0],
        "critical_min": 1.0,
        "conditional_caps": [],
        "consensus": {
            "mode": "two_then_third",
            "initial_samples": 2,
            "disagreement_tolerance": 0.0,
            "resolution": "median",
        },
        "anchor_groups": [
            {"id": "quality", "criterion_indices": [0, 1]},
        ],
    }
    return configured


def sampled_judge(first, second):
    rows = [first, second]
    judgments = [
        {
            "sample": sample,
            "criterion": criterion,
            "score": score,
            "model": "glm-5.3",
            "revision": "revision-a",
        }
        for sample, scores in enumerate(rows)
        for criterion, score in enumerate(scores)
    ]
    return {
        "status": "graded",
        "reward": sum(sum(scores) for scores in rows)
        / sum(len(scores) for scores in rows),
        "detail": {
            "provider": "glm",
            "requested_model": "glm-5.3",
            "temperature": 0.2,
            "judgments": judgments,
        },
    }


def adjudicator(scores, *, status="graded"):
    return {
        "status": status,
        "reward": sum(scores) / len(scores) if status == "graded" else None,
        "detail": {
            "judgments": [
                {
                    "sample": 0,
                    "criterion": criterion,
                    "score": score,
                    "model": "glm-5.3",
                    "revision": "revision-b",
                }
                for criterion, score in enumerate(scores)
            ],
            "provider": "glm",
            "requested_model": "glm-5.3",
            "temperature": 0.2,
        },
    }


def test_machine_gate_failure_forces_zero_even_when_rubric_passes():
    result = aggregate_composite(machine(gate=0.0), {}, policy())
    assert result["status"] == "graded"
    assert result["reward"] == 0.0
    assert result["detail"]["failed_machine_gates"] == ["fatal"]
    assert result["detail"]["judge_path"] == "skipped_machine_gate"


def test_critical_judge_failure_forces_zero_and_penalty_is_subtractive():
    assert aggregate_composite(machine(), judge(first=0.0), policy())["reward"] == 0.0
    unpenalized = aggregate_composite(machine(), judge(), policy())["reward"]
    penalized = aggregate_composite(machine(penalty=1.0), judge(), policy())["reward"]
    assert unpenalized == pytest.approx(5 / 6)
    assert penalized == pytest.approx(4 / 6)


def test_infrastructure_result_cannot_be_converted_to_zero():
    broken = machine()
    broken[0] = {"id": "fatal", "status": "infra_error", "reward": None}
    with pytest.raises(ValueError, match="finite graded reward"):
        aggregate_composite(broken, judge(), policy())


def test_conditional_cap_zeros_only_its_declared_judge_section():
    configured = policy()
    configured["judge"]["conditional_caps"] = [
        {
            "id": "register-overflag-cap",
            "trigger_check_id": "false-positive",
            "trigger_min": 1.0,
            "target_judge_indices": [1],
            "max_fraction": 0.0,
        }
    ]
    result = aggregate_composite(
        machine(measured=1.0, penalty=1.0), judge(), configured
    )
    assert result["reward"] == pytest.approx((2 + 1 - 1) / 6)
    assert result["detail"]["judge_criterion_scores"] == [1.0, 1.0]
    assert result["detail"]["effective_judge_criterion_scores"] == [1.0, 0.0]
    assert result["detail"]["applied_conditional_caps"][0]["id"] == (
        "register-overflag-cap"
    )


def test_config_is_bound_to_specification_and_adapter():
    adapter = hashlib.sha256(b"adapter").hexdigest()
    policy_hash = hashlib.sha256(b"policy").hexdigest()
    specification = hashlib.sha256(b"specification").hexdigest()
    document = {
        "schema_version": SCHEMA_VERSION,
        "specification_sha256": specification,
        "implementation": {
            "taskcompendium_revision": "dc6b501c8604bcd2e3c20c1e9947679845fdfef8",
            "adapter_sha256": adapter,
            "policy_sha256": policy_hash,
            "native_judge_protocol_sha256": hashlib.sha256(
                (
                    Path(__file__).parents[1]
                    / "capability_pipeline/native_judge_protocol.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "steps": [policy()],
    }
    assert validate_composite_config(
        document,
        specification_sha256=specification,
        adapter_sha256=adapter,
        policy_sha256=policy_hash,
        step_count=1,
    ) == {0: document["steps"][0]}
    document["implementation"]["native_judge_protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        validate_composite_config(
            document,
            specification_sha256=specification,
            adapter_sha256=adapter,
            policy_sha256=policy_hash,
            step_count=1,
        )
    document["implementation"]["native_judge_protocol_sha256"] = hashlib.sha256(
        (
            Path(__file__).parents[1] / "capability_pipeline/native_judge_protocol.py"
        ).read_bytes()
    ).hexdigest()
    document["specification_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        validate_composite_config(
            document,
            specification_sha256=specification,
            adapter_sha256=adapter,
            policy_sha256=policy_hash,
            step_count=1,
        )


def test_explicit_consensus_and_anchor_config_is_validated():
    adapter = hashlib.sha256(b"adapter").hexdigest()
    policy_hash = hashlib.sha256(b"policy").hexdigest()
    specification = hashlib.sha256(b"specification").hexdigest()
    configured = consensus_policy()
    document = {
        "schema_version": SCHEMA_VERSION,
        "specification_sha256": specification,
        "implementation": {
            "taskcompendium_revision": "dc6b501c8604bcd2e3c20c1e9947679845fdfef8",
            "adapter_sha256": adapter,
            "policy_sha256": policy_hash,
            "native_judge_protocol_sha256": hashlib.sha256(
                (
                    Path(__file__).parents[1]
                    / "capability_pipeline/native_judge_protocol.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "steps": [configured],
    }
    assert validate_composite_config(
        document,
        specification_sha256=specification,
        adapter_sha256=adapter,
        policy_sha256=policy_hash,
        step_count=1,
    ) == {0: configured}

    configured["judge"]["anchor_groups"][0]["criterion_indices"] = [1, 0]
    with pytest.raises(ValueError, match="anchor group criterion mapping"):
        validate_composite_config(
            document,
            specification_sha256=specification,
            adapter_sha256=adapter,
            policy_sha256=policy_hash,
            step_count=1,
        )


def test_two_agreeing_judges_need_no_adjudicator_and_retain_raw_receipt():
    configured = consensus_policy()
    initial = sampled_judge([1, 0], [1, 0])
    assert consensus_disagreements(initial, configured) == []
    scores, receipt = resolve_judge_scores(initial, None, configured)
    assert scores == [1.0, 0.0]
    assert receipt["disagreed_indices"] == []
    assert receipt["adjudicator_judgments"] == []
    assert receipt["native_grade_attempt_count"] == 1
    assert receipt["judge_vector_count"] == 2
    assert receipt["judge_call_count"] == 4
    assert receipt["anchor_scores"] == [
        {
            "id": "quality",
            "criterion_indices": [0, 1],
            "score": 1,
            "max_score": 2,
        }
    ]


def test_disagreement_triggers_one_third_pass_and_binary_majority():
    configured = consensus_policy()
    initial = sampled_judge([1, 0], [1, 1])
    assert consensus_disagreements(initial, configured) == [1]
    scores, receipt = resolve_judge_scores(initial, adjudicator([1, 0]), configured)
    assert scores == [1.0, 0.0]
    assert receipt["disagreed_indices"] == [1]
    assert receipt["native_grade_attempt_count"] == 2
    assert receipt["judge_vector_count"] == 3
    assert receipt["judge_call_count"] == 6
    assert len(receipt["initial_judgments"]) == 4
    assert len(receipt["adjudicator_judgments"]) == 2
    assert receipt["initial_judge_provenance"] == {
        "provider": "glm",
        "requested_model": "glm-5.3",
        "temperature": 0.2,
    }
    assert receipt["adjudicator_judge_provenance"] == {
        "provider": "glm",
        "requested_model": "glm-5.3",
        "temperature": 0.2,
    }
    assert receipt["initial_judgments"][0]["revision"] == "revision-a"
    assert receipt["adjudicator_judgments"][0]["revision"] == "revision-b"

    scores, receipt = resolve_judge_scores(initial, adjudicator([1, 1]), configured)
    assert scores == [1.0, 1.0]
    assert receipt["anchor_scores"][0]["score"] == 2


def test_consensus_receipt_retains_protocol_request_evidence():
    configured = consensus_policy()
    initial = sampled_judge([1, 0], [1, 1])
    third = adjudicator([1, 0])
    initial["detail"]["judge_protocol"] = {
        "schema_version": "capability-native-judge-protocol-v2",
        "actual_request_count": 5,
    }
    third["detail"]["judge_protocol"] = {
        "schema_version": "capability-native-judge-protocol-v2",
        "actual_request_count": 3,
    }

    _, receipt = resolve_judge_scores(initial, third, configured)

    assert (
        receipt["initial_judge_provenance"]["judge_protocol"]
        == initial["detail"]["judge_protocol"]
    )
    assert (
        receipt["adjudicator_judge_provenance"]["judge_protocol"]
        == third["detail"]["judge_protocol"]
    )
    assert receipt["actual_judge_request_count"] == 8


def test_multiple_disagreements_are_resolved_by_one_full_adjudicator_record():
    configured = consensus_policy()
    configured["judge"]["anchor_groups"] = []
    initial = sampled_judge([1, 0], [0, 1])
    assert consensus_disagreements(initial, configured) == [0, 1]
    scores, receipt = resolve_judge_scores(initial, adjudicator([1, 0]), configured)
    assert scores == [1.0, 0.0]
    assert receipt["disagreed_indices"] == [0, 1]
    assert receipt["native_grade_attempt_count"] == 2
    assert receipt["judge_vector_count"] == 3
    assert receipt["judge_call_count"] == 6


def test_c02_thirty_nine_thresholds_report_actual_completion_counts():
    configured = consensus_policy()
    configured["judge"]["criterion_weights"] = [1.0] * 39
    configured["judge"]["critical_indices"] = [15, 18, 21, 24]
    configured["judge"]["anchor_groups"] = [
        {"id": f"group-{group}", "criterion_indices": list(range(start, start + 3))}
        for group, start in enumerate(range(0, 39, 3))
    ]
    full = [1, 1, 1] * 13
    initial = sampled_judge(full, full)
    scores, receipt = resolve_judge_scores(initial, None, configured)
    assert scores == [1.0] * 39
    assert receipt["native_grade_attempt_count"] == 1
    assert receipt["judge_vector_count"] == 2
    assert receipt["judge_call_count"] == 78
    assert sum(group["score"] for group in receipt["anchor_scores"]) == 39

    disagreeing = full.copy()
    disagreeing[-1] = 0
    initial = sampled_judge(full, disagreeing)
    scores, receipt = resolve_judge_scores(initial, adjudicator(full), configured)
    assert scores == [1.0] * 39
    assert receipt["native_grade_attempt_count"] == 2
    assert receipt["judge_vector_count"] == 3
    assert receipt["judge_call_count"] == 117


def test_consensus_rejects_missing_adjudicator_and_nonmonotone_raw_anchor():
    configured = consensus_policy()
    disagreeing = sampled_judge([1, 0], [1, 1])
    with pytest.raises(ValueError, match="requires an adjudicator"):
        resolve_judge_scores(disagreeing, None, configured)

    incoherent = sampled_judge([0, 1], [0, 1])
    with pytest.raises(ValueError, match="not a monotone threshold sequence"):
        resolve_judge_scores(incoherent, None, configured)

    agreeing = sampled_judge([1, 0], [1, 0])
    with pytest.raises(ValueError, match="ran without a declared disagreement"):
        resolve_judge_scores(agreeing, adjudicator([1, 0]), configured)


def test_consensus_rejects_incomplete_boolean_and_infrastructure_evidence():
    configured = consensus_policy()
    incomplete = sampled_judge([1, 0], [1, 0])
    incomplete["detail"]["judgments"].pop()
    with pytest.raises(ValueError, match="coverage is incomplete"):
        resolve_judge_scores(incomplete, None, configured)

    boolean = sampled_judge([1, 0], [1, 0])
    boolean["detail"]["judgments"][0]["score"] = True
    with pytest.raises(ValueError, match="coverage is invalid"):
        resolve_judge_scores(boolean, None, configured)

    disagreeing = sampled_judge([1, 0], [1, 1])
    with pytest.raises(ValueError, match="finite graded reward"):
        resolve_judge_scores(
            disagreeing,
            adjudicator([1, 0], status="infra_error"),
            configured,
        )
