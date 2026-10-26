# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many.two_phase_benchmark_scoring_20260921 import (
    compare_models,
    evaluate_gates,
    score_predictions,
    summarize_predictions,
)


def prediction_menu(observed: list[float], predicted: list[float], model: str = "candidate") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "panel": "swarm",
            "target": "web_bpb",
            "repeat": 0,
            "outer_fold": 0,
            "row_id": [f"row_{index}" for index in range(len(observed))],
            "policy_id": [f"policy_{index}" for index in range(len(observed))],
            "group_id": [f"group_{index}" for index in range(len(observed))],
            "source": "local_archive",
            "region": "interior",
            "role": "test",
            "observed": observed,
            "predicted": predicted,
        }
    )


@pytest.fixture
def protocol() -> dict:
    return {
        "protocol_id": "frozen_test_protocol",
        "tolerances_frozen": True,
        "tolerance_provenance": {
            "kind": "external",
            "source": "registered_test_fixture",
            "frozen_before_candidate_evaluation": True,
        },
        "required_cells": [{"panel": "swarm", "target": "web_bpb", "role": "test", "repeat": 0, "outer_fold": 0}],
        "required_evidence": ["leakage_audit"],
        "evidence": {"leakage_audit": True},
        "gates": [
            {
                "panel": "swarm",
                "target": "web_bpb",
                "role": "test",
                "metrics": [
                    {"metric": "rmse", "allowed_increase": 0.0},
                    {"metric": "regret_at1", "allowed_increase": 0.0},
                ],
            }
        ],
    }


def test_lower_rmse_cannot_rescue_worse_policy_selection(protocol: dict) -> None:
    frame = pd.concat(
        [
            prediction_menu([0.0, 1.0, 2.0], [2.0, 3.0, 4.0], "reference"),
            prediction_menu([0.0, 1.0, 2.0], [0.6, 0.5, 2.0]),
        ]
    )
    result = compare_models(frame, "reference", protocol)
    assert result["status"] == "reject"
    checks = {check["metric"]: check for check in result["candidates"][0]["checks"]}
    assert checks["rmse"]["passed"] is True
    assert checks["regret_at1"]["passed"] is False
    assert checks["regret_at1"]["candidate"] == 1.0


@pytest.mark.parametrize("extra_menu", ["candidate_only", "unpaired"])
@pytest.mark.parametrize("candidate_error,expected", [(0.0, "advance"), (1.0, "reject")])
def test_nonrequired_menus_do_not_change_required_gate_verdict(
    extra_menu: str, candidate_error: float, expected: str, protocol: dict
) -> None:
    reference = prediction_menu([0.0, 1.0], [0.0, 1.0], "reference")
    candidate = reference.assign(model="candidate", predicted=reference.predicted + candidate_error)
    extra_candidate = candidate.assign(target="extra_target", row_id=["extra_0", "extra_1"])
    frames = [reference, candidate, extra_candidate]
    if extra_menu == "unpaired":
        frames.append(reference.assign(target="extra_target"))
    result = compare_models(pd.concat(frames), "reference", protocol)
    assert result["status"] == expected
    assert all(check["cell"]["target"] == "web_bpb" for check in result["candidates"][0]["checks"])
    if expected == "reject":
        assert result["candidates"][0]["rejection_class"] == "prediction"


def test_required_numeric_failure_is_decisive_despite_another_undefined_metric(protocol: dict) -> None:
    reference = prediction_menu([1.0, 1.0], [1.0, 1.0], "reference")
    candidate = reference.assign(model="candidate", predicted=2.0)
    protocol["gates"][0]["metrics"].append({"metric": "predictive_r2", "minimum": 0.0})
    result = compare_models(pd.concat([reference, candidate]), "reference", protocol)
    verdict = result["candidates"][0]
    assert verdict["status"] == "reject"
    assert verdict["rejection_class"] == "prediction"
    assert any(reason["code"] == "undefined_metric" for reason in verdict["reasons"])


def test_joint_menu_detects_uniform_asymmetric_optimism_despite_role_rankings(protocol: dict) -> None:
    asymmetric = prediction_menu([2.0, 3.0], [2.0, 3.0], "reference")
    tied = prediction_menu([0.0, 1.0], [0.0, 1.0], "reference").assign(
        row_id=["tied_0", "tied_1"], policy_id=["tied_0", "tied_1"], role="tied_test"
    )
    reference = pd.concat([asymmetric, tied])
    candidate = reference.assign(model="candidate").copy()
    candidate.loc[candidate.role.eq("test"), "predicted"] -= 4.0
    cell, gate = protocol["required_cells"][0], protocol["gates"][0]
    protocol["required_cells"] = [{**cell, "role": role} for role in ("test", "tied_test", "joint_test")]
    protocol["gates"] = [
        {**gate, "role": role, "metrics": [{"metric": "regret_at1", "allowed_increase": 0.0}]}
        for role in ("test", "tied_test", "joint_test")
    ]
    result = compare_models(pd.concat([reference, candidate]), "reference", protocol)
    checks = {check["cell"]["role"]: check for check in result["candidates"][0]["checks"]}
    assert checks["test"]["passed"] is True
    assert checks["tied_test"]["passed"] is True
    assert checks["joint_test"]["candidate"] == 2.0
    assert result["status"] == "reject"
    second_fold = candidate.assign(outer_fold=1, observed=candidate.observed + 10, predicted=candidate.predicted + 10)
    macro = summarize_predictions(pd.concat([candidate, second_fold]))
    joint = macro.loc[macro.role.eq("joint_test")].iloc[0]
    assert joint.regret_at1 == 2.0
    assert joint.selected_observed == 7.0


def test_unfrozen_protocol_does_not_reject_against_invalid_tolerances(protocol: dict) -> None:
    reference = prediction_menu([0.0, 1.0], [0.0, 1.0], "reference")
    candidate = prediction_menu([0.0, 1.0], [100.0, 99.0])
    protocol["tolerances_frozen"] = False
    result = compare_models(pd.concat([reference, candidate]), "reference", protocol)
    assert result["status"] == "incomplete"
    assert result["candidates"][0]["checks"] == []


def test_candidate_missing_available_required_cell_rejects_estimation(protocol: dict) -> None:
    reference = prediction_menu([0.0, 1.0], [0.0, 1.0], "reference")
    protocol["candidates"] = ["candidate"]
    result = compare_models(reference, "reference", protocol)
    candidate = result["candidates"][0]
    assert candidate["status"] == "reject"
    assert candidate["rejection_class"] == "estimation"
    assert any(reason["code"] == "missing_candidate_required_cell" for reason in candidate["reasons"])


@pytest.mark.parametrize("failed_model,expected", [("candidate", "reject"), ("reference", "incomplete")])
def test_nonfinite_predictions_in_required_joint_menu_have_model_specific_verdicts(
    failed_model: str, expected: str, protocol: dict
) -> None:
    reference = prediction_menu([0.0, 1.0], [0.0, 1.0], "reference")
    frame = pd.concat([reference, reference.assign(model="candidate")], ignore_index=True)
    frame.loc[frame.model.eq(failed_model) & frame.row_id.eq("row_0"), "predicted"] = np.nan
    protocol["required_cells"][0]["role"] = "joint_test"
    protocol["gates"][0]["role"] = "joint_test"
    result = compare_models(frame, "reference", protocol)
    assert result["status"] == expected
    assert {cell["role"] for cell in result["nonfinite_prediction_cells"]} == {"test", "joint_test"}
    if failed_model == "candidate":
        assert result["candidates"][0]["rejection_class"] == "estimation"


def test_selection_ties_are_stable_under_row_order_and_outcomes() -> None:
    frame = prediction_menu([3.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    original = score_predictions(frame).iloc[0]
    changed = frame.iloc[::-1].copy()
    changed.loc[changed.row_id == "row_0", "observed"] = -100.0
    selected = score_predictions(changed).iloc[0]
    assert original.selected_row_id == selected.selected_row_id == "row_0"
    assert original.regret_at1 == 2.0
    assert selected.regret_at1 == 0.0


def test_signed_errors_and_frontier_use_predicted_endpoint() -> None:
    frame = prediction_menu([10.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0])
    metrics = score_predictions(frame).iloc[0]
    assert metrics.signed_bias == -2.0
    assert metrics.optimism == 2.0
    assert metrics.selected_optimism == metrics.frontier_optimism == 10.0
    assert metrics.frontier_rmse == 10.0
    assert metrics.frontier_n == 1
    assert metrics.regret_at1 == 9.0
    assert metrics.regret_at3 == metrics.regret_at5 == 0.0


def test_predictive_r2_and_calibration_have_distinct_meanings() -> None:
    metrics = score_predictions(prediction_menu([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])).iloc[0]
    assert metrics.predictive_r2 == -6.0
    assert metrics.calibration_slope == 0.5
    assert metrics.calibration_intercept == 0.0
    assert metrics.mae == 2.0
    assert metrics.rmse == pytest.approx(np.sqrt(14.0 / 3.0))


@pytest.mark.parametrize("field", ["row_id", "policy_id", "group_id", "source", "region", "observed"])
def test_paired_gate_rejects_changed_menu_identity_or_outcome(field: str, protocol: dict) -> None:
    reference = prediction_menu([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "reference")
    candidate = reference.assign(model="candidate").copy()
    candidate.loc[0, field] = 1.001 if field == "observed" else "changed_identity"
    result = compare_models(pd.concat([reference, candidate]), "reference", protocol)
    assert result["status"] == "incomplete"
    assert any(reason["code"] == "unpaired_policy_menu" for reason in result["candidates"][0]["reasons"])


def test_missing_required_target_blocks_promotion(protocol: dict) -> None:
    reference = prediction_menu([1.0, 2.0], [1.0, 2.0], "reference")
    protocol["required_cells"].append({**protocol["required_cells"][0], "target": "code_bpb"})
    result = compare_models(pd.concat([reference, reference.assign(model="candidate")]), "reference", protocol)
    assert result["status"] == "incomplete"
    assert any(reason["code"] == "missing_required_cell" for reason in result["candidates"][0]["reasons"])


def test_empty_prediction_evidence_remains_machine_readable(protocol: dict) -> None:
    empty = prediction_menu([], [])
    protocol["candidates"] = ["candidate"]
    result = compare_models(empty, "reference", protocol)
    assert result["status"] == "incomplete"
    reasons = {reason["code"] for reason in result["candidates"][0]["reasons"]}
    assert {"missing_candidate", "missing_reference", "missing_required_cell"} <= reasons


def test_matching_missing_fingerprints_do_not_certify_paired_menus(protocol: dict) -> None:
    reference = prediction_menu([1.0, 2.0], [1.0, 2.0], "reference")
    metrics = score_predictions(pd.concat([reference, reference.assign(model="candidate")]))
    metrics.menu_fingerprint = None
    result = evaluate_gates(metrics, protocol["evidence"], {**protocol, "reference": "reference"})
    assert result["status"] == "incomplete"
    assert any(reason["code"] == "unpaired_policy_menu" for reason in result["candidates"][0]["reasons"])


@pytest.mark.parametrize(
    "fault", ["unfrozen", "candidate_tolerance", "missing_evidence", "undeclared_evidence", "missing_reference"]
)
def test_incomplete_protocol_or_evidence_cannot_advance(fault: str, protocol: dict) -> None:
    reference = prediction_menu([1.0, 2.0], [1.0, 2.0], "reference")
    frame = pd.concat([reference, reference.assign(model="candidate")])
    if fault == "unfrozen":
        protocol["tolerances_frozen"] = False
    elif fault == "candidate_tolerance":
        protocol["tolerance_provenance"]["kind"] = "candidate_rmse"
    elif fault == "missing_evidence":
        protocol["evidence"] = {}
    elif fault == "undeclared_evidence":
        del protocol["required_evidence"]
    else:
        frame = frame[frame.model == "candidate"]
    assert compare_models(frame, "reference", protocol)["status"] == "incomplete"


def test_undefined_r2_blocks_gate_instead_of_passing_nan(protocol: dict) -> None:
    reference = prediction_menu([1.0, 1.0, 1.0], [0.8, 1.0, 1.2], "reference")
    frame = pd.concat([reference, reference.assign(model="candidate")])
    protocol["gates"][0]["metrics"] = [{"metric": "predictive_r2", "minimum": 0.0}]
    assert np.isnan(score_predictions(frame).iloc[0].predictive_r2)
    result = compare_models(frame, "reference", protocol)
    assert result["status"] == "incomplete"
    assert result["candidates"][0]["reasons"][0]["code"] == "undefined_metric"


def test_duplicate_prediction_or_nonfinite_prediction_is_not_silently_dropped() -> None:
    frame = prediction_menu([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="Duplicate prediction identity"):
        score_predictions(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, "predicted"] = np.nan
    with pytest.raises(ValueError, match="Missing predictions"):
        score_predictions(frame)


def test_numeric_text_predictions_cannot_sort_lexicographically() -> None:
    frame = prediction_menu([1.0, 2.0, 3.0], [2.0, 10.0, 20.0])
    frame.predicted = frame.predicted.astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        score_predictions(frame)


def test_fold_macro_scores_preserve_menu_selection_and_pool_residuals(protocol: dict) -> None:
    first = prediction_menu([0.0, 1.0], [0.5, 0.4])
    second = prediction_menu([10.0, 11.0, 12.0], [10.0, 11.0, 12.0]).assign(outer_fold=1)
    candidate = pd.concat([first, second])
    summary = summarize_predictions(candidate).iloc[0]
    assert summary.outer_fold == "macro"
    assert summary.n == 5
    assert summary.regret_at1 == 0.5
    assert summary.selected_observed == 5.5
    assert summary.selected_optimism == pytest.approx(0.3)
    assert summary.rmse == pytest.approx(np.sqrt(0.61 / 5))
    assert summary.frontier_rmse == pytest.approx(np.sqrt(0.36 / 2))
    protocol["required_cells"][0]["outer_fold"] = "macro"
    protocol["aggregation"] = "fold_macro"
    result = compare_models(pd.concat([candidate, candidate.assign(model="reference")]), "reference", protocol)
    assert result["status"] == "advance"
    assert result["claim"] == "empirical_screen_only"


def test_macro_pairing_checks_fold_assignments(protocol: dict) -> None:
    first = prediction_menu([0.0, 1.0], [0.0, 1.0], "reference")
    second = prediction_menu([2.0, 3.0], [2.0, 3.0], "reference").assign(outer_fold=1)
    reference = pd.concat([first, second], ignore_index=True)
    candidate = reference.assign(model="candidate", outer_fold=1 - reference.outer_fold)
    protocol["required_cells"][0]["outer_fold"] = "macro"
    protocol["aggregation"] = "fold_macro"
    assert compare_models(pd.concat([reference, candidate]), "reference", protocol)["status"] == "incomplete"


def test_complete_screen_can_be_inconclusive_when_improvement_is_required(protocol: dict) -> None:
    reference = prediction_menu([1.0, 2.0], [1.0, 2.0], "reference")
    protocol["require_any_improvement"] = True
    frame = pd.concat([reference, reference.assign(model="candidate")])
    result = compare_models(frame, "reference", protocol)
    assert result["status"] == "inconclusive"
    candidate = frame[frame.model == "candidate"].copy()
    candidate.predicted -= 1000
    frozen = copy.deepcopy(protocol)
    assert compare_models(pd.concat([reference, candidate]), "reference", protocol)["status"] == "reject"
    assert protocol == frozen
