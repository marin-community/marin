# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented-policy tests; no registered teacher or fitted head is evaluated."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_tpp40_ridge_raw_policies_20261015 as run


def predictor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> run.FittedPredictor:
    monkeypatch.setattr(run, "resource_check", lambda _: None)
    config = run.oracle.feature_config(np.array([2.0, 5.0]), "equal_group")
    return run.FittedPredictor(
        {"shape": {"rate": 0.37, "power": 0.7, "threshold": 2.0}, "mu": 0.0, "ridge": 100.0},
        {"scale": 1.3, "floor": 0.6, "amplitudes": [0.2, 0.3, 0.04, 0.02], "normalization": 0.2},
        0.8,
        config,
        None,
        tmp_path,
    )


def test_saved_normalization_and_response_do_not_depend_on_query_or_ridge(monkeypatch, tmp_path) -> None:
    fitted = predictor(monkeypatch, tmp_path)
    early, late = np.array([1.1, 0.4]), np.array([0.2, 0.9])
    matrix, _ = run.models.design_and_offset(
        early[None], late[None], fitted.alpha, fitted.spec["shape"], run.models.MODEL, {"mu": 0}, fitted.config
    )
    expected = 0.6 + np.exp(np.log(1.3) + ((matrix @ fitted.head["amplitudes"])[0] - 0.2))
    assert fitted(early, late) == expected
    assert replace(fitted, spec=fitted.spec | {"ridge": 0.0})(early, late) == expected
    assert fitted.head["normalization"] == 0.2
    zero_scale = replace(fitted, head=fitted.head | {"scale": 0.0})
    assert zero_scale(early, late) == 0.6
    assert all(item["zero_scale"] for item in zero_scale.diagnostics.values())


def test_large_training_normalization_subtracts_before_adding_log_scale(monkeypatch, tmp_path) -> None:
    fitted = predictor(monkeypatch, tmp_path)
    monkeypatch.setattr(run.models, "design_and_offset", lambda *args: (np.array([[1e16, 0, 0, 0]]), np.zeros(1)))
    fitted = replace(fitted, head=fitted.head | {"amplitudes": [1.0, 0, 0, 0], "normalization": 1e16, "scale": np.e})
    assert fitted(np.zeros(2), np.zeros(2)) == pytest.approx(0.6 + np.e, rel=1e-15)
    assert fitted(np.zeros(2), np.zeros(2)) != 1.6


def test_unclipped_overflow_fails_and_underflow_remains_visible(monkeypatch, tmp_path) -> None:
    fitted = predictor(monkeypatch, tmp_path)
    monkeypatch.setattr(run.models, "design_and_offset", lambda *args: (np.zeros((1, 4)), np.zeros(1)))
    with pytest.raises(ArithmeticError):
        replace(fitted, head=fitted.head | {"normalization": -1000})(np.zeros(2), np.zeros(2))
    underflow = replace(fitted, head=fitted.head | {"normalization": 1000}, diagnostics={})
    assert underflow(np.zeros(2), np.zeros(2)) == 0.6
    assert underflow.underflows == 1
    assert next(iter(underflow.diagnostics.values()))["underflow"]


def test_resource_stop_is_retained_before_any_feature_evaluation(monkeypatch, tmp_path) -> None:
    fitted = predictor(monkeypatch, tmp_path)

    def stopped(_):
        raise run.ResourceStop("invented deadline")

    def forbidden(*args):
        raise AssertionError("features must not be evaluated")

    monkeypatch.setattr(run, "resource_check", stopped)
    monkeypatch.setattr(run.models, "design_and_offset", forbidden)
    with pytest.raises(run.ResourceStop):
        fitted(np.zeros(2), np.zeros(2))
    assert fitted.stopped


def test_signed_regret_and_distinct_static_native_and_phase_comparisons() -> None:
    result = {"best": {"predicted": 0.65}, "predicted_gain_same_aggregate": 0.04, "predicted_gain_vs_static": 0.02}
    values = {"selected_free": 0.7, "selected_static": 0.71, "selected_aggregate_tied": 0.73}
    reference = {"selected_free": 0.701, "selected_static": 0.72}
    value = run.accounting(result, values, reference, 0.75)
    assert value["signed_regret_vs_teacher_2p_incumbent"] == pytest.approx(-0.001)
    assert value["true_gain_vs_selected_static"] == pytest.approx(0.01)
    assert value["true_gain_same_aggregate"] == pytest.approx(0.03)
    assert value["teacher_advantage_over_native_selected"] == pytest.approx(0.05)
    assert value["selected_optimism"] == pytest.approx(0.05)
    assert run.accounting(result, values, reference, None)["teacher_advantage_over_native_selected"] is None


def test_complete_population_barrier_precedes_all_teacher_access(monkeypatch) -> None:
    def forbidden():
        raise AssertionError("Teacher geometry must not be opened before all selections")

    monkeypatch.setattr(run.prior, "receipt", lambda *args: False)
    monkeypatch.setattr(run.oracle, "inputs", forbidden)
    with pytest.raises(ValueError, match="all30"):
        run.score_all({})
    population = run.specs()
    assert len(population) == len({tuple(item.values()) for item in population}) == 30
    assert sum(item["candidate"] == "aggregate_mariner" for item in population) == 12


def test_teacher_reporting_uses_fixed_points_and_checks_every_callback(monkeypatch, tmp_path) -> None:
    visits = []
    monkeypatch.setattr(run, "resource_check", lambda _: visits.append("guard"))

    def truth(early, late):
        visits.append((early.tolist(), late.tolist()))
        return float(early[1] + 2 * late[1])

    result = {
        "best": {"phase0": [0.8, 0.2], "phase1": [0.3, 0.7]},
        "best_static": {"phase0": [0.6, 0.4], "phase1": [0.6, 0.4]},
        "aggregate": [0.7, 0.3],
    }
    values = run.teacher_values(result, truth, 0.8, np.array([2.0, 5.0]), tmp_path)
    assert visits[::2] == ["guard"] * 3
    assert values == pytest.approx({"selected_free": 2.2, "selected_static": 2.4, "selected_aggregate_tied": 1.8})


def test_observed_baseline_ties_use_identity_without_teacher_outcomes() -> None:
    values = np.array([0.8, 0.7, 0.7, 0.7])
    indices = np.array([5, 8, 9, 3])
    assert run.observed_choice(values, indices, ["a", "b", "a", "a"]) == 3


def test_constant_head_never_integrates_irrelevant_temporal_features(monkeypatch, tmp_path) -> None:
    fitted = predictor(monkeypatch, tmp_path)
    fitted = replace(fitted, spec=fitted.spec | {"mu": 30.0}, head=fitted.head | {"scale": 0.0}, diagnostics={})

    def forbidden(*args):
        raise ArithmeticError("A difficult irrelevant ODE must not invalidate a constant response")

    monkeypatch.setattr(run.models, "design_and_offset", forbidden)
    assert fitted(np.array([100.0, 0.0]), np.array([0.0, 0.1])) == 0.6
    assert next(iter(fitted.diagnostics.values())) == {"logit": None, "underflow": False, "zero_scale": True}


def test_baseline_deadline_precedes_clean_teacher_archive_access(monkeypatch) -> None:
    def stopped(_):
        raise run.ResourceStop("invented deadline")

    def forbidden(*args, **kwargs):
        raise AssertionError("Clean teacher archive must not be opened")

    monkeypatch.setattr(run, "resource_check", stopped)
    monkeypatch.setattr(run.np, "load", forbidden)
    with pytest.raises(run.ResourceStop):
        run.score_observed_baselines({})
