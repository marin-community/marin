# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Invented-only checks for the TP046-to-TP051 raw procedure adapter."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_tpp40_ridge_raw_policies_20261015 as original,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_tpp40_zero_ridge_raw_policies_20261015 as run,
)


def test_old_head_without_ridge_field_predicts_with_identical_fixed_normalization(tmp_path, monkeypatch):
    monkeypatch.setattr(run.backend, "resource_check", lambda _: None)
    monkeypatch.setattr(original, "resource_check", lambda _: None)
    spec = {"shape": {"rate": 0.37, "power": 0.7, "threshold": 2.0}, "mu": 0.0, "clock": "equal_group"}
    head = {"scale": 1.3, "floor": 0.6, "amplitudes": [0.2, 0.3, 0.04, 0.02], "normalization": 0.2}
    config = run.oracle.feature_config(np.array([2.0, 5.0]), "equal_group")
    new = run.backend.FittedPredictor(spec, head, 0.8, config, None, tmp_path)
    old = original.FittedPredictor(spec | {"ridge": 100.0}, head, 0.8, config, None, tmp_path)
    early, late = np.array([1.1, 0.4]), np.array([0.2, 0.9])
    assert new(early, late) == old(early, late)
    assert replace(new, head=head | {"scale": 0.0})(early, late) == 0.6
    assert original.UPSTREAM.name == "whole_response_ridge_stress"
    assert run.backend.UPSTREAM.name == "whole_response_fold3_noise"


def test_all30_and_all12_barrier_blocks_teacher_before_any_data_access(monkeypatch):
    def forbidden():
        raise AssertionError("No teacher access before the durable selection barrier")

    monkeypatch.setattr(run.backend.oracle, "inputs", forbidden)
    monkeypatch.setattr(run.backend.prior, "receipt", lambda *args: False)
    with pytest.raises(ValueError, match="all30"):
        run.backend.score_all({})
    monkeypatch.setattr(
        run.backend.prior, "receipt", lambda folder, stage, frozen: stage["stage"] == "surrogate_selection"
    )
    with pytest.raises(ValueError, match="all12"):
        run.backend.score_all({})


def test_upstream_failed_head_and_query_failure_are_saved_without_optimization(tmp_path, monkeypatch):
    spec = {"noise": "toy", "world": "toy", "candidate": "aggregate_mariner"}
    upstream, output = tmp_path / "source", tmp_path / "result"
    monkeypatch.setattr(run.backend, "UPSTREAM", upstream)
    monkeypatch.setattr(run.backend, "OUTPUT", output)
    source = run.cell(upstream, spec)
    fit = {"full_unresolved": True, "selected_candidate": {"mu": 0.0}, "selected": None}
    run.prior.write(source / "fit.json", fit)
    run.prior.write(source / "fit_complete.json", {"toy": True})
    failure = {"error": "original finite-head query overflow", "finite_predictions_saved": False}
    run.prior.write(source / "query_failure.json", failure)

    def forbidden():
        raise AssertionError("Missing heads cannot open scientific geometry or optimize")

    monkeypatch.setattr(run.backend.oracle, "inputs", forbidden)
    run.optimize_cell(spec, {"toy": True})
    result = run.prior.read(run.cell(output, spec) / "selection/optimization.json")
    assert result["status"] == "upstream_fit_failed" and result["optimization"] is None
    assert result["inherited"]["query_failure.json"] == failure
    assert (run.cell(output, spec) / "selection/complete.json").exists()


def test_signed_teacher_accounting_is_identical_to_frozen_procedure():
    result = {"best": {"predicted": 0.65}, "predicted_gain_same_aggregate": 0.04, "predicted_gain_vs_static": 0.02}
    values = {"selected_free": 0.7, "selected_static": 0.71, "selected_aggregate_tied": 0.73}
    truth = {"selected_free": 0.701, "selected_static": 0.72}
    actual = run.backend.accounting(result, values, truth, 0.75)
    assert actual == original.accounting(result, values, truth, 0.75)
    assert actual["signed_regret_vs_teacher_2p_incumbent"] == pytest.approx(-0.001)
    assert actual["true_gain_same_aggregate"] == pytest.approx(0.03)
