# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_mean_coordinate_starcoder_20261015 as runner,
)


def test_two_bucket_fit_excludes_held_labels_and_retains_complete_native_competitor() -> None:
    rng = np.random.default_rng(281)
    phase0, phase1 = rng.dirichlet(np.ones(2), 80), rng.dirichlet(np.ones(2), 80)
    phase1[-5:] = phase0[-5:]
    alpha = 3040 / 3814
    c0, c1 = alpha * np.array([1.0, 26.457868]), (1 - alpha) * np.array([1.0, 26.457868])
    shape = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
    early, late = phase0 * c0, phase1 * c1
    matrix, offset = runner.models.design_and_offset(
        early, late, alpha, shape, "hazard_coordinate_mean", {"lambda": 0.5}
    )
    response = runner.models.frozen.temporal.predict_head(
        runner.models.mariner.Head(0.1, np.array([0.5, 0.8, 0.03, 0.02]), 0.6), matrix, offset
    )
    panel = runner.benchmark.data.BenchPanel(
        name="two_bucket_behavior_test",
        buckets=("web", "code"),
        phase0=phase0,
        phase1=phase1,
        alpha=alpha,
        c0=c0,
        c1=c1,
        targets={"test_bpb": response},
        rows=pd.DataFrame({"tied": np.arange(80) >= 75}),
        input_hashes={},
        metadata={"realized_phase0_epochs": early, "realized_phase1_epochs": late},
    )
    train, query, inner = np.arange(60), np.arange(60, 80), np.arange(60) % 3
    poisoned = response.copy()
    poisoned[query] = np.resize(np.array([np.nan, np.inf, -1e100, 1e100]), len(query))
    altered_panel = replace(panel, targets={"test_bpb": poisoned})
    fitted_control, replay_control = None, None
    for family in runner.models.MODEL_NAMES:
        parameters = ({},) if family == "aggregate_mariner" else ({"lambda": 0.0}, {"lambda": 0.5}, {"lambda": 1.0})
        grid = runner.models.TestGrid(parameters, (0.01,), (1.0, 2.5, 6.0))
        fitted = runner.fit_partition(
            panel, "test_bpb", train, inner, family, fitted_control, test_shapes=(shape,), test_grid=grid
        )
        replay = runner.fit_partition(
            altered_panel, "test_bpb", train, inner, family, replay_control, test_shapes=(shape,), test_grid=grid
        )
        assert replay.to_json() == fitted.to_json()
        np.testing.assert_array_equal(
            replay.predict(early[query], late[query]), fitted.predict(early[query], late[query])
        )
        probe = np.column_stack([np.linspace(0, 1, 41), np.linspace(1, 0, 41)])
        expected = fitted.head.predict(runner.models.mariner.design_matrix(probe * (c0 + c1), fitted.shape))
        np.testing.assert_allclose(fitted.predict(probe * c0, probe * c1), expected, rtol=1e-14, atol=1e-14)
        if family == "aggregate_mariner":
            fitted_control, replay_control = fitted, replay
        else:
            assert fitted_control is not None
            assert fitted.inner_rmse <= fitted_control.inner_rmse + 1e-12
            assert fitted.search_space["complete_aggregate_source"] == fitted_control.to_json()


def test_empirical_target_reader_is_blocked_by_wrong_scope_and_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "verify_registration", lambda value: None)
    monkeypatch.setattr(runner.benchmark, "environment_identity", lambda: {})
    sources = {"toy": "first"}
    monkeypatch.setattr(runner, "source_hashes", lambda: sources)
    (tmp_path / "registration.json").write_text("{}\n")
    (tmp_path / "execution_identity.json").write_text(json.dumps({"sources": sources, "environment": {}}))
    approval = {
        "scope": "late_refresh_stage3_paired_noise_and_raw_qualification",
        "registration_sha256": runner.benchmark.sha256(tmp_path / "registration.json"),
        "execution_identity_sha256": runner.benchmark.sha256(tmp_path / "execution_identity.json"),
    }
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))

    def forbidden_reader(*args: object) -> None:
        raise AssertionError("Empirical outcomes were opened before authorization")

    monkeypatch.setattr(runner.benchmark, "read_panel", forbidden_reader)
    with pytest.raises(ValueError, match="Separate exact empirical"):
        runner.summarize()
    approval["scope"] = runner.APPROVAL_SCOPE
    (tmp_path / "execution_approval.json").write_text(json.dumps(approval))
    assert runner.identity()["sources"] == sources
    sources["toy"] = "changed"
    with pytest.raises(ValueError, match="Frozen screen source"):
        runner.summarize()


def test_noisy_prerequisite_requires_all_families_and_sealed_passing_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner.noisy, "OUTPUT", tmp_path)
    current = {"sources": {"toy": "first"}, "environment": {}}
    monkeypatch.setattr(runner.noisy, "source_identity", lambda: current)
    monkeypatch.setattr(runner.noisy, "registration", lambda: {})
    (tmp_path / "registration.json").write_text("{}\n")
    (tmp_path / "execution_identity.json").write_text(json.dumps(current))
    audit = tmp_path / "audit.json"
    audit.write_text('{"passed":true}\n')
    checks = []
    for spec in runner.noisy.context_specs():
        temporal = spec["matching_family"]
        families = ("aggregate_mariner", temporal)
        scores = {
            family: {
                "clean_actual_tied_rmse": 0.01,
                "max_positive_predicted_gain": 0.0,
                "clean_gain_rmse": 0.001 if family == temporal else 0.04,
                "clean_gain_rms": 0.04,
            }
            for family in families
        }
        raw = {
            family: {
                "predicted_gain_same_aggregate": 0.0,
                "phase_gain_overprediction": 0.0,
                "regret_vs_truth_numerical_incumbent": 0.0,
            }
            for family in families
        }
        fits = {
            family: SimpleNamespace(parameters={} if family == "aggregate_mariner" else {"lambda": 0.5})
            for family in families
        }
        checks.extend(
            {**item, "matching_family": temporal}
            for item in runner.noisy.RUNNERS[temporal].qualification_checks(spec, scores, raw, fits, runner.noisy.GATES)
        )
    assert len(checks) == 90 and all(item["passed"] for item in checks)
    qualification = {"identity": current, "contexts": 12, "cells": 24, "passed": True, "checks": checks}

    def write_receipts() -> dict:
        (tmp_path / "qualification.json").write_text(json.dumps(qualification))
        names = ("registration.json", "execution_identity.json", "qualification.json", "audit.json")
        manifest = {"files": {name: {"sha256": runner.benchmark.sha256(tmp_path / name)} for name in names}}
        (tmp_path / "result_manifest.json").write_text(json.dumps(manifest))
        return {
            "noisy_prerequisite": {
                "receipts": {
                    key: runner.benchmark.receipt(tmp_path / name)
                    for key, name in {
                        "registration": "registration.json",
                        "identity": "execution_identity.json",
                        "qualification": "qualification.json",
                        "manifest": "result_manifest.json",
                    }.items()
                },
                "independent_audits": [runner.benchmark.receipt(audit)],
            }
        }

    config = write_receipts()
    runner.verify_noisy_prerequisite(config)
    checks[-1]["passed"] = False
    with pytest.raises(ValueError, match="Every registered check"):
        runner.verify_noisy_prerequisite(write_receipts())
    checks[-1]["passed"] = True
    qualification["checks"] = [item for item in checks if item["matching_family"] == runner.noisy.FAMILIES[0]]
    with pytest.raises(ValueError, match="Every registered check"):
        runner.verify_noisy_prerequisite(write_receipts())
    qualification["checks"] = checks
    config = write_receipts()
    audit.write_text('{"passed":false}\n')
    with pytest.raises(ValueError):
        runner.verify_noisy_prerequisite(config)
