# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CV overflow preservation and immutable conditional-fit adoption regressions."""

import gzip
import json

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_whole_response_clean_v2_20261015 as model


@pytest.fixture
def checkpoint_root(tmp_path, monkeypatch):
    root = tmp_path / "continuation"
    root.mkdir()
    adoption = root / "adoption_manifest.json"
    adoption.write_text(json.dumps({"records": {}, "original_execution_identity_sha256": "old-identity"}))
    monkeypatch.setattr(model, "OUTPUT", root)
    monkeypatch.setattr(model, "ADOPTION", adoption)
    model.adoption_manifest.cache_clear()
    yield root
    model.adoption_manifest.cache_clear()


def finite_head():
    return {
        "selected": {
            "amplitudes": [1.0],
            "floor": 0.0,
            "scale": 1.0,
            "normalization": 0.0,
            "start_index": 0,
            "near_constant_phi": False,
            "nonregular_learned_floor": False,
            "training_exponential_underflow_count": 0,
            "first_order_tolerance_met": False,
        },
        "unresolved": True,
        "solves": [{"success": False, "start_index": 0, "termination": "evaluation_cap"}],
        "trials": [{"status": "finite", "objective": 0.5}],
    }


def write_checkpoint(path, matrix, observed, spec, identity):
    fit = finite_head()
    record = {
        "header": {
            "spec": spec,
            "identity_sha256": identity,
            "input_sha256": model.prior.canonical_hash({"matrix": matrix.tolist(), "observed": observed.tolist()}),
        },
        "fit_sha256": model.prior.canonical_hash(fit),
        "fit": fit,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(record, sort_keys=True).encode(), mtime=0))
    return record


def test_finite_predictions_with_unrepresentable_mse_remain_inspectable(checkpoint_root):
    matrix = np.full((9, 1), np.log(1e200))
    observed = np.ones(9)
    labels = np.arange(9) % 3
    spec = {"shape_index": 0, "strength_index": 0}
    for label in range(3):
        train = labels != label
        write_checkpoint(
            checkpoint_root / f"inner{label}.json.gz",
            matrix[train],
            observed[train],
            spec | {"inner_fold": label},
            "toy-identity",
        )
    profile = model.profile_candidate(matrix, observed, labels, checkpoint_root, spec, "toy-identity")
    assert not profile["eligible_for_selection"] and profile["inner_mse"] is None
    assert profile["cv_loss_failure"] == "cv_mean_squared_error_exceeds_float64"
    assert np.isfinite(profile["inner_predictions"]).all()
    np.testing.assert_allclose(profile["inner_predictions"], 1e200, rtol=3e-14, atol=0)
    assert len(profile["conditional_records"]) == 3 and not profile["prediction_failure_folds"]
    assert json.loads(json.dumps(profile, allow_nan=False)) == profile
    finite = {"eligible_for_selection": True, "inner_mse": 1.0, "spec": {"strength_index": 0, "shape_index": 1}}
    assert model.backend.choose([profile, finite]) == finite
    with pytest.raises(ArithmeticError):
        model.backend.choose([profile])


def test_scaled_fallback_recovers_representable_mse_without_changing_ordinary_scores():
    predicted = np.zeros(1000)
    predicted[0] = 1e155
    mse, status = model.finite_cv_mse(predicted, np.zeros_like(predicted))
    assert mse == pytest.approx(1e307, rel=4e-16)
    assert status == "scaled_mean_after_intermediate_overflow"
    observed = np.linspace(0.8, 1.4, 101)
    ordinary_prediction = observed + np.sin(np.arange(101)) * 0.017
    score, status = model.finite_cv_mse(ordinary_prediction, observed)
    assert score == float(np.mean((ordinary_prediction - observed) ** 2))
    assert status == "original_finite_arithmetic"


def test_adoption_preserves_complete_fit_and_checks_original_inputs(checkpoint_root, tmp_path):
    matrix, observed = np.arange(8, dtype=float).reshape(-1, 1), np.linspace(0.8, 1.2, 8)
    spec = {"world": "invented", "inner_fold": 0}
    old_path = tmp_path / "preserved" / "conditional.json.gz"
    old = write_checkpoint(old_path, matrix, observed, spec, "old-identity")
    old_bytes = old_path.read_bytes()
    entry = {
        "path": str(old_path),
        "sha256": model.prior.digest(old_path),
        "header": old["header"],
        "fit_sha256": old["fit_sha256"],
    }
    model.ADOPTION.write_text(
        json.dumps({"records": {"conditional.json.gz": entry}, "original_execution_identity_sha256": "old-identity"})
    )
    destination = checkpoint_root / "conditional.json.gz"
    fitted = model.conditional_fit(destination, matrix, observed, spec, "new-identity")
    new = json.loads(gzip.decompress(destination.read_bytes()))
    assert fitted == old["fit"] and new["fit"] == old["fit"]
    assert new["fit_sha256"] == old["fit_sha256"]
    assert new["header"]["identity_sha256"] == "new-identity"
    assert new["adopted_from"]["sha256"] == entry["sha256"]
    assert new["adopted_from"]["identity_sha256"] == "old-identity"
    assert old_path.read_bytes() == old_bytes
    destination_bytes = destination.read_bytes()
    assert model.conditional_fit(destination, matrix, observed, spec, "new-identity") == fitted
    with pytest.raises(ValueError):
        model.conditional_fit(destination, matrix, observed + 0.1, spec, "new-identity")
    assert destination.read_bytes() == destination_bytes and old_path.read_bytes() == old_bytes


def test_interrupted_destination_is_preserved_before_adoption(checkpoint_root):
    destination = checkpoint_root / "conditional.json.gz"
    partial = destination.with_name(destination.name + ".writing")
    partial.write_bytes(b"interrupted, not a completed fit")
    with pytest.raises(ValueError):
        model.conditional_fit(destination, np.zeros((4, 1)), np.ones(4), {"toy": True}, "new-identity")
    assert partial.read_bytes() == b"interrupted, not a completed fit"
    assert not destination.exists()
