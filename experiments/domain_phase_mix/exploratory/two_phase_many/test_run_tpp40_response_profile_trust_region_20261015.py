# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Durable selection before held-label access, and receipt adoption boundaries."""

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_response_profile_trust_region_20261015 as runner,
)


def toy_training():
    t = np.linspace(0.05, 2, 23)
    matrix = np.column_stack((-t, t**2))
    observed = 0.6 + np.exp(matrix @ np.array([0.7, 0.08]))
    return {
        "matrix": matrix.tolist(),
        "observed": observed.tolist(),
        "train_indices": list(range(len(t))),
        "starts": [{"amplitudes": [0.35 + 0.1 * i, 0.04 + 0.01 * i]} for i in range(5)],
    }


def test_held_label_poisoning_changes_scores_only_after_durable_fit(tmp_path, monkeypatch):
    saved = toy_training()
    matrix, observed = np.asarray(saved["matrix"]), np.asarray(saved["observed"])
    selected = runner.model.prior.evaluate(matrix, observed, np.array([0.7, 0.08]), 0.6)
    old_fit = {
        "selected": selected,
        "unresolved": False,
        "selected_from_successful_solve": True,
        "solves": [{"success": True}] * 5,
    }
    query_matrix = matrix[[1, 5, 1, 5, 10]]
    query = pd.DataFrame(
        {
            "role": ["asymmetric", "asymmetric", "matched_tied", "matched_tied", "actual_tied"],
            "source_index": [1, 5, 1, 5, 10],
            "row_id": ["a", "b", "a", "b", "c"],
        }
    )
    prediction = runner.model.prior.predict(query_matrix, selected)
    read_csv = pd.read_csv
    opened_after_saved = []
    for poison in (0.0, 100.0):
        target = tmp_path / str(poison)
        source = tmp_path / f"previous_{poison}.csv"
        query.assign(predicted=prediction, truth=prediction + poison).to_csv(source, index=False)

        def read_after_fit(path, target=target, **kwargs):
            opened_after_saved.append((target / "fit.json").exists())
            return read_csv(path, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(pd, "read_csv", read_after_fit)
            runner.fit_and_score(saved, runner.ARMS[0], target, query_matrix, query, old_fit, source)
    assert opened_after_saved == [True, True]
    assert (tmp_path / "0.0/fit.json").read_bytes() == (tmp_path / "100.0/fit.json").read_bytes()
    clean = runner.prior.read(tmp_path / "0.0/scores.json")
    poisoned = runner.prior.read(tmp_path / "100.0/scores.json")
    assert clean["comparison"] == poisoned["comparison"]
    assert clean["new"]["query_rmse"] < 1e-7
    assert poisoned["new"]["query_rmse"] > 99


def test_initialization_rejects_foreign_identity_and_modified_training_values(tmp_path):
    saved = toy_training()
    spec, frozen = {"world": "invented"}, {"source_hashes": {"solver.py": "original"}}
    runner.prior.write(tmp_path / "starts.json", saved)
    runner.prior.finish(tmp_path, {**spec, "stage": "five_gamma_starts"}, frozen, ("starts.json",), ())
    assert runner.saved_initialization(tmp_path, spec, frozen) == saved
    with pytest.raises(ValueError, match="another execution"):
        runner.saved_initialization(tmp_path, spec, {"source_hashes": {"solver.py": "changed"}})
    saved["observed"][0] += 1
    runner.prior.write(tmp_path / "starts.json", saved)
    with pytest.raises(ValueError, match="Frozen input changed"):
        runner.saved_initialization(tmp_path, spec, frozen)
