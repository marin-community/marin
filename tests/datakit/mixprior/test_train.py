# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from marin.execution.artifact import read_artifact

from experiments.datakit.mixprior.generate import main as generate
from experiments.datakit.mixprior.model import fit
from experiments.datakit.mixprior.objective import fit_objective
from experiments.datakit.mixprior.train import GPArtifact, load_model, save_model


def test_moved_gp_artifact_preserves_predictions_and_generates_named_candidates(tmp_path, monkeypatch, data):
    data = replace(data, weights=data.weights.copy())
    data.weights[1] = data.weights[0]
    objective = fit_objective(data, metrics=("reward", "guard"), targets=("reward",))
    model = fit(data, *objective(data.outcomes))
    points = np.random.default_rng(19).dirichlet([1, 2, 3], size=(5, 2))
    expected_mean, expected_variance = model.predict(points)

    monkeypatch.chdir(tmp_path)
    save_model(Path("gp"), model, data, objective, "test-revision")
    Path("gp").rename("moved-gp")
    metadata = read_artifact(str(tmp_path / "moved-gp"), GPArtifact)
    assert (metadata.swarm_id, metadata.hf_revision) == (data.name, "test-revision")
    assert metadata.components == data.components
    assert metadata.observation_ids == data.observation_ids
    assert metadata.objective_metrics == ["reward", "guard"]
    assert metadata.target_metrics == ["reward"]
    artifact, restored, observed = load_model(Path("moved-gp"))
    assert artifact.path == str(tmp_path / "moved-gp")
    np.testing.assert_array_equal(observed, data.weights)
    actual_mean, actual_variance = restored.predict(points)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_variance, expected_variance, rtol=1e-12, atol=1e-12)

    monkeypatch.setattr(
        "sys.argv",
        ["generate", "--model", "moved-gp", "--output", "candidates.parquet", "--pool-size", "128", "--batch-size", "2"],
    )
    generate()
    assert Path("candidates.parquet").is_file()
    rows = pq.read_table("candidates.parquet").to_pylist()
    assert len(rows) == 2
    assert all(row["swarm_id"] == data.name and row["hf_revision"] == "test-revision" for row in rows)
    weights = np.array(
        [[[row[f"phase{phase}_weights"][name] for name in data.components] for phase in range(2)] for row in rows]
    )
    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-12)
