# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pyarrow.parquet as pq
from marin.execution.artifact import read_artifact

from experiments.datakit.mixprior.acquisition import KernelConfig
from experiments.datakit.mixprior.calibration import fit_calibrated
from experiments.datakit.mixprior.generate import main as generate
from experiments.datakit.mixprior.objective import Objective
from experiments.datakit.mixprior.observations import group_observations
from experiments.datakit.mixprior.train import MetricArtifact, load_model, save_model


def test_moved_metric_artifact_preserves_acquisition_and_generates_named_candidates(tmp_path, monkeypatch, data):
    data = replace(data, weights=data.weights.copy())
    data.weights[1] = data.weights[0]
    objective = Objective(
        np.array([0, 1]), np.array([True, False]), np.full(2, 3.0), np.full(2, 0.1), np.eye(2) * 0.001, 0.0
    )
    grouped = group_observations(data)
    model = fit_calibrated(
        grouped.data,
        objective,
        grouped.counts,
        np.arange(10),
        jax.devices("cpu")[0],
        kernel_config=KernelConfig(matern=0.3, intercept=12.0, features=2.0, trend=0.2),
    )
    points = np.random.default_rng(19).dirichlet([1, 2, 3], size=(5, 2))
    expected_scores = model.acquisition(points)
    expected_moments = model.predict_metrics(points)

    monkeypatch.chdir(tmp_path)
    save_model(Path("gp"), model, data, objective, "test-revision", "previous-revision", data.observation_ids[:10])
    Path("gp").rename("moved-gp")
    metadata = read_artifact(str(tmp_path / "moved-gp"), MetricArtifact)
    assert (metadata.swarm_id, metadata.hf_revision) == (data.name, "test-revision")
    assert metadata.components == data.components
    assert metadata.observation_ids == data.observation_ids
    assert metadata.objective_metrics == ["reward", "guard"]
    assert metadata.target_metrics == ["reward"]
    artifact, restored, observed = load_model(Path("moved-gp"))
    assert artifact.path == str(tmp_path / "moved-gp")
    np.testing.assert_array_equal(observed, data.weights)
    np.testing.assert_allclose(restored.acquisition(points), expected_scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(restored.predict_metrics(points), expected_moments, rtol=1e-12, atol=1e-12)

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
