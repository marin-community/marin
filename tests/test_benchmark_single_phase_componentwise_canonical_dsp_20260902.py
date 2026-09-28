# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_componentwise_canonical_dsp_20260902 as benchmark,
)


@pytest.mark.parametrize(
    ("name", "expected_rows", "expected_buckets", "expected_components"),
    [
        ("60m_39bucket", 242, 39, (7, 51)),
        ("300m_39bucket", 280, 39, (7, 51)),
        ("delphi_3e18_39bucket", 280, 39, (7, 51)),
        ("dclm_10k", 363, 118, (42,)),
        ("high_quality_10k", 363, 120, (42,)),
    ],
)
def test_core_panel_contracts(name, expected_rows, expected_buckets, expected_components):
    panel = benchmark.load_panel(name, benchmark.DEFAULT_OUTPUT_DIR)

    assert len(panel.runs) == expected_rows
    assert len(panel.buckets) == expected_buckets
    assert tuple(len(group.components) for group in panel.groups) == expected_components
    assert panel.weights.shape == panel.exposures.shape == (expected_rows, expected_buckets)
    assert np.allclose(panel.weights.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(panel.exposures).all()

    for group in panel.groups:
        assert np.max(np.abs(group.outcomes @ group.aggregation_weights - group.aggregate)) < 3e-6


def test_target_group_rejects_wrong_aggregate():
    outcomes = np.asarray([[1.0, 2.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match="component reconstruction"):
        benchmark._target_group(
            name="broken",
            components=("a", "b"),
            outcomes=outcomes,
            aggregate=np.asarray([10.0, 10.0]),
            aggregation_weights=np.asarray([0.5, 0.5]),
            aggregation="mean",
        )


def test_reindex_unique_reorders_and_rejects_duplicates():
    frame = pd.DataFrame({"row_id": ["b", "a"], "value": [2.0, 1.0]})
    reordered = benchmark._reindex_unique(frame, "row_id", ("a", "b"), "test")
    assert reordered["value"].tolist() == [1.0, 2.0]

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate identities"):
        benchmark._reindex_unique(duplicate, "row_id", ("a", "b"), "test")


def test_shard_cache_is_bound_to_protocol(tmp_path: Path):
    group = benchmark.TargetGroup(
        name="metric",
        components=("component",),
        outcomes=np.asarray([[1.0], [2.0]]),
        aggregate=np.asarray([1.0, 2.0]),
        aggregation_weights=np.asarray([1.0]),
        aggregation="identity",
    )
    panel = benchmark.Panel(
        name="panel",
        runs=("a", "b"),
        buckets=("bucket",),
        weights=np.ones((2, 1)),
        exposures=np.ones((2, 1)),
        groups=(group,),
        input_hashes={},
    )
    split = benchmark.Fold(repeat=0, fold=0, train=np.asarray([0]), test=np.asarray([1]))
    path = tmp_path / "shard.npz"
    task = benchmark.FitTask(panel, group, 0, split, path, "protocol-a", ("protocol-a",), 1, 1)
    benchmark.atomic_save(
        path,
        protocol_hash=np.asarray("protocol-a"),
        component=np.asarray("component"),
        test=split.test,
        prediction=np.asarray([1.5]),
    )

    assert benchmark.valid_shard(task)
    assert not benchmark.valid_shard(
        dataclasses.replace(task, protocol_hash="protocol-b", compatible_protocol_hashes=("protocol-b",))
    )


def test_olmix_model_comparison_includes_canonical_and_reference_models():
    aggregate = pd.DataFrame(
        {
            "panel": ["dclm_10k", "high_quality_10k"],
            "target": ["native_42_task_mean", "native_42_task_mean"],
            "model": ["canonical_dsp", "canonical_dsp"],
            "rmse": [0.5, 0.3],
            "rmse_repeat_sd": [0.1, 0.1],
            "spearman": [0.8, 0.7],
            "spearman_repeat_sd": [0.01, 0.01],
            "mean_fold_selection_regret": [0.05, 0.08],
        }
    )

    comparison = benchmark.olmix_model_comparison(aggregate)

    assert len(comparison) == 8
    assert set(comparison["model"]) == {
        "canonical_dsp",
        "olmix_exact_macro",
        "linear_epoch_log_link",
        "dsp_benefit_log_link",
    }
