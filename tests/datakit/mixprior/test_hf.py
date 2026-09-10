# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.mixprior.acquisition import fit_additive
from experiments.datakit.mixprior.hf import load_data, write_candidates
from experiments.datakit.mixprior.objective import fit_objective
from experiments.datakit.mixprior.observations import group_observations
from experiments.datakit.mixprior.search import search


def test_named_hf_data_runs_through_fit_search_and_parquet_output(tmp_path, data):
    root = tmp_path / "registry/v1/swarms/test"
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"swarm_id": "test", "phase_budgets": data.phase_budgets.tolist()}]),
        root / "swarm.parquet",
    )
    cells = [
        {"cell": name, "domain": domain, "available_tokens": tokens, "quality": quality}
        for name, domain, tokens, quality in zip(
            data.components, data.domains, data.available_tokens.tolist(), data.quality.tolist(), strict=True
        )
    ]
    pq.write_table(pa.Table.from_pylist([{"cells": cells}]), root / "buckets.parquet")
    weights = data.weights.copy()
    weights[1] = weights[0]
    rows = [
        {
            "observation_id": f"test:{i}",
            "swarm_id": "test",
            "group": data.groups[i],
            "phase0_weights": dict(zip(data.components[::-1], phases[0, ::-1].tolist(), strict=True)),
            "phase1_weights": dict(zip(data.components[::-1], phases[1, ::-1].tolist(), strict=True)),
            "grouped_bpb": {
                "reward": data.outcomes[i, 0],
                "guard": data.outcomes[i, 1],
                "belebele_a": 1.0,
                "belebele_b": 3.0,
                "belebele_mean": 999.0,
            },
        }
        for i, phases in enumerate(weights)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "observations.parquet")
    loaded = load_data(tmp_path, "test")
    np.testing.assert_array_equal(loaded.weights, weights)
    np.testing.assert_array_equal(loaded.quality, data.quality)
    np.testing.assert_array_equal(loaded.outcomes[:, loaded.labels.index("belebele_mean")], 2)
    objective = fit_objective(loaded, metrics=("reward", "guard"), targets=("reward",))
    grouped = group_observations(loaded)
    model = fit_additive(grouped.data, objective, grouped.counts, jax.devices("cpu")[0])
    selected = search(model, loaded.available_tokens, loaded.phase_budgets, loaded.weights, pool_size=128, batch_size=2)
    path = tmp_path / "candidates.parquet"
    write_candidates(path, selected, loaded.name, loaded.components, "test-revision")
    records = pq.read_table(path).to_pylist()
    assert len(records) == 2
    restored = np.array(
        [[[row[f"phase{phase}_weights"][name] for name in data.components] for phase in range(2)] for row in records]
    )
    np.testing.assert_array_equal(restored, selected)
    assert all(row["swarm_id"] == "test" and row["hf_revision"] == "test-revision" for row in records)
