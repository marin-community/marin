# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the large fuzzy-cluster planner."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams

from experiments.datakit.scripts.fuzzy_large_clusters import _sample_indices, candidate_shard_paths, main


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_candidate_shard_paths_skip_a_source_with_no_candidate_files(tmp_path: Path) -> None:
    first_attr = tmp_path / "attributes" / "first"
    second_attr = tmp_path / "attributes" / "second"
    _write_parquet(first_attr / "part-1.parquet", [{"id": "a", "dup_cluster_id": "1"}])
    write_artifact(
        FuzzyDupsAttrData(
            params=MinHashParams(num_perms=16, num_bands=4, ngram_size=5, seed=0),
            sources={
                "normalized/second": FuzzyDupsPerSource(attr_dir="attributes/second"),
                "normalized/first": FuzzyDupsPerSource(attr_dir="attributes/first"),
            },
            counters={},
        ),
        str(tmp_path / "candidates"),
    )

    assert candidate_shard_paths(str(tmp_path), "candidates") == [str(first_attr / "part-1.parquet")]
    assert not second_attr.exists()


def test_hash_sample_is_independent_of_row_position() -> None:
    ids = [f"document-{index}" for index in range(100)]
    selected = {ids[index] for index in _sample_indices(ids, stride=8)}
    reversed_ids = list(reversed(ids))
    selected_reversed = {reversed_ids[index] for index in _sample_indices(reversed_ids, stride=8)}

    assert selected
    assert selected_reversed == selected


def test_planner_ignores_count_files_from_previous_runs(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "attributes" / "part-1.parquet",
        [{"id": "a", "dup_cluster_id": "7"}, {"id": "b", "dup_cluster_id": "7"}],
    )
    write_artifact(
        FuzzyDupsAttrData(
            params=MinHashParams(num_perms=16, num_bands=4, ngram_size=5, seed=0),
            sources={"normalized/source": FuzzyDupsPerSource(attr_dir="attributes")},
            counters={},
        ),
        str(tmp_path / "candidates"),
    )
    output = tmp_path / "plan"
    _write_parquet(output / "counts" / "part-99999.parquet", [{"dup_cluster_id": "7", "n": 999}])

    main(
        [
            "--prefix",
            str(tmp_path),
            "--candidates",
            "candidates",
            "--out",
            str(output),
            "--stride",
            "1",
            "--minimum-size",
            "1",
            "--max-workers",
            "1",
            "--worker-cpu",
            "1",
            "--worker-ram",
            "1g",
            "--task-ram",
            "1g",
        ]
    )

    assert pq.read_table(output / "large_clusters.parquet").to_pylist() == [{"dup_cluster_id": "7", "size": 2}]
    assert json.loads((output / "summary.json").read_text())["candidates"] == str(tmp_path / "candidates")
