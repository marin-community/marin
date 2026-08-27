# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the materialized fuzzy-cluster text artifact."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.cluster_text import (
    ClusterTextManifest,
    ClusterTextShard,
    read_cluster_text_manifest,
    write_cluster_text_manifest,
)
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams

import experiments.datakit.scripts.fuzzy_cluster_text as cluster_text_script
from experiments.datakit.scripts.fuzzy_cluster_text import (
    TextShard,
    _join_shard,
    build_shards,
    cluster_sort_key,
    load_oversized,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _candidate_artifact(prefix: Path, sources: dict[str, Path]) -> None:
    write_artifact(
        FuzzyDupsAttrData(
            params=MinHashParams(num_perms=16, num_bands=4, ngram_size=5, seed=0),
            sources={
                source_key: FuzzyDupsPerSource(attr_dir=str(path.relative_to(prefix)))
                for source_key, path in sources.items()
            },
            counters={},
        ),
        str(prefix / "candidates"),
    )


def test_build_shards_uses_candidate_source_keys_and_normalized_basenames(tmp_path: Path) -> None:
    first_key = "normalized/first"
    second_key = "normalized/second"
    first_attr = tmp_path / "attributes" / "first"
    second_attr = tmp_path / "attributes" / "second"
    _write_parquet(tmp_path / first_key / "part-1.parquet", [{"id": "a", "text": "first"}])
    _write_parquet(tmp_path / second_key / "part-2.parquet", [{"id": "b", "text": "second"}])
    _write_parquet(
        first_attr / "part-1.parquet",
        [{"id": "a", "dup_cluster_id": "1", "is_cluster_canonical": True}],
    )
    _candidate_artifact(tmp_path, {second_key: second_attr, first_key: first_attr})

    shards = build_shards(str(tmp_path), "candidates", str(tmp_path / "cluster-text"))

    assert [(shard.file_idx, shard.source_key, shard.source_tag, shard.basename) for shard in shards] == [
        (0, first_key, "source_000", "part-1.parquet"),
        (1, second_key, "source_001", "part-2.parquet"),
    ]
    assert shards[0].candidate_path == str(first_attr / "part-1.parquet")
    assert shards[1].candidate_path == str(second_attr / "part-2.parquet")


def test_build_shards_rejects_candidate_shard_absent_from_normalized_source(tmp_path: Path) -> None:
    source_key = "normalized/source"
    attr_dir = tmp_path / "attributes" / "source"
    _write_parquet(tmp_path / source_key / "part-1.parquet", [{"id": "a", "text": "first"}])
    _write_parquet(
        attr_dir / "part-extra.parquet",
        [{"id": "a", "dup_cluster_id": "1", "is_cluster_canonical": True}],
    )
    _candidate_artifact(tmp_path, {source_key: attr_dir})

    with pytest.raises(ValueError, match="unexpected shards"):
        build_shards(str(tmp_path), "candidates", str(tmp_path / "cluster-text"))


def test_join_shard_writes_candidate_text_and_provenance(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.parquet"
    candidates = tmp_path / "candidates.parquet"
    _write_parquet(normalized, [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}])
    _write_parquet(
        candidates,
        [{"id": "b", "dup_cluster_id": "7", "is_cluster_canonical": False}],
    )
    shard = TextShard(
        file_idx=3,
        normalized_path=str(normalized),
        candidate_path=str(candidates),
        source_key="normalized/source",
        source_tag="source_000",
        basename="part.parquet",
    )

    rows = list(_join_shard(shard, {}))

    assert rows == [
        {
            "cluster_key": "7",
            "dup_cluster_id": "7",
            "id": "b",
            "text": "second",
            "text_truncated": False,
            "file_idx": 3,
        }
    ]


def test_join_shard_rejects_candidate_without_normalized_text(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.parquet"
    candidates = tmp_path / "candidates.parquet"
    _write_parquet(normalized, [{"id": "a", "text": "first"}])
    _write_parquet(
        candidates,
        [{"id": "missing", "dup_cluster_id": "7", "is_cluster_canonical": False}],
    )
    shard = TextShard(
        file_idx=0,
        normalized_path=str(normalized),
        candidate_path=str(candidates),
        source_key="normalized/source",
        source_tag="source_000",
        basename="part.parquet",
    )

    with pytest.raises(ValueError, match="IDs absent"):
        list(_join_shard(shard, {}))


def test_oversized_cluster_keeps_equal_text_in_one_split(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.parquet"
    candidates = tmp_path / "candidates.parquet"
    text = "the same normalized text stays together in an oversized cluster"
    _write_parquet(normalized, [{"id": "a", "text": text}, {"id": "b", "text": text}])
    _write_parquet(
        candidates,
        [
            {"id": "a", "dup_cluster_id": "7", "is_cluster_canonical": True},
            {"id": "b", "dup_cluster_id": "7", "is_cluster_canonical": False},
        ],
    )
    shard = TextShard(
        file_idx=0,
        normalized_path=str(normalized),
        candidate_path=str(candidates),
        source_key="normalized/source",
        source_tag="source_000",
        basename="part.parquet",
    )

    rows = list(_join_shard(shard, {"7": 8}))

    assert len({row["cluster_key"] for row in rows}) == 1
    assert rows[0]["cluster_key"].startswith("7:")


def test_load_oversized_uses_the_required_split_count(tmp_path: Path) -> None:
    sizes = tmp_path / "large-clusters.parquet"
    (tmp_path / "summary.json").write_text(json.dumps({"minimum_size": 100}))
    _write_parquet(
        sizes,
        [
            {"dup_cluster_id": "at-cap", "size": 100},
            {"dup_cluster_id": "one-over", "size": 101},
            {"dup_cluster_id": "large", "size": 250},
        ],
    )

    assert load_oversized(str(sizes), max_cluster_size=100) == ({"one-over": 2, "large": 3}, 351)


def test_load_oversized_rejects_a_planner_threshold_above_the_cap(tmp_path: Path) -> None:
    sizes = tmp_path / "large-clusters.parquet"
    (tmp_path / "summary.json").write_text(json.dumps({"minimum_size": 200}))
    _write_parquet(sizes, [{"dup_cluster_id": "large", "size": 250}])

    with pytest.raises(ValueError, match="above the materializer cap"):
        load_oversized(str(sizes), max_cluster_size=100)


def test_cluster_sort_puts_longest_documents_first() -> None:
    records = [
        {"cluster_key": "2", "id": "b", "text": "short"},
        {"cluster_key": "1", "id": "b", "text": "the longest text"},
        {"cluster_key": "1", "id": "a", "text": "short"},
    ]

    ordered = sorted(records, key=cluster_sort_key)

    assert [(record["cluster_key"], record["id"]) for record in ordered] == [("1", "b"), ("1", "a"), ("2", "b")]


def test_cluster_text_manifest_round_trip_preserves_layout(tmp_path: Path) -> None:
    manifest = ClusterTextManifest(
        candidates="candidates",
        max_cluster_size=100,
        output_shards=8,
        groups_per_shard=2,
        split_ngram_size=5,
        oversized_clusters={"7": 3},
        oversized_cluster_members=250,
        shards=[
            ClusterTextShard(
                file_idx=0,
                source_key="normalized/source",
                source_tag="source_000",
                basename="part.parquet",
            )
        ],
    )

    write_cluster_text_manifest(str(tmp_path), manifest)

    assert read_cluster_text_manifest(str(tmp_path)) == manifest


def test_cluster_text_manifest_rejects_noncontiguous_file_indices() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        ClusterTextManifest(
            candidates="candidates",
            max_cluster_size=100,
            output_shards=8,
            groups_per_shard=2,
            split_ngram_size=5,
            oversized_clusters={},
            oversized_cluster_members=0,
            shards=[
                ClusterTextShard(
                    file_idx=1,
                    source_key="normalized/source",
                    source_tag="source_000",
                    basename="part.parquet",
                )
            ],
        )


@pytest.mark.parametrize("field", ["source_tag", "basename"])
def test_cluster_text_manifest_rejects_path_components(field: str) -> None:
    shard = {
        "file_idx": 0,
        "source_key": "normalized/source",
        "source_tag": "source_000",
        "basename": "part.parquet",
    }
    shard[field] = "../outside"

    with pytest.raises(ValueError, match="path components"):
        ClusterTextManifest(
            candidates="candidates",
            max_cluster_size=100,
            output_shards=8,
            groups_per_shard=2,
            split_ngram_size=5,
            oversized_clusters={},
            oversized_cluster_members=0,
            shards=[ClusterTextShard.model_validate(shard)],
        )


def test_duplicate_candidate_ids_are_rejected(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.parquet"
    candidates = tmp_path / "candidates.parquet"
    _write_parquet(normalized, [{"id": "a", "text": "first"}])
    _write_parquet(
        candidates,
        [
            {"id": "a", "dup_cluster_id": "7"},
            {"id": "a", "dup_cluster_id": "8"},
        ],
    )
    shard = TextShard(
        file_idx=0,
        normalized_path=str(normalized),
        candidate_path=str(candidates),
        source_key="normalized/source",
        source_tag="source_000",
        basename="part.parquet",
    )

    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        list(_join_shard(shard, {}))


def test_truncated_text_is_marked_for_conservative_verification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cluster_text_script, "MAXIMUM_VERIFICATION_TEXT_CHARS", 8)
    normalized = tmp_path / "normalized.parquet"
    candidates = tmp_path / "candidates.parquet"
    _write_parquet(normalized, [{"id": "a", "text": "a document longer than the test cap"}])
    _write_parquet(candidates, [{"id": "a", "dup_cluster_id": "7"}])
    shard = TextShard(
        file_idx=0,
        normalized_path=str(normalized),
        candidate_path=str(candidates),
        source_key="normalized/source",
        source_tag="source_000",
        basename="part.parquet",
    )

    (row,) = list(_join_shard(shard, {}))

    assert row["text"] == ""
    assert row["text_truncated"] is True


def test_repeated_normalized_id_requires_equal_text(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.parquet"
    candidates = tmp_path / "candidates.parquet"
    _write_parquet(normalized, [{"id": "a", "text": "first"}, {"id": "a", "text": "different"}])
    _write_parquet(candidates, [{"id": "a", "dup_cluster_id": "7"}])
    shard = TextShard(
        file_idx=0,
        normalized_path=str(normalized),
        candidate_path=str(candidates),
        source_key="normalized/source",
        source_tag="source_000",
        basename="part.parquet",
    )

    with pytest.raises(ValueError, match="inconsistent text"):
        list(_join_shard(shard, {}))
