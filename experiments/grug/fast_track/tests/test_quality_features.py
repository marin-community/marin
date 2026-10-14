# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from zephyr.stage_io import ZephyrWorkerError

from experiments.datakit import hero_data
from experiments.grug.fast_track.corpus_sample import CorpusSource
from experiments.grug.fast_track.quality_features import (
    FEATURE_BATCH_ROWS,
    HARRIER_DIM,
    HARRIER_FEATURE_IDENTITY,
    QualityFeatureSource,
    _join_source_shard,
    _NormalizedRowLocator,
    normalize_harrier_embeddings,
    pinned_quality_feature_sources,
    prepare_quality_features,
)


@pytest.fixture
def feature_pool(tmp_path, write_raw_pool):
    normalized_path = tmp_path / "normalized.parquet"
    harrier_path = tmp_path / "harrier" / normalized_path.name
    harrier_path.parent.mkdir()
    normalized_rows = [{"id": f"doc-{index}", "text": f"normalized text {index}"} for index in range(12)]
    embeddings = []
    for index in range(12):
        vector = np.zeros(1024, dtype=np.int8)
        vector[index] = 127
        embeddings.append({"id": f"doc-{index}", "embedding": vector})
    pq.write_table(pa.Table.from_pylist(normalized_rows), normalized_path, row_group_size=3)
    pq.write_table(
        pa.Table.from_pylist(
            embeddings,
            schema=pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.int8(), 1024))]),
        ),
        harrier_path,
        row_group_size=3,
    )
    raw_rows = []
    for index in range(12):
        raw_rows.append(
            {
                "source": "source",
                "id": f"doc-{index}",
                "sample_rank": hashlib.sha256(f"seed:{index}".encode()).hexdigest(),
                "duplicate_group": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
                "normalized_shard": str(normalized_path),
                "normalized_row": index,
                "text": f"normalized text {index}",
                "input_ids": [index + 1] * 5,
                "token_count": 5,
            }
        )
    raw_rows.sort(key=lambda row: (row["sample_rank"], row["source"], row["id"]))
    pool = write_raw_pool(raw_rows, sources=(CorpusSource("source", str(tmp_path), 60),))
    source = QualityFeatureSource("source", str(tmp_path), str(tmp_path / "harrier"))
    return pool, source, harrier_path


def test_prepared_features_keep_raw_prefix_order_and_cached_vectors(feature_pool, zephyr_context, tmp_path):
    pool, source, _ = feature_pool
    prepared = prepare_quality_features(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "features"),
        token_budget=40,
        sources=(source,),
    )
    expected_raw_ids = [row["id"] for path in prepared.raw_prefix_shards for row in pq.read_table(path).to_pylist()]
    feature_rows = [row for item in prepared.feature_shards for row in pq.read_table(item.path).to_pylist()]

    assert prepared.requested_tokens == 40
    assert prepared.actual_tokens == 40
    assert [row["id"] for row in feature_rows] == expected_raw_ids
    assert [row["raw_row_index"] for row in feature_rows] == list(range(6)) + list(range(2))
    assert prepared.feature_identity["model"] == HARRIER_FEATURE_IDENTITY["model"]
    for row in feature_rows:
        expected_index = int(row["id"].split("-")[-1])
        assert np.argmax(row["embedding"]) == expected_index


def test_prepared_features_fail_when_cached_harrier_id_differs(feature_pool, zephyr_context, tmp_path):
    pool, source, harrier_path = feature_pool
    rows = pq.read_table(harrier_path).to_pylist()
    rows[0]["id"] = "wrong-id"
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.int8(), 1024))]),
        ),
        harrier_path,
        row_group_size=3,
    )

    with pytest.raises(ZephyrWorkerError, match="Harrier ID differs from normalized ID"):
        prepare_quality_features(
            pool,
            ctx=zephyr_context,
            output_path=str(tmp_path / "mismatched-features"),
            token_budget=60,
            sources=(source,),
        )


def test_harrier_normalization_matches_fit_and_score_inputs():
    raw = np.asarray([[0, 127, 0], [0, 0, 0]], dtype=np.int8)

    normalized = normalize_harrier_embeddings(np.pad(raw, ((0, 0), (0, 1021))))

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 0.0], atol=1e-6)
    assert normalized[0, 1] == pytest.approx(1.0)


def test_pinned_feature_sources_resolve_complete_harrier_paths():
    sources = [
        CorpusSource("biocorpus", "s3://bucket/normalized/biocorpus", 1),
        CorpusSource("agenttrove", "s3://bucket/normalized/agenttrove", 1),
    ]

    pins = pinned_quality_feature_sources(sources)

    assert [pin.source for pin in pins] == ["agenttrove", "biocorpus"]
    for pin in pins:
        assert pin.harrier_path == f"{hero_data.MANIFEST_PREFIX}/{hero_data.pinned_map('harrier')[pin.source]}"
        assert "/embed/harrier-all/" in pin.harrier_path


def test_harrier_join_reads_sparse_rows_across_batches_and_row_groups(tmp_path):
    rows = 3 * FEATURE_BATCH_ROWS
    ids = [f"{index:032x}" for index in range(rows)]
    (tmp_path / "normalized").mkdir()
    (tmp_path / "harrier").mkdir()
    normalized_shard = str(tmp_path / "normalized" / "shard.parquet")
    pq.write_table(pa.table({"id": ids, "text": ["text"] * rows}), normalized_shard)
    embeddings = pa.array([[index % 127] * HARRIER_DIM for index in range(rows)], type=pa.list_(pa.int8(), HARRIER_DIM))
    pq.write_table(
        pa.table({"id": ids, "embedding": embeddings}),
        tmp_path / "harrier" / "shard.parquet",
        row_group_size=2 * FEATURE_BATCH_ROWS,
    )
    pins = {"source": QualityFeatureSource("source", str(tmp_path / "normalized"), str(tmp_path / "harrier"))}
    selected = [5, FEATURE_BATCH_ROWS - 1, FEATURE_BATCH_ROWS, 2 * FEATURE_BATCH_ROWS, rows - 1]
    locators = [
        _NormalizedRowLocator("source", normalized_shard, row, 0, index, ids[row]) for index, row in enumerate(selected)
    ]

    joined = list(_join_source_shard(normalized_shard, iter(locators), source_pins=pins))

    assert [(row["id"], row["embedding"][0]) for row in joined] == [(ids[row], row % 127) for row in selected]
