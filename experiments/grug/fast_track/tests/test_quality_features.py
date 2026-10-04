# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.stage_io import ZephyrWorkerError

from experiments.grug.fast_track.corpus_sample import CorpusSource, RawCorpusPool
from experiments.grug.fast_track.quality_features import (
    HARRIER_FEATURE_IDENTITY,
    QualityFeatureSource,
    normalize_harrier_embeddings,
    prepare_quality_features,
)
from experiments.grug.fast_track.ranked_pool import RangeTokenTotal


@pytest.fixture
def zephyr_context(tmp_path):
    client = LocalClient()
    context = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="quality-features-test",
    )
    yield context
    context.shutdown()
    client.shutdown(wait=True)


@pytest.fixture
def feature_pool(tmp_path):
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
    shard_paths = (tmp_path / "raw-0.parquet", tmp_path / "raw-1.parquet")
    pq.write_table(pa.Table.from_pylist(raw_rows[:6]), shard_paths[0])
    pq.write_table(pa.Table.from_pylist(raw_rows[6:]), shard_paths[1])
    pool = RawCorpusPool(
        tokenizer="quality-feature-test",
        tokenizer_hash="a" * 64,
        sources=(CorpusSource("source", str(tmp_path), 60),),
        seed=1,
        requested_tokens=60,
        actual_tokens=60,
        documents=12,
        shards=tuple(map(str, shard_paths)),
        range_totals=(
            RangeTokenTotal(0, str(shard_paths[0]), 6, 30),
            RangeTokenTotal(1, str(shard_paths[1]), 6, 30),
        ),
        source_tokens={"source": 60},
        manifest_path=str(tmp_path / "raw.json"),
    )
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
