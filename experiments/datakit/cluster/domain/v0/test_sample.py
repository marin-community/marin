# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig

from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs
from experiments.datakit.embeddings.luxical.pipeline import (
    _EMBEDDING_SCHEMA,
    LUXICAL_DIM,
    QUANT_RANGE,
    QUANT_SCALE,
    EmbeddingAttrData,
)

_SEED = 7


def _write_embedding_shard(path, embeddings: np.ndarray, row_group_size: int) -> None:
    ids = [f"doc-{i:04d}" for i in range(len(embeddings))]
    table = pa.table({"id": ids, "embedding": embeddings.tolist()}, schema=_EMBEDDING_SCHEMA)
    pq.write_table(table, str(path), row_group_size=row_group_size)


def _expected_sample(embeddings: np.ndarray, shard_idx: int, per_shard: int) -> list[list[int]]:
    rows = embeddings.tolist()
    if per_shard >= len(rows):
        return rows
    idx = np.random.default_rng(_SEED + shard_idx).choice(len(rows), size=per_shard, replace=False)
    return [rows[i] for i in idx]


def test_sample_centroid_inputs_draws_seeded_rows_per_shard_in_draw_order(tmp_path):
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir()
    rng = np.random.default_rng(0)
    shards = [
        rng.integers(-127, 128, size=(50, LUXICAL_DIM), dtype=np.int8),
        rng.integers(-127, 128, size=(3, LUXICAL_DIM), dtype=np.int8),
        np.empty((0, LUXICAL_DIM), dtype=np.int8),
    ]
    for i, embeddings in enumerate(shards):
        _write_embedding_shard(embed_dir / f"part-{i:05d}.parquet", embeddings, row_group_size=7)
    attr = EmbeddingAttrData(
        output_dir=str(embed_dir),
        source_key="web/a",
        model_name="test",
        embedding_dim=LUXICAL_DIM,
        quantization_scale=QUANT_SCALE,
        quantization_range=QUANT_RANGE,
        batch_size=7,
    )

    sample_centroid_inputs(
        str(tmp_path / "out"),
        {"web/a": attr},
        n_per_source=60,
        seed=_SEED,
        worker_resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=2,
        parallel_sources=1,
    )

    per_shard = 60 // len(shards)
    for i, embeddings in enumerate(shards):
        written = pq.read_table(str(tmp_path / "out" / "web-a" / f"sample-{i:06d}.parquet"))
        assert written.column("embedding").to_pylist() == _expected_sample(embeddings, i, per_shard)
        assert set(written.column("source").to_pylist()) <= {"web/a"}
        assert written.schema.field("embedding").type == _EMBEDDING_SCHEMA.field("embedding").type
