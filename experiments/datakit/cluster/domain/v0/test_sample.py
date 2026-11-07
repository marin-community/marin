# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig

from experiments.datakit.cluster.domain.v0.sample import sample_centroid_inputs
from experiments.datakit.embeddings.luxical.pipeline import (
    LUXICAL_DIM,
    QUANT_RANGE,
    QUANT_SCALE,
    EmbeddingAttrData,
)

_SEED = 7
_PER_SHARD = 12
_EMBEDDING_TYPE = pa.list_(pa.int8(), LUXICAL_DIM)


def _write_embedding_shard(path, embeddings: np.ndarray, row_group_size: int) -> None:
    ids = [f"doc-{i:04d}" for i in range(len(embeddings))]
    schema = pa.schema([pa.field("id", pa.string()), pa.field("embedding", _EMBEDDING_TYPE)])
    table = pa.table({"id": ids, "embedding": embeddings.tolist()}, schema=schema)
    pq.write_table(table, str(path), row_group_size=row_group_size)


def _expected_sample(embeddings: np.ndarray, shard_idx: int) -> list[list[int]]:
    # Restates the sampling rule on purpose: a rerun resumes over files written by earlier
    # code, so the draw per (seed, shard) must stay exactly what it has always been.
    rows = embeddings.tolist()
    if _PER_SHARD >= len(rows):
        return rows
    idx = np.random.default_rng(_SEED + shard_idx).choice(len(rows), size=_PER_SHARD, replace=False)
    return [rows[i] for i in idx]


def test_sample_centroid_inputs_draws_seeded_rows_per_shard_in_draw_order(tmp_path):
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir()
    rng = np.random.default_rng(0)
    # (rows, row_group_size): two sampled shards, a keep-all shard spanning three row groups,
    # one shard of exactly _PER_SHARD rows, and an empty shard.
    layout = [(50, 7), (40, 7), (10, 4), (_PER_SHARD, 5), (0, 7)]
    shards = [rng.integers(-127, 128, size=(rows, LUXICAL_DIM), dtype=np.int8) for rows, _ in layout]
    for i, (embeddings, (_, row_group_size)) in enumerate(zip(shards, layout, strict=True)):
        _write_embedding_shard(embed_dir / f"part-{i:05d}.parquet", embeddings, row_group_size)
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
        n_per_source=_PER_SHARD * len(shards),
        seed=_SEED,
        worker_resources=ResourceConfig(cpu=1, ram="1g"),
        max_workers=2,
        parallel_sources=1,
    )

    for i, embeddings in enumerate(shards):
        written = pq.read_table(str(tmp_path / "out" / "web-a" / f"sample-{i:06d}.parquet"))
        expected = _expected_sample(embeddings, i)
        assert written.column("embedding").to_pylist() == expected
        assert written.column("source").to_pylist() == ["web/a"] * len(expected)
        assert written.schema.field("embedding").type == _EMBEDDING_TYPE
