# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.cluster.domain.v1 import materialize_10pct_harrier_tokens
from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM


def test_attach_sample_records_uses_row_locators(tmp_path):
    path = tmp_path / "sample.parquet"
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    pq.write_table(pa.table({"id": ["a", "b", "c"], "text": ["alpha", "beta", "gamma"]}), path)
    pq.write_table(
        pa.table({"id": ["a", "b", "c"], "input_ids": pa.array([[1], [2], [3]], type=pa.list_(pa.int32()))}),
        token_dir / path.name,
    )
    items = iter(
        [
            {"source": "example", "id": "a", "sample_path": str(path), "row_index": 0, "basename": "x"},
            {"source": "example", "id": "c", "sample_path": str(path), "row_index": 2, "basename": "y"},
        ]
    )

    records = list(
        materialize_10pct_harrier_tokens._attach_sample_records(
            ("example", str(path)), items, marin_dirs={"example": str(token_dir)}
        )
    )

    assert records == [
        {
            "source": "example",
            "id": "a",
            "record": {"id": "a", "text": "alpha"},
            "basename": "x",
            "marin_input_ids": [1],
        },
        {
            "source": "example",
            "id": "c",
            "record": {"id": "c", "text": "gamma"},
            "basename": "y",
            "marin_input_ids": [3],
        },
    ]


def test_materialize_group_coalesces_embeddings_and_left_joins_attributes(tmp_path, monkeypatch):
    source = "example"
    basename = "part.parquet"
    sample_dir = tmp_path / "sample"
    main_dir = tmp_path / "main"
    fuzzy_dir = tmp_path / "fuzzy"
    nemotron_dir = tmp_path / "nemotron"
    quality_dir = tmp_path / "quality"
    output_dir = tmp_path / "output"
    for directory in (sample_dir, main_dir, fuzzy_dir, nemotron_dir, quality_dir):
        directory.mkdir()

    sample = pa.table({"id": ["a", "b"], "text": ["alpha", "beta"]})
    pq.write_table(sample, sample_dir / basename)
    embedding_type = pa.list_(pa.int8(), HARRIER_DIM)
    pq.write_table(
        pa.table({"id": ["a"], "embedding": pa.array([[1] * HARRIER_DIM], type=embedding_type)}),
        main_dir / basename,
    )
    pq.write_table(
        pa.table({"id": ["a", "b"], "embedding": pa.array([[9] * HARRIER_DIM, [2] * HARRIER_DIM], type=embedding_type)}),
        fuzzy_dir / basename,
    )
    pq.write_table(
        pa.table({"id": ["a", "b"], "input_ids": pa.array([[10, 11], [12]], type=pa.list_(pa.int32()))}),
        nemotron_dir / basename,
    )
    pq.write_table(pa.table({"id": ["b"], "score": [0.75]}), quality_dir / basename)

    class Index:
        def search(self, embeddings, _neighbors):
            clusters = (embeddings[:, :1] > 0.003).astype(np.int64)
            return np.zeros_like(clusters, dtype=np.float32), clusters

    monkeypatch.setattr(
        materialize_10pct_harrier_tokens,
        "_get_index",
        lambda _centroids, _lookups: {"index": Index(), "lookups": {40: np.asarray([7, 8], dtype=np.int32)}},
    )
    items = iter(
        [
            {
                "source": source,
                "id": "a",
                "record": {"id": "a", "text": "alpha"},
                "basename": basename,
                "marin_input_ids": [20],
            },
            {
                "source": source,
                "id": "b",
                "record": {"id": "b", "text": "beta"},
                "basename": basename,
                "marin_input_ids": None,
            },
        ]
    )

    stats = materialize_10pct_harrier_tokens._materialize_group(
        (source, basename),
        items,
        main_dirs={source: str(main_dir)},
        fuzzy_dirs={source: str(fuzzy_dir)},
        nemotron_dirs={source: str(nemotron_dir)},
        quality_dirs={source: str(quality_dir)},
        schema_paths={source: str(sample_dir / basename)},
        output_root=str(output_dir),
        cluster_root="cluster",
    )

    result = pq.read_table(output_dir / source / basename)
    assert result["id"].to_pylist() == ["a", "b"]
    assert result["embedding"].to_pylist() == [[1] * HARRIER_DIM, [2] * HARRIER_DIM]
    assert result["cluster_5000"].to_pylist() == [0, 1]
    assert result["dist_5000"].to_pylist() == [0.0, 0.0]
    assert result["domain_id"].to_pylist() == [7, 8]
    assert result["quality_score_pooled_junkgate2"].to_pylist() == [None, 0.75]
    assert result["nemotron_input_ids"].to_pylist() == [[10, 11], [12]]
    assert result["marin_input_ids"].to_pylist() == [[20], None]
    assert stats == {
        "source": source,
        "basename": basename,
        "rows": 2,
        "main": 1,
        "fuzzy": 1,
        "missing_nemotron": 0,
        "missing_marin": 1,
        "missing_quality": 1,
        "reused": False,
    }
