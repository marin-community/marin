# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metadata selection agrees with the production token-store join."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.store.datakit_store import _FilterStats, _iter_surviving_docs
from experiments.datakit.store.metadata import ShardManifest, export_metadata_shard


@pytest.mark.parametrize("fuzzy_exempt", [False, True])
@pytest.mark.parametrize(
    "provider_ids",
    [None, [10, 11, 12, 13, 14, 15], ["a", "b", "c", "d", "e", "f"], [{"page": i} for i in range(6)]],
)
def test_metadata_export_matches_token_store_and_preserves_occurrences(tmp_path, fuzzy_exempt, provider_ids):
    ids = ["repeat", "repeat", "contaminated", "exact", "fuzzy", "keep"]
    normalized = {"id": ids, "text": ["private text"] * len(ids)}
    if provider_ids is not None:
        normalized["source_id"] = provider_ids
    tables = {
        "normalized": normalized,
        "decontam": {"id": ids, "contaminated": [False, False, True, False, False, False]},
        "cluster": {"id": ids, "cluster_2": [0, 1, 0, 1, 1, 0]},
        "quality": {"id": ids, "quality_bucket": [0, 1, 2, 3, 4, 0], "score": [0.1, 0.3, 0.5, 0.7, 0.9, 0.1]},
        "exact_dedup": {"id": ["exact", "contaminated"]},
        "dedup": {"id": ["fuzzy", "contaminated"], "dup_doc": [True, True]},
        # Two consecutive documents have the same ID, with the first chunked.
        "tokenize": {
            "id": ["repeat", *ids],
            "chunk_index": [0, 1, 0, 0, 0, 0, 0],
            "input_ids": [[0], [0], [1], [2], [3], [4], [5]],
        },
    }
    paths = {}
    for name, data in tables.items():
        path = str(tmp_path / f"{name}.parquet")
        pq.write_table(pa.table(data), path)
        paths[name] = path
    if fuzzy_exempt:
        paths["dedup"] = ""
    manifest = ShardManifest(
        **{name: paths[name] for name in tables if name != "tokenize"},
        marin_source="source",
        bucket_edges=None,
        store_task_id=7,
    )
    spec = {**paths, "source_name": "source", "basename": "part.parquet"}
    stats = _FilterStats()
    store_rows = list(_iter_surviving_docs(spec, "cluster_2", stats=stats))
    report = export_metadata_shard(manifest, str(tmp_path / "out"), cluster_view=2, max_rows=10, batch_size=1)
    rows = []
    for output in report["outputs"]:
        table = pq.read_table(output["path"])
        assert "text" not in table.column_names and "input_ids" not in table.column_names
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: row["normalized_row"])
    expected_positions = [0, 1, 4, 5] if fuzzy_exempt else [0, 1, 5]
    assert [row["normalized_row"] for row in rows] == expected_positions
    assert [(row["cluster_id"], row["quality_bucket"]) for row in rows] == [
        (doc.cluster, doc.quality) for doc in store_rows
    ]
    assert [int(doc.input_ids[0]) for doc in store_rows] == expected_positions
    assert [(row["id"], row["normalized_row"]) for row in rows] == [(doc.id, doc.normalized_row) for doc in store_rows]
    assert [row["quality_score"] for row in rows] == [tables["quality"]["score"][i] for i in expected_positions]
    assert all(row["normalized_shard"] == manifest.normalized and row["store_task_id"] == 7 for row in rows)
    assert [row["id"] for row in rows] == [ids[position] for position in expected_positions]
    assert [row["source_id"] for row in rows] == [
        json.dumps(provider_ids[position]) if provider_ids is not None else None for position in expected_positions
    ]
    assert report["counters"] == stats.counters()
    assert not report["truncated"]
    prefix = export_metadata_shard(manifest, str(tmp_path / "prefix"), cluster_view=2, max_rows=1, batch_size=1)
    assert prefix["examined_rows"] == 1 and prefix["truncated"]
    assert sum(output["rows"] for output in prefix["outputs"]) == 1
    # Equal lengths with a different dense row order must never silently select wrong documents.
    tables["quality"]["id"] = ["wrong", *ids[1:]]
    pq.write_table(pa.table(tables["quality"]), paths["quality"])
    with pytest.raises(RuntimeError, match="dense id/order mismatch"):
        export_metadata_shard(manifest, str(tmp_path / "bad"), cluster_view=2, max_rows=10)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("with_content_types", [True, False])
def test_metadata_export_uses_frozen_store_calibration_and_default_fallback(tmp_path, with_content_types):
    """Boundary scores use the higher bucket; unknown types use the default fit."""
    ids = ["code-boundary", "prose", "unknown", "below", "upper", "middle"]
    scores = [0.4, 0.4, 0.4, -0.1, 1.1, 0.3]
    types = ["code", "prose", "new-type", "code", "prose", "code"]
    tables = {
        "normalized": {"id": ids},
        "decontam": {"id": ids, "contaminated": [False] * len(ids)},
        "cluster": {"id": ids, "cluster_2": [0] * len(ids)},
        # Frozen v5 scorer produces raw scores without precomputed buckets.
        "quality": {"id": ids, "score": scores},
        "content_type": {"id": ids, "content_type": types},
    }
    paths = {}
    for name, data in tables.items():
        path = str(tmp_path / f"{name}.parquet")
        pq.write_table(pa.table(data), path)
        paths[name] = path
    if not with_content_types:
        paths["content_type"] = ""
    manifest = ShardManifest(
        **paths,
        marin_source="source",
        exact_dedup=str(tmp_path / "absent-exact.parquet"),
        dedup="",
        bucket_edges={
            "default": [0.2, 0.4, 0.6, 0.8],
            "code": [0.3, 0.5, 0.7, 0.9],
            "prose": [0.1, 0.2, 0.3, 0.4],
        },
    )
    report = export_metadata_shard(manifest, str(tmp_path / "out"), cluster_view=2, max_rows=10, batch_size=2)
    rows = [row for output in report["outputs"] for row in pq.read_table(output["path"]).to_pylist()]
    rows.sort(key=lambda row: row["normalized_row"])
    assert [row["quality_bucket"] for row in rows] == ([1, 4, 2, 0, 4, 1] if with_content_types else [2, 2, 2, 0, 4, 1])
    assert [row["content_type"] for row in rows] == (types if with_content_types else [None] * len(ids))
    assert [row["quality_score"] for row in rows] == scores
    assert report["content_type_without_calibration"] == (1 if with_content_types else 0)
    if with_content_types:
        pq.write_table(pa.table({"id": list(reversed(ids)), "content_type": types}), paths["content_type"])
        with pytest.raises(RuntimeError, match="dense id/order mismatch"):
            export_metadata_shard(manifest, str(tmp_path / "bad"), cluster_view=2, max_rows=10)


@pytest.fixture
def embedding_inputs(tmp_path):
    ids = ["a", "a", "b", "c", "drop"]
    tables = {
        "normalized": {"id": ids},
        "decontam": {"id": ids, "contaminated": [False, False, False, False, True]},
        "cluster": {"id": ids, "cluster_2": [0, 1, 1, 0, 0]},
        "quality": {"id": ids, "quality_bucket": [2] * 5, "score": [0.5] * 5},
    }
    paths = {}
    for name, data in tables.items():
        path = str(tmp_path / f"{name}.parquet")
        pq.write_table(pa.table(data), path)
        paths[name] = path
    path = str(tmp_path / "embedding.parquet")
    schema = pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.int8(), 1024))])
    vectors = {"a": list(range(-127, 128)) * 4 + [0, 1, 2, 3], "c": [17] * 1024}
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "unselected", "embedding": [8] * 1024},
                {"id": "c", "embedding": vectors["c"]},
                {"id": "a", "embedding": vectors["a"]},
            ],
            schema=schema,
        ),
        path,
        row_group_size=1,
    )
    provenance = {
        "artifact": "frozen-harrier-artifact",
        "model_name": "microsoft/harrier-oss-v1-0.6b",
        "model_revision": "frozen-revision",
        "embedding_dim": 1024,
        "quantization_scale": 0.3 / 127,
        "quantization_range": 0.3,
    }
    manifest = ShardManifest(
        **paths,
        marin_source="source",
        exact_dedup="",
        dedup="",
        bucket_edges=None,
        harrier_embedding=path,
        harrier_embedding_provenance=provenance,
    )
    return manifest, vectors


def test_sparse_embedding_export_joins_by_id_and_preserves_quantized_coordinates(tmp_path, embedding_inputs):
    manifest, vectors = embedding_inputs
    parquet = pq.ParquetFile(manifest.harrier_embedding)
    selected_bytes = sum(
        parquet.metadata.row_group(group).column(column).total_compressed_size for group in (1, 2) for column in (0, 1)
    )
    report = export_metadata_shard(
        manifest, str(tmp_path / "out"), cluster_view=2, max_rows=10, batch_size=1, max_embedding_bytes=selected_bytes
    )
    rows = [row for output in report["outputs"] for row in pq.read_table(output["path"]).to_pylist()]
    rows.sort(key=lambda row: row["normalized_row"])
    assert [row["harrier_embedding"] for row in rows] == [vectors["a"], vectors["a"], None, vectors["c"]]
    assert [row["normalized_row"] for row in rows] == [0, 1, 2, 3]
    coverage = report["harrier_embedding"]
    assert coverage["embedded_occurrences"] == 3 and coverage["missing_occurrences"] == 1
    assert coverage["null_occurrences"] == 1
    assert coverage["unmatched_distinct_ids"] == 1 and coverage["status"] == "incomplete"
    assert not coverage["budget_exhausted"] and coverage["projected_bytes"] > 0
    assert coverage["provenance"] == manifest.harrier_embedding_provenance


def test_sparse_embedding_budget_leaves_unexamined_ids_unresolved(tmp_path, embedding_inputs):
    manifest, _ = embedding_inputs
    report = export_metadata_shard(manifest, str(tmp_path / "out"), cluster_view=2, max_rows=10, max_embedding_bytes=1)
    coverage = report["harrier_embedding"]
    assert coverage["budget_exhausted"] and coverage["projected_bytes"] == 0
    assert coverage["embedded_occurrences"] == 0 and coverage["unresolved_occurrences"] == 4
    assert coverage["missing_occurrences"] is None
    assert all(
        row["harrier_embedding"] is None
        for output in report["outputs"]
        for row in pq.read_table(output["path"]).to_pylist()
    )


@pytest.mark.parametrize("fault", ["dimension", "dtype", "null_coordinate", "conflicting_duplicate"])
def test_sparse_embedding_export_rejects_corrupt_selected_vectors(tmp_path, embedding_inputs, fault):
    manifest, vectors = embedding_inputs
    embedding_type = pa.list_(pa.int8(), 1023 if fault == "dimension" else 1024)
    if fault == "dtype":
        embedding_type = pa.list_(pa.float32(), 1024)
    vector = vectors["a"][:1023] if fault == "dimension" else list(vectors["a"])
    if fault == "null_coordinate":
        vector[7] = None
    rows = [{"id": "a", "embedding": vector}]
    if fault == "conflicting_duplicate":
        rows.append({"id": "a", "embedding": [1] * 1024})
    pq.write_table(
        pa.Table.from_pylist(rows, schema=pa.schema([("id", pa.string()), ("embedding", embedding_type)])),
        manifest.harrier_embedding,
        row_group_size=1,
    )
    with pytest.raises(ValueError, match="Harrier"):
        export_metadata_shard(manifest, str(tmp_path / "bad"), cluster_view=2, max_rows=10)
    assert not (tmp_path / "bad").exists()
