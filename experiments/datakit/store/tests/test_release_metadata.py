# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.datakit.scripts.metadata_upstream_locators import SourceFile, SourceManifest, SourceRef
from experiments.datakit.store.metadata import ShardManifest
from experiments.datakit.store.release_metadata import export_release_metadata_shard


@pytest.fixture
def release_inputs(tmp_path):
    doc_id = generate_id("Repeated retained text")
    columns = {
        "normalized": {"id": [doc_id, doc_id], "text": ["Repeated retained text"] * 2},
        "decontam": {"id": [doc_id] * 2, "contaminated": [False, False]},
        "cluster": {"id": [doc_id] * 2, "cluster_40": [3, 4]},
        "quality": {"id": [doc_id] * 2, "score": [0.5, 0.5], "quality_bucket": [2, 2]},
        "exact_dedup": {"id": pa.array([], type=pa.string())},
    }
    paths = {}
    for name, values in columns.items():
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table(values), path)
        paths[name] = str(path)
    selection = ShardManifest(marin_source="test", dedup="", bucket_edges=None, **paths)
    source = SourceManifest(
        source_ref=SourceRef("provider/test", "default", "train", "pinned-revision"),
        files=(SourceFile("z.parquet", 3), SourceFile("a.parquet", 5)),
        ordering_evidence="Explicit provider split order recorded before transformation",
    )
    locators = [
        {"source_ref": source.sha256, "row_offset": 5, "file": "a.parquet", "file_row_offset": 2},
        {"source_ref": source.sha256, "row_offset": 1, "file": "z.parquet", "file_row_offset": 1},
    ]
    rows = [{"id": doc_id, "normalized_row": i, "source_rows": [locator]} for i, locator in enumerate(locators)]
    return selection, source, rows


def test_release_preserves_occurrences_and_verified_upstream_coordinates(tmp_path, release_inputs):
    selection, source, rows = release_inputs
    # An assembled document can require several upstream rows, in transform order.
    rows[0]["source_rows"].append(rows[1]["source_rows"][0])
    sidecar = tmp_path / "upstream.parquet"
    pq.write_table(pa.Table.from_pylist(rows), sidecar)
    result = export_release_metadata_shard(
        selection,
        str(sidecar),
        {source.sha256: source},
        str(tmp_path / "release"),
        cluster_view=40,
        max_rows=64,
    )
    exported = []
    for output in result["outputs"]:
        table = pq.ParquetFile(output["path"]).read()
        assert not {"text", "input_ids", "normalized_row", "normalized_shard"}.intersection(table.column_names)
        exported.extend(table.to_pylist())
    assert result["selected_occurrences"] == 2
    assert result["harrier_embedding"]["status"] == "not_requested"
    assert result["harrier_embedding"]["null_occurrences"] == 2
    assert all(row["harrier_embedding"] is None for row in exported)
    assert [row["source_rows"] for row in exported] == [row["source_rows"] for row in rows]
    assert (tmp_path / "release" / "_complete.json").is_file()


@pytest.mark.parametrize("fault", ["missing", "empty", "file_offset_used_as_split_offset"])
def test_release_refuses_to_publish_incomplete_or_wrong_coordinates(tmp_path, release_inputs, fault):
    selection, source, rows = release_inputs
    if fault == "missing":
        rows = rows[:1]
    elif fault == "empty":
        rows[0]["source_rows"] = None
    else:
        rows[0]["source_rows"][0]["row_offset"] = 2
    sidecar = tmp_path / "upstream.parquet"
    pq.write_table(pa.Table.from_pylist(rows), sidecar)
    with pytest.raises((ValueError, IndexError)):
        export_release_metadata_shard(
            selection,
            str(sidecar),
            {source.sha256: source},
            str(tmp_path / "release"),
            cluster_view=40,
            max_rows=64,
        )
    assert not (tmp_path / "release").exists()


def test_release_preserves_harrier_vectors_and_upstream_locators(tmp_path, release_inputs):
    selection, source, rows = release_inputs
    vector = [i % 255 - 127 for i in range(1024)]
    path = tmp_path / "embedding.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": rows[0]["id"], "embedding": vector}],
            schema=pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.int8(), 1024))]),
        ),
        path,
    )
    selection = replace(
        selection,
        harrier_embedding=str(path),
        harrier_embedding_provenance={
            "artifact": "frozen-harrier",
            "model_name": "harrier",
            "model_revision": "pinned",
            "embedding_dim": 1024,
            "quantization_scale": 0.3 / 127,
            "quantization_range": 0.3,
        },
    )
    sidecar = tmp_path / "upstream.parquet"
    pq.write_table(pa.Table.from_pylist(rows), sidecar)
    result = export_release_metadata_shard(
        selection, str(sidecar), {source.sha256: source}, str(tmp_path / "release"), cluster_view=40, max_rows=64
    )
    exported = [row for output in result["outputs"] for row in pq.ParquetFile(output["path"]).read().to_pylist()]
    assert [row["harrier_embedding"] for row in exported] == [vector, vector]
    assert [row["source_rows"] for row in exported] == [row["source_rows"] for row in rows]
    assert result["harrier_embedding"]["embedded_occurrences"] == 2
    assert result["harrier_embedding"]["status"] == "complete"
    assert result["harrier_embedding"]["provenance"] == selection.harrier_embedding_provenance
