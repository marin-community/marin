# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the finestore archive: round-trip, dedup, schema evolution, blobs, compaction."""

from __future__ import annotations

import finestore
import pyarrow as pa
from finestore import CompositeReader, DataStore, compact


def _rows(reader: CompositeReader, table: str, **kwargs) -> list[dict]:
    result = reader.scan(table, **kwargs)
    return [] if result is None else result.to_pylist()


def test_round_trip_and_projection(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        samples = store.table("samples", primary_key=("task", "doc_id"))
        samples.append({"task": "arc", "doc_id": "1", "correct": True, "logprobs": [0.1, 0.2]})
        samples.append({"task": "arc", "doc_id": "2", "correct": False, "logprobs": [0.3]})
        store.flush()

    reader = CompositeReader(root)
    rows = _rows(reader, "samples")
    assert {(r["task"], r["doc_id"]) for r in rows} == {("arc", "1"), ("arc", "2")}

    # Projection returns only the requested columns; the fat column is absent from the result.
    projected = reader.scan("samples", columns=["task", "doc_id", "correct"])
    assert set(projected.column_names) == {"task", "doc_id", "correct"}


def test_point_lookup(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": str(i), "score": float(i)} for i in range(5)])
        store.flush()

    row = CompositeReader(root).point("samples", task="arc", doc_id="3")
    assert row is not None
    assert row["score"] == 3.0
    assert CompositeReader(root).point("samples", task="arc", doc_id="99") is None


def test_duplicate_delivery_latest_seq_wins(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.flush()
        # A retried trial re-delivers the same primary key with a fresh score in a later shard.
        table.append({"task": "arc", "doc_id": "1", "score": 2.0})
        store.flush()

    rows = _rows(CompositeReader(root), "samples")
    assert len(rows) == 1
    assert rows[0]["score"] == 2.0


def test_multi_writer_compose(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="worker-a") as a:
        ta = a.table("samples", primary_key=("task", "doc_id"))
        ta.append({"task": "arc", "doc_id": "a", "score": 1.0})
        a.flush()
    with DataStore.open(root, writer_id="worker-b") as b:
        tb = b.table("samples", primary_key=("task", "doc_id"))
        tb.append({"task": "arc", "doc_id": "b", "score": 2.0})
        b.flush()

    rows = _rows(CompositeReader(root), "samples")
    assert {r["doc_id"] for r in rows} == {"a", "b"}


def test_schema_evolution_null_promotes_missing_column(tmp_path):
    root = str(tmp_path / "run")
    # An "old" writer without the difficulty column.
    with DataStore.open(root, writer_id="old") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        store.flush()
    # A "new" writer that added a column.
    with DataStore.open(root, writer_id="new") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "2", "difficulty": 3})
        store.flush()

    result = CompositeReader(root).scan("samples")
    assert "difficulty" in result.column_names
    by_doc = {r["doc_id"]: r["difficulty"] for r in result.to_pylist()}
    assert by_doc == {"1": None, "2": 3}


def test_empty_dict_column_is_writable(tmp_path):
    # Every row's metrics is empty: the column would be a zero-field parquet struct, so it is dropped
    # at write time and reads back as absent rather than failing the flush.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.append({"task": "arc", "doc_id": "1", "metrics": {}})
        store.flush()
    rows = _rows(CompositeReader(root), "samples")
    assert rows[0]["doc_id"] == "1"


def test_blob_write_and_resolve(tmp_path):
    root = str(tmp_path / "run")
    payload = b'{"steps": [1, 2, 3]}'
    with DataStore.open(root, writer_id="w1") as store:
        uri = store.write("traj/t1.json", {"task": "arc"}, payload)
        store.flush()
    assert uri == "finestore://blobs/traj/t1.json"
    assert CompositeReader(root).resolve(uri) == payload
    assert CompositeReader(root).read_blob("missing") is None


def test_compaction_merges_and_deletes_source(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        for i in range(3):
            table.append({"task": "arc", "doc_id": str(i)})
            store.flush()  # three separate level-0 shards

    before = CompositeReader(root).list_shards("samples")
    assert len(before) == 3

    written = compact(root, "samples")
    assert written == 3

    after = CompositeReader(root).list_shards("samples")
    assert len(after) == 1
    assert after[0].generation == 1
    rows = _rows(CompositeReader(root), "samples")
    assert {r["doc_id"] for r in rows} == {"0", "1", "2"}


def test_crash_mid_compaction_does_not_double_count(tmp_path):
    # Simulate a crash after writing the compacted shard but before deleting the level-0 sources:
    # both generations are present. Dedup must still return one row per key.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": str(i)} for i in range(4)])
        store.flush()

    compact(root, "samples", delete_source=False)
    shards = CompositeReader(root).list_shards("samples")
    assert {s.generation for s in shards} == {0, 1}

    rows = _rows(CompositeReader(root), "samples")
    assert len(rows) == 4
    assert {r["doc_id"] for r in rows} == {"0", "1", "2", "3"}


def test_seal_marker(tmp_path):
    root = str(tmp_path / "run")
    reader = CompositeReader(root)
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        assert not reader.is_sealed()
        store.seal()
    assert reader.is_sealed()


def test_reader_none_for_absent_table(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        store.flush()
    assert CompositeReader(root).scan("does-not-exist") is None


def test_explicit_schema_is_enforced(tmp_path):
    root = str(tmp_path / "run")
    schema = pa.schema([("task", pa.string()), ("doc_id", pa.string()), ("_seq", pa.int64()), ("_writer", pa.string())])
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"), schema=schema)
        table.append({"task": "arc", "doc_id": "1"})
        store.flush()
    rows = _rows(CompositeReader(root), "samples")
    assert rows[0]["task"] == "arc"


def test_public_surface():
    assert {"DataStore", "DataTable", "CompositeReader", "compact"} <= set(finestore.__all__)
