# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the finestore archive: round-trip, dedup, schema evolution, blobs, compaction."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore import compaction, shard_writer
from finestore.compaction import compact
from finestore.layout import (
    FORMAT_VERSION,
    ArchiveMetadata,
    FineStoreLayout,
    OnConflict,
    SealMarker,
    TableMetadata,
)
from finestore.reader import CompositeReader
from finestore.store import DataStore, DataTable, PrimaryKeyConflict
from fsspec.implementations.memory import MemoryFileSystem
from rigging.filesystem.storage_path import StoragePath


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


def test_supersede_keeps_latest_seq(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"), on_conflict=OnConflict.SUPERSEDE)
        table.append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.flush()
        # A retried trial re-delivers the same primary key with a fresh score in a later shard.
        table.append({"task": "arc", "doc_id": "1", "score": 2.0})
        store.flush()

    rows = _rows(CompositeReader(root), "samples")
    assert len(rows) == 1
    assert rows[0]["score"] == 2.0


def test_conflicting_primary_key_raises(tmp_path):
    """A key repeat that would silently discard a differing row is rejected, not folded."""
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.append({"task": "gsm8k", "doc_id": "0", "score": 0.0})
        with pytest.raises(PrimaryKeyConflict, match="gsm8k"):
            table.append({"task": "gsm8k", "doc_id": "0", "score": 1.0})
        # The rejected row never entered the buffer, so the accepted one is intact.
        store.flush()

    rows = _rows(CompositeReader(root), "samples")
    assert [row["score"] for row in rows] == [0.0]


def test_identical_redelivery_is_not_a_conflict(tmp_path):
    """At-least-once delivery of the same row loses nothing, so it collapses without raising."""
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.flush()
        table.append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.flush()

    rows = _rows(CompositeReader(root), "samples")
    assert len(rows) == 1


def test_conflict_detected_across_flushes(tmp_path):
    """The earlier row already being durable does not hide the conflict."""
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.append({"task": "gsm8k", "doc_id": "0", "score": 0.0})
        store.flush()
        with pytest.raises(PrimaryKeyConflict):
            table.append({"task": "gsm8k", "doc_id": "0", "score": 1.0})


def test_seal_records_superseded_rows(tmp_path):
    """Supersession is legal but never silent: the seal marker carries the count."""
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"), on_conflict=OnConflict.SUPERSEDE)
        table.append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.flush()
        table.append({"task": "arc", "doc_id": "1", "score": 2.0})
        store.seal()

    marker = SealMarker.model_validate_json(StoragePath(FineStoreLayout(root).sealed_path).read_text())
    assert marker.superseded["samples"] == 1


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


def test_resume_seq_prevents_shadowing(tmp_path):
    # The no-shadow guarantee. A key compacted into a higher generation must still be overwritten by a
    # later append. A resuming writer starts its sequence above every persisted _seq, and dedup ranks
    # by _seq before generation, so the new low-generation row outranks the old compacted one -- which
    # a generation-first rule would have shadowed.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1", "score": 1.0})
    compact(root, "samples")
    assert CompositeReader(root).list_shards("samples")[0].generation == 1

    with DataStore.open(root, writer_id="w2") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1", "score": 2.0})
    rows = _rows(CompositeReader(root), "samples")
    assert len(rows) == 1
    assert rows[0]["score"] == 2.0


def test_keys_lists_deduped_primary_keys(tmp_path):
    # keys() is the resume primitive: the deduplicated set of primary-key tuples already committed, so a
    # writer can skip finished work. A re-delivered key appears once; an absent table is the empty set.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": "1"}, {"task": "arc", "doc_id": "2"}])
        store.flush()
        table.append({"task": "arc", "doc_id": "1"})
        store.flush()
    reader = CompositeReader(root)
    assert reader.keys("samples") == {("arc", "1"), ("arc", "2")}
    assert reader.keys("does-not-exist") == set()


def test_byte_cap_requests_an_early_flush(tmp_path):
    # The buffer flushes early when its estimated payload crosses the byte cap, whatever the row count:
    # one large blob trips a small cap that a same-count batch of tiny rows would not. DataTable drives
    # a flush through a narrow scheduler protocol, so a fake one records exactly when the cap fires.
    class RecordingScheduler:
        def __init__(self):
            self.flushes = 0

        def request_flush(self):
            self.flushes += 1

        def raise_if_failed(self):
            pass

    scheduler = RecordingScheduler()
    table = DataTable(
        "blobs",
        writer_id="w1",
        layout=FineStoreLayout(str(tmp_path / "run")),
        max_buffer_bytes=1024,
        scheduler=scheduler,
        primary_key=("name",),
        schema=None,
        schema_version=1,
    )
    table.append({"name": "a", "data": b"x" * 100})  # 100 bytes: under the 1 KiB cap
    assert scheduler.flushes == 0
    table.append({"name": "b", "data": b"y" * 2000})  # one row crosses the cap on its own
    assert scheduler.flushes == 1


def test_large_flush_splits_into_prunable_row_groups(tmp_path, monkeypatch):
    # A flush larger than the row-group cap writes several row groups, not one, so a reader can prune
    # to the groups a filter needs. Shrink the cap so a handful of rows spans several groups.
    monkeypatch.setattr(shard_writer, "ROW_GROUP_ROWS", 2)
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": f"{i:03d}"} for i in range(7)])
        store.flush()
    shard = CompositeReader(root).list_shards("samples")[0].path
    metadata = pq.ParquetFile(shard).metadata
    assert metadata.num_rows == 7
    assert metadata.num_row_groups == 4  # ceil(7 / 2)


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


def test_blob_rewrite_supersedes_by_name(tmp_path):
    # Blobs are a table keyed by name, so re-writing a name buffers a higher-_seq row that the reader's
    # dedup keeps -- the latest payload wins, like any other primary-key collision.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        uri = store.write("t1.json", None, b"first")
        store.write("t1.json", None, b"second")
        store.flush()
    assert CompositeReader(root).resolve(uri) == b"second"


def test_compaction_merges_and_deletes_source(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        for i in range(3):
            table.append({"task": "arc", "doc_id": str(i)})
            store.flush()  # three separate level-0 shards

    before = CompositeReader(root).list_shards("samples")
    assert len(before) == 3

    assert compact(root, "samples").written == 3

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


def test_recompaction_merges_all_generations(tmp_path):
    # A crash-style compaction leaves g0 and g1 both present; a second compaction merges them into
    # g2 (preferring the higher generation per key, the merge result) and drops the lower ones. This
    # exercises the streaming read of an already-compacted (multi-generation) shard.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": str(i), "score": float(i)} for i in range(4)])
        store.flush()
    compact(root, "samples", delete_source=False)
    assert {s.generation for s in CompositeReader(root).list_shards("samples")} == {0, 1}

    assert compact(root, "samples").written == 4
    assert {s.generation for s in CompositeReader(root).list_shards("samples")} == {2}
    rows = {r["doc_id"]: r["score"] for r in _rows(CompositeReader(root), "samples")}
    assert rows == {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0}


def test_compaction_unifies_evolved_schema(tmp_path):
    # Two writers wrote different columns; the streaming merge reads through the unified schema, so
    # the compacted shard carries both columns, null where a source lacked one.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="old") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1", "a": 1})
        store.flush()
    with DataStore.open(root, writer_id="new") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "2", "b": 2})
        store.flush()

    compact(root, "samples")
    rows = {r["doc_id"]: (r.get("a"), r.get("b")) for r in _rows(CompositeReader(root), "samples")}
    assert rows == {"1": (1, None), "2": (None, 2)}


def test_compaction_streams_multiple_row_groups(tmp_path, monkeypatch):
    # A tiny output row group forces a compacted shard to span several groups; re-compacting it with a
    # fresh level-0 shard must stream those groups in primary-key order and merge correctly, proving the
    # merge never materializes the whole table.
    monkeypatch.setattr(compaction, "_COMPACT_BATCH_ROWS", 2)
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": f"{i:03d}", "score": float(i)} for i in range(7)])
        store.flush()
    compact(root, "samples")  # g1 spanning ceil(7/2) row groups

    with DataStore.open(root, writer_id="w2") as store:
        store.table("samples", primary_key=("task", "doc_id")).extend(
            [{"task": "arc", "doc_id": f"{i:03d}", "score": float(i)} for i in range(7, 10)]
        )
        store.flush()

    assert compact(root, "samples").written == 10
    assert {s.generation for s in CompositeReader(root).list_shards("samples")} == {2}
    rows = {r["doc_id"]: r["score"] for r in _rows(CompositeReader(root), "samples")}
    assert rows == {f"{i:03d}": float(i) for i in range(10)}


def test_typed_on_disk_metadata(tmp_path):
    # The three on-disk contracts are typed pydantic objects with clear owners: the store stamps the
    # archive-wide metadata at open, the per-table metadata at registration, and the seal marker at
    # seal. Each round-trips through its model.
    root = str(tmp_path / "run")
    layout = FineStoreLayout(root)
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"), schema_version=3)
        table.append({"task": "arc", "doc_id": "1"})
        store.seal()

    archive = ArchiveMetadata.model_validate_json(StoragePath(layout.archive_path).read_text())
    assert archive.format_version == FORMAT_VERSION

    table_meta = TableMetadata.model_validate_json(StoragePath(layout.schema_path("samples")).read_text())
    assert table_meta.primary_key == ("task", "doc_id")
    assert table_meta.schema_version == 3

    seal = SealMarker.model_validate_json(StoragePath(layout.sealed_path).read_text())
    assert seal.writer == "w1"


def test_reader_refuses_a_future_format(tmp_path):
    # A reader must not silently misread an archive a newer finestore wrote; the archive metadata
    # records the format version and scan refuses one it does not understand.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        store.flush()
    StoragePath(FineStoreLayout(root).archive_path).write_text(
        ArchiveMetadata(format_version=FORMAT_VERSION + 1).model_dump_json()
    )
    with pytest.raises(ValueError, match="format"):
        CompositeReader(root).scan("samples")


def test_seal_marker(tmp_path):
    root = str(tmp_path / "run")
    reader = CompositeReader(root)
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        assert not reader.is_sealed()
        store.seal()
    assert reader.is_sealed()


def test_seal_compacts_each_table_to_one_generation(tmp_path):
    # seal is the materialize contract: it flushes then compacts every table, so a sealed archive is
    # one deduplicated Parquet shard per table. Two flushes of the same key collapse to a single
    # generation-1 shard holding the latest row -- readable without applying finestore's dedup rule.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"), on_conflict=OnConflict.SUPERSEDE)
        table.append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.flush()
        table.append({"task": "arc", "doc_id": "1", "score": 2.0})
        store.seal()
    shards = CompositeReader(root).list_shards("samples")
    assert len(shards) == 1
    assert shards[0].generation == 1
    assert CompositeReader(root).scan("samples", columns=["doc_id", "score"]).to_pylist() == [
        {"doc_id": "1", "score": 2.0}
    ]


def test_reads_resolve_the_filesystem_factory_at_call_time(monkeypatch):
    # finestore must resolve rigging's filesystem factory at call time, not bind it at import: a caller
    # that routes a remote scheme to a stand-in filesystem (as the harbor tests do) must be honored on
    # the read path too, or a write-then-read over the same mock silently reaches real object storage.
    # Route s3:// to an in-memory store and round-trip a write, seal (flush + compact), and scan --
    # every step touches the factory. If the read path bound url_to_fs at import, the scan would miss
    # the mock and try real S3.
    class MemoryS3(MemoryFileSystem):
        @classmethod
        def _strip_protocol(cls, path):
            if isinstance(path, str) and path.startswith("s3://"):
                path = path[len("s3://") :]
            return MemoryFileSystem._strip_protocol(path)

    MemoryS3.protocol = "s3"
    MemoryS3.store = {}
    MemoryS3.pseudo_dirs = [""]
    fs = MemoryS3()

    def fake_url_to_fs(url, **_kwargs):
        path = StoragePath(url)
        assert path.scheme == "s3"
        return fs, "/".join(part for part in (path.netloc, path.key) if part)

    def fake_open_url(url, mode="rb", **kwargs):
        target_fs, key = fake_url_to_fs(url)
        return target_fs.open(key, mode, **kwargs)

    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", fake_url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.open_url", fake_open_url)

    root = "s3://finestore-test/run"
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.seal()
    rows = _rows(CompositeReader(root), "samples")
    assert len(rows) == 1
    assert rows[0]["score"] == 1.0


def test_reader_none_for_absent_table(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        store.flush()
    assert CompositeReader(root).scan("does-not-exist") is None


def test_single_table_flush_is_independent(tmp_path):
    # DataTable.flush() persists just its own buffer; a sibling table's rows stay buffered. A long
    # flush interval keeps the background thread from flushing the sibling during the test.
    root = str(tmp_path / "run")
    reader = CompositeReader(root)
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        samples = store.table("samples", primary_key=("task", "doc_id"))
        steps = store.table("steps", primary_key=("task", "step_id"))
        samples.append({"task": "arc", "doc_id": "1"})
        steps.append({"task": "arc", "step_id": 0})
        samples.flush()
        assert len(reader.list_shards("samples")) == 1
        assert reader.scan("steps") is None


def test_table_returns_shared_handle(tmp_path):
    # Re-registering a table returns the same handle, so two appenders share one buffer and one
    # sequence counter rather than splitting rows across two.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        first = store.table("samples", primary_key=("task", "doc_id"))
        second = store.table("samples", primary_key=("task", "doc_id"))
        assert first is second
        first.append({"task": "arc", "doc_id": "1"})
        second.append({"task": "arc", "doc_id": "2"})
        store.flush()
    rows = _rows(CompositeReader(root), "samples")
    assert {r["doc_id"] for r in rows} == {"1", "2"}


def test_explicit_schema_pins_types_and_rejects_mismatch(tmp_path):
    # An explicit schema pins each column's type. A row that violates it fails the flush loudly rather
    # than silently writing an inferred type a later reader would see drift from; a conforming row
    # persists with exactly the declared types.
    root = str(tmp_path / "run")
    schema = pa.schema(
        [
            ("task", pa.string()),
            ("doc_id", pa.string()),
            ("count", pa.int64()),
            ("_seq", pa.int64()),
            ("_writer", pa.string()),
        ]
    )
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        store.table("samples", primary_key=("task", "doc_id"), schema=schema).append(
            {"task": "arc", "doc_id": "1", "count": "not-an-int"}
        )
        with pytest.raises(pa.ArrowInvalid):
            store.flush()

    with DataStore.open(root, writer_id="w2", flush_interval=3600) as store:
        store.table("samples", primary_key=("task", "doc_id"), schema=schema).append(
            {"task": "arc", "doc_id": "2", "count": 5}
        )
        store.flush()
    result = CompositeReader(root).scan("samples", columns=["task", "doc_id", "count"])
    assert result.schema.field("count").type == pa.int64()
    assert result.to_pylist() == [{"task": "arc", "doc_id": "2", "count": 5}]
