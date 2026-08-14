# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the finestore archive: round-trip, dedup, schema evolution, blobs, compaction."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore import compaction, shard_writer
from finestore.commit import CommitConflict, CommitCoordinator
from finestore.compaction import compact
from finestore.layout import (
    FORMAT_VERSION,
    ArchiveMetadata,
    FineStoreLayout,
    FormatVersionError,
    OnConflict,
)
from finestore.migrations import migrate
from finestore.reader import ReadView
from finestore.store import DataStore, DataTable, PrimaryKeyConflict, TransactionTooLarge
from rigging.filesystem import StoragePath


def _rows(reader: ReadView, table: str, **kwargs) -> list[dict]:
    result = reader.scan(table, **kwargs)
    return [] if result is None else result.to_pylist()


def test_round_trip_and_projection(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        samples = store.table("samples", primary_key=("task", "doc_id"))
        samples.append({"task": "arc", "doc_id": "1", "correct": True, "logprobs": [0.1, 0.2]})
        samples.append({"task": "arc", "doc_id": "2", "correct": False, "logprobs": [0.3]})
        store.flush()

    reader = ReadView(root)
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

    row = ReadView(root).point("samples", task="arc", doc_id="3")
    assert row is not None
    assert row["score"] == 3.0
    assert ReadView(root).point("samples", task="arc", doc_id="99") is None


def test_iter_rows_streams_latest_values_in_primary_key_order(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        table = store.table("samples", primary_key=("doc_id",), on_conflict=OnConflict.SUPERSEDE)
        table.extend([{"doc_id": "b", "score": 1}, {"doc_id": "a", "score": 2}])
        store.flush()
        table.append({"doc_id": "a", "score": 3})
        store.flush()

    assert list(ReadView(root).iter_rows("samples", columns=["doc_id", "score"])) == [
        {"doc_id": "a", "score": 3},
        {"doc_id": "b", "score": 1},
    ]


def test_supersede_keeps_latest_seq(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"), on_conflict=OnConflict.SUPERSEDE)
        table.append({"task": "arc", "doc_id": "1", "score": 1.0})
        store.flush()
        # A retried trial re-delivers the same primary key with a fresh score in a later shard.
        table.append({"task": "arc", "doc_id": "1", "score": 2.0})
        store.flush()

    rows = _rows(ReadView(root), "samples")
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

    rows = _rows(ReadView(root), "samples")
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

    rows = _rows(ReadView(root), "samples")
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

    marker = ReadView(root).seal_marker()
    assert marker is not None
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

    rows = _rows(ReadView(root), "samples")
    assert {r["doc_id"] for r in rows} == {"a", "b"}


def test_later_commit_prevents_compaction_shadowing(tmp_path):
    # A key compacted into a higher generation must still be overwritten by a later low-generation
    # append. The user commit sequence ranks before compaction generation.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1", "score": 1.0})
    compact(root, "samples")
    assert ReadView(root).list_shards("samples")[0].generation == 1

    with DataStore.open(root, writer_id="w2") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1", "score": 2.0})
    rows = _rows(ReadView(root), "samples")
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
    reader = ReadView(root)
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

        def flush_table(self, _table):
            raise AssertionError("flush_table is not used by this test")

    scheduler = RecordingScheduler()
    table = DataTable(
        "blobs",
        writer_id="w1",
        layout=FineStoreLayout(str(tmp_path / "run")),
        metadata_path=str(tmp_path / "schema.json"),
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
    shard = ReadView(root).list_shards("samples")[0].path
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

    result = ReadView(root).scan("samples")
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
    rows = _rows(ReadView(root), "samples")
    assert rows[0]["doc_id"] == "1"


def test_blob_write_and_resolve(tmp_path):
    root = str(tmp_path / "run")
    payload = b'{"steps": [1, 2, 3]}'
    with DataStore.open(root, writer_id="w1") as store:
        uri = store.write_object("traj/t1.json", payload, {"task": "arc"})
        store.flush()
    assert uri == "finestore://blobs/traj/t1.json"
    assert ReadView(root).resolve(uri) == payload
    assert ReadView(root).read_blob("missing") is None


def test_blob_rewrite_supersedes_by_name(tmp_path):
    # Blobs are a table keyed by name, so re-writing a name buffers a higher-_seq row that the reader's
    # dedup keeps -- the latest payload wins, like any other primary-key collision.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        uri = store.write_object("t1.json", b"first")
        store.write_object("t1.json", b"second")
        store.flush()
    assert ReadView(root).resolve(uri) == b"second"


def test_compaction_replaces_sources_logically_and_retains_objects(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        for i in range(3):
            table.append({"task": "arc", "doc_id": str(i)})
            store.flush()  # three separate level-0 shards

    before = ReadView(root).list_shards("samples")
    assert len(before) == 3

    assert compact(root, "samples").written == 3

    after = ReadView(root).list_shards("samples")
    assert len(after) == 1
    assert after[0].generation == 1
    assert all(StoragePath(shard.path).exists() for shard in before)
    rows = _rows(ReadView(root), "samples")
    assert {r["doc_id"] for r in rows} == {"0", "1", "2"}


def test_read_view_remains_valid_after_compaction(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": str(i)} for i in range(4)])
        store.flush()

    pinned = ReadView(root)
    compact(root, "samples")
    assert {s.generation for s in pinned.list_shards("samples")} == {0}
    assert {s.generation for s in ReadView(root).list_shards("samples")} == {1}
    assert {row["doc_id"] for row in _rows(pinned, "samples")} == {"0", "1", "2", "3"}


def test_recompaction_merges_all_generations(tmp_path):
    # Recompaction must stream an already compacted shard and advance its generation without changing
    # the visible rows.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table("samples", primary_key=("task", "doc_id"))
        table.extend([{"task": "arc", "doc_id": str(i), "score": float(i)} for i in range(4)])
        store.flush()
    compact(root, "samples")
    assert {s.generation for s in ReadView(root).list_shards("samples")} == {1}

    assert compact(root, "samples").written == 4
    assert {s.generation for s in ReadView(root).list_shards("samples")} == {2}
    rows = {r["doc_id"]: r["score"] for r in _rows(ReadView(root), "samples")}
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
    rows = {r["doc_id"]: (r.get("a"), r.get("b")) for r in _rows(ReadView(root), "samples")}
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
    assert {s.generation for s in ReadView(root).list_shards("samples")} == {2}
    rows = {r["doc_id"]: r["score"] for r in _rows(ReadView(root), "samples")}
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

    table_meta = ReadView(root).table_metadata("samples")
    assert table_meta.primary_key == ("task", "doc_id")
    assert table_meta.schema_version == 3

    seal = ReadView(root).seal_marker()
    assert seal is not None
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
    with pytest.raises(FormatVersionError, match="upgrade marin-finestore"):
        ReadView(root).scan("samples")
    with pytest.raises(FormatVersionError, match="upgrade marin-finestore"):
        DataStore.open(root)


def test_read_view_refuses_an_older_format_with_migration_instructions(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "_archive.json").write_text('{"format_version": 1}')

    with pytest.raises(FormatVersionError, match="migrate it explicitly"):
        ReadView(str(root))


def test_seal_marker(tmp_path):
    root = str(tmp_path / "run")
    reader = ReadView(root)
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        assert not reader.is_sealed()
        store.seal()
    assert not reader.is_sealed()
    assert ReadView(root).is_sealed()


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
    shards = ReadView(root).list_shards("samples")
    assert len(shards) == 1
    assert shards[0].generation == 1
    assert ReadView(root).scan("samples", columns=["doc_id", "score"]).to_pylist() == [{"doc_id": "1", "score": 2.0}]


def test_seal_compacts_tables_not_registered_by_the_sealing_writer(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="first", flush_interval=3600) as store:
        table = store.table("history", primary_key=("id",))
        table.append({"id": "1"})
        store.flush()
        table.append({"id": "2"})
        store.flush()

    with DataStore.open(root, writer_id="sealer", flush_interval=3600) as store:
        store.seal()

    view = ReadView(root)
    assert view.table_names() == ("history",)
    assert len(view.list_shards("history")) == 1


def test_reader_none_for_absent_table(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1") as store:
        store.table("samples", primary_key=("task", "doc_id")).append({"task": "arc", "doc_id": "1"})
        store.flush()
    assert ReadView(root).scan("does-not-exist") is None


def test_single_table_flush_is_independent(tmp_path):
    # DataTable.flush() persists just its own buffer; a sibling table's rows stay buffered. A long
    # flush interval keeps the background thread from flushing the sibling during the test.
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        samples = store.table("samples", primary_key=("task", "doc_id"))
        steps = store.table("steps", primary_key=("task", "step_id"))
        samples.append({"task": "arc", "doc_id": "1"})
        steps.append({"task": "arc", "step_id": 0})
        samples.flush()
        assert len(ReadView(root).list_shards("samples")) == 1
        assert ReadView(root).scan("steps") is None


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
    rows = _rows(ReadView(root), "samples")
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
    with pytest.raises(pa.ArrowInvalid):
        with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
            store.table("samples", primary_key=("task", "doc_id"), schema=schema).append(
                {"task": "arc", "doc_id": "1", "count": "not-an-int"}
            )
            store.flush()

    with DataStore.open(root, writer_id="w2", flush_interval=3600) as store:
        store.table("samples", primary_key=("task", "doc_id"), schema=schema).append(
            {"task": "arc", "doc_id": "2", "count": 5}
        )
        store.flush()
    result = ReadView(root).scan("samples", columns=["task", "doc_id", "count"])
    assert result.schema.field("count").type == pa.int64()
    assert result.to_pylist() == [{"task": "arc", "doc_id": "2", "count": 5}]


def test_transaction_publishes_tables_and_objects_with_one_token(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        store.table("purchases", primary_key=("order_id",))
        before = store.read_view()
        with store.transaction() as transaction:
            transaction.table("purchases").add({"order_id": "o1", "amount": 12})
            uri = transaction.write_object("receipts/o1", b"paid")
            assert transaction.lookup("receipts/o1") == b"paid"
            assert ReadView(root).scan("purchases") is None

        assert transaction.token is not None
        assert before.scan("purchases") is None
        assert before.resolve(uri) is None
        after = ReadView(root)
        assert after.token == transaction.token
        assert after.point("purchases", order_id="o1")["amount"] == 12
        assert after.resolve(uri) == b"paid"


def test_transaction_exception_publishes_nothing(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        store.table("purchases", primary_key=("order_id",))
        with pytest.raises(RuntimeError, match="abort"):
            with store.transaction() as transaction:
                transaction.table("purchases").add({"order_id": "o1"})
                transaction.write_object("receipts/o1", b"paid")
                raise RuntimeError("abort")

        view = ReadView(root)
        assert view.scan("purchases") is None
        assert view.read_blob("receipts/o1") is None


def test_transaction_rejects_payload_above_its_bound(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        with pytest.raises(TransactionTooLarge, match="smaller batch"):
            with store.transaction(max_bytes=16) as transaction:
                transaction.write_object("receipt", b"payload")

        assert ReadView(root).read_blob("receipt") is None


def test_transaction_lookup_is_pinned_to_its_starting_commit(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="first", flush_interval=3600) as first:
        transaction = first.transaction()
        with DataStore.open(root, writer_id="second", flush_interval=3600) as second:
            second.write_object("new", b"value")
            second.flush()
        assert transaction.lookup("new") is None
        transaction.commit()


def test_failed_multi_table_flush_restores_every_buffer(tmp_path, monkeypatch):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        store.table("purchases", primary_key=("order_id",)).append({"order_id": "o1"})
        store.table("refunds", primary_key=("refund_id",)).append({"refund_id": "r1"})

        real_write = shard_writer.write_table
        writes = 0

        def fail_second_write(path, table):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("injected shard failure")
            real_write(path, table)

        monkeypatch.setattr("finestore.store.write_table", fail_second_write)
        with pytest.raises(OSError, match="injected"):
            store.flush()
        assert ReadView(root).scan("purchases") is None
        assert ReadView(root).scan("refunds") is None

        monkeypatch.setattr("finestore.store.write_table", real_write)
        token = store.flush()
        assert token is not None
        assert ReadView(root).point("purchases", order_id="o1") is not None
        assert ReadView(root).point("refunds", refund_id="r1") is not None


def test_maintenance_compacts_after_shard_threshold(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600, compaction_shards=3) as store:
        table = store.table("samples", primary_key=("doc_id",))
        for index in range(3):
            table.append({"doc_id": str(index)})
            store.flush()
        assert len(ReadView(root).list_shards("samples")) == 3

        store.maintain()
        shards = ReadView(root).list_shards("samples")
        assert len(shards) == 1
        assert shards[0].generation == 1


def test_compaction_losing_an_input_race_is_a_benign_noop(tmp_path):
    root = str(tmp_path / "run")
    with DataStore.open(root, writer_id="w1", flush_interval=3600) as store:
        table = store.table("samples", primary_key=("doc_id",))
        table.append({"doc_id": "1"})
        store.flush()
        table.append({"doc_id": "2"})
        store.flush()

    real = CommitCoordinator(FineStoreLayout(root))

    class LosingCoordinator:
        def snapshot(self):
            return real.snapshot()

        def commit(self, _delta, *, base=None):
            raise CommitConflict("inputs changed")

    assert compact(root, "samples", coordinator=LosingCoordinator()).written == 0
    assert len(ReadView(root).list_shards("samples")) == 2


def test_manifest_migration_publishes_existing_shards_through_head(tmp_path):
    root = tmp_path / "run"
    shard = root / "samples" / "w=legacy" / "g=0" / "0000000000000000-old.parquet"
    shard.parent.mkdir(parents=True)
    (root / "_archive.json").write_text('{"format_version": 1}')
    (root / "SEALED").write_text('{"writer": "legacy", "superseded": {}}')
    (root / "samples" / "_schema.json").write_text(
        '{"primary_key": ["doc_id"], "schema_version": 3, "on_conflict": "error"}'
    )
    pq.write_table(
        pa.Table.from_pylist([{"doc_id": "1", "score": 0.5, "_seq": 7, "_writer": "legacy"}]),
        shard,
    )

    result = migrate(root.as_uri())
    token = result.token
    assert result.applied == ("0001_manifest",)
    assert token is not None
    view = ReadView(str(root))
    assert view.token == token
    assert view.is_sealed()
    assert view.schema_version("samples") == 3
    assert view.point("samples", doc_id="1")["score"] == 0.5
    assert view.list_shards("samples")[0].max_seq == 7
    assert shard.exists()
    assert (root / "samples" / "_schema.json").exists()
    repeated = migrate(root.as_uri())
    assert repeated.applied == ()
    assert repeated.token == token


def test_write_open_migrates_a_sealed_legacy_archive(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "_archive.json").write_text('{"format_version": 1}')
    (root / "SEALED").write_text('{"writer": "legacy", "superseded": {}}')

    with DataStore.open(str(root), writer_id="new") as store:
        view = store.read_view()
        assert view.token is not None
        assert view.token.sequence == 2
        assert not view.is_sealed()

    archive = ArchiveMetadata.model_validate_json((root / "_archive.json").read_bytes())
    assert archive.format_version == FORMAT_VERSION


@pytest.mark.parametrize("open_archive", [migrate, DataStore.open], ids=["explicit", "write-open"])
def test_manifest_migration_requires_a_sealed_archive(tmp_path, open_archive):
    root = tmp_path / "run"
    root.mkdir()
    (root / "_archive.json").write_text('{"format_version": 1}')

    with pytest.raises(ValueError, match="not sealed"):
        open_archive(str(root))
