# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import random
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import finestore.cache as cache_module
import finestore.store as store_module
import fsspec
import pyarrow as pa
import pyarrow.dataset as pds
import pytest
from finestore import shard_writer
from finestore.admin import drop_table
from finestore.cache import PersistentKvCache
from finestore.cache_storage import maintain_cache
from finestore.commit import read_snapshot
from finestore.compaction import compact_table
from finestore.layout import BlobTables, FineStoreLayout, OnConflict
from finestore.reader import ReadView
from finestore.store import OBJECT_PART_BYTES, DataStore
from finestore.testing.simulated_s3 import Request
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import factory
from rigging.filesystem.conditional_object import LocalConditionalObject
from rigging.filesystem.cross_region import MARIN_CROSS_REGION_OVERRIDE_ENV, CrossRegionGuardedFS, TransferBudget
from rigging.filesystem.storage_path import StoragePath

_BLOB_URI_SCHEME = "blobread"
_RANGE_DEFAULT_BLOCK_BYTES = 50 * 1024 * 1024

_CACHE_PROCESS = textwrap.dedent(
    """
    import atexit
    import sys
    import threading

    import finestore.cache as cache_module
    from finestore.cache import PersistentKvCache
    from finestore.store import DataStore
    from rigging.filesystem.storage_path import StoragePath

    root, mode = sys.argv[1:]
    StoragePath.is_remote = property(lambda self: True)
    cache_module._EXIT_FLUSH_TIMEOUT = 0.05 if mode == "stall" else 1.0
    real_commit = DataStore._commit_transaction
    commit_started = threading.Event()
    release_commit = threading.Event()

    def commit(store, rows):
        commit_started.set()
        if mode == "stall":
            threading.Event().wait()
        else:
            release_commit.wait()
        return real_commit(store, rows)

    DataStore._commit_transaction = commit
    PersistentKvCache.at(root).store("kernel", b"object-code")
    if not commit_started.wait(timeout=3):
        raise RuntimeError("cache commit did not start")
    if mode != "stall":
        atexit.register(release_commit.set)
    """
)

_CACHE_EXIT_COMPACTION_PROCESS = textwrap.dedent(
    """
    import sys
    import threading

    import finestore.cache as cache_module
    from finestore.cache import PersistentKvCache
    from finestore.layout import FineStoreLayout
    from fsspec.implementations.local import LocalFileSystem
    from rigging.filesystem.conditional_object import LocalConditionalObject
    from rigging.filesystem.storage_path import StoragePath

    root = sys.argv[1]
    StoragePath.is_remote = property(lambda self: True)
    cache_module._EXIT_FLUSH_TIMEOUT = 0.05
    committed = threading.Event()
    real_write = LocalConditionalObject.write

    def observe_commit(head, data, *, expected_version):
        version = real_write(head, data, expected_version=expected_version)
        if head.path == FineStoreLayout(root).head_path:
            committed.set()
        return version

    LocalConditionalObject.write = observe_commit
    cache = PersistentKvCache.at(root)
    for index in range(2):
        committed.clear()
        cache.store(str(index), b"object-code")
        if not committed.wait(timeout=3):
            raise RuntimeError("cache batch did not commit")

    real_open = LocalFileSystem.open

    def stall_compaction(filesystem, path, mode="rb", *args, **kwargs):
        if "/w=compactor/" in str(path) and "w" in mode:
            threading.Event().wait()
        return real_open(filesystem, path, mode, *args, **kwargs)

    LocalFileSystem.open = stall_compaction
    """
)


class _RangeFile(AbstractBufferedFile):
    def _fetch_range(self, start, end):
        self.fs.largest_fetch = max(self.fs.largest_fetch, end - start)
        with open(self.path, "rb") as source:
            source.seek(start)
            data = source.read(end - start)
        self.fs.fetched_bytes += len(data)
        return data


class _RangeFileSystem(LocalFileSystem):
    """Local bytes served through the same read-ahead cache as object storage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fetched_bytes = 0
        self.largest_fetch = 0

    def _open(self, path, mode="rb", *, cache_type="readahead", block_size=None, **_kwargs):
        assert mode == "rb"
        return _RangeFile(
            self, path, mode=mode, block_size=block_size or _RANGE_DEFAULT_BLOCK_BYTES, cache_type=cache_type
        )

    def cat_file(self, path, start=None, end=None, **_kwargs):
        with self.open(path, "rb", cache_type="none") as source:
            source.seek(start or 0)
            return source.read() if end is None else source.read(end - (start or 0))


class _BlobUriFileSystem(_RangeFileSystem):
    """Exercise URI routing with file options rejected by the storage constructor."""

    protocol = _BLOB_URI_SCHEME

    def __init__(self, **kwargs):
        if "cache_type" in kwargs:
            raise TypeError("cache_type belongs to the file open, not the filesystem constructor")
        super().__init__(**kwargs)
        self.read_cache_types = []

    @classmethod
    def _strip_protocol(cls, path):
        return str(path).removeprefix(f"{_BLOB_URI_SCHEME}://")

    def _open(self, path, mode="rb", *, cache_type="readahead", **kwargs):
        if path.endswith(".parquet"):
            self.read_cache_types.append(cache_type)
        return super()._open(path, mode=mode, cache_type=cache_type, **kwargs)


def test_read_view_reuses_ranges_across_queries_and_keeps_its_snapshot(tmp_path, monkeypatch):
    root = str(tmp_path / "archive")
    with DataStore.open(root, flush_interval=600) as store:
        samples = store.table("samples", primary_key=("id",), on_conflict=OnConflict.SUPERSEDE)
        samples.extend([{"id": "b", "status": "pending"}, {"id": "a", "status": "pending"}])
        store.flush()
        samples.append({"id": "a", "status": "done"})
        store.flush()
        view = ReadView(root)
        stored_bytes = sum(shard.size_bytes for shard in view.list_shards("samples"))
        samples.append({"id": "b", "status": "done"})
        store.flush()
        latest = ReadView(root)

    filesystem = _RangeFileSystem(skip_instance_cache=True)
    monkeypatch.setattr(factory, "url_to_fs", lambda path: (filesystem, path))
    # Concurrent cold lookups must share the download, including schema/footer reads.
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda key: view.point("samples", id=key), ["a", "b", "a", "b"]))
    assert [row["status"] for row in results] == ["done", "pending", "done", "pending"]
    assert filesystem.fetched_bytes == stored_bytes
    assert view.scan("samples", columns=["id"], where=[("status", "==", "pending")]).to_pylist() == [{"id": "b"}]
    assert list(view.iter_rows("samples", columns=["id", "status"])) == [
        {"id": "a", "status": "done"},
        {"id": "b", "status": "pending"},
    ]
    assert view.keys("samples") == {("a",), ("b",)}
    assert view.point("samples", id="missing") is None
    assert filesystem.fetched_bytes == stored_bytes
    assert latest.point("samples", id="b")["status"] == "done"
    assert filesystem.fetched_bytes > stored_bytes


def test_read_view_cache_budget_is_shared_by_tables_and_can_be_disabled(tmp_path, monkeypatch):
    root = str(tmp_path / "archive")
    with DataStore.open(root, flush_interval=600) as store:
        for name in ("first", "second"):
            store.table(name, primary_key=("id",)).append({"id": "a", "value": name})
    snapshot = read_snapshot(FineStoreLayout(root))
    capacity = max(sum(shard.size_bytes for shard in table.shards) for table in snapshot.manifest.tables.values())
    filesystem = _RangeFileSystem(skip_instance_cache=True)
    monkeypatch.setattr(factory, "url_to_fs", lambda path: (filesystem, path))
    view = ReadView(root, snapshot=snapshot, cache_bytes=capacity)
    assert view.point("first", id="a")["value"] == "first"
    warm_bytes = filesystem.fetched_bytes
    assert view.point("first", id="a")["value"] == "first"
    assert filesystem.fetched_bytes == warm_bytes
    assert view.point("second", id="a")["value"] == "second"
    warm_bytes = filesystem.fetched_bytes
    assert view.point("second", id="a")["value"] == "second"
    assert filesystem.fetched_bytes == warm_bytes
    assert view.point("first", id="a")["value"] == "first"
    assert filesystem.fetched_bytes > warm_bytes

    uncached = ReadView(root, snapshot=snapshot, cache_bytes=0)
    for _ in range(2):
        before = filesystem.fetched_bytes
        assert uncached.point("first", id="a")["value"] == "first"
        assert filesystem.fetched_bytes > before


def test_read_view_caches_selected_row_groups_from_a_shard_larger_than_its_budget(tmp_path, monkeypatch):
    root = str(tmp_path / "archive")
    monkeypatch.setattr(shard_writer, "ROW_GROUP_TARGET_BYTES", 512 * 1024)
    random_bytes = random.Random(1729)
    payloads = [random_bytes.randbytes(512 * 1024) for _ in range(20)]
    with DataStore.open(root, flush_interval=600, max_buffer_bytes=2**40) as store:
        store.table("samples", primary_key=("id",)).extend(
            {"id": index, "payload": payload} for index, payload in enumerate(payloads)
        )
    snapshot = read_snapshot(FineStoreLayout(root))
    stored_bytes = sum(shard.size_bytes for shard in snapshot.manifest.tables["samples"].shards)
    filesystem = _RangeFileSystem(skip_instance_cache=True)
    budget = TransferBudget(limit_bytes=2 * 1024 * 1024)
    guarded = CrossRegionGuardedFS(filesystem, cross_region_checker=lambda _: True, budget=budget)
    monkeypatch.delenv(MARIN_CROSS_REGION_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(factory, "url_to_fs", lambda path: (guarded, path))
    view = ReadView(root, snapshot=snapshot, cache_bytes=2 * 1024 * 1024)
    assert view.point("samples", id=7)["payload"] == payloads[7]
    warm_bytes = filesystem.fetched_bytes
    assert 0 < warm_bytes < stored_bytes // 2
    assert budget.bytes_used == warm_bytes
    assert view.point("samples", id=7)["payload"] == payloads[7]
    assert filesystem.fetched_bytes == warm_bytes
    assert budget.bytes_used == warm_bytes


def test_read_view_retries_a_short_storage_read_without_caching_it(tmp_path, monkeypatch):
    root = str(tmp_path / "archive")
    with DataStore.open(root, flush_interval=600) as store:
        store.table("samples", primary_key=("id",)).append({"id": "a", "value": "complete"})
    snapshot = read_snapshot(FineStoreLayout(root))
    filesystem = _RangeFileSystem(skip_instance_cache=True)
    full_read = filesystem.cat_file
    truncate = True

    def short_first_read(path, start=None, end=None, **kwargs):
        nonlocal truncate
        data = full_read(path, start=start, end=end, **kwargs)
        if truncate:
            truncate = False
            return data[:-1]
        return data

    monkeypatch.setattr(filesystem, "cat_file", short_first_read)
    monkeypatch.setattr(factory, "url_to_fs", lambda path: (filesystem, path))
    view = ReadView(root, snapshot=snapshot)
    with pytest.raises(ValueError, match=r"shard .*\.parquet.*: read \d+ of \d+ bytes at \d+"):
        view.point("samples", id="a")
    assert view.point("samples", id="a")["value"] == "complete"


def test_blob_uri_keeps_file_cache_options_out_of_storage_constructor(tmp_path):
    fsspec.register_implementation(_BLOB_URI_SCHEME, _BlobUriFileSystem, clobber=True)
    local_root = str(tmp_path / "archive")
    root = f"{_BLOB_URI_SCHEME}://{local_root}"
    chunked = b"x" * (OBJECT_PART_BYTES + 1)
    with DataStore.open(local_root, flush_interval=600) as store:
        store.write_object("inline", b"old")
        store.flush()
        store.write_object("inline", b"latest")
        store.write_object("chunked", chunked)
        store.flush()

    # Keep real local conditional commits; route immutable shard reads through the URI backend.
    snapshot = read_snapshot(FineStoreLayout(local_root))
    tables = {
        name: table.model_copy(
            update={
                "shards": [
                    shard.model_copy(update={"path": f"{_BLOB_URI_SCHEME}://{shard.path}"}) for shard in table.shards
                ]
            }
        )
        for name, table in snapshot.manifest.tables.items()
    }
    snapshot = replace(snapshot, manifest=snapshot.manifest.model_copy(update={"tables": tables}))
    filesystem, _ = factory.url_to_fs(root)
    filesystem.read_cache_types.clear()
    view = ReadView(root, snapshot=snapshot)
    assert view.read_blobs(["inline", "chunked", "missing"]) == {
        "inline": b"latest",
        "chunked": chunked,
    }
    diagnostics = view.read_diagnostics()
    assert diagnostics.descriptor_lookups == 1
    assert diagnostics.bytes_returned == len(b"latest") + len(chunked)
    assert (diagnostics.requested_names, diagnostics.found_names) == (3, 2)
    assert view.read_blobs(["inline", "missing", "inline"]) == {"inline": b"latest"}
    cumulative = view.read_diagnostics()
    assert (cumulative.read_calls, cumulative.requested_names, cumulative.found_names) == (2, 5, 3)
    assert cumulative.bytes_returned == len(chunked) + 2 * len(b"latest")
    diagnostics.bytes_returned = 0
    assert view.read_diagnostics().bytes_returned == cumulative.bytes_returned
    assert filesystem.read_cache_types
    assert set(filesystem.read_cache_types) == {"none"}


def test_blob_lookup_avoids_remote_readahead_of_unmatched_row_groups(tmp_path, monkeypatch):
    root = str(tmp_path / "archive")
    monkeypatch.setattr(shard_writer, "ROW_GROUP_TARGET_BYTES", 64 * 1024)
    # Exercise older inline values that exceed the current 128 KiB cutoff.
    monkeypatch.setattr(store_module, "INLINE_BLOB_BYTES", 2 * 1024 * 1024)
    random_bytes = random.Random(0)
    with DataStore.open(root, max_buffer_bytes=64 * 1024 * 1024, flush_interval=600) as store:
        store.write_object("matched", b"old")
        store.flush()
        store.write_object("matched", b"latest")
        for index in range(16):
            store.write_object(f"unmatched-{index:02d}", random_bytes.randbytes(1024 * 1024))
        store.flush()
        snapshot = read_snapshot(FineStoreLayout(root))

    filesystem = _RangeFileSystem(skip_instance_cache=True)
    monkeypatch.setattr(factory, "url_to_fs", lambda path: (filesystem, path))
    view = ReadView(root, snapshot=snapshot)
    assert view.read_blobs(["matched", "absent", "matched"]) == {"matched": b"latest"}
    diagnostics = view.read_diagnostics()
    bounded_bytes = filesystem.fetched_bytes
    assert diagnostics.bytes_returned == len(b"latest")
    assert diagnostics.descriptor_lookups == 1

    # The unchanged Arrow filter can prune the other row groups. Default fsspec
    # read-ahead still pulls their bytes after reading the small matched group.
    filesystem.fetched_bytes = 0
    dataset = pds.dataset(
        [shard.path for shard in view.list_shards(BlobTables.DESCRIPTORS)],
        filesystem=PyFileSystem(FSSpecHandler(filesystem)),
        format="parquet",
    )
    baseline = dataset.scanner(
        filter=pds.field("name") == "matched",
        fragment_scan_options=pds.ParquetFragmentScanOptions(pre_buffer=False),
        use_threads=False,
    ).to_table()
    assert set(baseline.column("data").to_pylist()) == {b"old", b"latest"}
    assert filesystem.fetched_bytes > 16 * 1024 * 1024
    assert bounded_bytes < 512 * 1024


def test_compaction_buffers_compressed_file_reads(tmp_path, monkeypatch):
    root = str(tmp_path / "archive")
    row_bytes = 1024 * 1024
    rows_per_flush = 9
    monkeypatch.setattr(shard_writer, "ROW_GROUP_TARGET_BYTES", row_bytes // 2)
    with DataStore.open(root, flush_interval=3600, max_buffer_bytes=2**40, compaction_levels=(1000,)) as store:
        table = store.table("samples", primary_key=("id",), on_conflict=OnConflict.SUPERSEDE)
        expected = {}
        for revision in range(2):
            for index in range(rows_per_flush):
                doc = random.Random(1729 + revision * rows_per_flush + index).randbytes(row_bytes)
                table.append({"id": index, "revision": revision, "doc": doc})
                if revision == 1:
                    expected[index] = doc
            store.flush()

        filesystem = _RangeFileSystem(skip_instance_cache=True)
        with monkeypatch.context() as patch:
            patch.setattr(factory, "url_to_fs", lambda path: (filesystem, path))
            result = compact_table(root, "samples")
        assert result.written + result.superseded == 2 * rows_per_flush
        assert filesystem.fetched_bytes > 0
        # Inputs above the whole-file prefetch budget must retain bounded streaming reads.
        assert filesystem.largest_fetch <= 2 * row_bytes
        rows = ReadView(root).iter_rows("samples")
        assert [(row["id"], row["revision"], row["doc"]) for row in rows] == [
            (index, 1, expected[index]) for index in range(rows_per_flush)
        ]


def test_persistent_cache_round_trips_and_supersedes_named_bytes(tmp_path):
    root = str(tmp_path / "cache")
    cache = PersistentKvCache.at(root)
    cache.store("kernel", b"first")
    cache.store("kernel", b"second")
    cache.close()
    assert len(ReadView(root).list_shards(BlobTables.DESCRIPTORS)) == 1

    reader = PersistentKvCache.at(root)
    assert reader.load("kernel") == b"second"
    assert reader.load("missing") is None
    reader.close()


def test_persistent_cache_keeps_a_loaded_value_in_memory(tmp_path):
    root = str(tmp_path / "cache")
    writer = PersistentKvCache.at(root)
    writer.store("kernel", b"object-code")
    writer.close()

    reader = PersistentKvCache.at(root)
    assert reader.load("kernel") == b"object-code"
    drop_table(root, BlobTables.DESCRIPTORS)
    assert reader.load("kernel") == b"object-code"


def test_batch_cache_reads_latest_persisted_values_and_preserves_memory_hits(tmp_path):
    root = str(tmp_path / "cache")
    writer = PersistentKvCache.at(root)
    writer.store("first", b"old")
    writer.store("first", b"new")
    writer.store("second", b"other")
    writer.close()

    reader = PersistentKvCache.at(root)
    assert reader.load_many(["first", "second", "missing", "first"]) == {"first": b"new", "second": b"other"}
    first_read = reader.read_diagnostics()
    assert (first_read.requested_keys, first_read.memory_hits, first_read.storage_hits, first_read.misses) == (
        3,
        0,
        2,
        1,
    )
    assert (first_read.blob_reads.read_calls, first_read.blob_reads.requested_names) == (1, 3)
    drop_table(root, BlobTables.DESCRIPTORS)
    assert reader.load_many(["first", "second", "missing"]) == {"first": b"new", "second": b"other"}
    diagnostics = reader.read_diagnostics()
    assert (diagnostics.load_calls, diagnostics.memory_hits, diagnostics.storage_hits, diagnostics.misses) == (
        2,
        2,
        2,
        2,
    )
    assert diagnostics.blob_reads.read_calls == 2
    first_read.blob_reads.bytes_returned = 0
    assert diagnostics.blob_reads.bytes_returned == len(b"new") + len(b"other")
    reader.close()


def test_batch_cache_lookup_bounds_decoding_of_unrequested_inline_values(tmp_path):
    root = str(tmp_path / "cache")
    # A compacted review cache has large inline values spread across row groups.
    # Requesting a few keys must not decode the whole shard at once.
    with DataStore.open(root, max_buffer_bytes=512 * 1024 * 1024, flush_interval=600) as store:
        for index in range(8192):
            store.write_object(f"{index:08d}", index.to_bytes(8, "little") + b"x" * (8192 - 8))

    cache = PersistentKvCache.at(root)
    default_pool = pa.default_memory_pool()
    pool = pa.proxy_memory_pool(default_pool)
    pa.set_memory_pool(pool)
    try:
        indexes = range(0, 8192, 128)
        values = cache.load_many([f"{index:08d}" for index in indexes])
        peak_bytes = pool.max_memory()
    finally:
        pa.set_memory_pool(default_pool)
        cache.close()
    assert values == {f"{index:08d}": index.to_bytes(8, "little") + b"x" * (8192 - 8) for index in indexes}
    # The requested values occupy 512 KiB. A batch read should not decode all
    # 64 MiB of inline payloads into Arrow memory at once.
    assert peak_bytes < 64 * 1024 * 1024


def test_cache_reads_an_archive_without_its_format_marker_and_the_next_store_recreates_it(tmp_path):
    # A TTL prefix expires the write-once marker while HEAD survives. Reads keep hitting, and
    # the writer open behind the next store puts the marker back.
    root = tmp_path / "cache"
    writer = PersistentKvCache.at(str(root))
    writer.store("kernel", b"object-code")
    writer.close()
    (root / "_archive.json").unlink()

    cache = PersistentKvCache.at(str(root))
    assert cache.load("kernel") == b"object-code"
    cache.store("other", b"more-code")
    cache.close()

    assert (root / "_archive.json").exists()
    assert PersistentKvCache.at(str(root)).load("other") == b"more-code"


def test_in_memory_cache_never_resolves_storage():
    cache = PersistentKvCache.in_memory()
    cache.store("k", b"v")
    assert cache.load("k") == b"v"
    assert cache.location() is None


def test_cache_root_resolves_lazily():
    calls = []

    def resolve() -> str:
        calls.append(1)
        return "/unused"

    PersistentKvCache(resolve)
    assert calls == []


def test_prefix_cache_uses_region_local_fine_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "marin_temp_bucket", lambda _ttl, prefix: str(tmp_path / prefix))
    cache = PersistentKvCache.for_prefix("cutlass-kernels")
    cache.store("kernel", b"value")
    cache.close()
    assert PersistentKvCache.for_prefix("cutlass-kernels").load("kernel") == b"value"


@pytest.mark.parametrize("expired", ["descriptor", "parts", "schema"])
def test_cache_reads_survivors_and_repairs_expired_objects_on_write(archive_root, expired):
    root = archive_root
    payload = b"x" * (129 * 1024)
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("lost", payload)
        store.flush()
        store.compact(BlobTables.DESCRIPTORS)
        old = ReadView(root)
        store.write_object("healthy", b"kept")
        store.flush()
    if expired == "schema":
        missing = old.table_metadata_path(BlobTables.DESCRIPTORS)
    else:
        table = BlobTables.DESCRIPTORS if expired == "descriptor" else BlobTables.PARTS
        missing = old.list_shards(table)[0].path
    assert missing is not None
    StoragePath(missing).rm()
    with pytest.raises(FileNotFoundError):
        ReadView(root).read_blob("lost")

    token = ReadView(root).token
    cache = PersistentKvCache.at(root)
    expected = {"healthy": b"kept"}
    if expired == "schema":
        expected["lost"] = payload
    assert cache.load_many(["lost", "healthy"]) == expected
    assert ReadView(root).token == token  # Reading the cache must never repair storage.

    cache.store("lost", b"recompiled")
    cache.store("new", b"new")
    cache.close()
    assert ReadView(root).read_blobs(["lost", "healthy", "new"]) == {
        "lost": b"recompiled",
        "healthy": b"kept",
        "new": b"new",
    }
    compact_table(root, BlobTables.DESCRIPTORS)
    compact_table(root, BlobTables.PARTS)
    assert PersistentKvCache.at(root).load("healthy") == b"kept"


def test_cache_repair_preserves_a_concurrent_commit(tmp_path, monkeypatch):
    root = str(tmp_path / "cache")
    with DataStore.open(root) as store:
        store.write_object("expired", b"old")
        store.flush()
    StoragePath(ReadView(root).list_shards(BlobTables.DESCRIPTORS)[0].path).rm()

    with DataStore.open(root, flush_interval=600) as competitor:
        real_write = LocalConditionalObject.write
        raced = False

        def publish_competitor(head, data, *, expected_version):
            nonlocal raced
            if head.path == FineStoreLayout(root).head_path and not raced:
                raced = True
                competitor.write_object("concurrent", b"preserved")
                competitor.flush()
            return real_write(head, data, expected_version=expected_version)

        monkeypatch.setattr(LocalConditionalObject, "write", publish_competitor)
        maintain_cache(root, refresh_before=None)
        assert ReadView(root).read_blobs(["expired", "concurrent"]) == {"concurrent": b"preserved"}


@pytest.mark.parametrize("expired_object", ["shard", "schema"])
def test_cache_read_requests_repair_from_an_existing_writer(tmp_path, expired_object):
    root = str(tmp_path / "cache")
    with DataStore.open(root) as store:
        store.write_object("expired", b"old")
        store.flush()
    view = ReadView(root)
    expired = (
        view.list_shards(BlobTables.DESCRIPTORS)[0].path
        if expired_object == "shard"
        else view.table_metadata_path(BlobTables.DESCRIPTORS)
    )
    assert expired is not None
    cache = PersistentKvCache.at(root)
    cache.store("healthy", b"kept")
    StoragePath(expired).rm()

    assert cache.load("expired") == (None if expired_object == "shard" else b"old")
    cache.store("new", b"new")
    cache.close()

    expected = {"healthy": b"kept", "new": b"new"}
    if expired_object == "schema":
        expected["expired"] = b"old"
    assert ReadView(root).read_blobs(["expired", "healthy", "new"]) == expected


@pytest.mark.parametrize("expired", ["descriptor", "parts", "later-compaction"])
def test_cache_expiry_does_not_revive_superseded_values(archive_root, expired):
    root = archive_root
    old = b"a" * (129 * 1024) if expired == "parts" else b"old"
    new = b"b" * len(old)
    with DataStore.open(root, flush_interval=600) as store:
        if expired == "later-compaction":
            # Keep two older level-two runs so they can be merged after the newest
            # value commits, without including that value in the compaction.
            for group in range(2):
                for item in range(2):
                    store.write_object(f"other-{group}-{item}", b"other")
                    if group == item == 0:
                        store.write_object("kernel", old)
                    store.flush()
                    store.compact(BlobTables.DESCRIPTORS, generation=0)
                store.compact(BlobTables.DESCRIPTORS, generation=1)
        else:
            store.write_object("kernel", old)
            store.flush()
            store.compact(BlobTables.DESCRIPTORS, generation=0)
            if expired == "parts":
                store.compact(BlobTables.PARTS, generation=0)
        store.write_object("kernel", new)
        store.flush()
        store.compact(BlobTables.DESCRIPTORS, generation=0)
        table = BlobTables.PARTS if expired == "parts" else BlobTables.DESCRIPTORS
        if expired == "parts":
            store.compact(BlobTables.PARTS, generation=0)
        missing = max(ReadView(root).list_shards(table), key=lambda shard: shard.commit_sequence).path
        if expired == "later-compaction":
            store.compact(BlobTables.DESCRIPTORS, generation=2)
        store.write_object("healthy", b"kept")
        store.flush()
    StoragePath(missing).rm()

    cache = PersistentKvCache.at(root)
    assert cache.load_many(["kernel", "healthy"]) == {"healthy": b"kept"}
    cache.store("new-key", b"new")
    cache.close()
    assert ReadView(root).read_blobs(["kernel", "healthy", "new-key"]) == {"healthy": b"kept", "new-key": b"new"}


@pytest.mark.parametrize("error_type", [PermissionError, TimeoutError])
def test_cache_retention_does_not_prune_on_storage_errors(tmp_path, monkeypatch, error_type):
    root = str(tmp_path / "cache")
    with DataStore.open(root) as store:
        store.write_object("kernel", b"preserved")
        store.flush()
    token = ReadView(root).token
    real_modified = LocalFileSystem.modified

    def unavailable(filesystem, path):
        if str(path).endswith(".parquet"):
            raise error_type(path)
        return real_modified(filesystem, path)

    with monkeypatch.context() as patch:
        patch.setattr(LocalFileSystem, "modified", unavailable)
        with pytest.raises(error_type):
            maintain_cache(root, refresh_before=None)
    assert ReadView(root).token == token
    assert ReadView(root).read_blob("kernel") == b"preserved"


def test_cache_write_refreshes_aging_objects_and_preserves_versions(tmp_path):
    root = str(tmp_path / "ttl=30d" / "cache")
    with DataStore.open(root) as store:
        store.write_object("kernel", b"old")
        store.flush()
        store.compact(BlobTables.DESCRIPTORS)
        store.write_object("kernel", b"latest")
        store.flush()
    pinned = ReadView(root)
    paths = [shard.path for shard in pinned.list_shards(BlobTables.DESCRIPTORS)]
    paths.append(pinned.table_metadata_path(BlobTables.DESCRIPTORS))
    aged = datetime.now(UTC) - timedelta(days=20)
    for path in paths:
        os.utime(path, (aged.timestamp(), aged.timestamp()))
    before = datetime.now(UTC)

    assert PersistentKvCache.at(root).load("kernel") == b"latest"
    assert all(StoragePath(path).mtime() < before for path in paths)
    with_cache = PersistentKvCache.at(root)
    with_cache.store("another", b"new")
    with_cache.close()

    assert all(StoragePath(path).mtime() >= before for path in paths)
    assert pinned.read_blob("kernel") == b"latest"
    assert ReadView(root).read_blob("kernel") == b"latest"


def test_cache_retention_copies_remote_payloads_only_when_aging(simulated_bucket):
    root = "s3://finestore-tests/ttl=30d/cache"
    cache = PersistentKvCache.at(root)
    cache.store("kernel", b"code")
    cache.close()
    before = simulated_bucket.log.snapshot()
    maintain_cache(root, refresh_before=datetime.now(UTC) - timedelta(days=15))
    assert simulated_bucket.log.delta(before).requests[Request.COPY] == 0

    # A future cutoff makes the real in-memory objects eligible without sleeping.
    maintain_cache(root, refresh_before=datetime.now(UTC) + timedelta(days=1))
    assert simulated_bucket.log.delta(before).requests[Request.COPY] >= 2
    assert ReadView(root).read_blob("kernel") == b"code"


def test_non_writer_cache_keeps_value_in_memory_without_persisting(tmp_path, monkeypatch):
    root = tmp_path / "cutlass-kernels"
    monkeypatch.setattr(cache_module, "marin_temp_bucket", lambda _ttl, _prefix: str(root))
    cache = PersistentKvCache.for_prefix("cutlass-kernels", is_writer=lambda: False)

    cache.store("kernel", b"value")
    assert cache.load("kernel") == b"value"
    cache.close()

    assert PersistentKvCache.at(str(root)).load("kernel") is None


def test_remote_cache_can_commit_object_larger_than_store_buffer(tmp_path, monkeypatch):
    root = str(tmp_path / "cache")
    open_store = DataStore.open
    monkeypatch.setattr(StoragePath, "is_remote", property(lambda _self: True))
    monkeypatch.setattr(cache_module, "_MAX_BATCH_DATA_BYTES", 64)
    monkeypatch.setattr(
        cache_module.DataStore,
        "open",
        classmethod(lambda _cls, path: open_store(path, max_buffer_bytes=64)),
    )

    cache = PersistentKvCache.at(root)
    cache.store("kernel", b"x" * 40)
    cache.close()

    assert ReadView(root).read_blob("kernel") == b"x" * 40


def test_cache_normal_process_exit_drains_pending_remote_commit(tmp_path):
    root = str(tmp_path / "cache")

    subprocess.run([sys.executable, "-c", _CACHE_PROCESS, root, "slow"], check=True, timeout=10)

    assert ReadView(root).read_blob("kernel") == b"object-code"


def test_cache_process_exit_abandons_stalled_remote_commit(tmp_path):
    root = str(tmp_path / "cache")

    result = subprocess.run([sys.executable, "-c", _CACHE_PROCESS, root, "stall"], timeout=10)

    assert result.returncode == 0


def test_cache_process_exit_does_not_start_remote_compaction(tmp_path):
    root = str(tmp_path / "cache")
    subprocess.run([sys.executable, "-c", _CACHE_EXIT_COMPACTION_PROCESS, root], check=True, timeout=10)
    view = ReadView(root)
    assert view.read_blobs(["0", "1"]) == {"0": b"object-code", "1": b"object-code"}
    assert len(view.list_shards(BlobTables.DESCRIPTORS)) == 2
