# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier

import finestore.reader as reader_module
import finestore.store as store_module
import pytest
import rigging.filesystem.factory as factory
from finestore.admin import drop_table
from finestore.cache import PersistentKvCache
from finestore.compaction import compact_table
from finestore.layout import BlobTables
from finestore.reader import BlobKeyIndex, BlobReadDiagnostics, ReadView
from finestore.store import DataStore


@pytest.fixture
def archive(tmp_path):
    root = str(tmp_path / "archive")
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("first", b"old")
        store.write_object("other", b"other")
        store.flush()
        store.write_object("first", b"new")
        store.flush()
    return root, tmp_path / "index"


def test_index_refresh_observes_external_commits_compaction_and_drop(archive):
    root, directory = archive
    index = BlobKeyIndex(directory)
    pinned = ReadView(root, blob_key_index=index)
    assert pinned.read_blobs(["first", "missing", "first"]) == {"first": b"new"}
    compact_table(root, BlobTables.DESCRIPTORS)
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("first", b"latest")
        store.write_object("added", b"added")
    cache = PersistentKvCache.at(root, key_index_directory=directory)
    assert cache.load_many(["first", "added", "other", "missing"]) == {
        "first": b"latest",
        "added": b"added",
        "other": b"other",
    }
    cache.close()
    # A concurrent refresh for an older pinned snapshot may rebuild its pruned
    # identities, but must retain that snapshot's values and ordering.
    assert pinned.read_blobs(["first", "added"]) == {"first": b"new"}
    assert ReadView(root, blob_key_index=index).read_blob("first") == b"latest"
    drop_table(root, BlobTables.DESCRIPTORS)
    assert ReadView(root, blob_key_index=index).read_blobs(["first", "added"]) == {}


def test_warm_index_opens_only_matching_payload_shards(archive, monkeypatch):
    root, directory = archive
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("isolated", b"only-this-shard")
    expected_shard = ReadView(root).list_shards(BlobTables.DESCRIPTORS)[-1].path
    assert PersistentKvCache.at(root, key_index_directory=directory).load("isolated") == b"only-this-shard"
    fs, _ = factory.url_to_fs(root)
    original_open = fs.open
    opened = set()

    def open_file(path, *args, **kwargs):
        if str(path).endswith(".parquet"):
            opened.add(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(fs, "open", open_file)
    assert PersistentKvCache.at(root, key_index_directory=directory).load_many(["absent", "missing"]) == {}
    assert opened == set()
    assert PersistentKvCache.at(root, key_index_directory=directory).load("isolated") == b"only-this-shard"
    assert opened == {expected_shard}


def test_warm_index_reads_committed_values_while_another_connection_reserves_writes(archive):
    root, directory = archive
    cold = BlobReadDiagnostics()
    assert PersistentKvCache.at(root, key_index_directory=directory).load_many(["first"], diagnostics=cold) == {
        "first": b"new"
    }
    assert cold.index_refreshed_shards == len(ReadView(root).list_shards(BlobTables.DESCRIPTORS))
    assert cold.index_warm_reads == 0
    diagnostics = BlobReadDiagnostics()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with closing(sqlite3.connect(next(directory.glob("*.sqlite")))) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("DELETE FROM names")
            writer.execute("DELETE FROM shards")
            try:
                future = pool.submit(
                    ReadView(root, blob_key_index=BlobKeyIndex(directory)).read_blobs,
                    ["first", "other", "missing"],
                    diagnostics=diagnostics,
                )
                # The reader must finish without releasing the reserved write lock.
                assert future.result(timeout=3) == {"first": b"new", "other": b"other"}
            finally:
                writer.rollback()
    assert diagnostics.indexed_reads == 1
    assert diagnostics.index_fallbacks == 0
    assert diagnostics.scan_reads == 0
    assert diagnostics.index_warm_reads == 1
    assert diagnostics.index_refreshed_shards == 0
    assert diagnostics.index_lock_seconds == 0
    assert diagnostics.index_refresh_seconds == 0


def test_concurrent_index_refresh_keeps_each_pinned_manifest_values(archive):
    root, directory = archive
    old = ReadView(root, blob_key_index=BlobKeyIndex(directory))
    assert old.read_blobs(["first", "other"]) == {"first": b"new", "other": b"other"}
    compact_table(root, BlobTables.DESCRIPTORS)
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("first", b"latest")
        store.write_object("added", b"added")
    current = ReadView(root, blob_key_index=BlobKeyIndex(directory))
    barrier = Barrier(2)

    def read(view):
        values = []
        for _ in range(4):
            barrier.wait(timeout=5)
            values.append(view.read_blobs(["first", "other", "added", "missing"]))
        return values

    with ThreadPoolExecutor(max_workers=2) as pool:
        old_values, current_values = pool.map(read, [old, current])
    assert old_values == [{"first": b"new", "other": b"other"}] * 4
    assert current_values == [{"first": b"latest", "other": b"other", "added": b"added"}] * 4


def test_failed_name_scan_is_not_published_and_can_retry(archive, monkeypatch):
    root, directory = archive
    shard = ReadView(root).list_shards(BlobTables.DESCRIPTORS)[-1].path
    fs, _ = factory.url_to_fs(root)
    original_open = fs.open

    def unavailable(path, *args, **kwargs):
        if str(path) == shard:
            raise OSError("temporarily unavailable")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(fs, "open", unavailable)
        diagnostics = BlobReadDiagnostics()
        assert (
            PersistentKvCache.at(root, key_index_directory=directory).load_many(["first"], diagnostics=diagnostics) == {}
        )
        assert diagnostics.index_fallbacks == 1
        assert diagnostics.index_refreshed_shards == 1
        assert diagnostics.index_refresh_seconds > 0
    assert PersistentKvCache.at(root, key_index_directory=directory).load_many(["first", "other"]) == {
        "first": b"new",
        "other": b"other",
    }
    # Loss of a selected immutable shard must never return a stale indexed payload.
    Path(shard).unlink()
    assert PersistentKvCache.at(root, key_index_directory=directory).load("first") is None


def test_incremental_refresh_reads_only_the_new_shard(archive, monkeypatch):
    root, directory = archive
    assert PersistentKvCache.at(root, key_index_directory=directory).load("first") == b"new"
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("added", b"added")
    new_shard = ReadView(root).list_shards(BlobTables.DESCRIPTORS)[-1].path
    fs, _ = factory.url_to_fs(root)
    original_open = fs.open
    opened = set()

    def open_file(path, *args, **kwargs):
        if str(path).endswith(".parquet"):
            opened.add(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(fs, "open", open_file)
    assert PersistentKvCache.at(root, key_index_directory=directory).load("added") == b"added"
    assert opened == {new_shard}


@pytest.mark.parametrize("failure", ["corrupt", "unavailable", "full"])
def test_local_index_failure_falls_back_to_committed_values(archive, monkeypatch, failure):
    root, directory = archive
    assert PersistentKvCache.at(root, key_index_directory=directory).load("first") == b"new"
    if failure == "corrupt":
        next(directory.glob("*.sqlite")).write_bytes(b"broken database")
    elif failure == "unavailable":
        directory = directory / "not-a-directory"
        directory.write_bytes(b"file")
    else:
        # Force the first schema allocation beyond the local disk budget.
        directory = directory / "small"
        monkeypatch.setattr(reader_module, "_KEY_INDEX_BYTES", 4096)
    diagnostics = BlobReadDiagnostics()
    assert PersistentKvCache.at(root, key_index_directory=directory).load_many(
        ["first", "other"], diagnostics=diagnostics
    ) == {
        "first": b"new",
        "other": b"other",
    }
    assert diagnostics.index_fallbacks == 1
    assert diagnostics.scan_reads == 1
    assert diagnostics.indexed_reads == 0
    assert diagnostics.bytes_returned == len(b"newother")


def test_index_preserves_chunked_blob_validation_and_rewrites(tmp_path, monkeypatch):
    root = str(tmp_path / "archive")
    directory = tmp_path / "index"
    monkeypatch.setattr(store_module, "OBJECT_PART_BYTES", 4)
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("chunked", b"abcdefghijk")
    assert PersistentKvCache.at(root, key_index_directory=directory).load("chunked") == b"abcdefghijk"
    with DataStore.open(root, flush_interval=600) as store:
        store.write_object("chunked", b"12345")
    assert PersistentKvCache.at(root, key_index_directory=directory).load("chunked") == b"12345"
    drop_table(root, BlobTables.PARTS)
    assert PersistentKvCache.at(root, key_index_directory=directory).load("chunked") is None


def test_index_is_shared_by_simultaneous_fresh_processes(archive):
    root, directory = archive
    script = """
import json, sys
from pathlib import Path
from finestore.cache import PersistentKvCache
cache = PersistentKvCache.at(sys.argv[1], key_index_directory=Path(sys.argv[2]))
values = cache.load_many(['first', 'other', 'missing'])
cache.close()
print(json.dumps({key: value.decode() for key, value in values.items()}))
"""

    def read():
        result = subprocess.run(
            [sys.executable, "-c", script, root, str(directory)], capture_output=True, text=True, check=True, timeout=20
        )
        return json.loads(result.stdout)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: read(), range(2)))
    assert results == [{"first": "new", "other": "other"}] * 2
