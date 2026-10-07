# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import random
import subprocess
import sys
import textwrap
from dataclasses import replace

import finestore.cache as cache_module
import finestore.store as store_module
import fsspec
import pyarrow.dataset as pds
from finestore import shard_writer
from finestore.admin import drop_table
from finestore.cache import PersistentKvCache
from finestore.commit import read_snapshot
from finestore.layout import BlobTables, FineStoreLayout
from finestore.reader import ReadView
from finestore.store import OBJECT_PART_BYTES, DataStore
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import factory
from rigging.filesystem.storage_path import StoragePath

_BLOB_URI_SCHEME = "blobread"

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


class _RangeFile(AbstractBufferedFile):
    def _fetch_range(self, start, end):
        self.fs.fetched_bytes += end - start
        with open(self.path, "rb") as source:
            source.seek(start)
            return source.read(end - start)


class _RangeFileSystem(LocalFileSystem):
    """Local bytes served through the same read-ahead cache as object storage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fetched_bytes = 0

    def _open(self, path, mode="rb", *, cache_type="readahead", **_kwargs):
        assert mode == "rb"
        return _RangeFile(self, path, mode=mode, block_size=50 * 1024 * 1024, cache_type=cache_type)


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

    filesystem = _RangeFileSystem(skip_instance_cache=True)
    monkeypatch.setattr(factory, "url_to_fs", lambda path: (filesystem, path))
    view = ReadView(root)
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


def test_persistent_cache_round_trips_and_supersedes_named_bytes(tmp_path):
    root = str(tmp_path / "cache")
    cache = PersistentKvCache.at(root)
    cache.store("kernel", b"first")
    cache.store("kernel", b"second")
    cache.close()

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

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import json
                import sys
                import pyarrow as pa
                from finestore.cache import PersistentKvCache

                indexes = range(0, 8192, 128)
                cache = PersistentKvCache.at(sys.argv[1])
                values = cache.load_many([f"{index:08d}" for index in indexes])
                cache.close()
                assert values == {
                    f"{index:08d}": index.to_bytes(8, "little") + b"x" * (8192 - 8)
                    for index in indexes
                }
                print(json.dumps({"peak_bytes": pa.default_memory_pool().max_memory()}))
                """
            ),
            root,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # The requested values occupy 512 KiB. A batch read should not decode all
    # 64 MiB of inline payloads into Arrow memory at once.
    assert json.loads(result.stdout)["peak_bytes"] < 64 * 1024 * 1024


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
