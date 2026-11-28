# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import finestore.fileset as fileset_module
import pytest
import rigging.timing as timing
from finestore.fileset import (
    FileSetFetchDeadlineError,
    FileSetFetchLimitError,
    FileSetFetchLimits,
    FileSetPublishLimitError,
    FineStoreDirectory,
    fetch_file_set,
)
from finestore.layout import BlobColumns, BlobTables
from finestore.reader import BlobCorruptionError, ReadView
from finestore.store import OBJECT_PART_BYTES, DataStore

_PRESERVED_MTIME_NS = 1_000_000_000


def test_file_set_round_trips_nested_files_in_one_level_zero(tmp_path):
    root = str(tmp_path / "remote")
    local = tmp_path / "local"
    writer = FineStoreDirectory(root, str(local), flush_interval=3600)
    for index in range(100):
        path = local / "nested" / f"{index}.textproto"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"value-{index}".encode())
    writer.close()

    assert len(ReadView(root).list_shards(BlobTables.DESCRIPTORS)) == 1
    reloaded = tmp_path / "reloaded"
    fetch_file_set(root, str(reloaded))
    assert sorted(path.read_bytes() for path in reloaded.rglob("*.textproto")) == sorted(
        f"value-{index}".encode() for index in range(100)
    )


def test_file_set_materializes_chunked_file(tmp_path):
    root = str(tmp_path / "remote")
    local = tmp_path / "local"
    payload = b"x" * (OBJECT_PART_BYTES + 1)
    writer = FineStoreDirectory(root, str(local), flush_interval=3600, max_batch_data_bytes=1024 * 1024)
    path = local / "logs" / "archive.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    writer.close()

    reloaded = tmp_path / "reloaded"
    assert fetch_file_set(root, str(reloaded)) == {"logs/archive.bin"}
    reloaded_path = reloaded / "logs" / "archive.bin"
    assert reloaded_path.read_bytes() == payload
    os.utime(reloaded_path, ns=(_PRESERVED_MTIME_NS, _PRESERVED_MTIME_NS))
    assert fetch_file_set(root, str(reloaded)) == {"logs/archive.bin"}
    assert reloaded_path.stat().st_mtime_ns == _PRESERVED_MTIME_NS


def test_file_set_preserves_matching_files_and_replaces_same_size_difference(tmp_path):
    root = str(tmp_path / "remote")
    with DataStore.open(root) as store:
        store.write_object("matching.textproto", b"one")
        store.write_object("different.textproto", b"two")
        store.flush()

    local = tmp_path / "local"
    assert fetch_file_set(root, str(local)) == {"matching.textproto", "different.textproto"}
    matching = local / "matching.textproto"
    different = local / "different.textproto"
    os.utime(matching, ns=(_PRESERVED_MTIME_NS, _PRESERVED_MTIME_NS))
    different.write_bytes(b"bad")

    assert fetch_file_set(root, str(local)) == {"matching.textproto", "different.textproto"}
    assert matching.stat().st_mtime_ns == _PRESERVED_MTIME_NS
    assert different.read_bytes() == b"two"


def test_file_set_replaces_matching_symlink(tmp_path):
    root = str(tmp_path / "remote")
    with DataStore.open(root) as store:
        store.write_object("entry.textproto", b"cached")
        store.flush()

    outside = tmp_path / "outside.textproto"
    outside.write_bytes(b"cached")
    local = tmp_path / "local"
    local.mkdir()
    entry = local / "entry.textproto"
    entry.symlink_to(outside)

    assert fetch_file_set(root, str(local)) == {"entry.textproto"}
    assert not entry.is_symlink()
    assert entry.read_bytes() == b"cached"
    assert outside.read_bytes() == b"cached"


def test_file_set_replaces_unreadable_local_file_and_continues(tmp_path, monkeypatch):
    root = str(tmp_path / "remote")
    with DataStore.open(root) as store:
        store.write_object("a.textproto", b"new")
        store.write_object("b.textproto", b"next")
        store.flush()

    local = tmp_path / "local"
    local.mkdir()
    blocked = local / "a.textproto"
    blocked.write_bytes(b"old")
    original_open = Path.open

    def deny_local_read(path, mode="r", *args, **kwargs):
        if path == blocked and mode == "rb":
            raise PermissionError("unreadable cache entry")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "open", deny_local_read)
        fetched = fetch_file_set(root, str(local))

    assert fetched == {"a.textproto", "b.textproto"}
    assert blocked.read_bytes() == b"new"
    assert (local / "b.textproto").read_bytes() == b"next"


def test_file_set_validates_inline_file_size(tmp_path):
    root = str(tmp_path / "remote")
    with DataStore.open(root, writer_id="w1") as store:
        table = store.table(BlobTables.DESCRIPTORS, primary_key=(BlobColumns.NAME,))
        table.append(
            {
                BlobColumns.NAME: "archive.bin",
                BlobColumns.DATA: b"payload",
                BlobColumns.SIZE: 1,
                BlobColumns.PART_COUNT: None,
            }
        )
        store.flush()

    with pytest.raises(BlobCorruptionError, match="declares 1 bytes"):
        fetch_file_set(root, str(tmp_path / "local"))


def test_file_set_concurrent_writers_rebase_disjoint_files(tmp_path):
    root = str(tmp_path / "remote")
    first_local = tmp_path / "first"
    second_local = tmp_path / "second"
    first = FineStoreDirectory(root, str(first_local), flush_interval=3600)
    second = FineStoreDirectory(root, str(second_local), flush_interval=3600)
    (first_local / "a").write_bytes(b"one")
    (second_local / "b").write_bytes(b"two")
    first.close()
    second.close()

    reloaded = tmp_path / "reloaded"
    fetch_file_set(root, str(reloaded))
    assert (reloaded / "a").read_bytes() == b"one"
    assert (reloaded / "b").read_bytes() == b"two"


def test_file_set_does_not_republish_files_fetched_at_open(tmp_path):
    root = str(tmp_path / "remote")
    seed = tmp_path / "seed"
    writer = FineStoreDirectory(root, str(seed), flush_interval=3600)
    (seed / "a").write_bytes(b"one")
    writer.close()

    local = tmp_path / "local"
    mirror = FineStoreDirectory(root, str(local), flush_interval=3600)
    (local / "a").write_bytes(b"tampered")
    mirror.close()
    reloaded = tmp_path / "reloaded"
    fetch_file_set(root, str(reloaded))
    assert (reloaded / "a").read_bytes() == b"one"


def test_file_set_mirror_excludes_preexisting_local_files(tmp_path):
    root = str(tmp_path / "remote")
    local = tmp_path / "local"
    local.mkdir()
    (local / "leftover").write_bytes(b"old")

    mirror = FineStoreDirectory(root, str(local), flush_interval=3600)
    (local / "fresh").write_bytes(b"new")
    mirror.close()

    assert fetch_file_set(root, str(tmp_path / "reloaded")) == {"fresh"}


@pytest.mark.parametrize(("max_files", "max_bytes"), [(1, 100), (10, 4)])
def test_file_set_mirror_refuses_to_publish_an_unfetchable_snapshot(tmp_path, max_files, max_bytes):
    root = str(tmp_path / "remote")
    local = tmp_path / "local"
    limits = FileSetFetchLimits(max_files=max_files, max_bytes=max_bytes, timeout=60)
    mirror = FineStoreDirectory(root, str(local), flush_interval=3600, fetch_limits=limits)
    (local / "a").write_bytes(b"one")
    (local / "b").write_bytes(b"two")

    with pytest.raises(FileSetPublishLimitError):
        mirror.close()
    assert fetch_file_set(root, str(tmp_path / "reloaded")) == set()


@pytest.mark.parametrize(
    ("max_files", "max_bytes", "message"),
    [(1, 100, "files"), (10, 4, "bytes")],
)
def test_file_set_fetch_rejects_oversized_snapshot(tmp_path, max_files, max_bytes, message):
    root = str(tmp_path / "remote")
    with DataStore.open(root) as store:
        store.write_object("a", b"one")
        store.write_object("b", b"two")
        store.flush()

    limits = FileSetFetchLimits(max_files=max_files, max_bytes=max_bytes, timeout=60)
    with pytest.raises(FileSetFetchLimitError, match=message):
        fetch_file_set(root, str(tmp_path / "local"), limits=limits)


def test_file_set_fetch_deadline_and_progress_are_visible(tmp_path, monkeypatch, caplog):
    root = str(tmp_path / "remote")
    with DataStore.open(root) as store:
        store.write_object("a", b"one")
        store.write_object("b", b"two")
        store.flush()

    ticks = count()
    fake_time = SimpleNamespace(monotonic=lambda: next(ticks) * 0.2)
    monkeypatch.setattr(fileset_module, "time", fake_time)
    monkeypatch.setattr(timing, "time", fake_time)
    limits = FileSetFetchLimits(max_files=10, max_bytes=100, timeout=1, log_interval=0.1)
    with caplog.at_level("INFO"), pytest.raises(FileSetFetchDeadlineError, match="deadline"):
        fetch_file_set(root, str(tmp_path / "local"), limits=limits)
    assert any(record.args[:4] == (1, 1, 0, 3) for record in caplog.records if isinstance(record.args, tuple))


def test_file_set_rejects_unsafe_committed_paths(tmp_path):
    root = str(tmp_path / "remote")
    with DataStore.open(root, writer_id="malicious") as store:
        store.write_object("../escape", b"payload")
        store.flush()
    with pytest.raises(ValueError, match="unsafe"):
        fetch_file_set(root, str(tmp_path / "local"))
