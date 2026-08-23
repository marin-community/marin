# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import pytest
from finestore.fileset import FineStoreDirectory, fetch_file_set
from finestore.layout import BlobColumns, BlobTables
from finestore.reader import BlobCorruptionError, ReadView
from finestore.store import OBJECT_PART_BYTES, DataStore

_FILE_SET_EXIT_PROCESS = textwrap.dedent(
    """
    import pathlib
    import sys
    import threading

    from finestore.fileset import FineStoreDirectory
    from finestore.store import DataStore

    root, local = sys.argv[1:]
    writer = FineStoreDirectory(root, local, flush_interval=3600)
    pathlib.Path(local, "new-cache-entry").write_bytes(b"value")

    def commit(_store, _rows):
        threading.Event().wait()

    DataStore._commit_transaction = commit
    """
)


def test_file_set_process_exit_does_not_wait_for_final_remote_commit(tmp_path):
    subprocess.run(
        [sys.executable, "-c", _FILE_SET_EXIT_PROCESS, str(tmp_path / "remote"), str(tmp_path / "local")],
        check=True,
        timeout=5,
    )


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
    assert (reloaded / "logs" / "archive.bin").read_bytes() == payload


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


def test_file_set_rejects_unsafe_committed_paths(tmp_path):
    root = str(tmp_path / "remote")
    with DataStore.open(root, writer_id="malicious") as store:
        store.write_object("../escape", b"payload")
        store.flush()
    with pytest.raises(ValueError, match="unsafe"):
        fetch_file_set(root, str(tmp_path / "local"))
