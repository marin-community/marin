# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from finestore.fileset import FineStoreDirectory, fetch_file_set
from finestore.layout import BLOBS_TABLE
from finestore.reader import ReadView
from finestore.store import DataStore


def test_file_set_round_trips_nested_files_in_one_level_zero(tmp_path):
    root = str(tmp_path / "remote")
    local = tmp_path / "local"
    writer = FineStoreDirectory(root, str(local), flush_interval=3600)
    for index in range(100):
        path = local / "nested" / f"{index}.textproto"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"value-{index}".encode())
    writer.close()

    assert len(ReadView(root).list_shards(BLOBS_TABLE)) == 1
    reloaded = tmp_path / "reloaded"
    fetch_file_set(root, str(reloaded))
    assert sorted(path.read_bytes() for path in reloaded.rglob("*.textproto")) == sorted(
        f"value-{index}".encode() for index in range(100)
    )


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
