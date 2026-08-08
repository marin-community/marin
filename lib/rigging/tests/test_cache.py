# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import rigging.cache as cache_module
from rigging.cache import PersistentKvCache, SyncedDirectory, flush_background_writes, sync_kv_cache
from rigging.provenance import Provenance


def test_store_then_load_round_trips_bytes(tmp_path):
    cache = PersistentKvCache.at(str(tmp_path))
    cache.store("k", b"object-code")
    assert cache.load("k") == b"object-code"


def test_load_of_an_absent_key_is_none(tmp_path):
    assert PersistentKvCache.at(str(tmp_path)).load("missing") is None


def test_store_overwrites_the_previous_value(tmp_path):
    cache = PersistentKvCache.at(str(tmp_path))
    cache.store("k", b"first")
    cache.store("k", b"second")
    assert cache.load("k") == b"second"


def test_distinct_keys_are_distinct_objects(tmp_path):
    cache = PersistentKvCache.at(str(tmp_path))
    cache.store("a", b"aaa")
    cache.store("b", b"bbb")

    assert cache.load("a") == b"aaa"
    assert cache.load("b") == b"bbb"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "b"]


def test_store_leaves_no_staging_file_behind(tmp_path):
    """A completed store renames its staged temp into place rather than leaving it."""
    cache = PersistentKvCache.at(str(tmp_path))
    cache.store("k", b"payload")
    assert [p.name for p in tmp_path.iterdir()] == ["k"]


def test_in_memory_cache_round_trips_without_a_directory():
    """A memory-only cache stores and loads within the process and reports no location."""
    cache = PersistentKvCache.in_memory()
    cache.store("k", b"v")
    assert cache.load("k") == b"v"
    assert cache.location() is None


def test_a_read_key_is_served_from_memory_after_its_object_is_removed(tmp_path):
    """A load populates the memory tier, so a repeated key is answered without re-reading the directory."""
    PersistentKvCache.at(str(tmp_path)).store("k", b"v")

    reader = PersistentKvCache.at(str(tmp_path))
    assert reader.load("k") == b"v"
    (tmp_path / "k").unlink()
    assert reader.load("k") == b"v"


def test_the_directory_resolves_lazily_not_at_construction():
    """Constructing a cache does not call its resolver; the first access does."""
    calls: list[int] = []

    def _resolve() -> str:
        calls.append(1)
        return "/unused"

    PersistentKvCache(_resolve)
    assert calls == []


def test_a_local_directory_is_written_inline(tmp_path, monkeypatch):
    """A local path is fast to write, so a store persists it before returning."""
    submitted: list = []
    monkeypatch.setattr(cache_module, "_submit_background_write", submitted.append)

    PersistentKvCache.at(str(tmp_path)).store("k", b"v")

    assert submitted == []
    assert (tmp_path / "k").read_bytes() == b"v"


def test_an_object_store_directory_is_written_in_the_background(monkeypatch):
    """A remote write is offloaded so it never blocks the thread that built the value."""
    submitted: list = []
    monkeypatch.setattr(cache_module, "_submit_background_write", submitted.append)

    PersistentKvCache(lambda: "gs://bucket/prefix").store("k", b"v")

    assert len(submitted) == 1


def test_for_prefix_round_trips_through_the_temp_object_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "marin_temp_bucket", lambda ttl_days, prefix: str(tmp_path / prefix))

    cache = PersistentKvCache.for_prefix("cutlass-kernels")
    cache.store("k", b"v")
    flush_background_writes()

    assert (tmp_path / "cutlass-kernels" / "k").read_bytes() == b"v"
    assert cache.load("k") == b"v"


def test_synced_directory_stages_remote_files_down_and_mirrors_new_ones_up(tmp_path):
    """A synced directory starts warm from the object store and mirrors new files back."""
    remote, local = tmp_path / "remote", tmp_path / "local"
    remote.mkdir()
    (remote / "a.txt").write_bytes(b"one")

    handle = SyncedDirectory(remote=lambda: str(remote), local=str(local))
    assert (local / "a.txt").read_bytes() == b"one"

    # Tamper with the staged-down file to prove it is not re-uploaded, and write a
    # new one the way the consumer (XLA) would.
    (remote / "a.txt").write_bytes(b"tampered")
    (local / "b.txt").write_bytes(b"two")
    handle.close()

    assert (remote / "b.txt").read_bytes() == b"two"
    assert (remote / "a.txt").read_bytes() == b"tampered"


def test_synced_directory_mirrors_nested_files_up_when_the_remote_starts_empty(tmp_path):
    """An absent object-store directory is tolerated; nested local files still mirror up."""
    remote, local = tmp_path / "remote", tmp_path / "local"

    handle = SyncedDirectory(remote=lambda: str(remote), local=str(local))
    (local / "sub").mkdir(parents=True)
    (local / "sub" / "k").write_bytes(b"v")
    handle.close()

    assert (remote / "sub" / "k").read_bytes() == b"v"


def test_sync_kv_cache_namespaces_the_object_store_by_tree_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "launch_provenance", lambda: _provenance("treehash"))
    monkeypatch.setattr(cache_module, "marin_temp_bucket", lambda ttl_days, prefix: str(tmp_path / prefix))
    local = tmp_path / "local"

    handle = sync_kv_cache("xla-autotune", str(local))
    (local / "k").write_bytes(b"v")
    handle.close()

    assert (tmp_path / "xla-autotune" / "treehash" / "k").read_bytes() == b"v"


def test_sync_kv_cache_is_a_noop_without_a_tree_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "launch_provenance", lambda: _provenance(""))
    assert sync_kv_cache("xla-autotune", str(tmp_path)) is None


def _provenance(tree_hash: str) -> Provenance:
    return Provenance(tree_hash=tree_hash, base_commit="", dirty=False, branch=None, built_by=None)
