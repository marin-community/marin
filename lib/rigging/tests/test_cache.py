# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import rigging.cache as cache_module
from rigging.cache import PersistentKvCache, flush_background_writes, marin_kv_cache, sync_kv_cache


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
    cache = PersistentKvCache.at(str(tmp_path), suffix=".o")
    cache.store("a", b"aaa")
    cache.store("b", b"bbb")

    assert cache.load("a") == b"aaa"
    assert cache.load("b") == b"bbb"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.o", "b.o"]


def test_store_leaves_no_staging_file_behind(tmp_path):
    """A completed store renames its staged temp into place rather than leaving it."""
    cache = PersistentKvCache.at(str(tmp_path), suffix=".o")
    cache.store("k", b"payload")
    assert [p.name for p in tmp_path.iterdir()] == ["k.o"]


def test_in_memory_cache_round_trips_without_a_directory():
    """A memory-only cache stores and loads within the process and reports no location."""
    cache = PersistentKvCache.in_memory()
    cache.store("k", b"v")
    assert cache.load("k") == b"v"
    assert cache.location() is None


def test_a_read_key_is_served_from_memory_after_its_object_is_removed(tmp_path):
    """A load populates the memory tier, so a repeated key is answered without re-reading the directory."""
    PersistentKvCache.at(str(tmp_path), suffix=".o").store("k", b"v")

    reader = PersistentKvCache.at(str(tmp_path), suffix=".o")
    assert reader.load("k") == b"v"
    (tmp_path / "k.o").unlink()
    assert reader.load("k") == b"v"


def test_a_store_writes_both_tiers_and_a_local_miss_falls_through_to_the_object_store(tmp_path):
    """The node-local tier is written inline and the object-store tier in the background."""
    local, remote = tmp_path / "local", tmp_path / "remote"
    cache = PersistentKvCache(local=lambda: str(local), remote=lambda: str(remote), suffix=".o")

    cache.store("k", b"v")
    flush_background_writes()
    assert (local / "k.o").read_bytes() == b"v"
    assert (remote / "k.o").read_bytes() == b"v"

    # A process that shares only the object store still serves the key.
    object_store_only = PersistentKvCache(remote=lambda: str(remote), suffix=".o")
    assert object_store_only.load("k") == b"v"


def test_the_directory_resolves_lazily_not_at_construction():
    """Constructing a cache does not call its resolver; the first access does."""
    calls: list[int] = []

    def _resolve() -> str:
        calls.append(1)
        return "/unused"

    PersistentKvCache(local=_resolve)
    assert calls == []


def test_marin_kv_cache_stacks_a_node_local_tier_over_the_object_store(tmp_path, monkeypatch):
    local, remote = tmp_path / "local", tmp_path / "remote"
    monkeypatch.setattr(cache_module, "marin_local_cache", lambda: str(local))
    monkeypatch.setattr(cache_module, "marin_temp_bucket", lambda ttl_days, prefix: str(remote / prefix))

    cache = marin_kv_cache("cutlass-kernels", suffix=".o")
    cache.store("k", b"v")
    flush_background_writes()

    assert (local / "cutlass-kernels" / "k.o").read_bytes() == b"v"
    assert (remote / "cutlass-kernels" / "k.o").read_bytes() == b"v"
    assert cache.load("k") == b"v"


def test_marin_kv_cache_omits_the_node_local_tier_off_cluster(tmp_path, monkeypatch):
    """With no node-local mount, the cache keeps to memory and the object store."""
    remote = tmp_path / "remote"
    monkeypatch.setattr(cache_module, "marin_local_cache", lambda: None)
    monkeypatch.setattr(cache_module, "marin_temp_bucket", lambda ttl_days, prefix: str(remote / prefix))

    cache = marin_kv_cache("p", suffix=".o")
    cache.store("k", b"v")
    flush_background_writes()

    assert (remote / "p" / "k.o").read_bytes() == b"v"
    assert cache.load("k") == b"v"


def test_sync_kv_cache_stages_remote_files_down_and_mirrors_new_ones_up(tmp_path):
    """A synced directory starts warm from the object store and mirrors new files back."""
    remote, local = tmp_path / "remote", tmp_path / "local"
    remote.mkdir()
    (remote / "a.txt").write_bytes(b"one")

    handle = sync_kv_cache(remote=lambda: str(remote), local=str(local))
    assert (local / "a.txt").read_bytes() == b"one"

    # Tamper with the staged-down file to prove it is not re-uploaded, and write a
    # new one the way the consumer (XLA) would.
    (remote / "a.txt").write_bytes(b"tampered")
    (local / "b.txt").write_bytes(b"two")
    handle.close()

    assert (remote / "b.txt").read_bytes() == b"two"
    assert (remote / "a.txt").read_bytes() == b"tampered"


def test_sync_kv_cache_mirrors_nested_files_up_when_the_remote_starts_empty(tmp_path):
    """An absent object-store directory is tolerated; nested local files still mirror up."""
    remote, local = tmp_path / "remote", tmp_path / "local"

    handle = sync_kv_cache(remote=lambda: str(remote), local=str(local))
    (local / "sub").mkdir(parents=True)
    (local / "sub" / "k").write_bytes(b"v")
    handle.close()

    assert (remote / "sub" / "k").read_bytes() == b"v"
