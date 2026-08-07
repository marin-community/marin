# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import rigging.cache as cache_module
from rigging.cache import PersistentKvCache, marin_kv_cache


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


def test_the_directory_resolves_lazily_not_at_construction():
    """Constructing a cache does not call its resolver; the first access does."""
    calls: list[int] = []

    def _resolve() -> str:
        calls.append(1)
        return "/unused"

    PersistentKvCache(_resolve)
    assert calls == []


def test_marin_kv_cache_maps_a_prefix_onto_the_region_store(monkeypatch):
    monkeypatch.setattr(cache_module, "marin_prefix", lambda: "gs://my-region-bucket/")
    cache = marin_kv_cache("levanter_kernel_autotune/fused_cross_entropy_loss", suffix=".json")
    assert cache.location() == "gs://my-region-bucket/levanter_kernel_autotune/fused_cross_entropy_loss"
