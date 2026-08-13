# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import finestore.cache as cache_module
from finestore.admin import drop_table
from finestore.cache import PersistentKvCache
from finestore.layout import BLOBS_TABLE


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
    drop_table(root, BLOBS_TABLE)
    assert reader.load("kernel") == b"object-code"


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
