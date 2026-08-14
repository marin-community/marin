# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import finestore.cache as cache_module
from finestore.admin import drop_table
from finestore.cache import PersistentKvCache
from finestore.layout import BLOBS_TABLE
from finestore.reader import ReadView

_CACHE_PROCESS = textwrap.dedent(
    """
    import atexit
    import sys
    import threading

    import finestore.cache as cache_module
    from finestore.cache import PersistentKvCache
    from finestore.store import DataStore
    from rigging.filesystem import StoragePath

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


def test_cache_normal_process_exit_drains_pending_remote_commit(tmp_path):
    root = str(tmp_path / "cache")

    subprocess.run([sys.executable, "-c", _CACHE_PROCESS, root, "slow"], check=True, timeout=10)

    assert ReadView(root).read_blob("kernel") == b"object-code"


def test_cache_process_exit_abandons_stalled_remote_commit(tmp_path):
    root = str(tmp_path / "cache")

    result = subprocess.run([sys.executable, "-c", _CACHE_PROCESS, root, "stall"], timeout=10)

    assert result.returncode == 0
