# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the distributed-locked model snapshot cache."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec

import pytest

import levanter.model_cache as model_cache
from levanter.model_cache import (
    DEFAULT_COMPLETE_MARKER,
    cache_hf_model,
    cache_to_prefix,
    resolve_cached_model_path,
)


def _make_populate(call_count: list[int], lock: threading.Lock, *, delay: float = 0.0):
    """Return a populate callback that records invocations and streams a payload to the cache."""

    def populate(fs: fsspec.AbstractFileSystem, dest: str) -> None:
        with lock:
            call_count.append(1)
        if delay:
            time.sleep(delay)
        with fs.open(f"{dest}/weights.bin", "w") as handle:
            handle.write("payload")

    return populate


def test_concurrent_callers_populate_once(tmp_path):
    """Under N concurrent callers only one populate runs; all get the cached path."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    # A non-trivial populate window forces losers to actually block on the lock
    # instead of all winning the marker fast-path on the first check.
    populate = _make_populate(calls, count_lock, delay=0.3)

    def run() -> str:
        return cache_to_prefix(cache_path, populate, poll_interval=0.02)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(), range(8)))

    assert len(calls) == 1, f"expected a single populate, got {len(calls)}"
    assert Path(cache_path, DEFAULT_COMPLETE_MARKER).exists()
    # Every caller returns the cache path holding the populated snapshot.
    for path in results:
        assert path == cache_path.rstrip("/")
        assert Path(path, "weights.bin").read_text() == "payload"


def test_cache_hit_skips_populate(tmp_path):
    """A second call after the marker exists returns the cache path without re-populating."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    populate = _make_populate(calls, count_lock)

    first = cache_to_prefix(cache_path, populate)
    second = cache_to_prefix(cache_path, populate)

    assert len(calls) == 1
    assert first == second == cache_path.rstrip("/")
    assert Path(first, "weights.bin").read_text() == "payload"


def test_custom_complete_marker(tmp_path):
    """The completion marker name is configurable and gates the cache-hit fast path."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    populate = _make_populate(calls, count_lock)

    cache_to_prefix(cache_path, populate, complete_marker=".done")
    assert Path(cache_path, ".done").exists()

    cache_to_prefix(cache_path, populate, complete_marker=".done")
    assert len(calls) == 1


def test_cache_hf_model_streams_one_file_at_a_time(tmp_path, monkeypatch):
    """cache_hf_model mirrors every repo file and keeps only one on local disk at a time."""
    repo_files = ["config.json", "model.safetensors", "tokenizer.json"]
    max_local_files = 0

    def fake_list_repo_files(model_id, revision=None):
        assert model_id == "org/model"
        return repo_files

    def fake_hf_hub_download(model_id, filename, revision=None, local_dir=None):
        nonlocal max_local_files
        local_path = Path(local_dir, filename)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(f"contents of {filename}")
        # The streamer must delete each file before fetching the next: peak local
        # footprint is one file, not the whole repo.
        present = [p for p in Path(local_dir).rglob("*") if p.is_file()]
        max_local_files = max(max_local_files, len(present))
        return str(local_path)

    monkeypatch.setattr(model_cache, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(model_cache, "hf_hub_download", fake_hf_hub_download)

    cache_path = str(tmp_path / "cache" / "model")
    result = cache_hf_model(cache_path, "org/model")

    assert result == cache_path
    assert max_local_files == 1, f"expected at most one staged file, saw {max_local_files}"
    for filename in repo_files:
        assert Path(cache_path, filename).read_text() == f"contents of {filename}"
    assert Path(cache_path, DEFAULT_COMPLETE_MARKER).exists()


@pytest.mark.parametrize(
    "path",
    [
        "gs://bucket/snapshot",  # object store
        "s3://bucket/snapshot",  # object store
        "hf://org/model",  # explicit fsspec HF URL
        "/local/checkpoint/dir",  # absolute local path
        "./relative/dir",  # relative local path
    ],
)
def test_resolve_loads_non_repo_paths_in_place(path, monkeypatch):
    """A path that already names a snapshot is returned unchanged, never mirrored.

    Mirroring it would feed the path to ``cache_hf_model`` as if it were a repo id and fail.
    """
    monkeypatch.setattr(model_cache, "cache_hf_model", lambda *a, **k: pytest.fail("must not mirror"))
    assert resolve_cached_model_path(path, cache_ttl_days=30, cache_prefix="models") == path


def test_resolve_disabled_ttl_skips_mirror(monkeypatch):
    monkeypatch.setattr(model_cache, "cache_hf_model", lambda *a, **k: pytest.fail("must not mirror"))
    assert resolve_cached_model_path("org/model", cache_ttl_days=0, cache_prefix="models") == "org/model"


def test_resolve_keeps_distinct_refs_in_distinct_cache_dirs(monkeypatch):
    """Two refs that a lossy slug would collide (``org/model_a`` vs ``org/model@a``) must mirror
    to different cache dirs, so a hit on one never loads the other's snapshot."""
    mirrored: dict[str, tuple[str, str | None]] = {}

    monkeypatch.setattr(model_cache, "marin_temp_bucket", lambda ttl_days, prefix: f"gs://temp/{prefix}")

    def fake_cache(cache_path, repo, *, revision=None, complete_marker):
        mirrored[cache_path] = (repo, revision)
        return cache_path

    monkeypatch.setattr(model_cache, "cache_hf_model", fake_cache)

    a = resolve_cached_model_path("org/model_a", cache_ttl_days=7, cache_prefix="models")
    b = resolve_cached_model_path("org/model@a", cache_ttl_days=7, cache_prefix="models")

    assert a != b
    assert mirrored[a] == ("org/model_a", None)
    assert mirrored[b] == ("org/model", "a")
