# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the distributed-locked model snapshot cache."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from levanter.model_cache import DEFAULT_COMPLETE_MARKER, cache_to_prefix


def _make_download(call_count: list[int], lock: threading.Lock, *, delay: float = 0.0):
    """Return a download callback that records invocations and writes a payload."""

    def download(local_dir: str) -> None:
        with lock:
            call_count.append(1)
        if delay:
            time.sleep(delay)
        Path(local_dir, "weights.bin").write_text("payload")

    return download


def test_concurrent_callers_download_once(tmp_path):
    """Under N concurrent callers only one download runs; the rest get the cached path."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    # A non-trivial download window forces losers to actually block on the lock
    # instead of all winning the marker fast-path on the first check.
    download = _make_download(calls, count_lock, delay=0.3)

    def run() -> str:
        return cache_to_prefix(cache_path, download, poll_interval=0.02)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(), range(8)))

    assert len(calls) == 1, f"expected a single download, got {len(calls)}"
    assert Path(cache_path, DEFAULT_COMPLETE_MARKER).exists()
    # Every caller returns a path containing the populated snapshot.
    for path in results:
        assert Path(path, "weights.bin").read_text() == "payload"


def test_cache_hit_skips_download(tmp_path):
    """A second call after the marker exists returns the cache path without downloading."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    download = _make_download(calls, count_lock)

    first = cache_to_prefix(cache_path, download)
    second = cache_to_prefix(cache_path, download)

    assert len(calls) == 1
    assert second == cache_path.rstrip("/")
    assert Path(first, "weights.bin").read_text() == "payload"


def test_custom_complete_marker(tmp_path):
    """The completion marker name is configurable and gates the cache-hit fast path."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    download = _make_download(calls, count_lock)

    cache_to_prefix(cache_path, download, complete_marker=".done")
    assert Path(cache_path, ".done").exists()

    cache_to_prefix(cache_path, download, complete_marker=".done")
    assert len(calls) == 1
