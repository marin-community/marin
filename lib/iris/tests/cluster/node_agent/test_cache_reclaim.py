# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

from iris.cluster.node_agent.cache_reclaim import reclaim_cache
from rigging.timing import Duration, Timestamp


def test_reclaim_cache_removes_stale_entries_and_preserves_fresh_entries(tmp_path):
    cache_dir = tmp_path / "iris-cache"
    cache_namespace = cache_dir / "cache"
    stale = cache_namespace / "old-model"
    fresh = cache_namespace / "current-model"
    stale.mkdir(parents=True)
    fresh.mkdir()
    (stale / "weights").write_bytes(b"stale")
    (fresh / "weights").write_bytes(b"fresh")
    os.utime(stale / "weights", (100.0, 100.0))
    os.utime(stale, (100.0, 100.0))
    os.utime(fresh, (100.0, 100.0))
    os.utime(fresh / "weights", (950.0, 950.0))

    reclaimed = reclaim_cache(
        cache_dir,
        max_age=Duration.from_seconds(500),
        now=Timestamp.from_seconds(1_000),
    )

    assert reclaimed == 1
    assert cache_namespace.is_dir()
    assert not stale.exists()
    assert (fresh / "weights").read_bytes() == b"fresh"
