# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

from iris.cluster.node_agent.cache_reclaim import reclaim_cache
from rigging.timing import Duration, Timestamp


def test_reclaim_cache_uses_file_writes_and_accesses_for_freshness(tmp_path):
    cache_dir = tmp_path / "iris-cache"
    cache_namespace = cache_dir / "cache"
    stale = cache_namespace / "old-model"
    recently_written = cache_namespace / "new-model"
    recently_accessed = cache_namespace / "used-model"
    stale.mkdir(parents=True)
    recently_written.mkdir()
    recently_accessed.mkdir()
    (stale / "weights").write_bytes(b"stale")
    (recently_written / "weights").write_bytes(b"written")
    (recently_accessed / "weights").write_bytes(b"accessed")
    os.utime(stale / "weights", (100.0, 100.0))
    os.utime(stale, (950.0, 100.0))
    os.utime(recently_written, (100.0, 100.0))
    os.utime(recently_written / "weights", (100.0, 950.0))
    os.utime(recently_accessed, (100.0, 100.0))
    os.utime(recently_accessed / "weights", (950.0, 100.0))

    reclaimed = reclaim_cache(
        cache_dir,
        max_age=Duration.from_seconds(500),
        now=Timestamp.from_seconds(1_000),
    )

    assert reclaimed == 1
    assert cache_namespace.is_dir()
    assert not stale.exists()
    assert (recently_written / "weights").read_bytes() == b"written"
    assert (recently_accessed / "weights").read_bytes() == b"accessed"
