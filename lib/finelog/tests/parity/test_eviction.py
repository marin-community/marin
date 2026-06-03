# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-4 eviction parity tests (over RPC + the --debug-admin surface).

Eviction trims a namespace's oldest L>=1 BOTH (locally + remotely durable)
segments once it exceeds its count/byte cap. The invariants pinned here, on BOTH
the Python and Rust backends:

- the OLDEST (smallest min_seq) eligible segment is evicted first;
- an evicted segment flips to REMOTE (local file gone, bucket copy preserved),
  and ListNamespaces stats stop counting it;
- a segment that has NOT been uploaded yet (still LOCAL, no remote copy) is NEVER
  evicted — eviction must not destroy data with no durable archive.

The seam is the (RPC + /debug/maintain + /debug/segments) socket; these never
import store internals.
"""

from __future__ import annotations

import pytest

from tests.parity.conftest import (
    RemoteServer,
    maintain,
    namespace_info,
    segments,
    stats_client,
    stats_pb2,
    worker_schema,
    write_worker_row,
)

pytestmark = pytest.mark.timeout(60)

_NS = "iris.worker"


def _register(url: str, *, max_segments: int | None = None) -> None:
    client = stats_client(url)
    policy = stats_pb2.StoragePolicy()
    if max_segments is not None:
        policy.max_segments = max_segments
    client.register_table(stats_pb2.RegisterTableRequest(namespace=_NS, schema=worker_schema(), storage_policy=policy))


def test_eviction_drops_oldest_both_segment_when_cap_exceeded(finelog_url_remote: RemoteServer) -> None:
    url = finelog_url_remote.base_url
    _register(url, max_segments=1)
    client = stats_client(url)

    # Cycle 1: write one row, force-compact to L1, sync uploads it (BOTH).
    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    after_first = segments(url, _NS)
    # The single L1 segment is now durable in the bucket.
    assert len(after_first) == 1
    first_min_seq = after_first[0].min_seq
    assert after_first[0].location == "BOTH"

    # Cycle 2: a second segment; cap=1 evicts the oldest BOTH.
    write_worker_row(client, _NS, "w-2", 200, 20)
    maintain(url, _NS, force_compact_l0=True)

    after_second = segments(url, _NS)
    # The catalog still has both rows, but the oldest is now REMOTE (evicted),
    # and exactly one local/BOTH segment remains.
    locations = {s.min_seq: s.location for s in after_second}
    assert locations[first_min_seq] == "REMOTE", "oldest segment evicted to REMOTE"
    queryable = [s for s in after_second if s.location != "REMOTE"]
    assert len(queryable) == 1, "one segment remains queryable under cap=1"
    assert queryable[0].min_seq > first_min_seq, "FIFO-from-front: newest kept"

    # The evicted segment's remote archive still exists.
    assert len(finelog_url_remote.remote_files(_NS)) == 2

    # ListNamespaces stats exclude the REMOTE-only segment.
    info = namespace_info(client, _NS)
    assert info is not None
    assert info.segment_count == 1
    assert info.row_count == 1


def test_eviction_skips_segments_not_yet_uploaded(finelog_url: str) -> None:
    # NO remote dir configured: segments stay LOCAL, nothing is BOTH, so the
    # cap=1 eviction must be a no-op (LOCAL-only data is never destroyed).
    _register(finelog_url, max_segments=1)
    client = stats_client(finelog_url)

    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(finelog_url, _NS, force_compact_l0=True)
    write_worker_row(client, _NS, "w-2", 200, 20)
    maintain(finelog_url, _NS, force_compact_l0=True)

    after = segments(finelog_url, _NS)
    # Both L1 segments survive — over the cap but nothing is evictable.
    l1 = [s for s in after if s.level == 1]
    assert len(l1) == 2, "LOCAL-only segments are never evicted"
    assert all(s.location == "LOCAL" for s in l1)


def test_eviction_preserves_remote_archive_across_ticks(finelog_url_remote: RemoteServer) -> None:
    url = finelog_url_remote.base_url
    _register(url, max_segments=1)
    client = stats_client(url)

    # Build two BOTH segments under cap=1 so one is evicted to REMOTE.
    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    write_worker_row(client, _NS, "w-2", 200, 20)
    maintain(url, _NS, force_compact_l0=True)

    evicted = [s for s in segments(url, _NS) if s.location == "REMOTE"]
    assert len(evicted) == 1
    remote_before = set(finelog_url_remote.remote_files(_NS))
    assert len(remote_before) == 2

    # Several more maintain ticks must NOT orphan-delete the REMOTE archive
    # (its catalog row keeps it from being treated as an orphan).
    for _ in range(3):
        maintain(url, _NS)
    remote_after = set(finelog_url_remote.remote_files(_NS))
    assert remote_after == remote_before, "REMOTE archive survives sync ticks"
    # Still exactly one REMOTE row, one queryable segment.
    after = segments(url, _NS)
    assert sum(1 for s in after if s.location == "REMOTE") == 1
    assert sum(1 for s in after if s.location != "REMOTE") == 1
