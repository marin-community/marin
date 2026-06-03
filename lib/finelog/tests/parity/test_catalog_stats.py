# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-4 catalog-stats lifecycle parity tests (RPC + --debug-admin).

ListNamespaces NamespaceInfo.row_count / segment_count is the externally-observed
roll-up over a namespace's LOCAL/BOTH segments + RAM (REMOTE-only excluded). This
pins, on BOTH backends:

- compaction collapses N L0 segments into one L1 segment ATOMICALLY — stats never
  observe a half-spliced set, the post-compaction Query returns every row in
  (key, seq) order, and segment_count drops from N to 1 with row_count unchanged;
- eviction of a BOTH segment drops it from the stats roll-up (segment_count and
  row_count fall), even though the durable remote copy survives.
"""

from __future__ import annotations

import pytest

from tests.parity.conftest import (
    RemoteServer,
    maintain,
    namespace_info,
    query_table,
    segments,
    stats_client,
    stats_pb2,
    worker_schema,
    write_worker_row,
)

pytestmark = pytest.mark.timeout(60)

_NS = "iris.worker"


def test_compaction_replaces_l0_rows_atomically(finelog_url: str) -> None:
    client = stats_client(finelog_url)
    client.register_table(stats_pb2.RegisterTableRequest(namespace=_NS, schema=worker_schema()))

    write_worker_row(client, _NS, "w-1", 100, 30)
    write_worker_row(client, _NS, "w-2", 200, 10)
    write_worker_row(client, _NS, "w-3", 300, 20)

    before = namespace_info(client, _NS)
    assert before is not None
    assert before.row_count == 3
    assert before.segment_count == 3

    maintain(finelog_url, _NS, force_compact_l0=True)

    # One L1 segment, all rows preserved, seq window spans all three.
    after_segs = segments(finelog_url, _NS)
    assert len(after_segs) == 1
    seg = after_segs[0]
    assert seg.level == 1
    assert seg.row_count == 3
    assert seg.min_seq == 1
    assert seg.max_seq == 3

    after = namespace_info(client, _NS)
    assert after.row_count == 3, "row_count unchanged across the swap"
    assert after.segment_count == 1, "N L0 segments collapsed to one L1"

    # The rows survive the swap and remain queryable in (key, seq) order. The
    # merge sorts by (timestamp_ms key default, seq) — order by seq recovers
    # write order.
    table = query_table(client, 'SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY seq')
    assert table.column("worker_id").to_pylist() == ["w-1", "w-2", "w-3"]
    assert table.column("mem_bytes").to_pylist() == [100, 200, 300]


def test_eviction_flips_remote_drops_from_stats(finelog_url_remote: RemoteServer) -> None:
    url = finelog_url_remote.base_url
    client = stats_client(url)
    client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=_NS,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_segments=1),
        )
    )

    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    one = namespace_info(client, _NS)
    assert one.row_count == 1
    assert one.segment_count == 1

    write_worker_row(client, _NS, "w-2", 200, 20)
    maintain(url, _NS, force_compact_l0=True)

    # cap=1 evicted the oldest BOTH -> REMOTE: stats now count only the survivor.
    after = namespace_info(client, _NS)
    assert after.segment_count == 1, "REMOTE-only segment excluded from segment_count"
    assert after.row_count == 1, "REMOTE-only segment excluded from row_count"
    # The catalog still holds the REMOTE row (durable archive pointer).
    rows = segments(url, _NS)
    assert sum(1 for s in rows if s.location == "REMOTE") == 1
    assert len(finelog_url_remote.remote_files(_NS)) == 2
