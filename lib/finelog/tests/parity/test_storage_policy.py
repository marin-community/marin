# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-4 per-namespace StoragePolicy parity tests (RPC + --debug-admin).

A per-namespace StoragePolicy overrides the cluster-wide CompactionConfig caps:
- ``max_segments`` evicts a namespace down to its own cap even though the global
  cap (1000) is loose;
- ``max_age_seconds`` drops L>=1 BOTH segments older than ``now - max_age``,
  scanning by ``created_at_ms`` (a compaction output can have a low min_seq but a
  fresh birth time), backdated over RPC via ``/debug/backdate`` (no sleep);
- the wire round-trips the effective policy (RegisterTableResponse.effective_policy
  and ListNamespaces NamespaceInfo.storage_policy), and re-register replaces a
  non-empty policy whole-record while an empty policy preserves the existing one.

Identical on Python and Rust.
"""

from __future__ import annotations

import pytest

from tests.parity.conftest import (
    RemoteServer,
    backdate,
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


def test_register_response_round_trips_effective_policy(finelog_url: str) -> None:
    client = stats_client(finelog_url)
    resp = client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=_NS,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_segments=1, max_age_seconds=10),
        )
    )
    assert resp.effective_policy.max_segments == 1
    assert resp.effective_policy.max_age_seconds == 10
    assert resp.effective_policy.max_bytes == 0  # unset -> proto3 zero

    # ListNamespaces reflects the same policy.
    info = namespace_info(client, _NS)
    assert info is not None
    assert info.storage_policy.max_segments == 1
    assert info.storage_policy.max_age_seconds == 10

    # Re-register with an empty policy PRESERVES the existing one (old-client-safe).
    resp2 = client.register_table(stats_pb2.RegisterTableRequest(namespace=_NS, schema=worker_schema()))
    assert resp2.effective_policy.max_segments == 1
    assert resp2.effective_policy.max_age_seconds == 10

    # Re-register with a non-empty policy REPLACES whole-record.
    resp3 = client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=_NS,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_bytes=4096),
        )
    )
    assert resp3.effective_policy.max_bytes == 4096
    assert resp3.effective_policy.max_segments == 0
    assert resp3.effective_policy.max_age_seconds == 0


def test_per_namespace_max_segments_overrides_global_cap(finelog_url_remote: RemoteServer) -> None:
    url = finelog_url_remote.base_url
    client = stats_client(url)
    # Per-namespace cap=1; the global cap (max_segments_per_namespace=1000) is
    # loose, so only the per-namespace override can trim down to one segment.
    client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=_NS,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_segments=1),
        )
    )

    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    write_worker_row(client, _NS, "w-2", 200, 20)
    maintain(url, _NS, force_compact_l0=True)

    after = segments(url, _NS)
    queryable = [s for s in after if s.location != "REMOTE"]
    assert len(queryable) == 1, "per-namespace cap=1 evicts down to one segment"
    info = namespace_info(client, _NS)
    assert info is not None
    assert info.segment_count == 1


def test_age_eviction_drops_backdated_segment_by_created_at(finelog_url_remote: RemoteServer) -> None:
    url = finelog_url_remote.base_url
    client = stats_client(url)
    client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=_NS,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_age_seconds=3600),
        )
    )

    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    seg_rows = segments(url, _NS)
    assert len(seg_rows) == 1
    assert seg_rows[0].location == "BOTH"
    seg_name = seg_rows[0].path

    # Within the window: a fresh maintain keeps the segment.
    maintain(url, _NS)
    assert namespace_info(client, _NS).segment_count == 1

    # Backdate past the cutoff (created_at_ms = 1, far older than now - 3600s),
    # then maintain age-evicts it. No wall-clock sleep.
    backdate(url, _NS, seg_name, 1)
    maintain(url, _NS)

    after = segments(url, _NS)
    assert all(s.location == "REMOTE" for s in after), "aged-out segment flipped to REMOTE"
    info = namespace_info(client, _NS)
    assert info.segment_count == 0, "aged-out segment dropped from queryable stats"
    # Remote archive preserved.
    assert len(finelog_url_remote.remote_files(_NS)) == 1


def test_age_eviction_scans_by_created_at_not_min_seq(finelog_url_remote: RemoteServer) -> None:
    """The age scan orders by created_at_ms, so a HIGH-min_seq but OLD segment is
    evicted while a LOW-min_seq but RECENT one is kept — a min_seq scan would
    short-circuit and miss the older sibling."""
    url = finelog_url_remote.base_url
    client = stats_client(url)
    client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=_NS,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_age_seconds=3600),
        )
    )

    # Two BOTH segments. seg A has the lower min_seq, seg B the higher.
    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    write_worker_row(client, _NS, "w-2", 200, 20)
    maintain(url, _NS, force_compact_l0=True)

    rows = sorted(segments(url, _NS), key=lambda s: s.min_seq)
    assert len(rows) == 2
    low_min_seq, high_min_seq = rows[0], rows[1]

    # Backdate the HIGHER-min_seq segment past the cutoff, keep the lower recent.
    backdate(url, _NS, high_min_seq.path, 1)
    maintain(url, _NS)

    after = {s.min_seq: s.location for s in segments(url, _NS)}
    assert after[high_min_seq.min_seq] == "REMOTE", "old high-min_seq segment aged out"
    assert after[low_min_seq.min_seq] == "BOTH", "recent low-min_seq segment kept"
