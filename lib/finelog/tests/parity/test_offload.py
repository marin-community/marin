# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-4 remote-sync + boot-reconcile parity tests (RPC + --debug-admin).

The sync step is the LOCAL -> BOTH -> REMOTE state machine's first hop:
- a compacted L>=1 LOCAL segment is uploaded and flipped to BOTH;
- a remote file with NO catalog row (a compaction input whose row was dropped) is
  orphan-deleted, but ONLY after the replacement is durable — the two-phase
  ordering that is the data-safety invariant.

Boot reconcile is the wiped-catalog recovery path: the bucket is the only durable
record of L>=1 segments after the local catalog + parquet are lost; restarting on
the same dirs re-adopts the remote files as REMOTE rows (never deletes them) and
drops a stale lower-level file that a surviving higher-level file fully covers.

These run identically on Python and Rust. The harness's RemoteServer /
RestartableServer expose the on-disk bucket so the tests observe uploads/deletes
directly.
"""

from __future__ import annotations

import pytest

from tests.parity.conftest import (
    RemoteServer,
    RestartableServer,
    maintain,
    segments,
    stats_client,
    stats_pb2,
    worker_schema,
    write_worker_row,
)

pytestmark = pytest.mark.timeout(90)

_NS = "iris.worker"


def _register(url: str) -> None:
    stats_client(url).register_table(stats_pb2.RegisterTableRequest(namespace=_NS, schema=worker_schema()))


def test_sync_uploads_compacted_segment(finelog_url_remote: RemoteServer) -> None:
    url = finelog_url_remote.base_url
    _register(url)
    client = stats_client(url)
    write_worker_row(client, _NS, "w-1", 100, 10)

    # No remote files until maintain runs the sync step.
    assert finelog_url_remote.remote_files(_NS) == []

    maintain(url, _NS, force_compact_l0=True)

    # The compacted L1 segment is uploaded and the catalog row flips to BOTH.
    files = finelog_url_remote.remote_files(_NS)
    assert len(files) == 1, "compacted L1 segment uploaded to the bucket"
    after = segments(url, _NS)
    assert len(after) == 1
    assert after[0].level == 1
    assert after[0].location == "BOTH"


def test_sync_deletes_orphan_remote_after_replacement_durable(finelog_url_remote: RemoteServer) -> None:
    url = finelog_url_remote.base_url
    _register(url)
    client = stats_client(url)

    # Land a real durable (BOTH) segment first.
    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    real = finelog_url_remote.remote_files(_NS)
    assert len(real) == 1

    # Drop an orphan parquet into the bucket with no catalog row — models a
    # compaction input left behind after promotion to a higher tier.
    orphan = finelog_url_remote.remote_dir / _NS / "seg_L1_0000000000000099999.parquet"
    orphan.write_bytes(b"orphan-bytes")
    assert orphan.name in finelog_url_remote.remote_files(_NS)

    # The next sync deletes the orphan (no catalog row) while preserving the real
    # BOTH archive (its row keeps it). all_durable holds (no failed uploads), so
    # phase 2 runs.
    maintain(url, _NS)

    remaining = finelog_url_remote.remote_files(_NS)
    assert orphan.name not in remaining, "orphan with no catalog row deleted"
    assert set(real).issubset(set(remaining)), "durable archive preserved"


def test_wiped_catalog_recovers_from_remote_at_boot(restartable_remote_server: RestartableServer) -> None:
    server = restartable_remote_server
    url = server.start()
    _register(url)
    client = stats_client(url)

    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    uploaded = server.remote_files(_NS)
    assert len(uploaded) == 1
    server.stop()

    # Wipe the local catalog + parquet; keep the remote bucket.
    for sidecar in server.log_dir_sidecars():
        sidecar.unlink()
    ns_dir = server.log_dir / _NS
    if ns_dir.is_dir():
        for p in ns_dir.glob("*.parquet"):
            p.unlink()

    # Reboot on the same dirs: boot reconcile adopts the remote file as a REMOTE
    # row and does NOT delete it. The namespace must be re-registered (deploy's
    # startup RegisterTable re-establishes the schema, matching production).
    url2 = server.start()
    _register(url2)
    after = segments(url2, _NS)
    assert len(after) == 1, "remote file adopted as a catalog row"
    assert after[0].location == "REMOTE"
    assert after[0].level == 1
    # The bucket file survives adoption.
    assert server.remote_files(_NS) == uploaded


def test_reconcile_drops_stale_input_covered_by_higher_level(restartable_remote_server: RestartableServer) -> None:
    server = restartable_remote_server
    url = server.start()
    _register(url)
    client = stats_client(url)

    write_worker_row(client, _NS, "w-1", 100, 10)
    maintain(url, _NS, force_compact_l0=True)
    real = server.remote_files(_NS)
    assert len(real) == 1
    real_basename = real[0]
    server.stop()

    # Plant a stale lower-level remote file whose [min_seq, max_seq] is fully
    # covered by the surviving L1 segment (which spans seq 1..1). An L0 file at
    # the same min_seq is a classic compaction-mid-crash leftover.
    stale = server.remote_dir / _NS / "seg_L0_0000000000000000001.parquet"
    # Copy the real L1 bytes so it is a readable parquet footer (same seq span).
    stale.write_bytes((server.remote_dir / _NS / real_basename).read_bytes())
    assert stale.name in server.remote_files(_NS)

    # Wipe local catalog + parquet; reboot. Boot reconcile must drop the stale L0
    # (covered by the L1) from both the bucket and the catalog, and adopt the L1.
    for sidecar in server.log_dir_sidecars():
        sidecar.unlink()
    ns_dir = server.log_dir / _NS
    if ns_dir.is_dir():
        for p in ns_dir.glob("*.parquet"):
            p.unlink()

    url2 = server.start()
    _register(url2)
    remaining = server.remote_files(_NS)
    assert stale.name not in remaining, "stale covered L0 dropped from bucket"
    assert real_basename in remaining, "uncovered L1 preserved"
    after = segments(url2, _NS)
    levels = sorted(s.level for s in after)
    assert levels == [1], "only the surviving L1 adopted; stale L0 redundancy-dropped"
