# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NCCL transport probe's log parsing and gate verdict."""

import hashlib

from experiments.post_training.nccl_transport_probe import (
    NcclCensus,
    gate_verdict,
    mapped_nccl_libraries,
    parse_nccl_census,
)


def test_parse_nccl_census_distinguishes_ib_and_socket():
    ib_log = [
        "g853134:13596:14951 [0] NCCL INFO Using network InfiniBand",
        "g853134:13596:14951 [0] NCCL INFO Channel 00 : 0[0] -> 1[0] via NET/IB/0/GDRDMA",
        "g853134:13596:14951 [0] NCCL INFO Channel 01 : 1[0] -> 0[0] via NET/IB/0/GDRDMA",
        "g853134:13596:14951 [0] NCCL INFO NCCL version 2.29.7+cuda13.2",
    ]
    census = parse_nccl_census(ib_log)
    assert census == NcclCensus(nccl_version="2.29.7", transports=["NET/IB/0/GDRDMA"])

    socket_log = [
        "g95c40:13657:14951 [0] NCCL INFO Channel 00 : 0[0] -> 2[0] via NET/Socket/0/0",
        "g95c40:13657:14951 [0] NCCL INFO NCCL version 2.29.7",
    ]
    assert parse_nccl_census(socket_log) == NcclCensus(nccl_version="2.29.7", transports=["NET/Socket/0/0"])


def test_parse_nccl_census_ignores_intra_node_p2p():
    """Intra-node NVLink paths are not the inter-node transport."""
    p2p_log = [
        "NCCL INFO Channel 00 : 0[0] -> 1[0] via P2P/IPC",
        "NCCL INFO NCCL version 2.29.7",
    ]
    assert parse_nccl_census(p2p_log) == NcclCensus(nccl_version="2.29.7", transports=[])


def test_gate_verdict_fails_closed_on_sockets_and_failures():
    """The gate passes only NET/IB or an empty (intra-node NVLink) census."""
    assert gate_verdict(True, [["NET/IB/0/GDRDMA"]]) == "pass"
    assert gate_verdict(True, [[]]) == "pass"
    assert gate_verdict(True, [[], ["NET/IB/0/GDRDMA"]]) == "pass"
    assert gate_verdict(True, [["NET/Socket/0"]]) == "socket_transport"
    assert gate_verdict(True, [[], ["NET/Socket/0/Shared"]]) == "socket_transport"
    assert gate_verdict(False, [["NET/IB/0/GDRDMA"]]) == "collective_failed"


def test_mapped_nccl_libraries_parses_proc_maps(tmp_path):
    """The maps scanner finds libnccl entries and hashes the real files."""
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"nccl-fixture-bytes")
    maps = tmp_path / "maps"
    maps.write_text(
        "7f0e00000000-7f0e10000000 r-xp 00000000 08:02 1234   "
        f"{library}\n"
        "7f0e20000000-7f0e21000000 r-xp 00000000 08:02 5678   "
        f"{tmp_path}/libother.so\n"
        "7f0e30000000-7f0e31000000 r--p 00000000 00:00 0      [vvar]\n"
    )
    records = mapped_nccl_libraries(maps)
    assert [record["path"] for record in records] == [str(library)]
    assert records[0]["sha256"] == hashlib.sha256(b"nccl-fixture-bytes").hexdigest()


def test_mapped_nccl_libraries_without_maps_is_empty(tmp_path):
    assert mapped_nccl_libraries(tmp_path / "missing") == []
