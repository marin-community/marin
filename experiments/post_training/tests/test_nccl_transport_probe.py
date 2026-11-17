# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NCCL transport probe's log parsing and gate verdict."""

from experiments.post_training.nccl_transport_probe import (
    NcclCensus,
    gate_verdict,
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
