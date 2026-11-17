# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NCCL transport probe's parsing and variant-matrix contract."""

from experiments.post_training.nccl_transport_probe import (
    NcclCensus,
    build_variants,
    parse_nccl_census,
)


def test_variant_matrix_isolates_each_workaround_variable():
    """The matrix must isolate each workaround variable and pin the exact prod overlay."""
    variants = build_variants()
    names = [variant.name for variant in variants]
    assert len(names) == len(set(names)), "variant names must be unique"

    for key, value in (("NCCL_NET", "Socket"), ("NCCL_IB_DISABLE", "1"), ("NCCL_SOCKET_FAMILY", "AF_INET")):
        alone = [variant for variant in variants if variant.env == {key: value}]
        assert alone, f"no single-variable variant for {key}"

    all_three = next(variant for variant in variants if variant.name == "all_three")
    prod_exact = next(variant for variant in variants if variant.name == "prod_exact")
    assert prod_exact.env == {**all_three.env, "NCCL_SOCKET_IFNAME": "^ibs,ibp,lo,docker,veth,cilium,lxc"}

    for variant in variants:
        assert all(key.startswith("NCCL_") for key in variant.env), variant.name


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


def test_empty_census_is_representable():
    assert parse_nccl_census([]) == NcclCensus(nccl_version="", transports=[])
