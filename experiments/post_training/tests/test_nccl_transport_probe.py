# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NCCL transport probe's pure parsing and matrix logic."""

from experiments.post_training.nccl_transport_probe import (
    build_variants,
    error_tail,
    parse_nccl_transport,
)


def test_build_variants_covers_workaround_and_hints():
    variants = build_variants()
    names = [variant.name for variant in variants]
    assert "baseline_default_env" in names
    assert "all_three" in names
    assert "prod_exact" in names
    assert {"ib_hca_mlx5_0", "ib_hca_mlx5_1", "ib_gid_index_0"} <= set(names)

    by_name = {variant.name: variant.env for variant in variants}
    assert by_name["net_socket_only"] == {"NCCL_NET": "Socket"}
    assert by_name["ib_disable_only"] == {"NCCL_IB_DISABLE": "1"}
    assert by_name["af_inet_only"] == {"NCCL_SOCKET_FAMILY": "AF_INET"}
    assert by_name["all_three"] == {
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_FAMILY": "AF_INET",
    }
    assert by_name["baseline_default_env"] == {}


def test_parse_nccl_transport_distinguishes_ib_and_socket():
    ib_log = [
        "g853134:13596:14951 [0] NCCL INFO Using network InfiniBand",
        "g853134:13596:14951 [0] NCCL INFO Channel 00 : 0[0] -> 1[0] via NET/IB/0/GDRDMA",
        "g853134:13596:14951 [0] NCCL INFO Channel 01 : 1[0] -> 0[0] via NET/IB/0/GDRDMA",
        "g853134:13596:14951 [0] NCCL INFO NCCL version 2.29.7+cuda13.2",
    ]
    parsed = parse_nccl_transport(ib_log)
    assert parsed["nccl_version"] == "2.29.7"
    assert parsed["transports"] == ["NET/IB/0/GDRDMA"]

    socket_log = [
        "g95c40:13657:14951 [0] NCCL INFO Channel 00 : 0[0] -> 2[0] via NET/Socket/0/0",
        "g95c40:13657:14951 [0] NCCL INFO NCCL version 2.29.7",
    ]
    parsed = parse_nccl_transport(socket_log)
    assert parsed["nccl_version"] == "2.29.7"
    assert parsed["transports"] == ["NET/Socket/0/0"]

    # Intra-node NVLink/P2P paths are not the inter-node transport.
    p2p_log = ["NCCL INFO Channel 00 : 0[0] -> 1[0] via P2P/IPC"]
    assert parse_nccl_transport(p2p_log)["transports"] == []


def test_error_tail_keeps_last_lines():
    lines = [f"line {index}" for index in range(30)]
    assert error_tail(lines) == lines[-12:]

