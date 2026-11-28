# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import socket
from multiprocessing.connection import Connection

import pytest
from rigging.port_lease import (
    DYNAMIC_TCP_PORTS,
    allocate_tcp_port,
    lease_tcp_port,
    lease_tcp_port_block,
    lease_tcp_ports,
    try_claim_free_tcp_port,
)


def _hold_process_port(pipe: Connection) -> None:
    pipe.send(allocate_tcp_port())
    pipe.recv()


@pytest.mark.parametrize("family, host", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")])
def test_claim_skips_occupied_tcp_port(family, host):
    if family == socket.AF_INET6 and not socket.has_ipv6:
        pytest.skip("IPv6 support is required")
    with socket.socket(family, socket.SOCK_STREAM) as occupied:
        occupied.bind((host, 0))
        port = occupied.getsockname()[1]
        assert try_claim_free_tcp_port(port) is None
    release = try_claim_free_tcp_port(port)
    assert release is not None
    release()


def test_named_port_honors_an_existing_claim(monkeypatch):
    with lease_tcp_port() as allocated:
        monkeypatch.setenv("IRIS_PORT_WEB", str(allocated))
        assert allocate_tcp_port(8080, name="web") == allocated

        monkeypatch.setenv("IRIS_PORT_WEB", "0")
        port = allocate_tcp_port(name="web")
        assert port in DYNAMIC_TCP_PORTS
        assert port != allocated


def test_allocated_port_stays_claimed_until_process_exit():
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe()
    child = context.Process(target=_hold_process_port, args=(sender,))
    child.start()
    sender.close()
    try:
        assert receiver.poll(10)
        port = receiver.recv()
        assert try_claim_free_tcp_port(port) is None

        child.kill()
        child.join(5)
        release = try_claim_free_tcp_port(port)
        assert release is not None
        release()
    finally:
        if child.is_alive():
            child.kill()
        child.join(10)
        receiver.close()


def test_contiguous_block_claims_every_port():
    with lease_tcp_port_block(3) as base:
        for port in range(base, base + 3):
            assert try_claim_free_tcp_port(port) is None
    for port in range(base, base + 3):
        release = try_claim_free_tcp_port(port)
        assert release is not None
        release()


def test_dynamic_leases_stay_in_the_shared_pool_and_honor_exclusions():
    with lease_tcp_ports(6, excluded=range(20000, 23000)) as ports:
        assert len(set(ports)) == 6
        assert all(23000 <= port < 24000 for port in ports)
