# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import socket
from multiprocessing.connection import Connection

import pytest
from rigging.port_lease import lease_tcp_port, lease_tcp_port_block, lease_tcp_ports


def _hold_lease(candidates: list[int], pipe: Connection) -> None:
    with lease_tcp_ports(candidates=candidates) as ports:
        pipe.send(ports[0])
        pipe.recv()


def _unused_ports() -> tuple[int, int]:
    with socket.socket() as first, socket.socket() as second:
        first.bind(("", 0))
        second.bind(("", 0))
        return first.getsockname()[1], second.getsockname()[1]


def test_independent_launchers_hold_distinct_ports_and_release_on_exit():
    first_port, second_port = _unused_ports()
    candidates = [first_port, second_port]
    context = multiprocessing.get_context("spawn")
    children: list[tuple[multiprocessing.Process, Connection]] = []
    try:
        for expected in candidates:
            receiver, sender = context.Pipe()
            child = context.Process(target=_hold_lease, args=(candidates, sender))
            child.start()
            sender.close()
            children.append((child, receiver))
            assert receiver.poll(10)
            assert receiver.recv() == expected

        first_child, first_receiver = children[0]
        first_receiver.send(None)
        first_child.join(10)
        assert first_child.exitcode == 0

        with lease_tcp_ports(candidates=candidates) as reclaimed:
            assert reclaimed == [first_port]

        second_child, _ = children[1]
        second_child.kill()
        second_child.join(10)
        assert second_child.exitcode is not None

        with lease_tcp_ports(candidates=[second_port]) as reclaimed:
            assert reclaimed == [second_port]
    finally:
        for child, receiver in children:
            if child.is_alive():
                child.kill()
            child.join(10)
            receiver.close()


@pytest.mark.parametrize("family, host", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")])
def test_lease_skips_occupied_tcp_port(family, host):
    if family == socket.AF_INET6 and not socket.has_ipv6:
        pytest.skip("IPv6 support is required")
    with socket.socket(family, socket.SOCK_STREAM) as occupied:
        occupied.bind((host, 0))
        occupied_port = occupied.getsockname()[1]
        available_port, other_port = _unused_ports()
        if available_port == occupied_port:
            available_port = other_port
        with lease_tcp_ports(candidates=[occupied_port, available_port]) as ports:
            assert ports == [available_port]


def test_named_port_honors_an_existing_claim(monkeypatch):
    with lease_tcp_ports() as (allocated,):
        monkeypatch.setenv("PORT_WEB", str(allocated))
        with lease_tcp_port(8080, name="web") as port:
            assert port == allocated

        monkeypatch.setenv("PORT_WEB", "0")
        with lease_tcp_port(name="web") as port:
            assert port != allocated


def test_contiguous_block_claims_every_port():
    with lease_tcp_port_block(3) as base:
        with pytest.raises(RuntimeError):
            with lease_tcp_ports(candidates=range(base, base + 3)):
                pytest.fail("A second allocation overlapped the contiguous block")


def test_dynamic_leases_stay_in_the_shared_pool_and_honor_exclusions():
    with lease_tcp_ports(6, excluded=range(20000, 23000)) as ports:
        assert len(set(ports)) == 6
        assert all(23000 <= port < 24000 for port in ports)
