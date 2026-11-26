# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import socket
import sys
from multiprocessing.connection import Connection

import pytest
from iris.cluster.runtime.port_lease import lease_tcp_port


def _hold_lease(candidates: list[int], pipe: Connection) -> None:
    with lease_tcp_port(iter(candidates)) as port:
        pipe.send(port)
        pipe.recv()


def _unused_ports() -> tuple[int, int]:
    with socket.socket() as first, socket.socket() as second:
        first.bind(("", 0))
        second.bind(("", 0))
        return first.getsockname()[1], second.getsockname()[1]


@pytest.mark.skipif(sys.platform != "linux", reason="abstract Unix sockets require Linux")
def test_independent_launchers_hold_distinct_ports_and_release_on_exit():
    first_port, second_port = _unused_ports()
    candidates = [first_port, second_port]
    context = multiprocessing.get_context("fork")
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

        with lease_tcp_port(iter(candidates)) as reclaimed:
            assert reclaimed == first_port

        second_child, _ = children[1]
        second_child.kill()
        second_child.join(10)
        assert second_child.exitcode is not None

        with lease_tcp_port(iter([second_port])) as reclaimed:
            assert reclaimed == second_port
    finally:
        for child, receiver in children:
            if child.is_alive():
                child.kill()
            child.join(10)
            receiver.close()


@pytest.mark.skipif(sys.platform != "linux", reason="abstract Unix sockets require Linux")
def test_lease_skips_occupied_tcp_port():
    occupied_port, available_port = _unused_ports()
    with socket.socket() as occupied:
        occupied.bind(("", occupied_port))
        with lease_tcp_port(iter([occupied_port, available_port])) as port:
            assert port == available_port
