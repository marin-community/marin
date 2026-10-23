# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docker workers' best-effort ALLOWLIST: per-task DNS filters in front of the host resolver."""

import errno
import ipaddress
import socket
import socketserver
import struct
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from iris.cluster.worker.egress_dns import EgressDnsFilters, HostResolver, read_host_resolver

_RCODE_REFUSED = 5
_ANSWER_ADDRESS = bytes([192, 0, 2, 1])
_UPSTREAM_BIND_ATTEMPTS = 20


def _query(name: str, query_id: int = 0x1234) -> bytes:
    header = struct.pack("!HHHHHH", query_id, 0x0100, 1, 0, 0, 0)  # RD set, one question
    labels = b"".join(bytes([len(label)]) + label.encode() for label in name.split("."))
    return header + labels + b"\x00" + struct.pack("!HH", 1, 1)  # A, IN


def _answer(query: bytes) -> bytes:
    """The fake upstream's reply: one A record for whatever was asked."""
    query_id, flags = struct.unpack_from("!HH", query)
    header = struct.pack("!HHHHHH", query_id, 0x8180 | (flags & 0x0100), 1, 1, 0, 0)
    record = struct.pack("!HHHIH", 0xC00C, 1, 1, 60, 4) + _ANSWER_ADDRESS
    return header + query[12:] + record


class _UpstreamUdp(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        query, sock = self.request
        sock.sendto(_answer(query), self.client_address)


class _UpstreamTcp(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        (length,) = struct.unpack("!H", self.rfile.read(2))
        response = _answer(self.rfile.read(length))
        self.wfile.write(struct.pack("!H", len(response)) + response)


def _bind_upstream() -> tuple[socketserver.ThreadingUDPServer, socketserver.ThreadingTCPServer]:
    """Bind UDP and TCP to one port, retrying when another socket already holds the TCP port."""
    for _ in range(_UPSTREAM_BIND_ATTEMPTS):
        udp = socketserver.ThreadingUDPServer(("127.0.0.1", 0), _UpstreamUdp)
        try:
            return udp, socketserver.ThreadingTCPServer(("127.0.0.1", udp.server_address[1]), _UpstreamTcp)
        except OSError as e:
            udp.server_close()
            if e.errno != errno.EADDRINUSE:
                raise
    raise RuntimeError(f"No port free for both UDP and TCP after {_UPSTREAM_BIND_ATTEMPTS} attempts")


@pytest.fixture
def upstream_port() -> Iterator[int]:
    """A resolver on 127.0.0.1 that answers every query over UDP and TCP."""
    udp, tcp = _bind_upstream()
    for server in (udp, tcp):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    yield udp.server_address[1]
    for server in (udp, tcp):
        server.shutdown()
        server.server_close()


@pytest.fixture
def filter_port() -> int:
    """A port for filters; a collision on a filter's address only moves it to the next one."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def make_filters(tmp_path, upstream_port, filter_port):
    created: list[EgressDnsFilters] = []

    def make(domains=("pypi.org", "*.github.com"), search=("c.proj.internal",)) -> EgressDnsFilters:
        filters = EgressDnsFilters(
            tmp_path / "egress-dns",
            domains,
            HostResolver("127.0.0.1", search, ("ndots:1",)),
            port=filter_port,
            upstream_port=upstream_port,
        )
        created.append(filters)
        return filters

    yield make
    for filters in created:
        filters.close()


def _nameserver(resolv_conf: Path) -> str:
    [line] = [line for line in resolv_conf.read_text().splitlines() if line.startswith("nameserver")]
    return line.split()[1]


def _ask(address: tuple[str, int], query: bytes, transport: str) -> bytes:
    if transport == "udp":
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(5)
            sock.sendto(query, address)
            return sock.recv(65535)
    with socket.create_connection(address, timeout=5) as sock:
        sock.sendall(struct.pack("!H", len(query)) + query)
        (length,) = struct.unpack("!H", sock.recv(2))
        response = b""
        while len(response) < length:
            response += sock.recv(length - len(response))
        return response


def _resolves(address: tuple[str, int], name: str, transport: str = "udp") -> bool:
    query = _query(name)
    response = _ask(address, query, transport)
    assert response[:2] == query[:2]  # the client matches answers to queries by ID
    rcode = response[3] & 0x0F
    if rcode == _RCODE_REFUSED:
        assert response[12 : len(query)] == query[12:]  # the question is echoed back
        return False
    assert rcode == 0
    assert response.endswith(_ANSWER_ADDRESS)
    return True


@pytest.mark.parametrize("transport", ["udp", "tcp"])
@pytest.mark.parametrize(
    ("name", "resolves"),
    [
        pytest.param("pypi.org", True, id="listed"),
        pytest.param("PyPI.ORG", True, id="case-insensitive"),
        pytest.param("api.github.com", True, id="one-label-wildcard"),
        pytest.param("a.b.github.com", False, id="wildcard-is-one-label"),
        pytest.param("github.com", False, id="wildcard-excludes-apex"),
        pytest.param("api.example.com", True, id="job-rule"),
        pytest.param("vm-1.c.proj.internal", True, id="host-search-domain"),
        pytest.param("metadata.google.internal", True, id="internal-names"),
        pytest.param("1.2.0.192.in-addr.arpa", True, id="reverse-lookup"),
        pytest.param("exfil.attacker.example", False, id="unlisted"),
    ],
)
def test_task_dns_filter_resolves_only_allowlisted_and_internal_names(
    make_filters, filter_port, transport, name, resolves
):
    filters = make_filters()
    resolv_conf = filters.start("task_attempt_0", ["api.example.com"])

    assert _resolves((_nameserver(resolv_conf), filter_port), name, transport) == resolves


def test_task_dns_filters_are_per_task_and_removed_on_stop(tmp_path, make_filters, filter_port):
    host_conf = tmp_path / "host-resolv.conf"
    host_conf.write_text(
        "# managed by the platform\nnameserver 169.254.169.254\nnameserver 8.8.8.8\n"
        "search c.proj.internal google.internal\noptions ndots:2 timeout:1\n"
    )
    host = read_host_resolver(host_conf)
    assert host == HostResolver("169.254.169.254", ("c.proj.internal", "google.internal"), ("ndots:2", "timeout:1"))

    filters = make_filters()
    first = filters.start("first_attempt_0", ["first.example.com"])
    second = filters.start("second_attempt_0", [])

    first_address, second_address = _nameserver(first), _nameserver(second)
    assert first_address != second_address
    assert all(ipaddress.ip_address(a) in ipaddress.ip_network("127.53.0.0/16") for a in (first_address, second_address))
    assert first.read_text().splitlines()[1:] == ["search c.proj.internal", "options ndots:1"]
    assert _resolves((first_address, filter_port), "first.example.com")
    assert not _resolves((second_address, filter_port), "first.example.com")

    filters.stop("first_attempt_0")

    assert not first.exists()
    assert _resolves((second_address, filter_port), "pypi.org")


def test_restarted_worker_restores_an_adopted_tasks_filter(make_filters, filter_port):
    """An adopted container's resolv.conf names the old filter's address, so the new worker must serve it."""
    previous = make_filters()
    resolv_conf = previous.start("task_attempt_0", ["api.example.com"])
    previous.close()  # the old worker exits and leaves the task's state behind

    restarted = make_filters(domains=("unrelated.example.org",))

    assert restarted.restore("task_attempt_0")
    assert not restarted.restore("other_attempt_0")
    address = (_nameserver(resolv_conf), filter_port)
    assert _resolves(address, "api.example.com")
    assert _resolves(address, "pypi.org")
    assert not _resolves(address, "unrelated.example.org")
