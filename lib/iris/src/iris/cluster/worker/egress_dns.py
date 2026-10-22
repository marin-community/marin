# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-effort ``EGRESS_POLICY_ALLOWLIST`` for Docker workers.

A Docker ALLOWLIST task shares the host's network, so the worker cannot filter
its connections. It filters the task's name lookups instead. Each task gets a
resolver on its own loopback address, and the task's ``/etc/resolv.conf``
names only that resolver. The resolver forwards a query for an allowlisted or
internal name to the host's resolver and answers every other query REFUSED.

This blocks a task that reaches an unlisted service by name. It does not stop
a task that connects to a literal address or queries another resolver
directly. CIDR rules and port lists therefore have no effect here: every
address and port is already reachable.

Internal names are the host's resolv.conf search domains, ``localhost``,
``internal`` (GCE instance and metadata names) and reverse-lookup names.
"""

import errno
import ipaddress
import json
import logging
import shutil
import socket
import socketserver
import struct
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DNS_PORT = 53
HOST_RESOLV_CONF = Path("/etc/resolv.conf")
# Per-task filter state, under the worker's cache directory.
EGRESS_DNS_DIRNAME = "egress-dns"
RESOLV_CONF_NAME = "resolv.conf"
_STATE_NAME = "filter.json"

# Per-task resolver addresses. systemd-resolved owns 127.0.0.53 and 127.0.0.54.
_LOOPBACK_POOL = ipaddress.ip_network("127.53.0.0/16")
_INTERNAL_SUFFIXES = ("localhost", "internal", "in-addr.arpa", "ip6.arpa")
_UPSTREAM_TIMEOUT = 5.0
# Each filter logs the first refusal of this many distinct names.
_MAX_LOGGED_REFUSALS = 100
# resolv.conf(5): with no nameserver line, the resolver uses the local machine.
_DEFAULT_NAMESERVER = "127.0.0.1"

_HEADER = struct.Struct("!HHHHHH")
_LENGTH = struct.Struct("!H")
_FLAG_RESPONSE = 0x8000
_FLAGS_OPCODE_AND_RD = 0x7900
_FLAG_RA = 0x0080
_RCODE_REFUSED = 5
_MAX_LABEL = 63
_WILDCARD = "*"


@dataclass(frozen=True)
class HostResolver:
    """The parts of the host's resolv.conf a task's resolv.conf keeps."""

    nameserver: str
    search: tuple[str, ...] = ()
    options: tuple[str, ...] = ()


def read_host_resolver(path: Path = HOST_RESOLV_CONF) -> HostResolver:
    """Parse the first nameserver, the last search line and every option from ``path``."""
    nameserver = ""
    search: tuple[str, ...] = ()
    options: list[str] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith(("#", ";")):
            continue
        keyword, values = fields[0], fields[1:]
        if keyword == "nameserver" and values and not nameserver:
            nameserver = values[0]
        elif keyword == "search":
            search = tuple(value.lower().rstrip(".") for value in values)
        elif keyword == "options":
            options.extend(values)
    return HostResolver(nameserver or _DEFAULT_NAMESERVER, search, tuple(options))


class DomainAllowlist:
    """Matches DNS names against exact names, wildcard patterns and internal suffixes.

    A ``*`` label in a pattern matches exactly one label, as in Cilium's
    ``matchPattern``. A suffix matches itself and every name below it.
    """

    def __init__(self, domains: Iterable[str], suffixes: Iterable[str]) -> None:
        self._names: set[str] = set()
        self._patterns: list[tuple[str, ...]] = []
        for domain in domains:
            labels = tuple(domain.split("."))
            if _WILDCARD in labels:
                self._patterns.append(labels)
            else:
                self._names.add(domain)
        self._suffixes = tuple(suffix for suffix in suffixes if suffix)

    def allows(self, name: str) -> bool:
        if name in self._names:
            return True
        if any(name == suffix or name.endswith(f".{suffix}") for suffix in self._suffixes):
            return True
        labels = name.split(".")
        return any(_matches(pattern, labels) for pattern in self._patterns)


def _matches(pattern: tuple[str, ...], labels: list[str]) -> bool:
    if len(pattern) != len(labels):
        return False
    return all(want == _WILDCARD or want == label for want, label in zip(pattern, labels, strict=True))


@dataclass(frozen=True)
class Question:
    name: str  # lowercase, without the trailing dot
    end: int  # offset of the first byte after the question section


def parse_question(query: bytes) -> Question | None:
    """The single question of a DNS query, or None for a response or a malformed query."""
    if len(query) < _HEADER.size:
        return None
    _, flags, question_count, _, _, _ = _HEADER.unpack_from(query)
    if flags & _FLAG_RESPONSE or question_count != 1:
        return None
    labels: list[str] = []
    offset = _HEADER.size
    while True:
        if offset >= len(query):
            return None
        length = query[offset]
        offset += 1
        if length == 0:
            break
        # Larger values are compression pointers, which a query's only name never needs.
        if length > _MAX_LABEL or offset + length > len(query):
            return None
        labels.append(query[offset : offset + length].decode("latin-1").lower())
        offset += length
    end = offset + 4  # QTYPE and QCLASS
    if end > len(query):
        return None
    return Question(".".join(labels), end)


def refused_response(query: bytes, question: Question) -> bytes:
    """A REFUSED answer to ``query`` that echoes its ID, opcode, RD flag and question."""
    query_id, flags = struct.unpack_from("!HH", query)
    response_flags = _FLAG_RESPONSE | (flags & _FLAGS_OPCODE_AND_RD) | _FLAG_RA | _RCODE_REFUSED
    return _HEADER.pack(query_id, response_flags, 1, 0, 0, 0) + query[_HEADER.size : question.end]


def answer(
    query: bytes,
    allowlist: DomainAllowlist,
    forward: Callable[[bytes], bytes | None],
    on_refused: Callable[[str], None],
) -> bytes | None:
    """Forward an allowed query, refuse any other; None means send no reply."""
    question = parse_question(query)
    if question is None:
        return None
    if allowlist.allows(question.name):
        return forward(query)
    on_refused(question.name)
    return refused_response(query, question)


def _family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def _forward_udp(query: bytes, upstream: tuple[str, int]) -> bytes | None:
    with socket.socket(_family(upstream[0]), socket.SOCK_DGRAM) as sock:
        sock.settimeout(_UPSTREAM_TIMEOUT)
        sock.connect(upstream)
        sock.send(query)
        try:
            return sock.recv(65535)
        except TimeoutError:
            return None  # the client retries


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _recv_message(sock: socket.socket) -> bytes | None:
    prefix = _recv_exact(sock, _LENGTH.size)
    if prefix is None:
        return None
    return _recv_exact(sock, _LENGTH.unpack(prefix)[0])


def _forward_tcp(query: bytes, upstream: tuple[str, int]) -> bytes | None:
    with socket.create_connection(upstream, timeout=_UPSTREAM_TIMEOUT) as sock:
        sock.sendall(_LENGTH.pack(len(query)) + query)
        return _recv_message(sock)


def _filter_of(server: socketserver.BaseServer) -> "DnsFilter":
    assert isinstance(server, _UdpServer | _TcpServer)
    return server.dns_filter


class _UdpHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        query, sock = self.request
        dns_filter = _filter_of(self.server)
        response = answer(
            query,
            dns_filter.allowlist,
            lambda message: _forward_udp(message, dns_filter.upstream),
            dns_filter.log_refused,
        )
        if response is not None:
            sock.sendto(response, self.client_address)


class _TcpHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock: socket.socket = self.request
        sock.settimeout(_UPSTREAM_TIMEOUT)
        dns_filter = _filter_of(self.server)
        while (query := _recv_message(sock)) is not None:
            response = answer(
                query,
                dns_filter.allowlist,
                lambda message: _forward_tcp(message, dns_filter.upstream),
                dns_filter.log_refused,
            )
            if response is None:
                return
            sock.sendall(_LENGTH.pack(len(response)) + response)


class _LoggingServerMixin:
    def handle_error(self, request, client_address) -> None:
        logger.warning("Task DNS filter failed a query from %s", client_address, exc_info=True)


class _UdpServer(_LoggingServerMixin, socketserver.ThreadingUDPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], dns_filter: "DnsFilter") -> None:
        self.dns_filter = dns_filter
        super().__init__(address, _UdpHandler)


class _TcpServer(_LoggingServerMixin, socketserver.ThreadingTCPServer):
    daemon_threads = True
    # A restored filter rebinds its predecessor's address while old connections linger in TIME_WAIT.
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], dns_filter: "DnsFilter") -> None:
        self.dns_filter = dns_filter
        super().__init__(address, _TcpHandler)


class DnsFilter:
    """A UDP and TCP resolver on one address that forwards only allowed names to ``upstream``.

    Binding happens in the constructor, so an address conflict raises before
    anything starts; ``port`` 0 binds an ephemeral port shared by both
    protocols.
    """

    def __init__(self, host: str, port: int, allowlist: DomainAllowlist, upstream: tuple[str, int]) -> None:
        self.allowlist = allowlist
        self.upstream = upstream
        self._refused: set[str] = set()
        self._refused_lock = threading.Lock()
        self._udp = _UdpServer((host, port), self)
        try:
            self._tcp = _TcpServer((host, self._udp.server_address[1]), self)
        except OSError:
            self._udp.server_close()
            raise
        self._threads: list[threading.Thread] = []

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._udp.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        for server in (self._udp, self._tcp):
            thread = threading.Thread(target=server.serve_forever, name=f"egress-dns-{self.address[0]}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        for server in (self._udp, self._tcp):
            if self._threads:
                server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join()

    def log_refused(self, name: str) -> None:
        with self._refused_lock:
            if name in self._refused or len(self._refused) >= _MAX_LOGGED_REFUSALS:
                return
            self._refused.add(name)
        logger.info("Task DNS filter %s refused %r, which the egress allowlist does not list", self.address[0], name)


class EgressDnsFilters:
    """The worker's per-task DNS filters.

    Each filter keeps its state, including the resolv.conf a task mounts at
    ``/etc/resolv.conf``, in ``state_root/<key>``, so a restarted worker can
    restore the filters of the containers it adopts.

    Args:
        state_root: Directory for per-task state; must be a host path Docker can mount.
        domains: Domains every task may resolve: Iris's base allowlist and the cluster's additions.
        host_resolver: The worker's resolv.conf, whose first nameserver answers forwarded queries.
        port: The port filters listen on; 53 outside tests, since resolv.conf cannot name a port.
        upstream_port: The nameserver's port.
    """

    def __init__(
        self,
        state_root: Path,
        domains: Sequence[str],
        host_resolver: HostResolver,
        port: int = DNS_PORT,
        upstream_port: int = DNS_PORT,
    ) -> None:
        self._state_root = state_root
        self._domains = tuple(domains)
        self._host = host_resolver
        self._port = port
        self._upstream = (host_resolver.nameserver, upstream_port)
        self._lock = threading.Lock()
        self._filters: dict[str, DnsFilter] = {}

    def start(self, key: str, domains: Sequence[str]) -> Path:
        """Start a filter for one task and return the resolv.conf naming it.

        ``domains`` are the job's own domain rules, added to the worker-wide ones.
        """
        allowed = [*self._domains, *domains]
        with self._lock:
            if key in self._filters:
                raise ValueError(f"Task DNS filter {key} is already running")
            in_use = {dns_filter.address[0] for dns_filter in self._filters.values()}
            dns_filter = self._bind_free_address(allowed, in_use)
            self._filters[key] = dns_filter
        state_dir = self._state_root / key
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / _STATE_NAME).write_text(json.dumps({"host": dns_filter.address[0], "domains": allowed}))
        resolv_conf = state_dir / RESOLV_CONF_NAME
        resolv_conf.write_text(self._resolv_conf(dns_filter.address[0]))
        dns_filter.start()
        logger.info("Started task DNS filter %s for %s", dns_filter.address[0], key)
        return resolv_conf

    def restore(self, key: str) -> bool:
        """Restart the filter a previous worker ran for ``key``; False when it ran none."""
        state_file = self._state_root / key / _STATE_NAME
        if not state_file.exists():
            return False
        state = json.loads(state_file.read_text())
        dns_filter = DnsFilter(state["host"], self._port, self._allowlist(state["domains"]), self._upstream)
        with self._lock:
            self._filters[key] = dns_filter
        dns_filter.start()
        logger.info("Restored task DNS filter %s for %s", state["host"], key)
        return True

    def stop(self, key: str) -> None:
        """Stop ``key``'s filter and remove its state; a no-op when none runs."""
        with self._lock:
            dns_filter = self._filters.pop(key, None)
        if dns_filter is not None:
            dns_filter.stop()
        state_dir = self._state_root / key
        if state_dir.exists():
            shutil.rmtree(state_dir)

    def close(self) -> None:
        """Stop every filter but keep its state, so the next worker can restore it."""
        with self._lock:
            filters = list(self._filters.values())
            self._filters.clear()
        for dns_filter in filters:
            dns_filter.stop()

    def _allowlist(self, domains: Sequence[str]) -> DomainAllowlist:
        return DomainAllowlist(domains, (*_INTERNAL_SUFFIXES, *self._host.search))

    def _bind_free_address(self, domains: Sequence[str], in_use: set[str]) -> DnsFilter:
        allowlist = self._allowlist(domains)
        for address in _LOOPBACK_POOL.hosts():
            host = str(address)
            if host in in_use:
                continue
            try:
                return DnsFilter(host, self._port, allowlist, self._upstream)
            except OSError as e:
                if e.errno != errno.EADDRINUSE:
                    raise
        raise RuntimeError(f"No free address in {_LOOPBACK_POOL} for a task DNS filter")

    def _resolv_conf(self, host: str) -> str:
        lines = [f"nameserver {host}"]
        if self._host.search:
            lines.append(f"search {' '.join(self._host.search)}")
        if self._host.options:
            lines.append(f"options {' '.join(self._host.options)}")
        return "\n".join(lines) + "\n"
