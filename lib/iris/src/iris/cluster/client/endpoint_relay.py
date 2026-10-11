# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a registered endpoint on loopback and follow it across re-registrations.

A client that resolves an endpoint once keeps the address of the task that
registered it. When that task is preempted and its retry registers on another
host, the old address refuses connections, hangs, or answers 404 from whatever
listens there now. The relay looks the name up in the endpoint registry, checks
it again periodically and after every upstream connection failure or stale 404,
and replays a request that has not received a response once the endpoint is
registered again. A request whose response has started is never replayed: the
relay drops the client connection so the client's own retry policy applies.

Run inside an Iris task as
``python -m iris.cluster.client.endpoint_relay /serve/name --port-file PATH``
and point the client at ``http://127.0.0.1:<port>``.
"""

import argparse
import http.client
import logging
import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from iris.client.client import iris_ctx

logger = logging.getLogger(__name__)

Resolver = Callable[[], list[str]]

DEFAULT_REFRESH_INTERVAL_SECONDS = 30.0
DEFAULT_RECOVERY_TIMEOUT_SECONDS = 3600.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 1800.0
# How often a request waiting for the endpoint to come back looks it up again.
_RECOVERY_POLL_SECONDS = 2.0
_STREAM_CHUNK_BYTES = 64 * 1024
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


class _Upstream:
    """One upstream HTTP connection, tagged with the address it was opened to."""

    def __init__(self, address: str, connect_timeout: float, response_timeout: float):
        parts = urlsplit(address if "://" in address else f"http://{address}")
        if parts.scheme != "http" or parts.hostname is None:
            raise ValueError(f"endpoint relay only forwards plain http addresses, got {address!r}")
        self.address = address
        self.base_path = parts.path.rstrip("/")
        self.connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=connect_timeout)
        self.reused = False
        self._response_timeout = response_timeout

    def send(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> http.client.HTTPResponse:
        self.connection.request(method, f"{self.base_path}{path}", body=body or None, headers=headers)
        # Connect with the short timeout, then wait for a long generation with the long one.
        self.connection.sock.settimeout(self._response_timeout)
        response = self.connection.getresponse()
        self.reused = True
        return response

    def abort(self) -> None:
        """Unblock a thread waiting on this connection and close it."""
        sock = self.connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.connection.close()


class EndpointRelay:
    """Loopback HTTP forwarder to whatever address an endpoint name currently resolves to.

    ``resolve`` returns the addresses (``http://host:port`` or ``host:port``)
    registered under the name, empty while nothing is registered; it may raise
    when the registry is unreachable, in which case the last known address
    stays in use. The relay stays on one address while it is registered and
    answering, and moves to another registered address after it fails.
    """

    def __init__(
        self,
        resolve: Resolver,
        *,
        name: str = "endpoint",
        host: str = "127.0.0.1",
        port: int = 0,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL_SECONDS,
        recovery_timeout: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        response_timeout: float = DEFAULT_RESPONSE_TIMEOUT_SECONDS,
    ):
        self._resolve = resolve
        self._name = name
        self._refresh_interval = refresh_interval
        self._recovery_timeout = recovery_timeout
        self._connect_timeout = connect_timeout
        self._response_timeout = response_timeout
        self._lock = threading.Lock()
        self._lookup_lock = threading.Lock()
        self._address: str | None = None
        self._looked_up_at = float("-inf")
        self._in_flight: set[_Upstream] = set()
        self._local = threading.local()
        self._stopped = threading.Event()
        self._server = ThreadingHTTPServer((host, port), self._handler_class())
        self._server.daemon_threads = True
        self._threads: list[threading.Thread] = []

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def address(self) -> str | None:
        """The upstream address requests currently go to."""
        with self._lock:
            return self._address

    def start(self) -> "EndpointRelay":
        self.refresh()
        for target, label in ((self._server.serve_forever, "serve"), (self._refresh_loop, "refresh")):
            thread = threading.Thread(target=target, name=f"endpoint-relay-{label}", daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info("Relaying %s on %s (upstream %s)", self._name, self.url, self.address)
        return self

    def close(self) -> None:
        self._stopped.set()
        self._server.shutdown()
        self._server.server_close()
        for thread in self._threads:
            thread.join(timeout=5)

    def wait(self) -> None:
        """Block until :meth:`close` is called."""
        self._stopped.wait()

    def __enter__(self) -> "EndpointRelay":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def refresh(self, *, newer_than: float = float("inf"), failed: str | None = None) -> str | None:
        """Look the endpoint up, reusing a lookup that started after ``newer_than`` (``time.monotonic()``).

        Requests that fail together share one lookup instead of each asking the registry. ``failed``
        is an address that just failed; another registered address is preferred over it.
        """
        with self._lookup_lock:
            if self._looked_up_at > newer_than:
                return self.address
            started = time.monotonic()
            try:
                addresses = self._resolve()
            except Exception:
                logger.warning("Endpoint lookup for %s failed; keeping %s", self._name, self.address, exc_info=True)
                return self.address
            self._looked_up_at = started
            candidates = [address for address in addresses if address != failed] or addresses
            current = self.address
            address = current if current in candidates else next(iter(candidates), None)
            self._set_address(address)
        return address

    def _set_address(self, address: str | None) -> None:
        with self._lock:
            previous = self._address
            if address == previous:
                return
            self._address = address
            stale = [upstream for upstream in self._in_flight if upstream.address != address]
        logger.warning("Endpoint %s moved: %s -> %s", self._name, previous, address)
        # A request blocked on the old address would otherwise wait out the full response timeout.
        for upstream in stale:
            upstream.abort()

    def _refresh_loop(self) -> None:
        while not self._stopped.wait(self._refresh_interval):
            self.refresh(newer_than=time.monotonic() - self._refresh_interval / 2)

    def _upstream(self, address: str) -> _Upstream:
        """This handler thread's pooled connection to ``address``."""
        upstream: _Upstream | None = getattr(self._local, "upstream", None)
        if upstream is None or upstream.address != address:
            if upstream is not None:
                upstream.connection.close()
            upstream = _Upstream(address, self._connect_timeout, self._response_timeout)
            self._local.upstream = upstream
        return upstream

    def _drop_upstream(self) -> None:
        upstream: _Upstream | None = getattr(self._local, "upstream", None)
        if upstream is not None:
            upstream.connection.close()
            self._local.upstream = None

    def open(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> http.client.HTTPResponse | None:
        """Send one request to the current address, following moves until it gets a response.

        Returns None when the endpoint stays unreachable for ``recovery_timeout`` seconds.
        """
        deadline = time.monotonic() + self._recovery_timeout
        while True:
            address = self.address
            if address is not None:
                upstream = self._upstream(address)
                with self._lock:
                    if self._address != address:
                        continue
                    self._in_flight.add(upstream)
                try:
                    response = upstream.send(method, path, body, headers)
                except (OSError, http.client.HTTPException) as exc:
                    self._drop_upstream()
                    if upstream.reused:
                        # The server closed an idle keep-alive connection; retry on a fresh one.
                        continue
                    logger.warning("Request to %s at %s failed: %s", self._name, address, exc)
                else:
                    if response.status != 404 or self.refresh(newer_than=time.monotonic(), failed=address) == address:
                        return response
                    response.read()
                    logger.warning("%s at %s answered 404 from a stale address", self._name, address)
                    continue
                finally:
                    with self._lock:
                        self._in_flight.discard(upstream)
            if time.monotonic() >= deadline:
                return None
            failed_at = time.monotonic()
            if self.refresh(newer_than=failed_at - _RECOVERY_POLL_SECONDS, failed=address) == address:
                time.sleep(_RECOVERY_POLL_SECONDS)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                logger.debug("%s - " + format, self.client_address[0], *args)

            def _forward(self) -> None:
                body = self._read_body()
                headers = {key: value for key, value in self.headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS}
                response = relay.open(self.command, self.path, body, headers)
                if response is None:
                    self._reply_unavailable()
                    return
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in _HOP_BY_HOP_HEADERS:
                        self.send_header(key, value)
                length = response.getheader("content-length")
                chunked = length is None and self.command != "HEAD"
                if length is not None:
                    self.send_header("Content-Length", length)
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while chunk := response.read1(_STREAM_CHUNK_BYTES):
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n" if chunked else chunk)
                        self.wfile.flush()
                except (OSError, http.client.HTTPException) as exc:
                    # The response started, so replaying could duplicate work; let the client retry.
                    logger.warning("Response from %s broke mid-stream: %s", relay._name, exc)
                    relay._drop_upstream()
                    self.close_connection = True
                    return
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
                if response.will_close:
                    relay._drop_upstream()

            def _read_body(self) -> bytes:
                if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                    parts = []
                    while size := int(self.rfile.readline().split(b";")[0], 16):
                        parts.append(self.rfile.read(size))
                        self.rfile.readline()
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(parts)
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

            def _reply_unavailable(self) -> None:
                message = f"endpoint {relay._name} unreachable for {relay._recovery_timeout:.0f}s\n".encode()
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _forward

        return Handler


def registry_resolver(endpoint_name: str) -> Resolver:
    """Resolve ``endpoint_name`` through the current task's Iris controller registry."""
    client = iris_ctx().client

    def resolve() -> list[str]:
        return [endpoint.address for endpoint in client.list_endpoint_instances(endpoint_name)]

    return resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("endpoint", help="registered endpoint name, e.g. /serve/inference-<id>")
    parser.add_argument("--port", type=int, default=0, help="loopback port to listen on (default: any free port)")
    parser.add_argument("--port-file", type=Path, help="write the listening port here once ready")
    parser.add_argument("--refresh-interval", type=float, default=DEFAULT_REFRESH_INTERVAL_SECONDS)
    parser.add_argument("--recovery-timeout", type=float, default=DEFAULT_RECOVERY_TIMEOUT_SECONDS)
    parser.add_argument("--response-timeout", type=float, default=DEFAULT_RESPONSE_TIMEOUT_SECONDS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    relay = EndpointRelay(
        registry_resolver(args.endpoint),
        name=args.endpoint,
        port=args.port,
        refresh_interval=args.refresh_interval,
        recovery_timeout=args.recovery_timeout,
        response_timeout=args.response_timeout,
    ).start()
    if args.port_file is not None:
        partial = args.port_file.with_suffix(".partial")
        partial.write_text(str(relay.port))
        partial.replace(args.port_file)
    relay.wait()


if __name__ == "__main__":
    main()
