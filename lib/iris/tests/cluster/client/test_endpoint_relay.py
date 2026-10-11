# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from iris.cluster.client.endpoint_relay import EndpointRelay


class FakeRegistry:
    """An endpoint registry whose entries the test moves between servers."""

    def __init__(self, *addresses: str):
        self.addresses = list(addresses)

    def resolve(self) -> list[str]:
        return list(self.addresses)

    def move_to(self, *addresses: str) -> None:
        self.addresses = list(addresses)


class Upstream:
    """A local HTTP server that answers every request with its own name, or as configured."""

    def __init__(
        self,
        name: str,
        *,
        status: int = 200,
        hang: threading.Event | None = None,
        chunks: int = 0,
        drop_idle: bool = False,
    ):
        upstream = self
        self.name = name
        self.requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                upstream.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if hang is not None:
                    hang.wait()
                    return
                if chunks:
                    self.send_response(200)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for index in range(chunks):
                        data = f"data: {upstream.name}-{index}\n\n".encode()
                        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    return
                body = json.dumps({"served_by": upstream.name, "path": self.path}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                # Close without announcing it, as a server timing out an idle keep-alive connection does.
                self.close_connection = drop_idle

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.address = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _post(url: str, payload: dict, timeout: float = 10.0) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@pytest.fixture
def upstreams():
    started: list[Upstream] = []

    def start(name: str, **kwargs) -> Upstream:
        upstream = Upstream(name, **kwargs)
        started.append(upstream)
        return upstream

    yield start
    for upstream in started:
        upstream.stop()


def test_relay_follows_endpoint_to_new_address_after_preemption(upstreams):
    first, second = upstreams("first"), upstreams("second")
    registry = FakeRegistry(first.address)
    with EndpointRelay(registry.resolve, refresh_interval=60) as relay:
        status, body = _post(f"{relay.url}/v1/completions", {"prompt": "a"})
        assert (status, json.loads(body)["served_by"]) == (200, "first")

        # The serve task is preempted and its retry registers elsewhere.
        first.stop()
        registry.move_to(second.address)
        status, body = _post(f"{relay.url}/v1/completions", {"prompt": "b"})

    assert status == 200
    assert json.loads(body) == {"served_by": "second", "path": "/v1/completions"}
    assert second.requests == [{"prompt": "b"}]


def test_relay_retries_a_stale_404_and_passes_a_current_one_through(upstreams):
    stranger, serve = upstreams("stranger", status=404), upstreams("serve")
    registry = FakeRegistry(stranger.address)
    with EndpointRelay(registry.resolve, refresh_interval=60) as relay:
        assert _post(f"{relay.url}/v1/completions", {"prompt": "a"})[0] == 404

        registry.move_to(serve.address)
        status, body = _post(f"{relay.url}/v1/completions", {"prompt": "b"})

    assert (status, json.loads(body)["served_by"]) == (200, "serve")


def test_relay_waits_while_no_endpoint_is_registered(upstreams):
    serve = upstreams("serve")
    registry = FakeRegistry()
    with EndpointRelay(registry.resolve, refresh_interval=60) as relay:
        threading.Timer(1.0, lambda: registry.move_to(serve.address)).start()
        status, body = _post(f"{relay.url}/v1/completions", {"prompt": "a"})

    assert (status, json.loads(body)["served_by"]) == (200, "serve")


def test_relay_abandons_a_request_hanging_on_the_old_address(upstreams):
    release = threading.Event()
    stuck, serve = upstreams("stuck", hang=release), upstreams("serve")
    registry = FakeRegistry(stuck.address)
    try:
        with EndpointRelay(registry.resolve, refresh_interval=0.5) as relay:
            threading.Timer(1.0, lambda: registry.move_to(serve.address)).start()
            started = time.monotonic()
            status, body = _post(f"{relay.url}/v1/completions", {"prompt": "a"})
            elapsed = time.monotonic() - started
    finally:
        release.set()

    assert (status, json.loads(body)["served_by"]) == (200, "serve")
    assert elapsed < 5


def test_relay_skips_a_dead_instance_still_listed_beside_the_new_one(upstreams):
    old, new = upstreams("old"), upstreams("new")
    old.stop()
    # The preempted attempt's registration can outlive it while the retry registers.
    with EndpointRelay(FakeRegistry(old.address, new.address).resolve, refresh_interval=60) as relay:
        status, body = _post(f"{relay.url}/v1/completions", {"prompt": "a"})

    assert (status, json.loads(body)["served_by"]) == (200, "new")


def test_relay_reconnects_at_once_when_the_server_drops_an_idle_connection(upstreams):
    serve = upstreams("serve", drop_idle=True)
    with EndpointRelay(FakeRegistry(serve.address).resolve, refresh_interval=60) as relay:
        assert _post(f"{relay.url}/v1/completions", {"prompt": "a"})[0] == 200
        started = time.monotonic()
        status, _ = _post(f"{relay.url}/v1/completions", {"prompt": "b"})
        elapsed = time.monotonic() - started

    assert status == 200
    assert serve.requests == [{"prompt": "a"}, {"prompt": "b"}]
    # A dead pooled connection is not an endpoint failure, so no recovery wait.
    assert elapsed < 1


def test_relay_streams_chunked_responses(upstreams):
    serve = upstreams("serve", chunks=3)
    with EndpointRelay(FakeRegistry(serve.address).resolve) as relay:
        status, body = _post(f"{relay.url}/v1/completions", {"prompt": "a", "stream": True})

    assert status == 200
    assert body == b"data: serve-0\n\ndata: serve-1\n\ndata: serve-2\n\n"


def test_relay_reports_503_when_endpoint_never_returns():
    with EndpointRelay(FakeRegistry().resolve, recovery_timeout=0.5) as relay:
        status, body = _post(f"{relay.url}/v1/completions", {"prompt": "a"})

    assert status == 503
    assert b"unreachable" in body
