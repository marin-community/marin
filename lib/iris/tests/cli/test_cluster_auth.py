# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from iris.cli import cluster as cluster_cli
from rigging.auth import StaticTokenProvider
from rigging.credentials import ClientCredentials


@pytest.fixture
def authenticated_upstream() -> Iterator[tuple[str, list[str | None]]]:
    proxy_authorizations: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            proxy_authorizations.append(self.headers.get("Proxy-Authorization"))
            body = json.dumps({"enabled": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", proxy_authorizations
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_proxy_sends_context_credentials_to_upstream(
    monkeypatch: pytest.MonkeyPatch,
    authenticated_upstream: tuple[str, list[str | None]],
) -> None:
    upstream_url, proxy_authorizations = authenticated_upstream
    credentials = ClientCredentials(iap_provider=StaticTokenProvider("iap-token"))

    monkeypatch.setattr(cluster_cli.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cluster_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(terminate=lambda: None, wait=lambda: None),
    )

    async def request_through_proxy(app) -> int:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
                return (await client.get("/auth/config")).status_code

    def serve_proxy(app, **_kwargs) -> None:
        assert asyncio.run(request_through_proxy(app)) == 200

    monkeypatch.setattr(cluster_cli.uvicorn, "run", serve_proxy)

    result = CliRunner().invoke(
        cluster_cli.cluster_dashboard_proxy,
        obj={"controller_url": upstream_url, "credentials": credentials},
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert proxy_authorizations == ["Bearer iap-token"]
