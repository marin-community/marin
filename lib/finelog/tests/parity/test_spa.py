# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-5d static SPA + base-href parity tests.

Drive the non-RPC HTTP surface (the Vue dashboard SPA) over real HTTP, with a
tmp ``dist`` (index.html + static/app.js) pointed at via the
``FINELOG_DASHBOARD_DIST`` env override. Assert:

  * ``GET /`` -> 200 text/html serving index.html,
  * ``GET /unknown/route`` -> 200 (the SPA fallback also serves index.html),
  * ``GET /static/app.js`` -> 200 with the written asset bytes,
  * ``GET /`` with ``X-Forwarded-Prefix: /p`` rewrites ``<base href="/">`` to
    ``<base href="/p/">`` (the byte-exact reverse-proxy base-href rewrite),
  * an RPC POST (RegisterTable) still 200s — confirming the SPA GET routes do
    NOT shadow the connect fallback (route precedence).

Backend coverage. Only the Rust backend resolves its dist from
``FINELOG_DASHBOARD_DIST`` (the Phase-5 design added the env override to the Rust
shell). The Python ASGI shell's SPA dist resolution predates that knob (fixed
in-repo / ``/app/dashboard/dist`` path), so the dist-serving assertions skip on
the Python backend rather than asserting against a non-feature — keeping the
contract honest. The byte-exact base-href rewrite is additionally covered
identically on both sides by unit tests (Python ``test_asgi.py``,
Rust ``server::spa::base_href_rewrite``). The RPC-route-precedence assertion runs
on BOTH backends (it does not depend on the dist).
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

from tests.parity.conftest import Backend

pytestmark = pytest.mark.timeout(60)

_INDEX_HTML = b'<html><head><base href="/" /></head><body><div id="app"></div></body></html>'
_APP_JS = b'console.log("finelog dashboard");\n'

# Only the Rust shell reads FINELOG_DASHBOARD_DIST; the dist-serving assertions
# are Rust-only (see the module docstring).
_DIST_ENV_BACKENDS = {"rust"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(0.05)
    raise TimeoutError(f"{base_url}/health did not come up within {timeout}s: {last_err}")


def _make_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "static").mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_bytes(_INDEX_HTML)
    (dist / "static" / "app.js").write_bytes(_APP_JS)
    return dist


@pytest.fixture
def spa_server(server_backend: Backend, tmp_path: Path) -> Iterator[str]:
    """Spawn the backend with FINELOG_DASHBOARD_DIST pointed at a tmp dist."""
    dist = _make_dist(tmp_path)
    port = _free_port()
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["FINELOG_DASHBOARD_DIST"] = str(dist)
    proc = subprocess.Popen(
        server_backend.command(port=port, log_dir=log_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        _wait_for_health(base_url, proc, timeout=20.0)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def _skip_if_no_dist_env(backend: Backend) -> None:
    if backend.name not in _DIST_ENV_BACKENDS:
        pytest.skip(f"{backend.name} backend does not resolve dist from FINELOG_DASHBOARD_DIST")


def test_spa_index_served(spa_server: str, server_backend: Backend) -> None:
    _skip_if_no_dist_env(server_backend)
    resp = httpx.get(f"{spa_server}/", timeout=5.0)
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert b'id="app"' in resp.content


def test_spa_fallback_serves_index(spa_server: str, server_backend: Backend) -> None:
    _skip_if_no_dist_env(server_backend)
    # An unknown GET route falls through to the SPA index (Vue Router takes over
    # client-side), NOT to the connect 404.
    resp = httpx.get(f"{spa_server}/some/client/route", timeout=5.0)
    assert resp.status_code == 200
    assert b'id="app"' in resp.content


def test_spa_static_asset_served(spa_server: str, server_backend: Backend) -> None:
    _skip_if_no_dist_env(server_backend)
    resp = httpx.get(f"{spa_server}/static/app.js", timeout=5.0)
    assert resp.status_code == 200
    assert resp.content == _APP_JS


def test_spa_base_href_rewrite_via_forwarded_prefix(spa_server: str, server_backend: Backend) -> None:
    _skip_if_no_dist_env(server_backend)
    resp = httpx.get(f"{spa_server}/", headers={"X-Forwarded-Prefix": "/p"}, timeout=5.0)
    assert resp.status_code == 200
    # The rewrite normalizes the prefix to leading+trailing slash and replaces
    # the first <base href="/">.
    assert b'<base href="/p/"' in resp.content
    assert b'<base href="/"' not in resp.content


def test_rpc_post_still_reaches_connect(spa_server: str, server_backend: Backend) -> None:
    # Runs on BOTH backends: the SPA GET routes must not shadow the connect
    # fallback, so an RPC POST still reaches the StatsService. (RegisterTable is
    # the cheapest write-free probe and is independent of the SPA dist.)
    client = StatsServiceClientSync(address=spa_server)
    resp = client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace="iris.worker",
            schema=stats_pb2.Schema(
                columns=[
                    stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
                    stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
                ],
            ),
        )
    )
    names = [c.name for c in resp.effective_schema.columns]
    assert "worker_id" in names and "timestamp_ms" in names
