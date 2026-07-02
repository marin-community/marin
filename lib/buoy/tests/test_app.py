# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve-layer behavior over a pre-populated cache: read API, async contract, proxy."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from buoy.app import _index_html, build_app
from buoy.mirror import RunRef, mirror_run
from fakes import FakeArtifact, FakeRun
from starlette.testclient import TestClient

REF = RunRef("marin-community", "marin_moe", "run-1")
Q = {"entity": REF.entity, "project": REF.project, "run": REF.run_id}


def _populate(cfg, patch_wandb, profile_logdir=None):
    arts = [FakeArtifact("jax_profile", "prof:v0", profile_logdir)] if profile_logdir else []
    run = FakeRun(
        name="r1",
        summary_dict={"train/loss": 0.5, "optim/learning_rate": 1e-3},
        rows=[{"_step": i, "train/loss": float(i), "optim/learning_rate": 1e-3} for i in range(6)],
        artifacts=arts,
    )
    patch_wandb(run)
    return mirror_run(cfg, REF)


def test_index_serves_dashboard_and_metrics(cfg, patch_wandb, tmp_path, monkeypatch):
    _populate(cfg, patch_wandb)
    dist = tmp_path / "dist"
    (dist / "static").mkdir(parents=True)
    (dist / "index.html").write_text('<!doctype html><base href="/"><title>buoy</title><div id="app"></div>')
    monkeypatch.setenv("BUOY_DASHBOARD_DIST", str(dist))
    with TestClient(build_app(cfg)) as client:
        home = client.get("/")
        assert home.status_code == 200 and "buoy" in home.text

        manifest = client.get("/api/manifest", params=Q).json()
        assert manifest["history"]["rows"] == 6

        body = client.get("/api/metrics", params={**Q, "keys": "train/loss"}).json()
        series = body["metrics"]["train/loss"]  # columnar {x, y}
        assert series["x"] == [0, 1, 2, 3, 4, 5]
        assert series["y"] == [0, 1, 2, 3, 4, 5]


def test_index_html_rewrites_base_href(tmp_path):
    (tmp_path / "index.html").write_text('<html><head><base href="/"><title>buoy</title></head></html>')
    # Behind the controller proxy the base is rewritten so relative URLs resolve under it.
    assert '<base href="/proxy/buoy/"' in _index_html(tmp_path, "/proxy/buoy").body.decode()
    # Direct access leaves the base at root; a missing build is a 503, not a crash.
    assert '<base href="/"' in _index_html(tmp_path, "").body.decode()
    assert _index_html(tmp_path / "missing", "").status_code == 503


def test_manifest_404_when_absent(cfg):
    with TestClient(build_app(cfg)) as client:
        assert client.get("/api/manifest", params=Q).status_code == 404


def test_async_mirror_contract(cfg, patch_wandb):
    # POST returns 202 immediately; status flips to done once the background mirror lands.
    run = FakeRun(summary_dict={"train/loss": 1.0}, rows=[{"_step": 0, "train/loss": 1.0}])
    patch_wandb(run)
    with TestClient(build_app(cfg)) as client:
        resp = client.post("/api/mirror", json={"entity": REF.entity, "project": REF.project, "run_id": REF.run_id})
        assert resp.status_code == 202
        assert resp.json()["state"] == "running"

        deadline = time.monotonic() + 5
        state = "running"
        while state == "running" and time.monotonic() < deadline:
            state = client.get("/api/mirror_status", params=Q).json()["state"]
            time.sleep(0.05)
        assert state == "done"


def test_xprof_proxy_forwards_to_backend(cfg, patch_wandb, profile_logdir, monkeypatch):
    _populate(cfg, patch_wandb, profile_logdir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"XPROF-STUB:" + self.path.encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    with TestClient(build_app(cfg)) as client:
        # Stand in for the real xprof subprocess with our stub's port.
        monkeypatch.setattr(client.app.state.xprof, "ensure", lambda run_key, logdir: port)
        resp = client.get("/xprof/marin-community/marin_moe/run-1/data/plugin/profile/x", params={"a": "1"})
        assert resp.status_code == 200
        assert resp.content == b"XPROF-STUB:/data/plugin/profile/x?a=1"
    server.shutdown()


def test_xprof_proxy_409_without_profile(cfg, patch_wandb):
    _populate(cfg, patch_wandb)  # no profile artifact
    with TestClient(build_app(cfg)) as client:
        resp = client.get("/xprof/marin-community/marin_moe/run-1/data/plugin/profile/")
        assert resp.status_code == 409


def test_profile_prepare_contract(cfg, patch_wandb, profile_logdir, monkeypatch):
    _populate(cfg, patch_wandb, profile_logdir)
    with TestClient(build_app(cfg)) as client:
        # prepare runs xprof in the background; stub the actual launch.
        monkeypatch.setattr(client.app.state.xprof, "start_prepare", lambda run_key, logdir: None)
        monkeypatch.setattr(client.app.state.xprof, "prepare_status", lambda run_key: {"state": "ready"})
        resp = client.post(
            "/api/profile_prepare", json={"entity": REF.entity, "project": REF.project, "run_id": REF.run_id}
        )
        assert resp.status_code == 202
        assert client.get("/api/profile_status", params=Q).json()["state"] == "ready"


def test_profile_prepare_409_without_profile(cfg, patch_wandb):
    _populate(cfg, patch_wandb)
    with TestClient(build_app(cfg)) as client:
        resp = client.post(
            "/api/profile_prepare", json={"entity": REF.entity, "project": REF.project, "run_id": REF.run_id}
        )
        assert resp.status_code == 409
