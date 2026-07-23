# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import threading
from pathlib import Path
from urllib.parse import urlencode
from wsgiref.util import setup_testing_defaults

import pytest

from infra.xprof.gateway import (
    ProfileCache,
    ProfileSourceError,
    ProfileStageManager,
    XprofGateway,
)


def _request(app, path: str, query: str = "", method: str = "GET"):
    environ = {}
    setup_testing_defaults(environ)
    environ.update(PATH_INFO=path, QUERY_STRING=query, REQUEST_METHOD=method)
    response = {}

    def start_response(status, headers, _exc_info=None):
        response["status"] = status
        response["headers"] = dict(headers)

    response["body"] = b"".join(app(environ, start_response))
    return response


class _BlockingStager:
    def __init__(self, local_path: Path):
        self.local_path = local_path
        self.started = threading.Event()
        self.release = threading.Event()

    def validate(self, uri: str) -> str:
        return uri

    def stage(self, uri: str) -> Path:
        self.started.set()
        self.release.wait(timeout=5)
        return self.local_path


def _xprof_app(environ, start_response):
    body = b"const api = '/data/plugin/profile/runs';"
    start_response("200 OK", [("Content-Type", "application/javascript"), ("Content-Length", str(len(body)))])
    return [body]


def _compressed_xprof_app(environ, start_response):
    body = gzip.compress(b"const api = '/data/plugin/profile/runs';")
    start_response(
        "200 OK",
        [
            ("Content-Type", "application/javascript"),
            ("Content-Encoding", "gzip"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def test_cache_accepts_any_gcs_or_s3_path(tmp_path):
    cache = ProfileCache(tmp_path / "cache")

    assert cache.validate("gs://other/checkpoints/run") == "gs://other/checkpoints/run"
    assert cache.validate("s3://another-bucket/profiles/run") == "s3://another-bucket/profiles/run"
    with pytest.raises(ProfileSourceError, match="must use"):
        cache.validate("https://example.com/profile")


def test_cache_returns_the_session_parent_xprof_expects(tmp_path):
    source = tmp_path / "source" / "run-1"
    session = source / "plugins" / "profile" / "session-1"
    session.mkdir(parents=True)
    (session / "host.xplane.pb").write_bytes(b"xplane")

    class LocalProfileCache(ProfileCache):
        def validate(self, uri: str) -> str:
            return uri

    cache = LocalProfileCache(tmp_path / "cache")

    source_uri = f"file://{source}"
    run_path = cache.stage(source_uri)
    assert run_path.name == "profile"
    assert run_path.parent.name == "plugins"
    assert (run_path / "session-1" / "host.xplane.pb").read_bytes() == b"xplane"
    assert cache.stage(source_uri) == run_path


def test_open_stages_outside_request_then_redirects_to_proxy_path(tmp_path):
    stager = _BlockingStager(tmp_path / "cached profile")
    manager = ProfileStageManager(stager, max_workers=1)
    app = XprofGateway(_xprof_app, manager, "/proxy/xprof")
    query = "uri=gs%3A%2F%2Fmarin-us-east5%2Ftmp%2Fttl%3D7d%2Fxprof%2Frun-1"
    try:
        pending = _request(app, "/open", query)
        assert pending["status"] == "202 Accepted"
        assert stager.started.wait(timeout=1)

        stager.release.set()
        manager.future("gs://marin-us-east5/tmp/ttl=7d/xprof/run-1").result(timeout=1)
        ready = _request(app, "/open", query)
        assert ready["status"] == "303 See Other"
        assert ready["headers"]["Location"] == f"./?{urlencode({'run_path': str(stager.local_path)})}"
    finally:
        app.shutdown()


def test_gateway_health_and_xprof_frontend_proxy_rewrite(tmp_path):
    stager = _BlockingStager(tmp_path)
    app = XprofGateway(_xprof_app, ProfileStageManager(stager), "/proxy/xprof")
    try:
        health = _request(app, "/healthz")
        assert health["status"] == "200 OK"
        assert health["body"] == b"ok\n"

        frontend = _request(app, "/bundle.js")
        assert frontend["status"] == "200 OK"
        assert frontend["body"] == b"const api = '/proxy/xprof/data/plugin/profile/runs';"
        assert frontend["headers"]["Content-Length"] == str(len(frontend["body"]))
    finally:
        app.shutdown()


def test_gateway_rewrites_compressed_xprof_assets(tmp_path):
    stager = _BlockingStager(tmp_path)
    app = XprofGateway(_compressed_xprof_app, ProfileStageManager(stager), "/proxy/xprof")
    try:
        frontend = _request(app, "/bundle.js")
        assert frontend["headers"]["Content-Encoding"] == "gzip"
        assert gzip.decompress(frontend["body"]) == b"const api = '/proxy/xprof/data/plugin/profile/runs';"
        assert frontend["headers"]["Content-Length"] == str(len(frontend["body"]))
    finally:
        app.shutdown()
