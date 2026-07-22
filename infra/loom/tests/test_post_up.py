# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "post_up.py"
SPEC = importlib.util.spec_from_file_location("loom_post_up", MODULE_PATH)
assert SPEC and SPEC.loader
post_up = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = post_up
SPEC.loader.exec_module(post_up)


class Response:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


def test_health_requires_exact_200_ok_response() -> None:
    with patch.object(post_up.urllib.request, "urlopen", return_value=Response(200, b"ok\n")):
        assert post_up.health_is_ready("loom.example.com")

    for status, body in [(200, b"login"), (204, b""), (404, b"not found")]:
        with patch.object(post_up.urllib.request, "urlopen", return_value=Response(status, body)):
            assert not post_up.health_is_ready("loom.example.com")


def test_health_rejects_caddy_upstream_errors() -> None:
    error = urllib.error.HTTPError("https://loom.example.com/api/health", 502, "bad gateway", {}, None)
    with patch.object(post_up.urllib.request, "urlopen", side_effect=error):
        assert not post_up.health_is_ready("loom.example.com")


def test_readiness_requires_ready_database_response() -> None:
    with patch.object(
        post_up.urllib.request,
        "urlopen",
        return_value=Response(200, b'{"status":"ready","database":true,"migrations":[]}'),
    ):
        assert post_up.readiness_is_ready("loom.example.com")

    for body in (
        b'{"status":"not_ready","database":true}',
        b'{"status":"ready","database":false}',
        b"[]",
        b"not-json",
    ):
        with patch.object(post_up.urllib.request, "urlopen", return_value=Response(200, body)):
            assert not post_up.readiness_is_ready("loom.example.com")
