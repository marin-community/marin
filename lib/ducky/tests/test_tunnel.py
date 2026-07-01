# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shlex
import sys

import click
import pytest
from ducky.tunnel import cluster_tunnel


def _fake_iris(code: str) -> str:
    """A DUCKY_IRIS_CMD that runs `python -c <code>` instead of the real iris CLI."""
    return f"{sys.executable} -c {shlex.quote(code)}"


def test_cluster_tunnel_yields_raw_controller_url(monkeypatch):
    # fake iris prints the tunnel line (flushed, as a pipe is block-buffered), then blocks
    monkeypatch.setenv(
        "DUCKY_IRIS_CMD", _fake_iris("import time; print('http://127.0.0.1:9999', flush=True); time.sleep(30)")
    )
    with cluster_tunnel("marin") as url:
        assert url == "http://127.0.0.1:9999"  # raw, no /proxy suffix


def test_cluster_tunnel_raises_when_no_url(monkeypatch):
    # fake iris exits without ever printing a tunnel URL
    monkeypatch.setenv("DUCKY_IRIS_CMD", _fake_iris("pass"))
    with pytest.raises(click.ClickException, match="tunnel"):
        with cluster_tunnel("marin", timeout=5):
            pass
