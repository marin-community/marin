# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client and lazy launcher for Echo's user-local file-search daemon."""

import json
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import file_search

DAEMON_SCRIPT = Path(__file__).with_name("local_search_daemon.py")


def daemon_request(root: Path, query: str, limit: int) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(30)
        client.connect(str(file_search.echo_cache_dir() / "search.sock"))
        client.sendall(
            json.dumps({"action": "search", "root": str(root), "query": query, "limit": limit}).encode() + b"\n"
        )
        response = bytearray()
        while not response.endswith(b"\n"):
            block = client.recv(64 * 1024)
            if not block:
                break
            response.extend(block)
    return json.loads(response)


def start_daemon(root: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to start local Echo file search")
    cache = file_search.echo_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "daemon.log").open("ab") as log:
        subprocess.Popen(
            [uv, "run", "--script", str(DAEMON_SCRIPT), "--warm-root", str(root)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def search(query: str, limit: int, root: Path | None = None) -> tuple[list[dict[str, object]], str | None]:
    resolved_root = file_search.repository_root(root)
    try:
        response = daemon_request(resolved_root, query, limit)
    except (TimeoutError, ConnectionError, FileNotFoundError, json.JSONDecodeError):
        start_daemon(resolved_root)
        return [], "local file search is starting; its first index builds in the background"
    status = response.get("status")
    if status == "ready":
        return response["results"], None
    message = response.get("message", "local file search is warming")
    return [], f"local file search: {message}"
