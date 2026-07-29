# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client and lazy launcher for Echo's user-local file-search daemon."""

import json
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import file_search
from search_result import SearchResult

DAEMON_SCRIPT = Path(__file__).with_name("local_search_daemon.py")
CACHE_ROOT = Path.home() / ".cache" / "echo"


@dataclass(frozen=True)
class LocalSearchResponse:
    results: tuple[SearchResult, ...]
    message: str | None


def daemon_request(root: Path, query: str, limit: int) -> object:
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(30)
        client.connect(str(CACHE_ROOT / file_search.DAEMON_SOCKET_NAME))
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
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with (CACHE_ROOT / "daemon.log").open("ab") as log:
        subprocess.Popen(
            [uv, "run", "--script", str(DAEMON_SCRIPT), "--warm-root", str(root)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def search(query: str, limit: int, root: Path | None = None) -> LocalSearchResponse:
    resolved_root = file_search.repository_root(root)
    try:
        value = daemon_request(resolved_root, query, limit)
    except (TimeoutError, ConnectionError, FileNotFoundError, json.JSONDecodeError):
        start_daemon(resolved_root)
        return LocalSearchResponse((), "local file search is starting; its first index builds in the background")
    if not isinstance(value, dict):
        raise ValueError("local search daemon response must be an object")
    status = value.get("status")
    if status == "ready":
        raw_results = value.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("local search daemon results must be a list")
        return LocalSearchResponse(tuple(SearchResult.from_json(result) for result in raw_results), None)
    message = str(value.get("message", "local file search is warming"))
    return LocalSearchResponse((), f"local file search: {message}")
