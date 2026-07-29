#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fastembed>=0.8,<0.9",
# ]
# ///

"""User-local Echo file-search daemon with lazy indexing and cache cleanup."""

import argparse
import json
import os
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

import file_search
import search_config
from fastembed import TextEmbedding

REFRESH_INTERVAL = 60
CACHE_MAX_AGE = 30 * 24 * 60 * 60
CLEANUP_INTERVAL = 24 * 60 * 60


class SearchState:
    def __init__(self, warm_root: Path | None) -> None:
        self._lock = threading.Lock()
        self._model: TextEmbedding | None = None
        self._indexing: set[Path] = set()
        self._ready: set[Path] = set()
        self._refreshed_at: dict[Path, float] = {}
        self._warm_root = warm_root
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        model = TextEmbedding(
            search_config.EMBED_MODEL,
            cache_dir=str(file_search.echo_cache_dir() / "models"),
        )
        with self._lock:
            self._model = model
        if self._warm_root is not None:
            self._start_refresh(self._warm_root)

    def _embed(self, mode: str, texts: list[str]) -> list[list[float]]:
        with self._lock:
            model = self._model
        if model is None:
            raise RuntimeError("embedding model is not ready")
        values = model.query_embed(texts) if mode == "query" else model.passage_embed(texts)
        return [[float(value) for value in vector] for vector in values]

    def _start_refresh(self, root: Path) -> None:
        with self._lock:
            if self._model is None or root in self._indexing:
                return
            self._indexing.add(root)
        threading.Thread(target=self._refresh, args=(root,), daemon=True).start()

    def _refresh(self, root: Path) -> None:
        try:
            with file_search.connect_index(file_search.cache_path(root)) as conn:
                file_search.refresh_index(root, conn, self._embed)
            with self._lock:
                self._ready.add(root)
                self._refreshed_at[root] = time.time()
        finally:
            with self._lock:
                self._indexing.discard(root)

    def search(self, root_value: str, query: str, limit: int) -> dict[str, Any]:
        root = file_search.repository_root(Path(root_value))
        now = time.time()
        with self._lock:
            model_ready = self._model is not None
            ready = root in self._ready
            refreshed_at = self._refreshed_at.get(root, 0.0)
        if not model_ready:
            return {"status": "warming", "message": "loading the local embedding model", "results": []}
        if not ready:
            self._start_refresh(root)
            return {"status": "warming", "message": f"indexing tracked files in {root}", "results": []}
        if now - refreshed_at >= REFRESH_INTERVAL:
            self._start_refresh(root)
        query_vector = self._embed("query", [query])[0]
        index_path = file_search.cache_path(root)
        results = file_search.search_index(query, query_vector, limit, index_path)
        os.utime(index_path)
        return {"status": "ready", "results": results}


class SearchHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline())
            if request.get("action") != "search":
                raise ValueError("unknown action")
            assert isinstance(self.server, SearchServer)
            response = self.server.state.search(
                str(request["root"]),
                str(request["query"]),
                int(request["limit"]),
            )
        except Exception as error:
            response = {"status": "error", "message": str(error), "results": []}
        self.wfile.write(json.dumps(response).encode() + b"\n")


class SearchServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: Path, state: SearchState) -> None:
        self.state = state
        super().__init__(str(socket_path), SearchHandler)


def remove_stale_indexes() -> None:
    repository_cache = file_search.echo_cache_dir() / "repositories"
    if not repository_cache.exists():
        return
    cutoff = time.time() - CACHE_MAX_AGE
    for path in repository_cache.glob("*.sqlite3*"):
        if path.stat().st_mtime < cutoff:
            path.unlink()


def cleanup_loop() -> None:
    while True:
        remove_stale_indexes()
        time.sleep(CLEANUP_INTERVAL)


def socket_path() -> Path:
    return file_search.echo_cache_dir() / "search.sock"


def serve(warm_root: Path | None) -> None:
    cache = file_search.echo_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    endpoint = socket_path()
    if endpoint.exists():
        try:
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(0.2)
                client.connect(str(endpoint))
            raise SystemExit("local Echo search daemon is already running")
        except OSError:
            endpoint.unlink()
    remove_stale_indexes()
    state = SearchState(warm_root)
    with SearchServer(endpoint, state) as server:
        endpoint.chmod(0o600)
        threading.Thread(target=cleanup_loop, daemon=True).start()
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-root", type=Path)
    args = parser.parse_args()
    serve(args.warm_root)
