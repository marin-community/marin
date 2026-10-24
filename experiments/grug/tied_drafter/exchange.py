# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One outstanding asynchronous snapshot/proposal exchange between two jobs."""

import hmac
import io
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue

import numpy as np
import requests

MAX_PACKET_BYTES = 1024**3


def encode_packet(metadata: dict, arrays: list[np.ndarray]) -> bytes:
    """Encode numeric arrays and JSON metadata without executable serialization."""
    stream = io.BytesIO()
    np.savez(
        stream,
        metadata=np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8),
        **{f"array_{i}": array for i, array in enumerate(arrays)},
    )
    return stream.getvalue()


def decode_packet(payload: bytes) -> tuple[dict, list[np.ndarray]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
        arrays = [archive[f"array_{i}"] for i in range(len(archive.files) - 1)]
    return metadata, arrays


class Exchange:
    """Keep the hero thread independent of network waits and draft execution."""

    def __init__(self):
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.finished = threading.Event()
        self.snapshots = Queue(maxsize=1)
        self.proposals = Queue(maxsize=1)
        self.generation = None
        self.lock = threading.Lock()

    def offer(self, generation: int, payload: bytes):
        with self.lock:
            if self.generation is not None or self.closed.is_set():
                raise RuntimeError("Cannot overwrite an outstanding snapshot")
            self.generation = generation
            self.snapshots.put_nowait(payload)

    def receive(self, generation: int, payload: bytes):
        with self.lock:
            if generation != self.generation or self.closed.is_set():
                raise ValueError("Proposal does not match the outstanding snapshot")
            self.proposals.put_nowait(payload)

    def poll(self) -> bytes | None:
        with self.lock:
            try:
                payload = self.proposals.get_nowait()
            except Empty:
                return None
            self.generation = None
            return payload


@contextmanager
def serve_exchange(exchange: Exchange, *, host: str, port: int, token: str):
    """Serve one private experiment connection; no network work runs on the hero thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message_format, *args):
            return

        def respond(self, status: int, body: bytes = b""):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self) -> bool:
            actual = self.headers.get("Authorization", "")
            if not hmac.compare_digest(actual, f"Bearer {token}"):
                self.respond(403)
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/status":
                self.respond(
                    200, json.dumps({"ready": exchange.ready.is_set(), "closed": exchange.closed.is_set()}).encode()
                )
                return
            if self.path != "/snapshot":
                self.respond(404)
                return
            if exchange.closed.is_set():
                self.respond(410)
                return
            try:
                snapshot = exchange.snapshots.get(timeout=20)
            except Empty:
                self.respond(204)
                return
            self.respond(200, snapshot)

        def do_POST(self):
            if not self.authorized():
                return
            if self.path == "/ready":
                exchange.ready.set()
                self.respond(200)
                return
            if self.path == "/done":
                exchange.finished.set()
                self.respond(200)
                return
            if self.path != "/proposal":
                self.respond(404)
                return
            count = int(self.headers["Content-Length"])
            if count < 1 or count > MAX_PACKET_BYTES:
                self.respond(413)
                return
            payload = self.rfile.read(count)
            if len(payload) != count:
                self.respond(400)
                return
            generation = int(self.headers["X-Generation"])
            exchange.receive(generation, payload)
            self.respond(200)

    with ThreadingHTTPServer((host, port), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            exchange.closed.set()
            server.shutdown()
            thread.join()


def exchange_session(token: str) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers["Authorization"] = f"Bearer {token}"
    return session
