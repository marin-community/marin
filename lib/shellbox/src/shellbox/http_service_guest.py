# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library loopback HTTP listener carried by one foreground process."""

import base64
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 4 * 1024 * 1024
MAX_PENDING = 16
MAX_PATH = 4096
REQUEST_TIMEOUT = 600
WIRE_LIMIT = MAX_BODY * 2 + 16384


class Pending:
    def __init__(self):
        self.ready = threading.Event()
        self.response = None


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port):
        self.pending = {}
        self.lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(MAX_PENDING)
        super().__init__(("127.0.0.1", port), Handler)

    def emit(self, message):
        frame = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        if len(frame) > WIRE_LIMIT:
            raise ValueError("HTTP request frame exceeds its limit")
        with self.output_lock:
            sys.stdout.buffer.write(frame)
            sys.stdout.buffer.flush()

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, message, *args):
        pass

    def setup(self):
        self.request.settimeout(REQUEST_TIMEOUT)
        super().setup()

    def send_body(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_request(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_body(400, b'{"error":"invalid content length"}')
            return
        if not 0 <= length <= MAX_BODY * 2 or self.headers.get("Transfer-Encoding"):
            self.send_body(400, b'{"error":"unsupported request"}')
            return
        if length > MAX_BODY:
            # Drain a bounded oversized body before replying so a client can
            # receive the rejection while its HTTP library finishes sending.
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
            self.send_body(400, b'{"error":"unsupported request"}')
            return
        if len(self.path) > MAX_PATH:
            self.send_body(400, b'{"error":"unsupported request"}')
            return
        body = self.rfile.read(length)
        if len(body) != length:
            return
        request_id = uuid.uuid4().hex
        pending = Pending()
        with self.server.lock:
            self.server.pending[request_id] = pending
        try:
            self.server.emit(
                {
                    "type": "request",
                    "id": request_id,
                    "method": self.command,
                    "path": self.path,
                    "body": base64.urlsafe_b64encode(body).decode(),
                }
            )
            if not pending.ready.wait(REQUEST_TIMEOUT):
                self.send_body(504, b'{"error":"service timed out"}')
                return
            response = pending.response
            if response is None:
                self.send_body(502, b'{"error":"host service closed"}')
                return
            self.send_body(
                response["status"],
                base64.b64decode(response["body"], altchars=b"-_", validate=True),
                response["content_type"],
            )
        finally:
            with self.server.lock:
                del self.server.pending[request_id]

    do_GET = handle_request
    do_POST = handle_request


def receive_responses(server):
    try:
        while frame := sys.stdin.buffer.readline(WIRE_LIMIT + 1):
            if len(frame) > WIRE_LIMIT or not frame.endswith(b"\n"):
                raise ValueError("Invalid HTTP response frame")
            response = json.loads(frame)
            with server.lock:
                pending = server.pending.get(response["id"])
                if pending is not None:
                    pending.response = response
                    pending.ready.set()
    finally:
        with server.lock:
            for pending in server.pending.values():
                pending.ready.set()
        server.shutdown()


def serve(port):
    with Server(port) as server:
        server.emit({"type": "ready"})
        threading.Thread(target=receive_responses, args=(server,), daemon=True).start()
        server.serve_forever(poll_interval=0.05)


if __name__ == "__main__":
    serve(int(sys.argv[1]))
