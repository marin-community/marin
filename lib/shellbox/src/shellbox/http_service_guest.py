# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-library HTTP listener copied into a machine by http_service."""

import base64
import json
import queue
import select
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 4 * 1024 * 1024
MAX_PENDING = 16
REQUEST_TIMEOUT = 600
CONTROL = "/__shellbox_http__/"


class Pending:
    def __init__(self):
        self.ready = threading.Event()
        self.response = None


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port):
        self.pending = {}
        self.lock = threading.Lock()
        self.requests = queue.Queue(MAX_PENDING)
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, message, *args):
        pass

    def send_body(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_request(self):
        self.connection.settimeout(REQUEST_TIMEOUT)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_body(400, b'{"error":"invalid content length"}')
            return
        if not 0 <= length <= MAX_BODY * 2 or self.headers.get("Transfer-Encoding"):
            self.send_body(413, b'{"error":"request too large or chunked"}')
            return
        body = self.rfile.read(length)
        if self.path == CONTROL + "ready":
            self.send_body(200, b"{}")
            return
        if self.path == CONTROL + "poll":
            try:
                request = self.server.requests.get(timeout=1)
            except queue.Empty:
                request = None
            self.send_body(200, json.dumps(request).encode())
            return
        if self.path == CONTROL + "respond":
            response = json.loads(body)
            with self.server.lock:
                pending = self.server.pending.get(response["id"])
                if pending is not None:
                    pending.response = response
                    pending.ready.set()
            self.send_body(200, b"{}")
            return
        if self.path == CONTROL + "stop":
            self.send_body(200, b"{}")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path.startswith(CONTROL) or length > MAX_BODY or len(self.path) > 4096:
            self.send_body(400, b'{"error":"unsupported request"}')
            return
        request_id = uuid.uuid4().hex
        pending = Pending()
        with self.server.lock:
            if len(self.server.pending) >= MAX_PENDING:
                self.send_body(503, b'{"error":"service busy"}')
                return
            self.server.pending[request_id] = pending
        try:
            try:
                self.server.requests.put_nowait(
                    {
                        "id": request_id,
                        "method": self.command,
                        "path": self.path,
                        "body": base64.b64encode(body).decode(),
                    }
                )
            except queue.Full:
                self.send_body(503, b'{"error":"service busy"}')
                return
            if not pending.ready.wait(REQUEST_TIMEOUT):
                self.send_body(504, b'{"error":"service timed out"}')
                return
            response = pending.response
            self.send_body(response["status"], base64.b64decode(response["body"]), response["content_type"])
        finally:
            with self.server.lock:
                del self.server.pending[request_id]

    do_GET = handle_request
    do_POST = handle_request


def start(port):
    process = subprocess.Popen(
        [sys.executable, __file__, "serve", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 10)
        if not ready or process.stdout.readline() != b"ready\n":
            raise RuntimeError("HTTP listener failed to bind its port")
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


if __name__ == "__main__":
    if sys.argv[1] == "start":
        start(int(sys.argv[2]))
    else:
        with Server(int(sys.argv[2])) as server:
            print("ready", flush=True)
            server.serve_forever(poll_interval=0.01)
