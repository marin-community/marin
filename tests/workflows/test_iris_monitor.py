# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import socket
import sys
import textwrap
import urllib.request
from pathlib import Path

from scripts.workflows.iris_monitor import open_supervised_tunnel, terminate_pid


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_supervised_tunnel_uses_file_backed_logs(tmp_path: Path) -> None:
    port = _free_port()
    log_path = tmp_path / "port-forward.log"
    server_path = tmp_path / "server.py"
    server_path.write_text(
        textwrap.dedent(
            """
            import http.server
            import socketserver
            import sys

            port = int(sys.argv[1])
            sys.stdout.write("x" * 2_000_000)
            sys.stdout.flush()

            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path != "/health":
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def log_message(self, format, *args):
                    return

            class Server(socketserver.TCPServer):
                allow_reuse_address = True

            with Server(("127.0.0.1", port), Handler) as httpd:
                httpd.serve_forever()
            """
        )
    )

    url, pid = open_supervised_tunnel(
        [sys.executable, str(server_path), str(port)],
        url=f"http://127.0.0.1:{port}",
        health_path="/health",
        timeout=5.0,
        poll_interval=0.05,
        log_path=log_path,
        restart_interval=0.1,
    )
    try:
        with urllib.request.urlopen(url + "/health", timeout=1.0) as response:
            assert response.status == 200
            assert response.read() == b"ok"
        assert log_path.stat().st_size > 1_000_000
    finally:
        terminate_pid(pid)
