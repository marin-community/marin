# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from iris.runtime.health_probe import main, probe_http_health


class _HealthHandler(BaseHTTPRequestHandler):
    status = 200
    redirected = False

    def do_GET(self) -> None:
        if self.path == "/redirected":
            type(self).redirected = True
            self.send_response(500)
        else:
            self.send_response(type(self).status)
            if type(self).status == 302:
                self.send_header("Location", "/redirected")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


@contextmanager
def _health_server(status: int):
    _HealthHandler.status = status
    _HealthHandler.redirected = False
    with ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            thread.join()


def test_probe_accepts_redirect_status_without_following_it():
    with _health_server(302) as port:
        result = probe_http_health(port, timeout=1)

    assert result.healthy
    assert result.detail == "HTTP 302"
    assert not _HealthHandler.redirected


def test_live_probe_writes_a_termination_reason_at_the_failure_threshold(monkeypatch, tmp_path):
    port_file = tmp_path / "port"
    failures_file = tmp_path / "failures"
    termination_file = tmp_path / "termination"
    monkeypatch.setenv("IRIS_HEALTH_PORT_FILE", str(port_file))
    monkeypatch.setenv("IRIS_HEALTH_FAILURE_COUNT_FILE", str(failures_file))
    monkeypatch.setenv("IRIS_HEALTH_TERMINATION_FILE", str(termination_file))

    with _health_server(503) as port:
        port_file.write_text(str(port))
        assert main(["--phase", "live", "--timeout", "1", "--failure-threshold", "2"]) == 1
        assert failures_file.read_text().strip() == "1"
        assert not termination_file.exists()

        assert main(["--phase", "live", "--timeout", "1", "--failure-threshold", "2"]) == 1

    assert failures_file.read_text().strip() == "2"
    assert "HTTP 503" in termination_file.read_text()

    with _health_server(200) as port:
        port_file.write_text(str(port))
        assert main(["--phase", "live", "--timeout", "1", "--failure-threshold", "2"]) == 0

    assert failures_file.read_text().strip() == "0"
    assert not termination_file.exists()


def test_startup_probe_does_not_write_a_termination_reason(monkeypatch, tmp_path):
    monkeypatch.setenv("IRIS_HEALTH_PORT_FILE", str(tmp_path / "missing-port"))
    monkeypatch.setenv("IRIS_HEALTH_FAILURE_COUNT_FILE", str(tmp_path / "failures"))
    termination_file = tmp_path / "termination"
    monkeypatch.setenv("IRIS_HEALTH_TERMINATION_FILE", str(termination_file))

    assert main(["--phase", "startup", "--timeout", "1"]) == 1
    assert not termination_file.exists()
