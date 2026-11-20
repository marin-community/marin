"""Shared pytest fixtures for harbor tests."""

import asyncio
import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@dataclass
class TelemetryCapture:
    endpoint: str
    _batches: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    hold_terminal_for: str | None = None
    terminal_received: threading.Event = field(default_factory=threading.Event)
    release_terminal: threading.Event = field(default_factory=threading.Event)

    def accept(self, batch: dict) -> None:
        with self._lock:
            self._batches.append(batch)

    @property
    def records(self) -> list[tuple[dict, dict]]:
        with self._lock:
            return [
                (batch["resource"], record)
                for batch in self._batches
                for record in batch["records"]
            ]


@pytest.fixture
def telemetry_capture():
    import zstandard
    from rigging import telemetry

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "zstd":
                body = zstandard.ZstdDecompressor().decompress(body)
            batch = json.loads(body)
            capture.accept(batch)
            if any(
                record["name"] == "job_terminal"
                and record["attributes"]["harbor_job_uid"] == capture.hold_terminal_for
                for record in batch["records"]
            ):
                capture.terminal_received.set()
                assert capture.release_terminal.wait(5)
            response = json.dumps(
                {"batch_id": batch["batch_id"], "status": "accepted"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    telemetry.shutdown(0.1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    capture = TelemetryCapture(f"http://127.0.0.1:{server.server_port}/v1/telemetry")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield capture
    finally:
        capture.release_terminal.set()
        telemetry.shutdown(0.1)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def pytest_addoption(parser):
    parser.addoption(
        "--harbor-local-cgroup-parent",
        help="Delegated cgroup v2 subtree for real local Harbor tests",
    )


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use ProactorEventLoop on Windows to support asyncio subprocess.

    On Windows, the default SelectorEventLoop does not support subprocess
    operations. The ProactorEventLoop is required for asyncio.create_subprocess_exec
    and asyncio.create_subprocess_shell to work.
    """
    if sys.platform == "win32":
        return asyncio.WindowsProactorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    env = AsyncMock()
    env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    env.upload_file.return_value = None
    return env


@pytest.fixture
def mock_logs_dir(temp_dir):
    """Create a temporary logs directory."""
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir()
    return logs_dir


def run_validator_cli(trajectory_path: Path, extra_args: list[str] | None = None):
    """Run the trajectory validator CLI on a file.

    Args:
        trajectory_path: Path to the trajectory file to validate.
        extra_args: Optional list of additional CLI arguments.

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    cmd = [
        sys.executable,
        "-m",
        "harbor.utils.trajectory_validator",
        str(trajectory_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr
