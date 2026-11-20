# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise task-local cache lifetimes through the serving subprocess boundary."""

import contextlib
import errno
import json
import os
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import psutil
import pytest
import requests
from iris.client.context_state import reset_context, set_context
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import JobName
from marin.inference.backend import ModelSpec
from marin.inference.config import VllmCompilationCacheMode, VllmEngineConfig, VllmRouterConfig
from marin.inference.vllm_backend import VllmBackend
from marin.inference.vllm_router import available_port
from rigging.timing import Duration, ExponentialBackoff

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Serving uses POSIX process groups")
_EVENTS_ENV = "CACHE_TEST_EVENTS"

_FAKE_SERVER = """\
import json
import os
import signal
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

kind = "lmcache" if Path(sys.argv[0]).name == "python" else Path(sys.argv[0]).name

def record(event, **fields):
    with Path(os.environ[EVENTS_ENV]).open("a") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\\n")

def stop(*_):
    record(kind + "-stop")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
if kind == "lmcache" and os.environ.get("CACHE_TEST_ORPHAN"):
    read_fd, write_fd = os.pipe()
    if os.fork():
        os.close(write_fd)
        os.read(read_fd, 1)
        raise SystemExit(19)
    os.close(read_fd)
    record("orphan-start")
    os.write(write_fd, b"1")
    os.close(write_fd)
    threading.Event().wait(600)

flag = "--http-port" if kind == "lmcache" else "--port"
port = int(sys.argv[sys.argv.index(flag) + 1])
fields = {}
if kind == "lmcache":
    transport = socket.socket()
    transport.bind(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])))
    transport.listen(1)
    fields["ram_gib"] = float(sys.argv[sys.argv.index("--l1-size-gb") + 1])
record(kind + "-start", **fields)
if kind == "vllm" and os.environ.get("CACHE_TEST_VLLM_FAIL"):
    raise SystemExit(23)
if kind == "vllm" and os.environ.get("CACHE_TEST_VLLM_HANG"):
    threading.Event().wait(600)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path == "/v1/models":
            self.wfile.write(b'{"data":[{"id":"fake-model"}]}')
    def log_message(self, *_):
        pass

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def cache_processes(tmp_path, monkeypatch):
    for name in ("vllm", "python"):
        path = tmp_path / name
        path.write_text(f"#!{sys.executable}\nEVENTS_ENV = {_EVENTS_ENV!r}\n{_FAKE_SERVER}")
        path.chmod(0o755)
    events = tmp_path / "events.jsonl"
    events.touch()
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv(_EVENTS_ENV, str(events))
    yield events
    # A failing assertion must not leave this test's fake services running.
    for row in _events(events):
        with contextlib.suppress(psutil.NoSuchProcess):
            process = psutil.Process(row.pid)
            if any(str(tmp_path) in arg for arg in process.cmdline()):
                process.kill()


@dataclass(frozen=True)
class _ProcessEvent:
    event: str
    pid: int
    ram_gib: float | None = None


def _events(path: Path) -> list[_ProcessEvent]:
    return [_ProcessEvent(**json.loads(line)) for line in path.read_text().splitlines()]


def _live(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait(predicate, timeout: float = 10) -> None:
    ExponentialBackoff(initial=0.01, maximum=0.1).wait_until_or_raise(
        predicate,
        timeout=Duration.from_seconds(timeout),
        error_message="Fake serving process did not reach expected state",
    )


def _backend() -> VllmBackend:
    return VllmBackend(
        VllmEngineConfig(
            lmcache_ram_gib=1.5,
            compilation_cache=VllmCompilationCacheMode.CALLER_MANAGED,
            startup_timeout_seconds=5,
        )
    )


def _spec() -> ModelSpec:
    return ModelSpec(
        weights="fake-model",
        api_model="fake-model",
        num_chips=1,
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=1024,
        chat_template_content="{{ messages }}",
    )


def test_default_backend_runs_without_a_cache_process(cache_processes):
    backend = VllmBackend(
        VllmEngineConfig(
            compilation_cache=VllmCompilationCacheMode.CALLER_MANAGED,
            startup_timeout_seconds=5,
        )
    )
    with backend.start(_spec()) as environment:
        environment.wait_until_ready(poll_interval_seconds=0.01)
        assert requests.get(environment.server_url + "/models", timeout=2).json()["data"][0]["id"] == "fake-model"
    rows = _events(cache_processes)
    assert [row.event for row in rows] == ["vllm-start", "vllm-stop"]
    assert all(not _live(row.pid) for row in rows)


@pytest.mark.parametrize("startup_failure", [False, True])
def test_backend_owns_cache_until_vllm_stops(cache_processes, monkeypatch, startup_failure):
    if startup_failure:
        monkeypatch.setenv("CACHE_TEST_VLLM_FAIL", "1")
    expectation = pytest.raises(RuntimeError) if startup_failure else contextlib.nullcontext()
    with expectation:
        with _backend().start(_spec()) as environment:
            environment.wait_until_ready(poll_interval_seconds=0.01)
            assert requests.get(environment.server_url + "/models", timeout=2).json()["data"][0]["id"] == "fake-model"
    rows = _events(cache_processes)
    names = [row.event for row in rows]
    assert names.index("lmcache-start") < names.index("vllm-start") < names.index("lmcache-stop")
    if not startup_failure:
        assert names.index("vllm-stop") < names.index("lmcache-stop")
    assert rows[0].ram_gib == 1.5
    assert all(not _live(row.pid) for row in rows)


def test_cli_sigterm_during_vllm_startup_stops_both_services(cache_processes, monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_TEST_VLLM_HANG", "1")
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    command = [
        sys.executable,
        "-c",
        "from marin.inference.serve_cli import main; main()",
        "local",
        "fake-model",
        "--chat-template",
        str(template),
        "--lmcache-ram-gib",
        "1",
        "--startup-timeout",
        "30",
    ]
    with (tmp_path / "owner.log").open("w") as log:
        owner = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        _wait(lambda: any(row.event == "vllm-start" for row in _events(cache_processes)), timeout=30)
        owner.send_signal(signal.SIGTERM)
        assert owner.wait(timeout=15) == 128 + signal.SIGTERM
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
    rows = _events(cache_processes)
    names = [row.event for row in rows]
    assert names.index("vllm-stop") < names.index("lmcache-stop")
    assert all(not _live(row.pid) for row in rows)


def test_cache_startup_failure_releases_orphaned_server_child(cache_processes, monkeypatch):
    monkeypatch.setenv("CACHE_TEST_ORPHAN", "1")
    with pytest.raises(RuntimeError):
        with _backend().start(_spec()):
            pytest.fail("An exited cache parent must fail startup")
    rows = _events(cache_processes)
    assert any(row.event == "orphan-start" for row in rows)
    _wait(lambda: all(not _live(row.pid) for row in rows))
    assert not any(row.event == "vllm-start" for row in rows)


def test_cache_death_is_reported_and_releases_vllm(cache_processes):
    with _backend().start(_spec()) as environment:
        environment.wait_until_ready(poll_interval_seconds=0.01)
        cache_pid = next(row.pid for row in _events(cache_processes) if row.event == "lmcache-start")
        os.kill(cache_pid, signal.SIGTERM)
        _wait(lambda: not _live(cache_pid))
        with pytest.raises(RuntimeError):
            environment.check_alive()
    rows = _events(cache_processes)
    assert any(row.event == "vllm-stop" for row in rows)
    assert all(not _live(row.pid) for row in rows)


@pytest.mark.parametrize("reuse_closed_ports", [False, True])
def test_colocated_backends_keep_separate_cache_servers_alive(cache_processes, monkeypatch, reuse_closed_ports):
    if reuse_closed_ports:
        # Reproduce bind(0) reusing a recently closed port. Keep real sockets,
        # so the fake cache, vLLM and router still have to bind and serve HTTP.
        recent_ports = []

        class ReusingSocket(socket.socket):
            def bind(self, address):
                if address[1] != 0:
                    return super().bind(address)
                for port in recent_ports:
                    try:
                        super().bind((address[0], port))
                    except OSError as error:
                        if error.errno != errno.EADDRINUSE:
                            raise
                    else:
                        return None
                super().bind(address)
                recent_ports.append(self.getsockname()[1])
                return None

        monkeypatch.setattr(socket, "socket", ReusingSocket)

    def backend():
        config = _backend().config
        if reuse_closed_ports:
            config = replace(config, router=VllmRouterConfig())
        return VllmBackend(config, port=available_port("127.0.0.1"))

    previous = get_job_info()
    context_token = set_context(None)
    try:
        set_job_info(JobInfo(task_id=JobName.from_wire("/test/cache-first/0")))
        with backend().start(_spec()) as first:
            first.wait_until_ready(poll_interval_seconds=0.01)
            set_job_info(JobInfo(task_id=JobName.from_wire("/test/cache-second/0")))
            with backend().start(_spec()) as second:
                second.wait_until_ready(poll_interval_seconds=0.01)
                first.check_alive()
                second.check_alive()
                for environment in (first, second):
                    assert requests.get(environment.server_url + "/models", timeout=2).status_code == 200
                caches = [row for row in _events(cache_processes) if row.event == "lmcache-start"]
                assert len(caches) == 2
                assert all(_live(row.pid) for row in caches)
    finally:
        set_job_info(previous)
        reset_context(context_token)
    assert all(not _live(row.pid) for row in _events(cache_processes))
