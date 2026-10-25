# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

from experiments.weight_merging.rsi.levanter_267 import eval_watchdog


def test_dead_endpoint_stops_client(monkeypatch):
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: False)
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        eval_watchdog.run_watched(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            "unused",
            timeout=5,
            interval=0.01,
            failure_seconds=0.03,
        )


def test_recovered_endpoint_allows_client_completion(monkeypatch):
    health = iter([False, False, True])
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: next(health, True))
    assert (
        eval_watchdog.run_watched(
            [sys.executable, "-c", "import time; time.sleep(.1)"],
            os.environ.copy(),
            "unused",
            timeout=5,
            interval=0.01,
            failure_seconds=0.03,
        )
        == 0
    )


def test_runtime_bound_stops_client(monkeypatch):
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: True)
    with pytest.raises(TimeoutError, match="runtime bound"):
        eval_watchdog.run_watched(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            "unused",
            timeout=0.05,
            interval=0.01,
        )


def test_dead_endpoint_kills_term_ignoring_descendant(monkeypatch, tmp_path):
    pid_file = tmp_path / "descendant.pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = "import subprocess,sys,time; " f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: not pid_file.exists())
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        eval_watchdog.run_watched(
            [sys.executable, "-c", parent],
            os.environ.copy(),
            "unused",
            timeout=5,
            interval=0.02,
            failure_seconds=0.1,
        )
    pid = int(pid_file.read_text())
    # An orphaned zombie may await init's reap; it must not remain executable.
    status = Path(f"/proc/{pid}/stat")
    for _ in range(100):
        if not status.exists() or status.read_text().split()[2] == "Z":
            return
        time.sleep(0.01)
    pytest.fail("TERM-ignoring descendant survived process-group cleanup")


def test_terminal_gang_stops_client_despite_healthy_http(monkeypatch):
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: True)
    client = SimpleNamespace(
        list_tasks=lambda request, **kwargs: SimpleNamespace(
            tasks=[SimpleNamespace(state=job_pb2.TASK_STATE_WORKER_FAILED)]
        )
    )
    with pytest.raises(RuntimeError, match="terminal task"):
        eval_watchdog.run_watched(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            "unused",
            timeout=5,
            interval=0.01,
            check_gang=lambda: eval_watchdog.gang_healthy(client, JobName.from_wire("/owner/serving")),
        )


def test_gang_query_outage_is_unhealthy_without_leaking_error():
    def unavailable(*args, **kwargs):
        raise OSError("private-controller-url")

    assert not eval_watchdog.gang_healthy(SimpleNamespace(list_tasks=unavailable), JobName.from_wire("/owner/serving"))


@pytest.mark.parametrize(
    "states",
    [
        [job_pb2.TASK_STATE_RUNNING] * 3,
        [job_pb2.TASK_STATE_RUNNING] * 3 + [job_pb2.TASK_STATE_BUILDING],
    ],
)
def test_incomplete_or_nonrunning_gang_is_unhealthy(states):
    tasks = [SimpleNamespace(task_id=f"/owner/serving/{i}", state=s) for i, s in enumerate(states)]
    rpc = SimpleNamespace(list_tasks=lambda request, **kwargs: SimpleNamespace(tasks=tasks))
    assert not eval_watchdog.gang_healthy(rpc, JobName.from_wire("/owner/serving"))


def test_controller_deadline_bounds_watchdog_teardown(monkeypatch):
    class DelayedController(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(0.2)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedController)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rpc = ControllerServiceClientSync(address=f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(eval_watchdog, "models_healthy", lambda _: True)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            eval_watchdog.run_watched(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                os.environ.copy(),
                "unused",
                timeout=2,
                interval=0.01,
                failure_seconds=0.04,
                check_gang=lambda: eval_watchdog.gang_healthy(rpc, JobName.from_wire("/owner/serving"), timeout_ms=30),
            )
        assert time.monotonic() - started < 1
    finally:
        rpc.close()
        server.shutdown()
        server.server_close()
        thread.join()
