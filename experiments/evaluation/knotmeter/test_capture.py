# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise result capture across a real command, HTTP reset and filesystem writes."""

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from marin.profiling.trace_summary import sha256_for_path

from experiments.evaluation.knotmeter.artifacts import read_record, unpack
from experiments.evaluation.knotmeter.collect import validate_window
from experiments.evaluation.knotmeter.config import Stage, load_profile, resolved_series
from experiments.evaluation.knotmeter.run import run_point
from experiments.evaluation.knotmeter.workload import prepare_workload


@pytest.mark.parametrize("outcome", ["success", "shutdown-failure", "benchmark-failure"])
def test_capture_preserves_completion_and_original_failure(tmp_path, monkeypatch, outcome):
    fixture = Path(__file__).with_name("fixtures")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable = binaries / "uvx"
    executable.write_text(
        f"#!{sys.executable}\nimport runpy, sys\n"
        f"runpy.run_path({str(fixture)!r} + "
        "('/replay.py' if '--output-artifact-dir' in sys.argv else '/prepare.py'), run_name='__main__')\n"
    )
    executable.chmod(0o755)
    hardware = binaries / "nvidia-smi"
    hardware.write_text(f"#!{sys.executable}\nprint('\\n'.join(f'fixture-{{i}}' for i in range(8)))\n")
    hardware.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("AGENTX_FIXTURE_EVENTS", str(tmp_path / "events"))
    monkeypatch.setenv("CAPTURE_FIXTURE_EXIT", "9" if outcome == "benchmark-failure" else "0")
    for name, value in {
        "IRIS_TASK_ID": "/fixture/capture/serve/0",
        "IRIS_ATTEMPT_UID": "capture-attempt",
        "IRIS_BUNDLE_ID": "fixture-source",
    }.items():
        monkeypatch.setenv(name, value)
    profile = load_profile(Path("snowball-h100"), ["checks=[]"])
    series = resolved_series(profile, Stage.SCREEN)
    series_path = tmp_path / "series.json"
    series_path.write_text(json.dumps(series))
    workload = prepare_workload(tmp_path / "prepare", series_path=series_path, cache_root=str(tmp_path / "cache"))
    manifest = {
        "series": series,
        "placement": {},
        "workload": workload,
        "concurrency": series["concurrency"],
        "vllm_release": {"source_commit": "0" * 40},
        "diagnostic": False,
    }
    plan = {"manifest": manifest, "trial_uri": str(tmp_path / "retained")}
    info = JobInfo(task_id=JobName.from_string(os.environ["IRIS_TASK_ID"]), num_tasks=1, advertise_host="127.0.0.1")
    task = SimpleNamespace(
        task_id=info.task_id,
        state=TaskState.RUNNING,
        current_attempt_number=0,
        attempts=[SimpleNamespace(attempt_number=0, attempt_uid="capture-attempt")],
    )
    client = SimpleNamespace(list_tasks=lambda _: [task], task_status=lambda _: task)
    monkeypatch.setattr("experiments.evaluation.knotmeter.run.get_job_info", lambda: info)
    monkeypatch.setattr("experiments.evaluation.knotmeter.run.iris_ctx", lambda: SimpleNamespace(client=client))
    logs = tmp_path / "native-logs"
    logs.mkdir()
    (logs / "stderr.log").write_text("fixture server\n")
    receipt_uri = f"{plan['trial_uri']}/attempts/capture-attempt/hosts/0/receipt.json"
    checkpoints = []
    shutdown_error = RuntimeError("co-scheduled server shutdown failed")

    class ResetHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/reset_prefix_cache"
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    class Server:
        command = ("fixture-server",)
        model_id = profile.model.repo
        router = None
        vllm_server = SimpleNamespace(log_dir=logs)

        def __enter__(self):
            return self

        def wait_until_ready(self):
            pass

        def check_alive(self):
            pass

        def __exit__(self, *_args):
            if outcome != "benchmark-failure":
                checkpoint = read_record(receipt_uri)
                assert Path(checkpoint["archive_uri"]).is_file()
                checkpoints.append(checkpoint)
            if outcome == "shutdown-failure":
                # Also lose the logs: that secondary capture fault must not
                # replace the server shutdown error or discard the exports.
                shutil.rmtree(logs)
                raise shutdown_error

    with ThreadingHTTPServer(("127.0.0.1", 0), ResetHandler) as http:
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        server = Server()
        server.server_url = server.worker_server_url = f"http://127.0.0.1:{http.server_port}/v1"
        monkeypatch.setattr("experiments.evaluation.knotmeter.run.create_server", lambda *_: server)
        try:
            if outcome == "success":
                run_point(plan, tmp_path / "outputs/point", tmp_path / "restored/bundle", None)
            elif outcome == "shutdown-failure":
                with pytest.raises(RuntimeError) as stopped:
                    run_point(plan, tmp_path / "outputs/point", tmp_path / "restored/bundle", None)
                assert stopped.value is shutdown_error
            else:
                with pytest.raises(subprocess.CalledProcessError) as stopped:
                    run_point(plan, tmp_path / "outputs/point", tmp_path / "restored/bundle", None)
                assert stopped.value.returncode == 9
        finally:
            http.shutdown()
            thread.join()

    receipt = read_record(receipt_uri)
    assert sha256_for_path(Path(receipt["archive_uri"])) == receipt["archive_sha256"]
    unpack(Path(receipt["archive_uri"]), tmp_path / "recovered")
    record = read_record(str(tmp_path / "recovered/attempt.json"))
    assert record["capture_phase"] == receipt["capture_phase"] == "final"
    assert record["status"] == ("succeeded" if outcome == "success" else "failed")
    validity = validate_window(tmp_path / "recovered", manifest, record)
    assert validity["submission_valid"]
    assert validity["request_counts_by_phase"] == {
        "warmup": {"successes": 0, "errors": 1},
        "profiling": {"successes": 1, "errors": 0},
    }
    assert validity["measurement_completed"] == validity["protocol_valid"] == (outcome != "benchmark-failure")
    if outcome != "benchmark-failure":
        assert checkpoints[0]["capture_phase"] == "before-shutdown"
        assert checkpoints[0]["status"] == "measured"
    if outcome == "shutdown-failure":
        assert record["error"].endswith(str(shutdown_error))
        assert record["capture_errors"] and "shutdown_completed_unix" not in record
    elif outcome == "success":
        assert record["measured_unix"] <= record["shutdown_started_unix"] <= record["shutdown_completed_unix"]
    assert not (tmp_path / "outputs/host.tar.zst").exists()
