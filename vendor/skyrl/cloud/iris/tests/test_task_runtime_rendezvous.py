import contextlib
import json
import platform
import signal
import socket
import threading
from types import SimpleNamespace

import pytest
import ray

from cloud.iris import task_runtime
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV
from cloud.iris.task_runtime import (
    DONE_FILENAME,
    RENDEZVOUS_FILENAME,
    RendezvousPayload,
    head_succeeded,
    validate_rendezvous_runtime,
    write_head_result,
    write_rendezvous,
)


def _head_payload() -> RendezvousPayload:
    return RendezvousPayload(
        gang_epoch="gang-epoch",
        head_ip="10.0.0.1",
        head_node="head-node",
        port=6379,
        num_tasks=2,
        python_version="3.12.13",
        ray_version="2.51.1",
        written_at=1.0,
    )


def test_rendezvous_publishes_head_runtime_identity(tmp_path):
    write_rendezvous(str(tmp_path), "10.0.0.1", 6379, "gang-epoch")

    payload = json.loads((tmp_path / RENDEZVOUS_FILENAME).read_text())
    assert payload["gang_epoch"] == "gang-epoch"
    assert payload["head_node"] == socket.gethostname()
    assert payload["python_version"] == platform.python_version()
    assert payload["ray_version"] == ray.__version__


def test_matching_rendezvous_runtime_is_accepted():
    head = _head_payload()

    validated = validate_rendezvous_runtime(
        head,
        worker_node="worker-node",
        python_version="3.12.13",
        ray_version="2.51.1",
    )

    assert validated == head


@pytest.mark.parametrize(
    ("python_version", "ray_version", "expected_versions"),
    [
        ("3.12.14", "2.51.1", ("Python 3.12.13", "Python 3.12.14")),
        ("3.12.13", "2.52.0", ("Ray 2.51.1", "Ray 2.52.0")),
    ],
)
def test_runtime_skew_names_both_nodes_and_versions(python_version, ray_version, expected_versions):
    with pytest.raises(RuntimeError) as error:
        validate_rendezvous_runtime(
            _head_payload(),
            worker_node="worker-node",
            python_version=python_version,
            ray_version=ray_version,
        )

    message = str(error.value)
    assert "head-node" in message
    assert "worker-node" in message
    for version in expected_versions:
        assert version in message


def test_head_result_only_completes_its_gang_epoch(tmp_path):
    write_head_result(str(tmp_path), "current-epoch")

    result = json.loads((tmp_path / DONE_FILENAME).read_text())
    assert result["gang_epoch"] == "current-epoch"
    assert result["outcome"] == "succeeded"
    assert head_succeeded(str(tmp_path), "current-epoch")
    assert not head_succeeded(str(tmp_path), "prior-epoch")


def test_unknown_head_result_outcome_fails_instead_of_parking_worker(tmp_path):
    (tmp_path / DONE_FILENAME).write_text(
        json.dumps({"gang_epoch": "gang-epoch", "outcome": "failed", "written_at": 1.0})
    )

    with pytest.raises(RuntimeError):
        head_succeeded(str(tmp_path), "gang-epoch")


def test_head_result_object_store_error_keeps_worker_parked(monkeypatch):
    class UnavailableStore:
        def exists(self, _path):
            raise OSError(16, "Service Unavailable")

    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (UnavailableStore(), "ray_head.done"))

    assert not head_succeeded("s3://bucket/rendezvous", "gang-epoch")


class _FakeRayLogSyncSession:
    def __init__(self, *_args):
        pass

    def start_periodic(self, _rendezvous_dir):
        return threading.Event()

    def sync_bounded(self, _reason):
        return None


def _worker_args(tmp_path):
    return SimpleNamespace(
        ray_log_dir=None,
        rendezvous_dir=str(tmp_path),
        rendezvous_timeout=1,
        ray_spill_backend=task_runtime.RaySpillBackend.LOCAL,
        ray_spill_dir=str(tmp_path / "spill"),
        cluster_join_timeout=1,
    )


def _isolate_worker_runtime(tmp_path, monkeypatch, payload):
    monkeypatch.setenv(DEBUG_ARTIFACT_DIR_ENV, str(tmp_path / "debug"))
    handlers = {}
    monkeypatch.setattr(task_runtime, "_rank", lambda: 1)
    monkeypatch.setattr(task_runtime, "_num_tasks", lambda: 2)
    monkeypatch.setattr(task_runtime, "_own_ip", lambda: "10.0.0.2")
    monkeypatch.setattr(task_runtime, "poll_rendezvous", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(
        task_runtime,
        "_runtime_versions",
        lambda: (payload.python_version, payload.ray_version),
    )
    monkeypatch.setattr(task_runtime, "ray_start_worker", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "wait_for_nodes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_runtime, "ray_stop", lambda: None)
    monkeypatch.setattr(task_runtime, "RayLogSyncSession", _FakeRayLogSyncSession)
    monkeypatch.setattr(
        task_runtime,
        "ray_metrics_telemetry",
        lambda *_args: contextlib.nullcontext(),
    )
    monkeypatch.setattr(task_runtime, "capture_termination_artifacts", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "sync_debug_artifacts", lambda *_args: None)
    monkeypatch.setattr(signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler))
    return handlers


def test_worker_exits_zero_after_current_head_succeeds(tmp_path, monkeypatch):
    payload = _head_payload()
    _isolate_worker_runtime(tmp_path, monkeypatch, payload)
    write_head_result(str(tmp_path), payload.gang_epoch)

    assert task_runtime.run_worker(_worker_args(tmp_path)) == 0


def test_worker_termination_without_current_head_success_is_nonzero(tmp_path, monkeypatch):
    payload = _head_payload()
    handlers = _isolate_worker_runtime(tmp_path, monkeypatch, payload)
    write_head_result(str(tmp_path), "prior-epoch")

    monkeypatch.setattr(
        task_runtime.time,
        "sleep",
        lambda _duration: handlers[signal.SIGTERM](signal.SIGTERM, None),
    )

    assert task_runtime.run_worker(_worker_args(tmp_path)) == 128 + signal.SIGTERM
