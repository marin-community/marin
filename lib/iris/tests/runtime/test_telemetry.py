# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import zstandard
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.client.job_info import JobInfo, set_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.hooks.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.resources.names import JobName
from iris.runtime import telemetry
from rigging import telemetry as rigging_telemetry


class _FakeClient:
    def __init__(self, endpoint: str, *, malformed: bool = False) -> None:
        self.endpoint = endpoint
        self.malformed = malformed

    def resolve_endpoint(self, name: str) -> str:
        assert name == LOG_SERVER_ENDPOINT_NAME
        if self.malformed:
            raise ValueError("malformed Iris metadata")
        return self.endpoint


class TelemetryReceiver:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.received = threading.Event()

        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "zstd":
                    body = zstandard.ZstdDecompressor().decompress(body)
                receiver.payloads.append(json.loads(body))
                receiver.received.set()
                batch_id = self.headers["Idempotency-Key"]
                response = json.dumps({"batch_id": batch_id, "status": "accepted"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format: str, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/"

    def latest_resource(self) -> dict[str, Any]:
        assert self.received.wait(timeout=5), "telemetry receiver did not receive a batch"
        return self.payloads[-1]["resource"]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    rigging_telemetry.shutdown(0.01)
    set_job_info(None)
    yield
    rigging_telemetry.shutdown(0.1)
    set_job_info(None)


@pytest.fixture
def telemetry_receiver() -> Iterator[TelemetryReceiver]:
    receiver = TelemetryReceiver()
    yield receiver
    receiver.close()


@contextmanager
def _iris_metadata(info: JobInfo, endpoint: str, *, malformed_endpoint: bool = False) -> Iterator[None]:
    set_job_info(info)
    client = IrisClient(_FakeClient(endpoint, malformed=malformed_endpoint))
    with iris_ctx_scope(IrisContext.from_job_info(info, client=client)):
        yield


def test_configure_stamps_canonical_resource_identity(
    monkeypatch: pytest.MonkeyPatch, telemetry_receiver: TelemetryReceiver
) -> None:
    monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")
    monkeypatch.setenv("IRIS_NODE_NAME", "g83d142")
    info = JobInfo(
        task_id=JobName.from_wire("/alice/train/worker/3"),
        worker_id="w-7",
        attempt_id=1,
        worker_region="us-east5",
    )
    with _iris_metadata(info, telemetry_receiver.endpoint):
        telemetry.configure(
            "levanter",
            root_run_uid="pretrain-42",
            attributes={"model_revision": "abc123", "role": rigging_telemetry.TelemetryRole.TRAINER.value},
        )
        rigging_telemetry.gauge("identity_probe").set(1)

    assert telemetry_receiver.latest_resource() == {
        "service": "levanter",
        "attributes": {
            "job_id": "/alice/train/worker",
            "task_id": "/alice/train/worker/3",
            "attempt": "1",
            "worker": "w-7",
            "region": "us-east5",
            "process_index": "2",
            "node_name": "g83d142",
            "model_revision": "abc123",
            "role": "trainer",
            "root_run_uid": "pretrain-42",
            "execution_uid": "iris:/alice/train/worker/3:attempt:1",
        },
    }


def test_vllm_resource_exposes_serving_job_join(telemetry_receiver: TelemetryReceiver) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/serve/0"), worker_id="w-1", attempt_id=0)
    with _iris_metadata(info, telemetry_receiver.endpoint):
        telemetry.configure("vllm", attributes={"role": rigging_telemetry.TelemetryRole.INFERENCE.value})
        rigging_telemetry.gauge("identity_probe").set(1)

    attributes = telemetry_receiver.latest_resource()["attributes"]
    assert attributes["serving_job_id"] == "/alice/serve"
    assert attributes["root_run_uid"] == "/alice/serve"


def test_configure_stamps_explicit_distributed_process_index(telemetry_receiver: TelemetryReceiver) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/train/7"), worker_id="w-1", attempt_id=2)
    with _iris_metadata(info, telemetry_receiver.endpoint):
        telemetry.configure("levanter", root_run_uid="run-42", process_index=0)
        rigging_telemetry.gauge("identity_probe").set(1)

    attributes = telemetry_receiver.latest_resource()["attributes"]
    assert attributes["task_id"] == "/alice/train/7"
    assert attributes["attempt"] == "2"
    assert attributes["root_run_uid"] == "run-42"
    assert attributes["process_index"] == "0"


@pytest.mark.parametrize("ambiguous_name", ["run", "run_id"])
def test_configure_rejects_ambiguous_identity(ambiguous_name: str) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"), worker_id="w-1", attempt_id=0)
    with _iris_metadata(info, "http://127.0.0.1:1/"):
        telemetry.configure("levanter", attributes={ambiguous_name: "ambiguous"})
        rigging_telemetry.gauge("identity_probe").set(1)

    assert rigging_telemetry.runtime_status().configured is False


@pytest.mark.parametrize("source", ["job_env", "endpoint"])
def test_configure_contains_malformed_iris_metadata(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    info = JobInfo(task_id=JobName.from_wire("/alice/train/0"), worker_id="w-1", attempt_id=0)
    if source == "job_env":
        monkeypatch.setenv("IRIS_TASK_ID", "not-a-task-id")
        # Malformed optional task metadata leaves telemetry inert and must not interrupt startup.
        telemetry.configure("levanter")
    else:
        with _iris_metadata(info, "http://127.0.0.1:1/", malformed_endpoint=True):
            telemetry.configure("levanter")

    assert rigging_telemetry.runtime_status().configured is False
