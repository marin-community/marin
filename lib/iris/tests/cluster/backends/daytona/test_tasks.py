# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Daytona provider runs each attempt in one sandbox and reports it to the controller."""

import base64
import io
import tarfile
import threading
import zipfile
from types import SimpleNamespace

import pytest
from daytona.v2 import (
    ConnectionError as ProviderConnectionError,
)
from daytona.v2 import (
    LimitExceededError,
    LogFrame,
    NotFoundError,
    Resources,
    StreamLogEofEvent,
    StreamLogFrameEvent,
    StreamLogStateEvent,
    WaitResult,
)
from iris.cluster.backends.daytona import tasks
from iris.cluster.backends.daytona.tasks import DaytonaTaskProvider, sandbox_resources
from iris.cluster.bundle import BundleStore
from iris.cluster.config import DaytonaProviderConfig
from iris.cluster.controller.backend import (
    BackendCapability,
    BackendDescriptor,
    BackendKind,
    DirectReconcileRequest,
    TaskTarget,
)
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.types import AttemptUid, JobName
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.test_util import wait_for_condition
from rigging.secrets import SECRETS_DIR
from rigging.timing import Duration

TASK = "/rav/job/0"
EXITED = WaitResult(reason="exited", exit_code=0)


def log_event(index: int, channel: str, data: bytes) -> StreamLogFrameEvent:
    # The API sends URL-safe base64 without padding.
    encoded = base64.urlsafe_b64encode(data).rstrip(b"=").decode()
    cursor = f"c{index}"
    return StreamLogFrameEvent(
        frame=LogFrame(cursor=cursor, channel=channel, data=encoded, timestamp_ms=index), cursor=cursor
    )


class FakeProcess:
    """A process whose log is ``events``; the stream breaks once after event ``break_after``."""

    def __init__(self, events, break_after=None):
        self.events, self.break_after = events, break_after
        self.cursors: list[str | None] = []

    def stream_logs(self, *, cursor, encoding):
        assert encoding == "base64"
        self.cursors.append(cursor)
        start = 0 if cursor is None else next(i for i, event in enumerate(self.events) if event.cursor == cursor) + 1
        for index, event in enumerate(self.events[start:], start):
            if index == self.break_after:
                self.break_after = None
                raise ProviderConnectionError("stream reset")
            yield event


class FakeSandbox:
    def __init__(self, client, labels, resources):
        self.id = f"sbx_{len(client.sandboxes)}"
        self.state, self.error = "started", None
        self.labels, self.resources = labels, resources
        self.files: dict[str, bytes] = {}
        self.unpacked: dict[str, bytes] = {}
        self.started = None
        self.execs: list = []
        self.fs = SimpleNamespace(write_bytes=self.files.__setitem__)
        self.process = SimpleNamespace(run=self._run, start=self._start)
        self._client = client

    def _run(self, *, argv, **options):
        if argv[0] != "sh":
            # A command exec'd into the running task.
            self.execs.append((argv, options))
            return SimpleNamespace(exit_code=0, stdout="exec output", stderr="", reason="exited")
        # The provider unpacks the archive it uploaded at /.
        (archive,) = self.files.values()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    self.unpacked["/" + member.name] = tar.extractfile(member).read()
        return SimpleNamespace(exit_code=0, output="")

    def _start(self, *, argv, cwd, env, tag):
        self.started = SimpleNamespace(argv=argv, env=env)
        return self._client.process


class FakeDaytona:
    def __init__(self, process=None):
        self.process = process or FakeProcess([StreamLogStateEvent(result=EXITED, cursor="c0")])
        self.sandboxes: list[FakeSandbox] = []
        self.deleted: list[str] = []
        self.create_errors: list[Exception] = []
        self.by_key: dict[str, FakeSandbox] = {}
        self.image_requested = threading.Event()
        self.image_gate = threading.Event()
        self.image_gate.set()
        self.images = SimpleNamespace(get=self._get_image, pull=self._pull)
        self._images: dict[str, SimpleNamespace] = {}

    def _get_image(self, name, **_options):
        self.image_requested.set()
        self.image_gate.wait()
        if name not in self._images:
            raise NotFoundError("no such image", status_code=404)
        return self._images[name]

    def _pull(self, reference, *, name, wait):
        self._images[name] = SimpleNamespace(id=name, name=name, status="ready", error=None)
        return self._images[name]

    def create(self, image, *, idempotency_key, labels, resources, **_options):
        if self.create_errors and isinstance(self.create_errors[0], LimitExceededError):
            raise self.create_errors.pop(0)
        if idempotency_key not in self.by_key:
            self.by_key[idempotency_key] = FakeSandbox(self, labels, resources)
            self.sandboxes.append(self.by_key[idempotency_key])
        if self.create_errors:
            # Daytona created the sandbox, but its answer was lost.
            raise self.create_errors.pop(0)
        return self.by_key[idempotency_key]

    def delete(self, sandbox, *, timeout):
        self.deleted.append(sandbox.id)

    def find_by_name(self, name):
        return [sandbox for sandbox in self.sandboxes if sandbox.id not in self.deleted]

    def close(self):
        pass


class RecordedLogs:
    def __init__(self):
        self.entries: dict[str, list] = {}

    def write_batch(self, key, messages):
        self.entries.setdefault(key, []).extend(messages)


@pytest.fixture
def bundle(tmp_path):
    store = BundleStore(storage_dir=str(tmp_path / "bundles"))
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("main.py", "print('hello')\n")
    return store, store.write(payload.getvalue())


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(tasks, "sleep", lambda _seconds: None)


def provider_for(client, store, logs=None, cluster_env=None):
    return DaytonaTaskProvider(
        DaytonaProviderConfig(api_key_env="DAYTONA_TEST_KEY", default_task_image="python:3.12-slim"),
        descriptor=BackendDescriptor(
            backend_id="daytona",
            kind=BackendKind.DAYTONA,
            capabilities=frozenset({BackendCapability.DIRECT_DISPATCH}),
        ),
        client=client,
        bundles=store,
        logs=logs or RecordedLogs(),
        cluster_name="daytona-test",
        cluster_env=cluster_env or {},
    )


def run_request(bundle_id: str) -> job_pb2.RunTaskRequest:
    request = job_pb2.RunTaskRequest(task_id=TASK, attempt_id=0, attempt_uid="uid-0", num_tasks=1, bundle_id=bundle_id)
    request.entrypoint.run_command.argv.extend(["python", "main.py"])
    request.entrypoint.workdir_files["config.json"] = b"{}"
    request.environment.env_vars["GREETING"] = "hi"
    request.resources.cpu_millicores = 3000
    request.resources.memory_bytes = 3 * 2**30
    request.termination_grace.milliseconds = 30_000
    return request


def running() -> DirectReconcileRequest:
    return DirectReconcileRequest(
        running_tasks=[RunningTaskEntry(JobName.from_wire(TASK), 0, AttemptUid("uid-0"), job_pb2.TASK_STATE_ASSIGNED)]
    )


def final_update(provider):
    """Reconcile until the attempt reaches a terminal state; return every update seen."""
    seen = []

    def terminal():
        seen.extend(provider.reconcile(running()).task_updates)
        return any(update.new_state != job_pb2.TASK_STATE_RUNNING for update in seen)

    wait_for_condition(terminal, timeout=Duration.from_seconds(5))
    return seen


@pytest.mark.parametrize(
    ("result", "state", "exit_code"),
    [
        (EXITED, job_pb2.TASK_STATE_SUCCEEDED, 0),
        (WaitResult(reason="exited", exit_code=3), job_pb2.TASK_STATE_FAILED, 3),
        (WaitResult(reason="signaled", signal="KILL"), job_pb2.TASK_STATE_FAILED, 137),
    ],
    ids=["exited", "failed", "signaled"],
)
def test_daytona_attempt_runs_in_one_sandbox_and_reports_its_exit(bundle, result, state, exit_code):
    store, bundle_id = bundle
    events = [
        log_event(0, "stdout", b"hel"),
        log_event(1, "stdout", b"lo\nwor"),
        log_event(2, "stderr", b"warn\n"),
        StreamLogStateEvent(result=result, cursor="c3"),
        StreamLogEofEvent(cursor="c4"),
    ]
    client = FakeDaytona(FakeProcess(events))
    logs = RecordedLogs()
    provider = provider_for(client, store, logs, cluster_env={"CLUSTER_VAR": "on", "GREETING": "cluster"})
    request = run_request(bundle_id)
    request.environment.secrets["HF_TOKEN"] = "hf-secret"

    provider.reconcile(DirectReconcileRequest(tasks_to_run=[request]))
    *_, final = final_update(provider)

    assert (final.new_state, final.exit_code) == (state, exit_code)
    (sandbox,) = client.sandboxes
    assert sandbox.unpacked == {
        "/app/main.py": b"print('hello')\n",
        "/app/config.json": b"{}",
        f"{SECRETS_DIR}/HF_TOKEN": b"hf-secret",
    }
    assert sandbox.started.argv[-1].endswith("exec python main.py")
    env = sandbox.started.env
    # Cluster defaults, then the job's own variables, then Iris's, as a Kubernetes pod receives them.
    assert (env["CLUSTER_VAR"], env["GREETING"], env["IRIS_TASK_ID"]) == ("on", "hi", f"{TASK}:0")
    assert env["HF_TOKEN_PATH"] == f"{SECRETS_DIR}/HF_TOKEN" and "hf-secret" not in env.values()
    # One core is two vCPUs, and the shape is fixed so the guest sees its own share.
    assert sandbox.resources == Resources(cpu=1.5, cpu_max=1.5, memory=3, memory_max=3, disk=None)
    lines = [(entry.source, entry.data) for entry in logs.entries[f"{TASK}:0"]]
    assert lines == [("stdout", "hello"), ("stderr", "warn"), ("stdout", "wor")]

    # A finished attempt leaves the desired set, and its sandbox is deleted.
    provider.reconcile(DirectReconcileRequest())
    provider.close()
    assert client.deleted == [sandbox.id]


def test_daytona_log_stream_resumes_from_its_cursor_after_a_disconnect(bundle, no_wait):
    store, bundle_id = bundle
    events = [
        log_event(0, "stdout", b"one\n"),
        log_event(1, "stdout", b"two\n"),
        StreamLogStateEvent(result=EXITED, cursor="c2"),
        StreamLogEofEvent(cursor="c3"),
    ]
    process = FakeProcess(events, break_after=1)
    client = FakeDaytona(process)
    logs = RecordedLogs()
    provider = provider_for(client, store, logs)

    provider.reconcile(DirectReconcileRequest(tasks_to_run=[run_request(bundle_id)]))
    *_, final = final_update(provider)

    assert final.new_state == job_pb2.TASK_STATE_SUCCEEDED
    assert process.cursors == [None, "c0"]
    assert [entry.data for entry in logs.entries[f"{TASK}:0"]] == ["one", "two"]
    provider.close()


def test_daytona_cancelled_launch_never_creates_a_sandbox(bundle):
    store, bundle_id = bundle
    client = FakeDaytona()
    client.image_gate.clear()
    provider = provider_for(client, store)
    provider.reconcile(DirectReconcileRequest(tasks_to_run=[run_request(bundle_id)]))
    assert client.image_requested.wait(timeout=5)

    # The job is cancelled while the launch waits for its image.
    provider.reconcile(DirectReconcileRequest())
    client.image_gate.set()
    provider.close()

    assert client.sandboxes == []


@pytest.mark.parametrize(
    ("errors", "state", "sandboxes"),
    [
        # Out of capacity until the launch gives up: the attempt is still ASSIGNED, so it retries for free.
        ([LimitExceededError("concurrent sandbox limit reached", status_code=429)] * 2, "WORKER_FAILED", 0),
        # Daytona created the sandbox but the answer was lost: the repeat returns the same sandbox.
        ([ProviderConnectionError("connection reset")], "SUCCEEDED", 1),
        # Every answer was lost: the launch gives up and finds the sandbox by its attempt label.
        ([ProviderConnectionError("connection reset")] * 2, "WORKER_FAILED", 1),
    ],
    ids=["capacity", "lost-answer", "unanswered"],
)
def test_daytona_create_waits_for_capacity_and_deletes_what_it_created(
    bundle, no_wait, monkeypatch, errors, state, sandboxes
):
    store, bundle_id = bundle
    client = FakeDaytona()
    client.create_errors = list(errors)
    # The launch waits 600 s for capacity: two create requests fit in it.
    clock = iter(range(0, 10_000, 300))
    monkeypatch.setattr(tasks, "monotonic", lambda: next(clock))
    provider = provider_for(client, store)

    provider.reconcile(DirectReconcileRequest(tasks_to_run=[run_request(bundle_id)]))
    updates = final_update(provider)
    provider.reconcile(DirectReconcileRequest())
    provider.close()

    assert job_pb2.TaskState.Name(updates[-1].new_state) == f"TASK_STATE_{state}"
    if state == "WORKER_FAILED":
        assert all(update.new_state != job_pb2.TASK_STATE_RUNNING for update in updates)
    assert len(client.sandboxes) == sandboxes
    assert sorted(client.deleted) == [sandbox.id for sandbox in client.sandboxes]


def test_daytona_rejects_workdir_files_outside_the_workdir(bundle):
    store, bundle_id = bundle
    client = FakeDaytona()
    provider = provider_for(client, store)
    request = run_request(bundle_id)
    request.entrypoint.workdir_files["../etc/passwd"] = b"root::0:0"

    provider.reconcile(DirectReconcileRequest(tasks_to_run=[request]))
    (update,) = final_update(provider)

    # A bad request charges the task's failure budget, unlike a Daytona capacity failure.
    assert update.new_state == job_pb2.TASK_STATE_FAILED
    assert client.deleted == [client.sandboxes[0].id]
    provider.close()


def test_daytona_reports_an_attempt_it_does_not_run_as_lost(bundle):
    store, _ = bundle
    provider = provider_for(FakeDaytona(), store)

    (update,) = provider.reconcile(running()).task_updates

    assert update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
    provider.close()


def test_daytona_rejects_accelerator_and_coscheduled_jobs(bundle):
    store, _ = bundle
    provider = provider_for(FakeDaytona(), store)
    gpu = controller_pb2.Controller.LaunchJobRequest()
    gpu.resources.device.gpu.count = 1
    coscheduled = controller_pb2.Controller.LaunchJobRequest()
    coscheduled.coscheduling.group_by = "tpu-name"

    assert "CPU" in provider.validate_job(gpu)
    assert "coscheduled" in provider.validate_job(coscheduled)
    assert provider.validate_job(controller_pb2.Controller.LaunchJobRequest()) is None
    provider.close()


def test_daytona_requests_a_fixed_shape_with_daytona_minimums() -> None:
    small = job_pb2.ResourceSpecProto(cpu_millicores=500, memory_bytes=256 * 2**20, disk_bytes=20 * 2**30)

    assert sandbox_resources(small) == Resources(cpu=0.5, cpu_max=0.5, memory=1, memory_max=1, disk=20)
    assert sandbox_resources(job_pb2.ResourceSpecProto()) == Resources(cpu=0.5, cpu_max=0.5, memory=1, memory_max=1)


@pytest.mark.parametrize(("timeout", "execution_timeout"), [(-1, None), (30, 30)], ids=["unbounded", "bounded"])
def test_daytona_exec_runs_in_the_attempt_sandbox(bundle, timeout, execution_timeout):
    store, bundle_id = bundle
    client = FakeDaytona()
    provider = provider_for(client, store)
    provider.reconcile(DirectReconcileRequest(tasks_to_run=[run_request(bundle_id)]))
    # The fake process exits at once; its sandbox stays until the attempt leaves the desired set.
    final_update(provider)
    target = TaskTarget(task_id=TASK, attempt_id=0, worker_id=None, address=None, attempt_uid="uid-0")
    request = worker_pb2.Worker.ExecInContainerRequest(task_id=TASK, command=["ls", "/app"])

    response = provider.exec_in_container(target, request, timeout_seconds=timeout)

    assert (response.exit_code, response.stdout) == (0, "exec output")
    # A negative timeout means none, as for the other backends.
    ((argv, options),) = client.sandboxes[0].execs
    assert argv == ["ls", "/app"] and options["execution_timeout"] == execution_timeout
    provider.close()
