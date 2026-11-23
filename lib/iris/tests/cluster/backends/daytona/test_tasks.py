# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Daytona provider runs each attempt in one sandbox and reports it to the controller."""

import base64
import io
import tarfile
import time
import zipfile
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona.v2 import NotFoundError, Resources, WaitResult
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

TASK = "/rav/job/0"


def frame(data: bytes) -> str:
    """A v2 log frame as the API sends it: URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class FakeProcess:
    def __init__(self, sandbox, argv, env, outcome):
        self.sandbox, self.argv, self.env, self.outcome = sandbox, argv, env, outcome

    def stream_logs(self, *, encoding):
        assert encoding == "base64"
        for channel, data in self.outcome["logs"]:
            yield SimpleNamespace(type="log", frame=SimpleNamespace(channel=channel, data=frame(data)))
        yield SimpleNamespace(type="state", result=self.outcome["result"])
        yield SimpleNamespace(type="eof")


class FakeSandbox:
    def __init__(self, client, name, labels, resources):
        self.id = f"sbx_{len(client.sandboxes)}"
        self.state = "started"
        self.error = None
        self.labels, self.resources = labels, resources
        self.files: dict[str, bytes] = {}
        self.workdir: dict[str, bytes] = {}
        self.started: FakeProcess | None = None
        self.execs: list = []
        outcome = client.outcome
        self.fs = SimpleNamespace(write_bytes=self.files.__setitem__)

        def run(*, argv, **options):
            if argv[0] != "sh":
                # A command exec'd into the running task.
                self.execs.append((argv, options))
                return SimpleNamespace(exit_code=0, stdout="exec output", stderr="", reason="exited")
            # The provider unpacks the workdir archive it uploaded.
            archive = next(data for path, data in self.files.items() if path.endswith(".tar.gz"))
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        self.workdir[member.name.removeprefix("./")] = tar.extractfile(member).read()
            return SimpleNamespace(exit_code=0, output="")

        def start(*, argv, cwd, env, tag):
            self.started = FakeProcess(self, argv, env, outcome)
            return self.started

        self.process = SimpleNamespace(run=run, start=start)


class FakeDaytona:
    def __init__(self, outcome):
        self.outcome = outcome
        self.sandboxes: list[FakeSandbox] = []
        self.deleted: list[str] = []
        self.pulled: list[str] = []
        self.fail_create: Exception | None = None
        self.images = SimpleNamespace(get=self._get_image, pull=self._pull)
        self._images: dict[str, SimpleNamespace] = {}

    def _get_image(self, name, **_options):
        if name not in self._images:
            raise NotFoundError("no such image", status_code=404)
        return self._images[name]

    def _pull(self, reference, *, name, wait):
        self.pulled.append(reference)
        self._images[name] = SimpleNamespace(id=f"img_{name}", name=name, status="ready", error=None)
        return self._images[name]

    def create(self, image, *, name, labels, resources, **_options):
        if self.fail_create is not None:
            raise self.fail_create
        sandbox = FakeSandbox(self, name, labels, resources)
        self.sandboxes.append(sandbox)
        return sandbox

    def delete(self, sandbox, *, timeout):
        self.deleted.append(sandbox.id)

    def find_by_name(self, name):
        return ()

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


def provider_for(client, store, logs):
    return DaytonaTaskProvider(
        DaytonaProviderConfig(api_key_env="DAYTONA_TEST_KEY", default_task_image="python:3.12-slim"),
        descriptor=BackendDescriptor(
            backend_id="daytona",
            kind=BackendKind.DAYTONA,
            capabilities=frozenset({BackendCapability.DIRECT_DISPATCH}),
        ),
        client=client,
        bundles=store,
        logs=logs,
        cluster_name="daytona-test",
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


def running(state: int) -> DirectReconcileRequest:
    return DirectReconcileRequest(
        running_tasks=[RunningTaskEntry(JobName.from_wire(TASK), 0, AttemptUid("uid-0"), state)]
    )


def reconcile_until(provider, request, state, deadline=5.0):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        updates = provider.reconcile(request).task_updates
        if any(update.new_state == state for update in updates):
            return next(update for update in updates if update.new_state == state)
        time.sleep(0.01)
    raise AssertionError(f"no update to state {state}")


@pytest.mark.parametrize(
    ("result", "state", "exit_code"),
    [
        (WaitResult(reason="exited", exit_code=0), job_pb2.TASK_STATE_SUCCEEDED, 0),
        (WaitResult(reason="exited", exit_code=3), job_pb2.TASK_STATE_FAILED, 3),
        (WaitResult(reason="signaled", signal="KILL"), job_pb2.TASK_STATE_FAILED, 137),
    ],
    ids=["exited", "failed", "signaled"],
)
def test_daytona_attempt_runs_in_one_sandbox_and_reports_its_exit(bundle, result, state, exit_code):
    store, bundle_id = bundle
    client = FakeDaytona({"logs": [("stdout", b"hel"), ("stdout", b"lo\nwor"), ("stderr", b"warn\n")], "result": result})
    logs = RecordedLogs()
    provider = provider_for(client, store, logs)

    building = provider.reconcile(DirectReconcileRequest(tasks_to_run=[run_request(bundle_id)])).task_updates
    assert [update.new_state for update in building] == [job_pb2.TASK_STATE_BUILDING]
    final = reconcile_until(provider, running(job_pb2.TASK_STATE_BUILDING), state)

    assert final.exit_code == exit_code
    (sandbox,) = client.sandboxes
    assert sandbox.workdir == {"main.py": b"print('hello')\n", "config.json": b"{}"}
    assert sandbox.started.argv[-1].endswith("exec python main.py")
    assert sandbox.started.env["GREETING"] == "hi" and sandbox.started.env["IRIS_TASK_ID"] == f"{TASK}:0"
    # One core is two vCPUs, and the shape is fixed so the guest sees its own share.
    assert sandbox.resources == Resources(cpu=1.5, cpu_max=1.5, memory=3, memory_max=3, disk=None)
    lines = [(entry.source, entry.data) for entry in logs.entries[f"{TASK}:0"]]
    assert lines == [("stdout", "hello"), ("stderr", "warn"), ("stdout", "wor")]

    # A finished attempt leaves the desired set, and its sandbox is deleted.
    provider.reconcile(DirectReconcileRequest())
    provider.close()
    assert client.deleted == [sandbox.id]


def test_daytona_launch_failure_retries_without_charging_the_task(bundle):
    store, bundle_id = bundle
    client = FakeDaytona({"logs": [], "result": WaitResult(reason="exited", exit_code=0)})
    client.fail_create = RuntimeError("concurrent sandbox limit reached")
    provider = provider_for(client, store, RecordedLogs())

    provider.reconcile(DirectReconcileRequest(tasks_to_run=[run_request(bundle_id)]))
    update = reconcile_until(provider, running(job_pb2.TASK_STATE_BUILDING), job_pb2.TASK_STATE_WORKER_FAILED)

    assert "concurrent sandbox limit reached" in update.error
    provider.close()


def test_daytona_reports_an_attempt_it_does_not_run_as_lost(bundle):
    store, _ = bundle
    provider = provider_for(FakeDaytona({}), store, RecordedLogs())

    (update,) = provider.reconcile(running(job_pb2.TASK_STATE_RUNNING)).task_updates

    assert update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
    provider.close()


def test_daytona_rejects_accelerator_and_coscheduled_jobs(bundle):
    store, _ = bundle
    provider = provider_for(FakeDaytona({}), store, RecordedLogs())
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
    client = FakeDaytona({"logs": [], "result": WaitResult(reason="exited", exit_code=0)})
    provider = provider_for(client, store, RecordedLogs())
    provider.reconcile(DirectReconcileRequest(tasks_to_run=[run_request(bundle_id)]))
    # The fake process exits at once; its sandbox stays until the attempt leaves the desired set.
    reconcile_until(provider, running(job_pb2.TASK_STATE_BUILDING), job_pb2.TASK_STATE_SUCCEEDED)
    target = TaskTarget(task_id=TASK, attempt_id=0, worker_id=None, address=None, attempt_uid="uid-0")
    request = worker_pb2.Worker.ExecInContainerRequest(task_id=TASK, command=["ls", "/app"])

    response = provider.exec_in_container(target, request, timeout_seconds=timeout)

    assert (response.exit_code, response.stdout) == (0, "exec output")
    # A negative timeout means none, as for the other backends.
    ((argv, options),) = client.sandboxes[0].execs
    assert argv == ["ls", "/app"] and options["execution_timeout"] == execution_timeout
    provider.close()
