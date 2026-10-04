# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ray-only behaviour of the fray Ray backend.

The backend-agnostic contract is covered by test_client.py and test_actor.py
through the parametrised ``client`` fixture; this module holds what only a
multi-process backend can exhibit (restarts, cross-process handles, log
capture, resource mapping, runtime ownership) plus a Zephyr smoke on the Ray backend.
"""

import os
import pickle
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fray.actor import ActorHandle, ActorUnavailableError, current_actor
from fray.client import wait_all
from fray.current_client import BACKEND_ENV, current_client
from fray.local_backend import LocalClient
from fray.types import (
    ActorConfig,
    Entrypoint,
    EnvironmentConfig,
    GpuConfig,
    JobRequest,
    JobStatus,
    ResourceConfig,
    TpuConfig,
)
from rigging.timing import Duration

pytest.importorskip("ray")

import ray
from fray.ray_backend import RayClient, RemoteTraceback, ray_options, ray_runtime_env

# Generous: under a fully loaded xdist run a subprocess driver can take a while to start.
RESTART_WAIT = 90.0
SUBPROCESS_TEST_TIMEOUT = 120

# Classes and functions below are not importable inside Ray workers; ship them by value.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


@pytest.fixture
def client(ray_client):
    yield ray_client
    ray_client.kill_actors()


def _sleep_forever():
    time.sleep(3600)


def _mark_then_sleep_forever(path: str):
    Path(path).touch()
    _sleep_forever()


def _client_namespace() -> str:
    client = current_client()
    assert isinstance(client, RayClient)
    return client.namespace


def _print_namespace():
    print(f"NAMESPACE={_client_namespace()}")


def _increment_through_handle(handle: ActorHandle, amount: int):
    handle.increment(amount)


def _record_attempt(path: str) -> int:
    with open(path, "a") as f:
        f.write("attempt\n")
    with open(path) as f:
        return sum(1 for _ in f)


def _crash_after_recording(path: str):
    _record_attempt(path)
    os._exit(1)


def _load_shard(name: str):
    raise KeyError(name)


def _failing_shard():
    _load_shard("shard-3")


def _write_progress_bar():
    sys.stdout.write("\rprogress 1\rprogress 2\r")
    print("last diagnostic", end="")


class Counter:
    def __init__(self, start: int = 0):
        self._value = start
        self._ctx = current_actor()

    def increment(self, amount: int = 1) -> int:
        self._value += amount
        return self._value

    def get(self) -> int:
        return self._value

    def client_namespace(self) -> str:
        return _client_namespace()

    def exit_cleanly(self) -> None:
        assert self._ctx.shutdown_event is not None
        self._ctx.shutdown_event.set()

    def crash(self) -> None:
        self._ctx.fail(RuntimeError("worker asked to fail"))


class _ActorConstructionError(RuntimeError):
    pass


class FailSecondActor:
    def __init__(self):
        self._index = current_actor().index
        if self._index == 1:
            raise _ActorConstructionError("second replica refuses to start")

    def get(self) -> int:
        return self._index


class FlakyInitActor:
    """Fails construction until the ``succeed_on`` attempt, counting attempts in a file."""

    def __init__(self, attempts_path: str, succeed_on: int):
        self._attempts = _record_attempt(attempts_path)
        if self._attempts < succeed_on:
            raise _ActorConstructionError(f"init attempt {self._attempts} fails")

    def attempts(self) -> int:
        return self._attempts


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.25)
    return predicate()


def _named_actors(namespace: str) -> set[str]:
    return {
        entry["name"] for entry in ray.util.list_named_actors(all_namespaces=True) if entry["namespace"] == namespace
    }


# ---------------------------------------------------------------------------
# Resource mapping (no cluster needed)
# ---------------------------------------------------------------------------


def test_ray_options_maps_task_and_actor_resources():
    resources = ResourceConfig(cpu=2, ram="512m", device=GpuConfig(variant="h100", count=2))
    task = ray_options(resources)
    assert (task["num_cpus"], task["memory"], task["num_gpus"]) == (2, 512 * 1024**2, 2)
    assert task["accelerator_type"] == "H100"
    assert "accelerator_type" not in ray_options(ResourceConfig.with_gpu("auto"))

    actor = ray_options(resources, ActorConfig(max_concurrency=8, max_task_retries=3))
    assert actor["num_cpus"] == 0
    assert (actor["max_concurrency"], actor["max_restarts"], actor["max_task_retries"]) == (8, 3, 0)


def test_ray_options_rejects_unschedulable_requests():
    with pytest.raises(ValueError, match="TPU"):
        ray_options(ResourceConfig(device=TpuConfig(variant="v5p-8")))
    with pytest.raises(ValueError, match="regions"):
        ray_options(ResourceConfig(regions=["us-central1"]))
    with pytest.raises(ValueError, match="accelerator_type"):
        ray_options(ResourceConfig.with_gpu("B100"))


def test_ray_runtime_env_rejects_unrunnable_setup():
    with pytest.raises(ValueError, match="pip_packages"):
        ray_runtime_env(EnvironmentConfig(workspace=os.getcwd(), pip_packages=["six"]))
    with pytest.raises(ValueError, match="setup_scripts"):
        ray_runtime_env(EnvironmentConfig(workspace=os.getcwd(), setup_scripts=["echo hi"]))
    assert ray_runtime_env(EnvironmentConfig(workspace=os.getcwd(), setup_scripts=[])) == {
        "py_executable": sys.executable
    }


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def test_submit_rejects_multi_process_tasks(client: RayClient):
    request = JobRequest(name="multi", entrypoint=Entrypoint.from_callable(_sleep_forever), processes_per_task=2)
    with pytest.raises(ValueError, match="processes_per_task"):
        client.submit(request)


def test_crashing_job_attempts_follow_max_retries_failure(client: RayClient, tmp_path: Path):
    once = tmp_path / "once"
    entrypoint = Entrypoint.from_callable(_crash_after_recording, args=(str(once),))
    handle = client.submit(JobRequest(name="crash", entrypoint=entrypoint))
    assert handle.wait(raise_on_failure=False) == JobStatus.FAILED
    assert once.read_text().count("attempt") == 1

    twice = tmp_path / "twice"
    entrypoint = Entrypoint.from_callable(_crash_after_recording, args=(str(twice),))
    handle = client.submit(JobRequest(name="crash-retry", entrypoint=entrypoint, max_retries_failure=1))
    assert handle.wait(raise_on_failure=False) == JobStatus.FAILED
    assert twice.read_text().count("attempt") == 2


def test_terminate_marks_running_job_stopped(client: RayClient):
    handle = client.submit(JobRequest(name="hang", entrypoint=Entrypoint.from_callable(_sleep_forever)))
    assert handle.status() == JobStatus.RUNNING
    handle.terminate()
    assert handle.status() == JobStatus.STOPPED
    assert handle.wait(timeout=30) == JobStatus.STOPPED
    assert handle.wait(timeout=30, raise_on_failure=False) == JobStatus.STOPPED


def test_wait_all_timeout_leaves_job_terminable(client: RayClient):
    handle = client.submit(JobRequest(name="hang", entrypoint=Entrypoint.from_callable(_sleep_forever)))
    with pytest.raises(TimeoutError):
        wait_all([handle], timeout=0.2)
    handle.terminate()
    assert handle.status() == JobStatus.STOPPED


def test_job_timeout_stops_job(client: RayClient):
    request = JobRequest(
        name="deadline", entrypoint=Entrypoint.from_callable(_sleep_forever), timeout=Duration.from_seconds(1)
    )
    handle = client.submit(request)
    assert handle.wait() == JobStatus.STOPPED
    assert handle.status() == JobStatus.STOPPED


def test_job_timeout_seen_by_status_makes_wait_return_stopped(client: RayClient):
    request = JobRequest(
        name="deadline", entrypoint=Entrypoint.from_callable(_sleep_forever), timeout=Duration.from_seconds(1)
    )
    handle = client.submit(request)
    assert _wait_until(lambda: handle.status() == JobStatus.STOPPED, RESTART_WAIT)
    assert handle.wait() == JobStatus.STOPPED


def test_job_timeout_cancels_task_without_polling(client: RayClient, tmp_path: Path):
    started = tmp_path / "started"
    total_cpus = ray.cluster_resources()["CPU"]
    request = JobRequest(
        name="unpolled-deadline",
        entrypoint=Entrypoint.from_callable(_mark_then_sleep_forever, args=(str(started),)),
        resources=ResourceConfig(cpu=1),
        timeout=Duration.from_seconds(5),
    )
    handle = client.submit(request)
    assert _wait_until(started.exists, RESTART_WAIT)
    assert ray.available_resources().get("CPU", 0.0) < total_cpus
    assert _wait_until(lambda: ray.available_resources().get("CPU", 0.0) >= total_cpus, RESTART_WAIT)
    assert handle.status() == JobStatus.STOPPED


def test_job_failure_carries_remote_traceback(client: RayClient):
    handle = client.submit(JobRequest(name="shard", entrypoint=Entrypoint.from_callable(_failing_shard)))
    with pytest.raises(KeyError, match="shard-3") as info:
        handle.wait()
    assert isinstance(info.value.__cause__, RemoteTraceback)
    assert "_load_shard" in str(info.value.__cause__)


def test_binary_job_logs_tail(client: RayClient):
    handle = client.submit(JobRequest(name="echo", entrypoint=Entrypoint.from_binary("echo", ["hello", "ray"])))
    assert handle.wait() == JobStatus.SUCCEEDED
    assert any(line.endswith("hello ray") for line in handle.logs())
    assert len(handle.logs(max_lines=1)) == 1


def test_job_logs_keep_carriage_return_progress_and_unterminated_line(client: RayClient):
    handle = client.submit(JobRequest(name="progress", entrypoint=Entrypoint.from_callable(_write_progress_bar)))
    assert handle.wait() == JobStatus.SUCCEEDED
    assert {"progress 1", "progress 2", "last diagnostic"} <= set(handle.logs())


def test_current_client_inside_job_uses_driver_namespace(client: RayClient):
    handle = client.submit(JobRequest(name="ns", entrypoint=Entrypoint.from_callable(_print_namespace)))
    assert handle.wait() == JobStatus.SUCCEEDED
    assert f"NAMESPACE={client.namespace}" in handle.logs()


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------


def test_current_client_inside_actor_uses_driver_namespace(client: RayClient):
    actor = client.create_actor(Counter, name="counter")
    assert actor.client_namespace() == client.namespace


def test_handle_pickled_into_job_reaches_same_actor(client: RayClient):
    actor = client.create_actor(Counter, 10, name="counter")
    restored = pickle.loads(pickle.dumps(actor))
    assert restored.increment(5) == 15

    entrypoint = Entrypoint.from_callable(_increment_through_handle, args=(actor, 7))
    assert client.submit(JobRequest(name="callback", entrypoint=entrypoint)).wait() == JobStatus.SUCCEEDED
    assert actor.get() == 22


def test_actor_group_member_construction_failure_spares_siblings(client: RayClient):
    group = client.create_actor_group(FailSecondActor, name="failing-actors", count=2)
    first = group.wait_ready(count=1)[0]
    assert first.get() == 0
    assert group.discover_new() == []
    with pytest.raises(_ActorConstructionError, match="second replica"):
        group.wait_ready()

    assert first.get() == 0
    assert not group.is_done()
    assert _wait_until(lambda: _named_actors(client.namespace) == {"failing-actors-0"}, RESTART_WAIT)

    group.shutdown()
    assert group.is_done()
    with pytest.raises(ActorUnavailableError):
        first.get()


def test_actor_init_retried_up_to_max_task_retries(client: RayClient, tmp_path: Path):
    attempts = tmp_path / "attempts"
    actor = client.create_actor(
        FlakyInitActor, str(attempts), 3, name="flaky", actor_config=ActorConfig(max_task_retries=2)
    )
    assert actor.attempts() == 3


def test_failed_create_actor_leaves_name_reusable(client: RayClient, tmp_path: Path):
    attempts = tmp_path / "attempts"
    with pytest.raises(_ActorConstructionError, match="attempt 1"):
        client.create_actor(FlakyInitActor, str(attempts), 2, name="flaky")
    assert attempts.read_text().count("attempt") == 1
    assert _wait_until(lambda: "flaky-0" not in _named_actors(client.namespace), RESTART_WAIT)

    counter = client.create_actor(Counter, name="flaky")
    assert counter.increment() == 1


def test_actor_fail_restarts_replica(client: RayClient):
    actor = client.create_actor(Counter, name="restarting", actor_config=ActorConfig(max_task_retries=1))
    assert actor.increment() == 1
    # The reply races the watcher thread's os._exit, so do not wait for it.
    actor.crash.remote()

    def restarted() -> bool:
        try:
            return actor.get() == 0
        except ActorUnavailableError:
            return False

    assert _wait_until(restarted, RESTART_WAIT), "replica did not come back with fresh state"


def test_actor_group_is_done_after_clean_exit(client: RayClient):
    group = client.create_actor_group(Counter, name="exiting", count=1)
    handle = group.wait_ready()[0]
    assert not group.is_done()
    handle.exit_cleanly.remote()
    assert _wait_until(group.is_done, RESTART_WAIT)
    with pytest.raises(ActorUnavailableError):
        handle.get()


def test_actor_group_pickles_to_identity(client: RayClient):
    group = client.create_actor_group(Counter, name="counters", count=2)
    group.wait_ready()
    restored = pickle.loads(pickle.dumps(group))
    handles = restored.wait_ready(timeout=RESTART_WAIT)
    assert [h.increment(i + 1) for i, h in enumerate(handles)] == [1, 2]


def test_wait_ready_times_out_when_unschedulable(client: RayClient):
    group = client.create_actor_group(Counter, name="starved", count=1, resources=ResourceConfig(ram="1000t"))
    with pytest.raises(TimeoutError):
        group.wait_ready(timeout=1.0)
    assert not group.is_done()
    group.shutdown()


ORPHAN_DRIVER = textwrap.dedent(
    """
    import sys
    import time

    from fray.ray_backend import RayClient


    class Noop:
        def ping(self) -> str:
            return "pong"


    client = RayClient.connect(address=sys.argv[1], namespace=sys.argv[2])
    print(client.create_actor(Noop, name="orphan").ping(), flush=True)
    time.sleep(600)
    """
)


@pytest.mark.timeout(SUBPROCESS_TEST_TIMEOUT)
def test_actors_die_with_their_driver(client: RayClient):
    address = ray.get_runtime_context().gcs_address
    driver = subprocess.Popen(
        [sys.executable, "-c", ORPHAN_DRIVER, address, client.namespace],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert driver.stdout is not None
        output = []
        for line in driver.stdout:
            output.append(line)
            if line.strip() == "pong":
                break
        assert "orphan-0" in _named_actors(client.namespace), "".join(output)
        driver.send_signal(signal.SIGKILL)
        driver.wait()
        assert _wait_until(lambda: "orphan-0" not in _named_actors(client.namespace), RESTART_WAIT)
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()


# ---------------------------------------------------------------------------
# Runtime ownership
# ---------------------------------------------------------------------------


def test_second_connect_shares_runtime_without_owning_it(client: RayClient):
    with pytest.raises(ValueError, match="namespace"):
        RayClient.connect(address="auto", namespace="elsewhere")
    second = RayClient.connect(address="auto")
    assert second.namespace == client.namespace

    second.shutdown()
    second.shutdown()
    assert ray.is_initialized()
    assert current_client() is client
    handle = client.submit(JobRequest(name="after-second", entrypoint=Entrypoint.from_callable(_print_namespace)))
    assert handle.wait() == JobStatus.SUCCEEDED


def test_failed_connect_leaves_current_client_resolution_alone(monkeypatch):
    def refuse(**kwargs):
        raise ConnectionError("no head")

    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", refuse)
    with pytest.raises(ConnectionError):
        RayClient.connect(address="auto")
    assert BACKEND_ENV not in os.environ
    assert isinstance(current_client(), LocalClient)


OWNER_DRIVER = textwrap.dedent(
    """
    import time

    import ray
    from fray.ray_backend import RayClient
    from fray.types import Entrypoint, JobRequest, JobStatus


    def say_hi():
        print("hi")


    def hang():
        time.sleep(600)


    client = RayClient.connect(address="local", num_cpus=1, object_store_memory=200_000_000)
    handle = client.submit(JobRequest(name="hi", entrypoint=Entrypoint.from_callable(say_hi)))
    assert handle.wait() == JobStatus.SUCCEEDED
    hanging = client.submit(JobRequest(name="hang", entrypoint=Entrypoint.from_callable(hang)))
    client.shutdown(wait=False)
    client.shutdown(wait=False)
    assert not ray.is_initialized()
    assert handle.status() == JobStatus.SUCCEEDED
    assert "hi" in handle.logs()
    assert hanging.status() == JobStatus.STOPPED
    assert not ray.is_initialized()
    print("OK")
    """
)


@pytest.mark.timeout(SUBPROCESS_TEST_TIMEOUT)
def test_owner_shutdown_is_idempotent_and_handles_answer_afterwards():
    result = subprocess.run([sys.executable, "-c", OWNER_DRIVER], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("OK")


# ---------------------------------------------------------------------------
# Zephyr smoke
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_zephyr_pipeline_on_ray(client: RayClient, tmp_path, monkeypatch):
    zephyr_context = pytest.importorskip("zephyr.context")
    zephyr_dataset = pytest.importorskip("zephyr.dataset")
    zephyr_runners = pytest.importorskip("zephyr.runners")
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))

    ctx = zephyr_context.ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="256m"),
        stage_runner_factory=zephyr_runners.InlineRunner,
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="ray-smoke",
    )
    dataset = zephyr_dataset.Dataset.from_list(list(range(1, 11))).map(lambda x: x * 2).filter(lambda x: x > 5)
    try:
        results = ctx.execute(dataset).results
    finally:
        ctx.shutdown()
    assert sorted(results) == [6, 8, 10, 12, 14, 16, 18, 20]
