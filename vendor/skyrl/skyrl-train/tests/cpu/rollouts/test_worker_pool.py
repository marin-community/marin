import asyncio
import json
import os
from pathlib import Path
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace

import psutil
import pytest
import ray
from omegaconf import DictConfig, OmegaConf, open_dict

from tests.cpu.tiny_training.tiny_model import build_tiny_policy

from skyrl_train.rollout_observability import measure_rollout, observe_rollout_call, rollout_wait
from skyrl_train.utils.algorithm_registry import sync_registries
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.taskcompendium import TaskCompendiumExp
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.workers import (
    RolloutWorkerPool,
    RolloutWorkerResources,
    RolloutWorkerStalledError,
    WorkerShard,
)
from skyrl_train.trajectory_runners.types import BatchMetadata, TrainingPhase, TrajectoryID, TaskRequestBatch
from tests.cpu.task_specs import lowered_task, text_task, task_input


@dataclass(frozen=True)
class _Spec:
    config: DictConfig

    def build(self, tokenizer, shard: WorkerShard):
        raise AssertionError("these tests supply already-started workers")


@pytest.fixture
def spec() -> _Spec:
    return _Spec(OmegaConf.create({}))


class _RemoteMethod:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        self._call = call

    def remote(self, *args, **kwargs):
        return self._call(*args, **kwargs)


@pytest.mark.asyncio
async def test_preconverted_tasks_send_harbor_settings_to_worker_process(monkeypatch):
    """Already-lowered Harbor rows have no terminal-bench directory sidechannel."""
    transmitted = []

    async def ready():
        pass

    def remote(spec, _shard, _sink, _executor_threads):
        transmitted.append(spec.harbor_config)
        return SimpleNamespace(startup=_RemoteMethod(ready), shutdown=_RemoteMethod(ready))

    # Observe the Ray actor construction payload, without starting Ray or model engines.
    monkeypatch.setattr(
        "skyrl_train.rollouts.workers.RolloutWorker",
        SimpleNamespace(options=lambda **_kwargs: SimpleNamespace(remote=remote)),
    )
    node_id = ray.NodeID.from_random().hex()
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: node_id))
    monkeypatch.setattr(ray, "kill", lambda _actor: None)
    config = get_default_config()
    with open_dict(config):
        config.terminal_bench_config = {"harbor": {"max_retries": 0}}
    config.trajectory_runner.rollout_workers.num_workers = 1
    config.trajectory_runner.rollout_workers.start_interval_seconds = 0
    experiment = object.__new__(TaskCompendiumExp)
    experiment.sessions = {}
    pool = experiment.get_task_runner(config, SimpleNamespace(engines=[]))
    await pool.startup()
    try:
        assert transmitted == [config.terminal_bench_config]
    finally:
        await pool.shutdown()


class _Worker:
    def __init__(self, call: Callable[..., Awaitable[dict]]):
        async def run(input_batch, _observe):
            return await call(input_batch), None

        self.run = _RemoteMethod(run)


class _SessionWorker(_Worker):
    def __init__(self, name: str, calls: list[tuple[str, str]]):
        async def run(input_batch):
            phase = input_batch.batch_metadata.training_phase
            calls.append((name, phase))
            return _output(input_batch.trajectory_ids)

        async def run_task(task, _writer, _observe):
            calls.append((name, f"task {task.prompt['uid']}"))
            return 0, None

        async def start_eval_session(**_kwargs):
            calls.append((name, "start_eval"))

        async def stop_eval_session():
            calls.append((name, "stop_eval"))

        super().__init__(run)
        self.run_task = _RemoteMethod(run_task)
        self.start_eval_session = _RemoteMethod(start_eval_session)
        self.stop_eval_session = _RemoteMethod(stop_eval_session)


@ray.remote
class _BlockingWorker:
    def __init__(self):
        self._started = asyncio.Event()
        self._cleaned = asyncio.Event()

    async def run(self, *_args):
        self._started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self._cleaned.set()

    async def wait_for_start(self):
        await self._started.wait()

    async def wait_for_cleanup(self):
        await self._cleaned.wait()
        return True


def _request(ids: list[TrajectoryID], phase: TrainingPhase | None = None) -> TaskRequestBatch:
    return TaskRequestBatch(
        rows=tuple(task_input(lowered_task(text_task(item.instance_id))) for item in ids),
        sampling_params={},
        trajectory_ids=tuple(ids),
        batch_metadata=BatchMetadata(global_step=1, training_phase=phase) if phase is not None else None,
    )


def _output(ids: list[TrajectoryID]) -> dict:
    values = [trajectory_id.repetition_id + (100 if trajectory_id.instance_id == "b" else 0) for trajectory_id in ids]
    return {
        "prompt_token_ids": [[value] for value in values],
        "response_ids": [[value] for value in values],
        "rewards": [float(value) for value in values],
        "loss_masks": [[1] for _ in values],
        "rollout_metrics": {},
        "rollout_logprobs": None,
        "trajectory_ids": list(ids),
    }


def _pool(workers: list[object], spec: _Spec, *, timeout: float = 1) -> RolloutWorkerPool:
    pool = RolloutWorkerPool(
        spec,
        RolloutWorkerResources(
            num_workers=len(workers),
            cpus_per_worker=1,
            executor_threads=1,
            progress_timeout_seconds=timeout,
            start_interval_seconds=0,
        ),
    )
    pool._actors = list(workers)
    return pool


@pytest.mark.asyncio
async def test_training_tasks_wait_for_all_eval_workers(spec):
    calls: list[tuple[str, str]] = []
    pool = _pool(
        [_SessionWorker("eval", calls), _SessionWorker("train", calls)],
        spec,
    )
    task = RolloutTask(RolloutLease("lease", 1, 1), {"uid": "a"}, _request([TrajectoryID("a", 0)], "train"))

    await pool.start_eval_session(run_name="run", eval_step=0)
    training = asyncio.create_task(pool.run_task(task, writer=None))
    await pool.stop_eval_session()
    await training

    assert calls == [
        ("eval", "start_eval"),
        ("train", "start_eval"),
        ("eval", "stop_eval"),
        ("train", "stop_eval"),
        ("eval", "task a"),
    ]


@pytest.mark.asyncio
async def test_pool_isolates_eval_from_concurrent_training(spec):
    calls: list[tuple[str, str]] = []
    pool = _pool(
        [_SessionWorker("eval", calls), _SessionWorker("train", calls)],
        spec,
    )

    await pool.start_eval_session(run_name="run", eval_step=0)
    await pool.run(_request([TrajectoryID("heldout", 0)], "eval"))
    training = asyncio.create_task(pool.run(_request([TrajectoryID("training", 0)], "train")))
    await pool.stop_eval_session()
    await training

    assert calls == [
        ("eval", "start_eval"),
        ("train", "start_eval"),
        ("eval", "eval"),
        ("eval", "stop_eval"),
        ("train", "stop_eval"),
        ("eval", "train"),
    ]


@pytest.mark.asyncio
async def test_a_single_worker_defers_training_until_the_eval_session_ends(spec):
    calls: list[tuple[str, str]] = []
    pool = _pool([_SessionWorker("only", calls)], spec)
    task = RolloutTask(RolloutLease("lease", 1, 1), {"uid": "a"}, _request([TrajectoryID("a", 0)], "train"))

    await pool.start_eval_session(run_name="run", eval_step=0)
    training = asyncio.create_task(pool.run_task(task, writer=None))
    await pool.run(_request([TrajectoryID("heldout", 0)], "eval"))
    await pool.stop_eval_session()
    await training

    assert calls == [("only", "start_eval"), ("only", "eval"), ("only", "stop_eval"), ("only", "task a")]


@pytest.mark.asyncio
async def test_evaluation_uses_every_worker_without_writing_training_groups(spec):
    calls = []
    pool = _pool([_SessionWorker("left", calls), _SessionWorker("right", calls)], spec)
    await pool.start_eval_session(run_name="run", eval_step=2)
    results = await asyncio.gather(*(pool.run(_request([TrajectoryID(name, 0)], "eval")) for name in ("a", "b")))
    await pool.stop_eval_session()
    assert {name for name, action in calls if action == "eval"} == {"left", "right"}
    assert all(not action.startswith("task ") for _, action in calls)
    assert [result["response_ids"] for result in results] == [[[0]], [[100]]]


@pytest.mark.asyncio
async def test_failed_session_start_releases_every_worker_for_training(spec):
    calls = []
    left, right = _SessionWorker("left", calls), _SessionWorker("right", calls)

    async def fail(**kwargs):
        raise ConnectionError("session initialization failed")

    right.start_eval_session = _RemoteMethod(fail)
    pool = _pool([left, right], spec)
    with pytest.raises(ExceptionGroup) as failure:
        await pool.start_eval_session(run_name="run", eval_step=2)
    assert isinstance(failure.value.exceptions[0], ConnectionError)
    output = await pool.run(_request([TrajectoryID("a", 0)], "train"))
    assert output["response_ids"] == [[0]]
    assert {name for name, action in calls if action == "stop_eval"} == {"left", "right"}


@pytest.mark.asyncio
async def test_waits_measured_in_a_worker_join_the_callers_rollout_observation(spec):
    class _MeasuringWorker:
        async def _run_task(self, task, _writer, observe):
            # A worker process has no caller observation; its waits reach the caller only through the return value.
            with measure_rollout(enabled=observe) as observation:
                with rollout_wait("env_queue"):
                    await asyncio.sleep(0)
            return 7, None if observation is None else observation.timings()

        def __init__(self):
            self.run_task = _RemoteMethod(self._run_task)

    pool = _pool([_MeasuringWorker()], spec)
    task = RolloutTask(RolloutLease("lease", 1, 1), {"uid": "a"}, _request([TrajectoryID("a", 0)], "train"))

    with observe_rollout_call(step=1, mode="async", enabled=True) as observation:
        response_tokens = await pool.run_task(task, writer=None)

    assert response_tokens == 7
    assert len(observation.waits["env_queue"]) == 1


@pytest.mark.asyncio
async def test_progress_deadline_resets_when_the_same_worker_completes_a_request(spec, monkeypatch):
    slow_result = None
    slow_task = None

    class _FakeDeadline:
        def __init__(self):
            self._expired = False

        async def __aenter__(self):
            nonlocal slow_task
            current_task = asyncio.current_task()
            if slow_task is None:
                slow_task = current_task
            elif current_task is slow_task and not slow_result.done():
                slow_result.set_result((_output([TrajectoryID("a", 0)]), None))
            return self

        async def __aexit__(self, exception_type, _exception, _traceback):
            if exception_type is asyncio.CancelledError:
                self._expired = True
                raise TimeoutError
            return False

        def expired(self):
            return self._expired

    class _TimedRemoteMethod:
        def remote(self, input_batch, _observe):
            nonlocal slow_result
            instance_id = input_batch.trajectory_ids[0].instance_id
            result = asyncio.get_running_loop().create_future()
            if instance_id == "a":
                slow_result = result
                return result

            result.set_result((_output(input_batch.trajectory_ids), None))
            asyncio.get_running_loop().call_soon(slow_task.cancel)
            return result

    class _TimedWorker:
        run = _TimedRemoteMethod()

    pool = _pool([_TimedWorker()], spec, timeout=0.5)
    monkeypatch.setattr(asyncio, "timeout_at", lambda _deadline: _FakeDeadline())

    slow, fast = await asyncio.gather(
        pool.run(_request([TrajectoryID("a", 0)])), pool.run(_request([TrajectoryID("b", 0)]))
    )

    assert slow["trajectory_ids"] == [TrajectoryID("a", 0)]
    assert fast["trajectory_ids"] == [TrajectoryID("b", 0)]


@pytest.mark.asyncio
async def test_stalled_worker_request_is_cancelled_remotely(ray_init, spec, monkeypatch):
    actor = _BlockingWorker.remote()
    async with asyncio.timeout(30):
        await actor.__ray_ready__.remote()
    pool = _pool([actor], spec, timeout=0.1)
    cancel_calls = []
    original_cancel = ray.cancel

    def capture_cancel(ref, *, force, recursive):
        cancel_calls.append((ref, force, recursive))
        original_cancel(ref, force=force, recursive=recursive)

    monkeypatch.setattr(ray, "cancel", capture_cancel)

    run = asyncio.create_task(pool.run(_request([TrajectoryID("a", 0)])))
    try:
        async with asyncio.timeout(30):
            await actor.wait_for_start.remote()
            with pytest.raises(RolloutWorkerStalledError):
                await run

        assert len(cancel_calls) == 1
        cancelled_ref, force, recursive = cancel_calls[0]
        assert isinstance(cancelled_ref, ray.ObjectRef)
        assert force is False
        assert recursive is True
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        ray.kill(actor)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_caller_cancellation_runs_remote_cleanup_without_pool_shutdown(ray_init, spec):
    actor = _BlockingWorker.remote()
    pool = _pool([actor], spec, timeout=60)
    run = asyncio.create_task(pool.run(_request([TrajectoryID("a", 0)])))
    try:
        async with asyncio.timeout(30):
            await actor.wait_for_start.remote()

        run.cancel()
        async with asyncio.timeout(10):
            with pytest.raises(asyncio.CancelledError):
                await run
            assert await actor.wait_for_cleanup.remote()
    finally:
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        ray.kill(actor)


@pytest.mark.asyncio
async def test_pool_preserves_a_remote_timeout_error(spec):
    remote_error = TimeoutError("remote post-processing timed out")

    async def failed_request(_input_batch):
        raise remote_error

    pool = _pool([_Worker(failed_request)], spec)

    with pytest.raises(TimeoutError) as raised:
        await pool.run(_request([TrajectoryID("a", 0)]))

    assert raised.value is remote_error


@dataclass(frozen=True)
class _FailingBuildSpec:
    config: DictConfig
    failure_phase: str

    def build(self, tokenizer, shard: WorkerShard):
        records = Path(self.config.startup_records)
        (records / f"worker-{shard.index}.json").write_text(json.dumps({"pid": os.getpid()}))
        if shard.index == 1 and self.failure_phase == "constructor":
            raise RuntimeError("runner build failed")

        async def lifecycle():
            if shard.index == 1 and self.failure_phase == "startup":
                raise RuntimeError("runner startup failed")

        return SimpleNamespace(startup=lifecycle, shutdown=lifecycle)


@pytest.mark.slow
@pytest.mark.parametrize(("start_interval_seconds", "failure_phase"), [(0, "constructor"), (0.01, "startup")])
@pytest.mark.asyncio
async def test_shutdown_kills_every_worker_after_startup_failure(
    ray_init, tmp_path, start_interval_seconds, failure_phase
):
    model = build_tiny_policy(tmp_path / "model")
    # The driver owns registries so a failed worker cannot terminate them during imports.
    sync_registries()
    records = tmp_path / "records"
    records.mkdir()
    spec = _FailingBuildSpec(
        OmegaConf.create(
            {
                "trainer": {"policy": {"model": {"tokenizer_path": str(model)}}, "disable_fast_tokenizer": False},
                "startup_records": str(records),
            }
        ),
        failure_phase,
    )
    pool = RolloutWorkerPool(
        spec,
        RolloutWorkerResources(
            num_workers=3,
            cpus_per_worker=1,
            executor_threads=1,
            progress_timeout_seconds=30,
            start_interval_seconds=start_interval_seconds,
        ),
    )
    startup_error = ray.exceptions.RayActorError if failure_phase == "constructor" else ray.exceptions.RayTaskError
    with pytest.raises(startup_error, match="runner (build|startup) failed"):
        await asyncio.wait_for(pool.startup(), timeout=120)
    expected_workers = 3 if start_interval_seconds == 0 else 2
    async with asyncio.timeout(30):
        while len(list(records.glob("worker-*.json"))) < expected_workers:
            await asyncio.sleep(0.01)
    live_workers = (0, 2) if failure_phase == "constructor" else range(expected_workers)
    processes = [json.loads((records / f"worker-{index}.json").read_text())["pid"] for index in live_workers]
    assert all(psutil.pid_exists(pid) for pid in processes)
    with pytest.raises(ExceptionGroup) as shutdown_failure:
        await asyncio.wait_for(pool.shutdown(), timeout=30)
    assert all(isinstance(error, startup_error) for error in shutdown_failure.value.exceptions)
    for pid in processes:
        try:
            await asyncio.to_thread(psutil.Process(pid).wait, timeout=15)
        except psutil.NoSuchProcess:
            pass
        assert not psutil.pid_exists(pid)
