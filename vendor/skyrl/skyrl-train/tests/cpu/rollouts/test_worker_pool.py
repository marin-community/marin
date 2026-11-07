import asyncio
import json
import os
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
import psutil
import pytest
import ray
from omegaconf import DictConfig, OmegaConf
from tests.cpu.tiny_training.tiny_model import build_tiny_policy
from skyrl_train.utils.algorithm_registry import sync_registries
from skyrl_train.rollouts.workers import RolloutWorkerPool, RolloutWorkerResources, WorkerShard


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


def _worker_resources(start_interval_seconds):
    return RolloutWorkerResources(
        num_workers=3,
        cpus_per_worker=1,
        executor_threads=1,
        progress_timeout_seconds=30,
        start_interval_seconds=start_interval_seconds,
    )


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
                "generator": {"chat_template": None},
                "startup_records": str(records),
            }
        ),
        failure_phase,
    )
    pool = RolloutWorkerPool(spec, _worker_resources(start_interval_seconds))
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
