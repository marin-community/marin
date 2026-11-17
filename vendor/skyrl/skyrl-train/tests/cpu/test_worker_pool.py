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

    def build(self, tokenizer, shard: WorkerShard):
        records = Path(self.config.startup_records)
        (records / f"worker-{shard.index}.json").write_text(json.dumps({"pid": os.getpid()}))

        async def startup():
            if shard.index == 1:
                raise RuntimeError("runner startup failed")

        async def shutdown():
            pass

        return SimpleNamespace(startup=startup, shutdown=shutdown)


@pytest.mark.slow
@pytest.mark.parametrize("start_interval_seconds", [0, 0.01])
@pytest.mark.asyncio
async def test_shutdown_kills_every_worker_after_startup_failure(ray_init, tmp_path, start_interval_seconds):
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
    # Retain the failed startup task while shutdown releases its workers.
    startup = asyncio.create_task(pool.startup())
    with pytest.raises(ray.exceptions.RayTaskError, match="runner startup failed"):
        await asyncio.wait_for(startup, timeout=120)
    expected_workers = 3 if start_interval_seconds == 0 else 2
    async with asyncio.timeout(30):
        while len(list(records.glob("worker-*.json"))) < expected_workers:
            await asyncio.sleep(0.01)
    processes = [json.loads((records / f"worker-{index}.json").read_text())["pid"] for index in range(expected_workers)]
    assert all(psutil.pid_exists(pid) for pid in processes)
    await asyncio.wait_for(pool.shutdown(), timeout=30)
    for pid in processes:
        try:
            await asyncio.to_thread(psutil.Process(pid).wait, timeout=15)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
        assert not psutil.pid_exists(pid), "worker survived shutdown"
