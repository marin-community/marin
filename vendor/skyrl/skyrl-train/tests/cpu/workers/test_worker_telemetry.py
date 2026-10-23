from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

from skyrl_train.workers.worker import PPORayActorGroup, Worker

_WORKER_CONFIG = OmegaConf.create(
    {
        "trainer": {
            "progress": {
                "mode": "tqdm",
                "min_interval_seconds": 0.5,
                "heartbeat_seconds": 15,
                "percent_step": 5,
                "count_step": 1000,
            },
            "numerics": {"mode": "native"},
        }
    }
)


@pytest.fixture(autouse=True)
def _isolated_rendezvous_environment(monkeypatch):
    # DistributedTorchRayActor writes the rendezvous variables into os.environ; keep them out of other tests.
    for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)


def _build_worker() -> Worker:
    return Worker(
        cfg=_WORKER_CONFIG,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )


@pytest.mark.parametrize("endpoint", [True, False])
def test_a_worker_reports_its_lifecycle_only_with_an_endpoint(telemetry_endpoint, monkeypatch, endpoint):
    if not endpoint:
        monkeypatch.delenv("SKYRL_TELEMETRY_ENDPOINT")
    worker = _build_worker()
    worker.close_telemetry()

    delivered = [(row["name"], row["attributes"]["role"], row["resource"]["run_id"]) for row in telemetry_endpoint.rows]
    assert delivered == (
        [("lifecycle", "worker", "telemetry-test"), ("terminal", "worker", "telemetry-test")] if endpoint else []
    )


def test_the_actor_group_drains_every_worker_before_killing_it(monkeypatch):
    """ray.kill runs no atexit handler in the actor, so the drain must be requested and awaited first."""
    order: list[str] = []
    drained = object()

    class Handle:
        def __init__(self, name):
            self.name = name
            self.close_telemetry = SimpleNamespace(remote=self._close)

        def _close(self):
            order.append(f"drain {self.name}")
            return drained

    handles = [Handle("rank0"), Handle("rank1")]
    monkeypatch.setattr(
        ray, "wait", lambda refs, *, num_returns, timeout: order.append(f"wait {num_returns} {timeout}")
    )
    monkeypatch.setattr(ray, "kill", lambda actor, *, no_restart: order.append(f"kill {actor.name} {no_restart}"))
    group = PPORayActorGroup.__new__(PPORayActorGroup)
    group._actor_handlers = handles

    group.kill_actors()

    assert order == ["drain rank0", "drain rank1", "wait 2 5.0", "kill rank0 True", "kill rank1 True"]
