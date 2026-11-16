"""Rollout actor events reach the exporter and flush before the actor exits."""

from types import SimpleNamespace

import pytest

from skyrl_train.rollouts import workers
from skyrl_train.telemetry import record_event


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_fails", [False, True])
async def test_rollout_actor_exports_events_and_drains_on_shutdown(monkeypatch, telemetry_endpoint, shutdown_fails):
    class Producer:
        async def run(self, _request):
            record_event("rollout_completion_probe", {"uid": "task-7"}, attributes={"source": "science"})
            return {"response_ids": [[7]]}

        async def shutdown(self):
            if shutdown_fails:
                raise RuntimeError("runner shutdown failed")

    for name in ("configure_worker_process", "configure_ray_worker_logging", "start_fd_monitor"):
        monkeypatch.setattr(workers, name, lambda: None)
    monkeypatch.setattr(workers, "tokenizer_from_config", lambda _config: None)
    spec = SimpleNamespace(config=None, build=lambda _tokenizer, _shard: Producer())
    actor = workers.RolloutWorker.__ray_metadata__.modified_class(spec, workers.WorkerShard(0, 1), None, 1)
    await actor.run(None, False)
    if shutdown_fails:
        with pytest.raises(RuntimeError, match="runner shutdown failed"):
            await actor.shutdown()
    else:
        await actor.shutdown()
    [completion] = telemetry_endpoint.select("rollout_completion_probe", source="science")
    assert completion["body"] == {"uid": "task-7"}
    assert completion["resource"]["role"] == "worker"
    [terminal] = telemetry_endpoint.select("terminal", role="worker")
    assert terminal["body"]["status"] == ("failed" if shutdown_fails else "completed")
