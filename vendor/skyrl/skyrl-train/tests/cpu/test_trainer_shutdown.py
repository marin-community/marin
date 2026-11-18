import asyncio
from types import SimpleNamespace

import pytest
from skyrl_train.trainer import RayPPOTrainer


class Clock:
    def __init__(self):
        self.now = 0
        self.timers = []

    def advance(self, seconds):
        self.now += seconds
        for timer in self.timers:
            if not timer.cancelled and self.now >= timer.deadline:
                timer.cancelled = True
                timer.callback()

    def timer(self, timeout, callback):
        clock = self

        class Timer:
            cancelled = False
            daemon = False

            def start(self):
                self.deadline = clock.now + timeout
                self.callback = callback
                clock.timers.append(self)

            def cancel(self):
                self.cancelled = True

        return Timer()


@pytest.mark.asyncio
async def test_shutdown_releases_all_roles_without_a_late_process_failure(monkeypatch):
    clock = Clock()
    exits = []
    releases = []
    monkeypatch.setattr("skyrl_train.trainer.threading.Timer", clock.timer)
    monkeypatch.setattr("skyrl_train.trainer.os._exit", exits.append)

    async def context_close():
        releases.append("context")

    class Resource:
        def __init__(self, label, seconds):
            self.label, self.seconds = label, seconds

        async def close(self):
            clock.advance(self.seconds)
            releases.append(self.label)

        shutdown = close
        teardown = close

    class Sink:
        def close(self):
            releases.append("sink")

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.context = SimpleNamespace(close=context_close)
    trainer._expert_block_sync = Resource("sync", 30)
    trainer._distillation_runtime = Resource("teacher", 30)
    trainer.trajectory_runner = Resource("rollout", 60)
    trainer.eval_trajectory_runner = Resource("evaluation", 60)
    trainer.inference_engine_client = Resource("inference", 30)
    trainer.inference_engine_client.shutdown_http_endpoint = lambda: releases.append("http")
    trainer.inference_engine_client.engines = []
    trainer.trajectory_sink = Sink()
    trainer.eval_trajectory_sink = trainer.trajectory_sink
    trainer._speculator_refresh_task = None
    trainer.policy_model = trainer.critic_model = trainer.ref_model = None

    await trainer.shutdown()
    await trainer.shutdown()
    clock.advance(1000)

    assert exits == []
    assert releases == ["context", "sync", "teacher", "http", "rollout", "evaluation", "sink", "inference"]


def test_stuck_shutdown_still_exits_with_failure(monkeypatch):
    clock = Clock()
    exits = []
    monkeypatch.setattr("skyrl_train.trainer.threading.Timer", clock.timer)
    monkeypatch.setattr("skyrl_train.trainer.os._exit", exits.append)
    RayPPOTrainer._start_exit_watchdog()
    clock.advance(301)
    assert exits == [1]
