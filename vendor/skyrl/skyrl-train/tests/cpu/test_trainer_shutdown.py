import asyncio
from collections import Counter
from dataclasses import dataclass
import threading
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


@dataclass
class ShutdownCase:
    trainer: RayPPOTrainer
    clock: Clock
    exits: list[int]
    releases: list[str]


@pytest.fixture
def shutdown_case(monkeypatch):
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
    # Each service finishes within its own deadline; their sum exceeds the old watchdog.
    short_cleanup = 20
    runner_cleanup = 50
    trainer._expert_block_sync = Resource("sync", short_cleanup)
    trainer._distillation_runtime = Resource("teacher", short_cleanup)
    trainer.trajectory_runner = Resource("rollout", runner_cleanup)
    trainer.eval_trajectory_runner = Resource("evaluation", runner_cleanup)
    trainer.inference_engine_client = Resource("inference", short_cleanup)
    trainer.inference_engine_client.shutdown_http_endpoint = lambda: releases.append("http")
    trainer.inference_engine_client.engines = []
    trainer.trajectory_sink = Sink()
    trainer.eval_trajectory_sink = trainer.trajectory_sink
    trainer._speculator_refresh_task = None
    trainer.policy_model = trainer.critic_model = trainer.ref_model = None

    return ShutdownCase(trainer, clock, exits, releases)


def test_shutdown_releases_all_roles_without_a_late_process_failure(shutdown_case):
    trainer, clock = shutdown_case.trainer, shutdown_case.clock
    asyncio.run(trainer.shutdown())
    asyncio.run(trainer.shutdown())
    trainer.finish_shutdown()
    clock.advance(1000)

    assert shutdown_case.exits == []
    assert Counter(shutdown_case.releases) == Counter(["context", "sync", "teacher", "http", "rollout", "evaluation", "sink", "inference"])


def test_shutdown_keeps_process_guard_while_executor_work_remains(shutdown_case, monkeypatch):
    trainer, clock = shutdown_case.trainer, shutdown_case.clock
    release_thread = threading.Event()

    def process_exit(code):
        shutdown_case.exits.append(code)
        release_thread.set()

    monkeypatch.setattr("skyrl_train.trainer.os._exit", process_exit)

    async def shutdown_with_executor():
        pending = asyncio.get_running_loop().run_in_executor(None, release_thread.wait)
        try:
            await trainer.shutdown()
            clock.advance(1000)
        finally:
            release_thread.set()
            await pending

    asyncio.run(shutdown_with_executor())
    trainer.finish_shutdown()
    assert shutdown_case.exits == [1]
