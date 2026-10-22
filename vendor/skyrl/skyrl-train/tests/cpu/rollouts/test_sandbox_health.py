"""Sandbox canary and event-loop lag alarms, driven by a fake machine factory and a fake clock."""

import asyncio
from dataclasses import dataclass, field, replace

import pytest
from loguru import logger
from shellbox.machine import Backend, Command, ExitReason, MachineSpec, Result

from skyrl_train.rollouts.sandbox_health import (
    SandboxHealthConfig,
    SandboxHealthProbe,
    SandboxHealthWatch,
    WorkerHealthReport,
)

PREFIX = "rollout_workers"
CONFIG = SandboxHealthConfig(
    enabled=True,
    backend="local",
    canary_interval_seconds=10.0,
    canary_timeout_seconds=60.0,
    event_loop_lag_interval_seconds=1.0,
    collect_interval_seconds=30.0,
    baseline_samples=3,
    slow_baseline_ratio=10.0,
    slow_seconds=2.0,
)


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeMachine:
    factory: "FakeFactory"
    closed: bool = False

    async def run(self, command: Command) -> Result:
        self.factory.clock.now += self.factory.run_seconds
        if self.factory.run_error is not None:
            raise self.factory.run_error
        if self.factory.run_blocks:
            await asyncio.Event().wait()
        return Result(0, b"", b"", False, False, ExitReason.EXITED)

    async def upload(self, source, target) -> None:
        raise NotImplementedError

    async def download(self, source, target) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        self.factory.clock.now += 0.01
        self.closed = True


@dataclass
class FakeFactory:
    """A machine backend whose command latency, failure and hang the test sets."""

    clock: FakeClock
    run_seconds: float = 0.08
    run_error: Exception | None = None
    run_blocks: bool = False
    machines: list[FakeMachine] = field(default_factory=list)
    backend: Backend = Backend.LOCAL

    def check_environment(self, requirements, spec: MachineSpec) -> None:
        pass

    async def create(self, spec: MachineSpec) -> FakeMachine:
        self.clock.now += 0.01
        machine = FakeMachine(self)
        self.machines.append(machine)
        return machine


@pytest.fixture
def alarms():
    """The alarm field of each warning or error the canary code logs."""
    records = []
    handler = logger.add(lambda message: records.append(message.record), level="WARNING")
    yield lambda: [(record["level"].name, record["extra"].get("alarm")) for record in records]
    logger.remove(handler)


def report_of(probe: SandboxHealthProbe):
    async def report() -> WorkerHealthReport:
        return probe.report()

    return report


@pytest.mark.asyncio
async def test_slow_canary_against_worker_baseline_raises_alarm_and_step_metric(alarms):
    clock = FakeClock()
    factory = FakeFactory(clock)
    probe = SandboxHealthProbe(CONFIG, factory, host="node-a", clock=clock)
    watch = SandboxHealthWatch(CONFIG)
    for _ in range(3):
        await probe.canary()
    # 1.2 s is under the 2 s absolute bound but over 10x this worker's 0.1 s baseline.
    factory.run_seconds = 1.2
    await probe.canary()
    await watch.collect([report_of(probe)])

    metrics = watch.step_metrics(PREFIX)

    assert metrics[f"{PREFIX}/sandbox_canary_slow"] == 1
    assert metrics[f"{PREFIX}/sandbox_canary_alarm"] == 1
    assert metrics[f"{PREFIX}/sandbox_canary_slow_threshold_seconds"] == pytest.approx(1.0)
    assert metrics[f"{PREFIX}/sandbox_canary_seconds/p50"] == pytest.approx(0.1)
    assert metrics[f"{PREFIX}/sandbox_canary_seconds/max"] == pytest.approx(1.22)
    assert metrics[f"{PREFIX}/sandbox_canary_run_seconds/max"] == pytest.approx(1.2)
    assert metrics[f"{PREFIX}/worker_0/sandbox_canary_max_seconds"] == pytest.approx(1.22)
    assert alarms() == [("WARNING", "sandbox_canary_slow"), ("WARNING", "sandbox_canary")]
    assert all(machine.closed for machine in factory.machines)

    factory.run_seconds = 0.08
    await probe.canary()
    await watch.collect([report_of(probe)])
    recovered = watch.step_metrics(PREFIX)

    assert recovered[f"{PREFIX}/sandbox_canary_alarm"] == 0
    assert recovered[f"{PREFIX}/sandbox_canary_samples"] == 1


@pytest.mark.asyncio
async def test_failed_and_hung_canaries_are_counted_and_their_machines_closed(alarms):
    clock = FakeClock()
    factory = FakeFactory(clock, run_error=OSError("bwrap: clone failed"))
    probe = SandboxHealthProbe(replace(CONFIG, canary_timeout_seconds=0.05), factory, host="node-a", clock=clock)
    watch = SandboxHealthWatch(CONFIG)
    await probe.canary()
    factory.run_error = None
    factory.run_blocks = True
    await probe.canary()
    await watch.collect([report_of(probe)])

    metrics = watch.step_metrics(PREFIX)

    assert metrics[f"{PREFIX}/sandbox_canary_failures"] == 1
    assert metrics[f"{PREFIX}/sandbox_canary_timeouts"] == 1
    assert metrics[f"{PREFIX}/sandbox_canary_alarm"] == 1
    assert [machine.closed for machine in factory.machines] == [True, True]
    assert ("WARNING", "sandbox_canary_failed") in alarms()
    assert ("WARNING", "sandbox_canary_timed_out") in alarms()


@pytest.mark.asyncio
async def test_unreachable_hung_or_silent_workers_are_errors(alarms):
    watch = SandboxHealthWatch(CONFIG, report_timeout_seconds=0.05)

    async def unreachable() -> WorkerHealthReport:
        raise ConnectionError("actor died")

    async def hung() -> WorkerHealthReport:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def silent() -> WorkerHealthReport:
        return WorkerHealthReport(True, "node-c", (), (0.01,), 0, CONFIG.silent_after_seconds + 1)

    async def without_backend() -> WorkerHealthReport:
        return WorkerHealthReport(False, "node-d", (), (0.01,), 0, 10_000.0)

    reports = [unreachable, hung, silent, without_backend]
    await watch.collect(reports)
    await watch.collect(reports)

    metrics = watch.step_metrics(PREFIX)

    assert metrics[f"{PREFIX}/sandbox_canary_missing_workers"] == 3
    assert metrics[f"{PREFIX}/sandbox_canary_samples"] == 0
    assert metrics[f"{PREFIX}/sandbox_canary_alarm"] == 1
    # One error per worker when its readings go missing, not one per collection.
    assert alarms().count(("ERROR", "sandbox_canary_missing")) == 3


@pytest.mark.asyncio
async def test_event_loop_lag_reaches_step_metrics_without_a_canary_backend():
    clock = FakeClock()
    lags = iter([0.0, 0.5, 3.0])
    blocked = asyncio.Event()

    async def wait(seconds: float) -> None:
        lag = next(lags, None)
        if lag is None:
            blocked.set()
            await asyncio.Event().wait()
        clock.now += seconds + lag

    probe = SandboxHealthProbe(CONFIG, None, host="node-a", clock=clock, wait=wait)
    watch = SandboxHealthWatch(CONFIG)
    probe.start()
    await blocked.wait()
    await watch.collect([report_of(probe)])
    await probe.stop()

    metrics = watch.step_metrics(PREFIX)

    assert metrics[f"{PREFIX}/event_loop_lag_seconds/max"] == pytest.approx(3.0)
    assert metrics[f"{PREFIX}/event_loop_lag_seconds/p50"] == pytest.approx(0.5)
    assert metrics[f"{PREFIX}/sandbox_canary_workers"] == 0
    assert metrics[f"{PREFIX}/sandbox_canary_alarm"] == 0
