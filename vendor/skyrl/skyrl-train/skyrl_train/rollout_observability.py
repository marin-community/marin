"""Wall time and waits of each rollout call and of the loop that dispatches rollout tasks."""

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from concurrent.futures import Executor
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
from uuid import uuid4

from skyrl_train.telemetry import TRAINER_ROLE, record_event, telemetry
from skyrl_train.timing_observability import PhaseBreakdown


RolloutPhase = Literal["collect", "assemble", "finalize", "tokenize", "retain"]
_PARENTS = {"rollout_tokenize": "rollout_collect", "rollout_retain": "rollout_finalize"}
_CURRENT: ContextVar["RolloutObservation | None"] = ContextVar("rollout_observation", default=None)
wait_seconds = telemetry.histogram("rollout_wait_seconds", unit="s")
waits = telemetry.counter("rollout_waits", unit="{wait}")
buffer_dwell = telemetry.histogram("rollout_buffer_dwell_seconds", unit="s")
groups = telemetry.counter("rollout_groups", unit="{group}")
group_tokens = telemetry.counter("rollout_group_tokens", unit="{token}")
event_loop_lag = telemetry.histogram("event_loop_lag_seconds", unit="s")


def _unix_ms() -> int:
    return time.time_ns() // 1_000_000


def _window(duration: float) -> dict[str, int]:
    """Unix start and finish of an interval that just ended, spaced by its monotonic duration."""
    finished = _unix_ms()
    return {"started_unix_ms": finished - round(duration * 1000), "finished_unix_ms": finished}


@contextlib.contextmanager
def async_phase_window(phase: str, *, step: int, enabled: bool) -> Iterator[None]:
    """Record a driver phase's window for joins with other processes' records."""
    if not enabled:
        yield
        return
    started = time.perf_counter()
    outcome = "success"
    try:
        yield
    except BaseException:
        outcome = "failure"
        raise
    finally:
        duration = time.perf_counter() - started
        record_event(
            "async_phase_window",
            {**_window(duration), "duration_seconds": duration},
            attributes={"phase": phase, "step": str(step), "role": TRAINER_ROLE, "outcome": outcome},
        )


async def monitor_event_loop_lag(
    *,
    step_fn: Callable[[], int],
    mode: str,
    interval: float = 1.0,
    clock: Callable[[], float] = time.perf_counter,
    wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Sample driver scheduling delay until cancelled, without replaying missed ticks."""
    while True:
        expected = clock() + interval
        await wait(interval)
        event_loop_lag.record(
            max(0.0, clock() - expected),
            attributes={"role": TRAINER_ROLE, "step": str(step_fn()), "mode": mode},
        )


def publish_wait(name: str, durations: Sequence[float], *, step: int, mode: str) -> None:
    attributes = {"wait": name, "role": TRAINER_ROLE, "step": str(step), "mode": mode}
    wait_seconds.record(sum(durations), attributes={**attributes, "stat": "sum"})
    wait_seconds.record(max(durations), attributes={**attributes, "stat": "max"})
    waits.add(len(durations), attributes=attributes)


@dataclass(frozen=True)
class RolloutTimings:
    """The waits and phase durations one process measured for a rollout call."""

    waits: dict[str, list[float]]
    phases: dict[str, float]


@dataclass
class RolloutObservation:
    phases: PhaseBreakdown
    call_id: str = field(default_factory=lambda: uuid4().hex)
    waits: dict[str, list[float]] = field(default_factory=dict)
    response_tokens: int = 0

    def add_wait(self, name: str, seconds: float) -> None:
        self.waits.setdefault(name, []).append(seconds)

    def timings(self) -> RolloutTimings:
        return RolloutTimings(
            {name: list(durations) for name, durations in self.waits.items()}, self.phases.durations()
        )

    def absorb(self, timings: RolloutTimings) -> None:
        """Add what a rollout worker measured while serving this call."""
        for name, durations in timings.waits.items():
            self.waits.setdefault(name, []).extend(durations)
        for phase, seconds in timings.phases.items():
            self.phases.add(phase, seconds)


def current_rollout_observation() -> RolloutObservation | None:
    return _CURRENT.get()


@contextlib.contextmanager
def measure_rollout(
    *, enabled: bool, clock: Callable[[], float] = time.perf_counter
) -> Iterator[RolloutObservation | None]:
    """Collect the enclosed rollout call's waits and phases in a new observation, without publishing them."""
    if not enabled:
        yield None
        return
    observation = RolloutObservation(PhaseBreakdown("rollout_call", _PARENTS, enabled=True, clock=clock))
    token = _CURRENT.set(observation)
    try:
        yield observation
    finally:
        _CURRENT.reset(token)


@contextlib.contextmanager
def observe_rollout_call(
    *, step: int, mode: str, enabled: bool, clock: Callable[[], float] = time.perf_counter
) -> Iterator[RolloutObservation | None]:
    if not enabled:
        yield None
        return
    outcome = "success"
    with measure_rollout(enabled=True, clock=clock) as observation:
        try:
            yield observation
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "failure"
            raise
        finally:
            attributes = {"role": TRAINER_ROLE, "step": str(step), "mode": mode, "outcome": outcome}
            duration = observation.phases.publish(clock_domain="driver_monotonic", attributes=attributes)
            for name, durations in observation.waits.items():
                publish_wait(name, durations, step=step, mode=mode)
            record_event(
                "rollout_call",
                {"call_id": observation.call_id, **_window(duration), "response_tokens": observation.response_tokens},
                attributes=attributes,
            )


@contextlib.contextmanager
def rollout_phase(name: RolloutPhase) -> Iterator[None]:
    observation = _CURRENT.get()
    with observation.phases.span(f"rollout_{name}") if observation is not None else contextlib.nullcontext():
        yield


@contextlib.contextmanager
def rollout_wait(name: str) -> Iterator[None]:
    observation = _CURRENT.get()
    if observation is None:
        yield
        return
    started = observation.phases.clock()
    try:
        yield
    finally:
        observation.add_wait(name, observation.phases.clock() - started)


@contextlib.contextmanager
def dispatch_wait(name: str, *, step: int, mode: str, enabled: bool) -> Iterator[None]:
    """Measure an await of the rollout dispatch loop outside the rollout call."""
    if not enabled:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        publish_wait(name, [time.perf_counter() - started], step=step, mode=mode)


async def run_environment(executor: Executor | None, func: Callable, *args, **kwargs):
    """Run an environment operation in a thread and record queue, execution, and resume delays."""
    observation = _CURRENT.get()
    call = partial(func, *args, **kwargs)
    if observation is None:
        return await asyncio.get_running_loop().run_in_executor(executor, call)
    clock = observation.phases.clock
    submitted = clock()
    stamps: list[float] = []

    def invoke():
        stamps.append(clock())
        try:
            return call()
        finally:
            stamps.append(clock())

    try:
        with rollout_wait("env_await"):
            return await asyncio.get_running_loop().run_in_executor(executor, invoke)
    finally:
        # A cancelled call can leave its thread running; that thread must not add to a published call.
        if len(stamps) == 2:
            observation.add_wait("env_queue", stamps[0] - submitted)
            observation.add_wait("env_exec", stamps[1] - stamps[0])
            observation.add_wait("env_resume", clock() - stamps[1])


def record_group_disposition(*, disposition: str, tokens: int, step: int, dwell_seconds: float | None) -> None:
    """Count a group's fate and, when known, how long it waited in the buffer."""
    attributes = {"role": TRAINER_ROLE, "step": str(step), "disposition": disposition}
    groups.add(1, attributes=attributes)
    group_tokens.add(tokens, attributes=attributes)
    if dwell_seconds is not None:
        buffer_dwell.record(dwell_seconds, attributes=attributes)
