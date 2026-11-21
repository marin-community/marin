# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import FlakySink, Number

from taskforge.effects.context import Context
from taskforge.effects.events import Event, Fingerprint, Note, ReplayPolicy, Resolved, ScopeEntered, ScopeExited
from taskforge.effects.issues import Hold, LogFailure, PublicationFailed, Restart
from taskforge.effects.log import EventLog, Missing, Record, Sink

pytestmark = pytest.mark.asyncio


async def test_inner_handler_wins_and_all_observers_see_resolutions() -> None:
    log = EventLog()
    ctx = Context(log=log)
    outer: list[Event[object]] = []
    inner: list[Event[object]] = []

    async def observe_outer(event: Event[object]) -> None:
        outer.append(event)

    async def observe_inner(event: Event[object]) -> None:
        inner.append(event)

    async def doubled(event: Number) -> int:
        return event.value * 2

    async def tripled(event: Number) -> int:
        return event.value * 3

    ctx.observe(Event, observe_outer)
    ctx.handle(Number, doubled)
    async with ctx.scope("outer"):
        assert await ctx.emit(Number(2)) == 4
        async with ctx.scope("inner"):
            ctx.handle(Number, tripled)
            ctx.observe(Event, observe_inner)
            assert await ctx.emit(Number(3)) == 9
        assert await ctx.emit(Number(4)) == 8
    await ctx.drain()
    assert [e.value for e in outer if isinstance(e, Resolved)] == [4, 9, 8]
    assert [e.value for e in inner if isinstance(e, Resolved)] == [9]
    assert any(isinstance(e, ScopeEntered) and e.name == "inner" for e in outer)
    assert any(isinstance(e, ScopeExited) and not e.failed for e in outer)


async def test_child_owns_parent_observer_issues_and_clean_exit_groups_them() -> None:
    ctx = Context(log=EventLog())

    async def stop(event: Note) -> None:
        raise Hold(event.msg)

    ctx.observe(Note, stop)
    with pytest.raises(ExceptionGroup) as caught:
        async with ctx.scope("child"):
            ctx.notify(Note("first"))
            ctx.notify(Note("second"))
    assert [str(e) for e in caught.value.exceptions] == ["first", "second"]
    await ctx.drain()


async def test_exceptional_scope_cancels_observers_without_masking_primary_error() -> None:
    ctx = Context(log=EventLog())
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def waiting(_event: Note) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    ctx.observe(Note, waiting)
    with pytest.raises(Restart, match="machine gone"):
        async with ctx.scope("child"):
            ctx.notify(Note("pending"))
            await started.wait()
            raise Restart("machine gone")
    assert cancelled.is_set()
    await ctx.drain()


async def test_check_raises_observer_issues_so_far_and_scope_exit_raises_later_ones() -> None:
    ctx = Context(log=EventLog())

    async def hold(event: Note) -> None:
        raise Hold(event.msg)

    with pytest.raises(ExceptionGroup) as at_exit:
        async with ctx.scope("work"):
            ctx.observe(Note, hold)
            ctx.notify(Note("checked"))
            with pytest.raises(ExceptionGroup) as checked:
                await ctx.check()
            assert [str(e) for e in checked.value.exceptions] == ["checked"]
            await ctx.check()
            ctx.notify(Note("later"))
    assert [str(e) for e in at_exit.value.exceptions] == ["later"]
    await ctx.drain()


async def test_persistence_retries_and_failure_is_deferred_to_root_drain() -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    transient = FlakySink(2)
    log = EventLog(sink=transient, sleep=fake_sleep)
    await log.record(Note("durable"))
    assert [record.event for record in transient.records] == [Note("durable")]
    assert delays == [1, 2]
    ctx = Context(log=EventLog(sink=FlakySink(100), sleep=fake_sleep))
    async with ctx.scope("work"):
        ctx.notify(Note("buffered"))
        await ctx.check()
        with pytest.raises(RuntimeError):
            await ctx.drain()
    with pytest.raises(ExceptionGroup) as caught:
        await ctx.drain()
    assert all(isinstance(e, LogFailure) for e in caught.value.exceptions)


@dataclass(slots=True)
class Shifting:
    current: Fingerprint

    def fingerprint(self) -> Fingerprint:
        return self.current


async def test_cached_replay_keys_on_the_event_and_the_resource_fingerprint_at_emission() -> None:
    policy = ReplayPolicy.CACHE
    fingerprint = Fingerprint("fake", "before", policy)
    previous = EventLog()
    await previous.record(Resolved(Number(1), (fingerprint,), 10))
    await previous.record(Resolved(Number(2), (fingerprint,), 20))
    log = EventLog(replay_from=previous.records)
    ctx = Context(log=log)
    resource = Shifting(fingerprint)
    executed: list[tuple[int, int | Missing]] = []

    async def execute(event: Number, obj: Shifting) -> int:
        assert obj is resource
        executed.append((event.value, Missing()))
        return event.value * 100

    ctx.handle(Number, execute, resource)
    async with ctx.scope("replay"):
        assert await ctx.emit(Number(2)) == 20
        resource.current = Fingerprint("fake", "after", policy)
        assert await ctx.emit(Number(1)) == 100
        resource.current = fingerprint
        assert await ctx.emit(Number(1)) == 10
    await ctx.drain()
    assert executed == [(1, Missing())]
    assert [r.event for r in log.records if isinstance(r.event, Resolved)][-1].replayed
    assert isinstance(log.lookup(Number(9), (fingerprint,)), Missing)


async def test_concurrent_scopes_keep_handlers_and_observers_separate() -> None:
    ctx = Context(log=EventLog())
    arrived = [asyncio.Event(), asyncio.Event()]
    seen: list[list[int]] = [[], []]

    async def worker(index: int) -> int:
        async def execute(event: Number) -> int:
            return event.value + index

        async def observe(event: Number) -> None:
            seen[index].append(event.value)

        async with ctx.scope(str(index)):
            ctx.handle(Number, execute)
            ctx.observe(Number, observe)
            arrived[index].set()
            await arrived[1 - index].wait()
            return await ctx.emit(Number(index * 10))

    assert await asyncio.gather(worker(0), worker(1)) == [0, 11]
    await ctx.drain()
    assert seen == [[0], [10]]


async def test_none_result_is_a_cache_hit_and_executor_failures_propagate_immediately() -> None:
    @dataclass(frozen=True, slots=True)
    class NoValue(Event[None]):
        kind = "test.no_value"

    log = EventLog()
    await log.record(Resolved(NoValue(), (), None))
    ctx = Context(log=log)

    async def execute(_event: NoValue) -> None:
        raise Restart("must not execute")

    async def fail(_event: Number) -> int:
        raise Restart("executor exhausted")

    ctx.handle(NoValue, execute)
    ctx.handle(Number, fail)
    async with ctx.scope("work"):
        assert await ctx.emit(NoValue()) is None
        with pytest.raises(Restart, match="executor exhausted"):
            await ctx.emit(Number(1))
    await ctx.drain()


async def test_scope_raises_failures_of_effects_its_observers_run() -> None:
    log = EventLog()
    ctx = Context(log=log)

    async def start_effect(_event: Note) -> None:
        await ctx.emit(Number(1))

    async def fail(_event: Number) -> int:
        raise Restart("observer child failed")

    ctx.handle(Number, fail)
    ctx.observe(Note, start_effect)
    with pytest.raises(ExceptionGroup) as caught:
        async with ctx.scope("work"):
            ctx.notify(Note("start"))
    assert isinstance(caught.value.exceptions[0], Restart)
    await ctx.drain()


async def test_failed_scope_exit_reaches_observers_without_masking_original_failure() -> None:
    log = EventLog()
    ctx = Context(log=log)
    seen: list[ScopeExited] = []

    async def exited(event: ScopeExited) -> None:
        seen.append(event)
        raise Hold("exit observer")

    ctx.observe(ScopeExited, exited)
    with pytest.raises(Restart, match="primary"):
        async with ctx.scope("failed"):
            raise Restart("primary")
    await ctx.drain()
    assert seen == [ScopeExited("failed", True)]
    assert any(record.event == seen[0] for record in log.records)


class ReentrantSink(Sink):
    """Notes a diagnostic while it writes ``outer``, and refuses to write that diagnostic."""

    def __init__(self) -> None:
        self.ctx: Context | None = None

    @override
    async def write(self, records: Sequence[Record]) -> None:
        events = [record.event for record in records]
        if Note("diagnostic") in events:
            raise PublicationFailed("diagnostic refused")
        if Note("outer") in events and self.ctx is not None:
            self.ctx.notify(Note("diagnostic"))


# Review 2026-11-20: a recording or observer run started while another was starting replaced the collection the outer
# one was appended to, so its failure was lost and drain returned with a row pending.
async def test_a_recording_started_while_another_is_written_surfaces_its_failure_at_drain() -> None:
    sink = ReentrantSink()
    ctx = Context(log=(log := EventLog(sink=sink)))
    sink.ctx = ctx
    ctx.notify(Note("outer"))
    with pytest.raises(ExceptionGroup) as caught:
        await ctx.drain()
    assert [type(error) for error in caught.value.exceptions] == [PublicationFailed]
    assert [record.event for record in log.records] == [Note("outer"), Note("diagnostic")]


async def test_an_observer_run_started_while_another_starts_reports_its_hold() -> None:
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        ctx = Context(log=EventLog())

        async def relay(event: Note) -> None:
            if event.msg == "outer":
                ctx.notify(Note("nested"))

        async def refuse(event: Note) -> None:
            if event.msg == "nested":
                raise Hold("nested observer")

        with pytest.raises(ExceptionGroup) as caught:
            async with ctx.scope("work"):
                ctx.observe(Note, relay)
                ctx.observe(Note, refuse)
                ctx.notify(Note("outer"))
        assert [str(error) for error in caught.value.exceptions] == ["nested observer"]
        await ctx.drain()
    finally:
        loop.set_task_factory(None)
