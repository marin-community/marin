# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass

import pytest

from forge.effects.context import Context
from forge.effects.events import Event, Fingerprint, Note, ReplayPolicy, Resolved, ScopeEntered, ScopeExited
from forge.effects.issues import Hold, LogFailure, Restart
from forge.effects.log import EventLog, Missing

from .effect_fakes import FlakySink, Number

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
            ctx.emit(Note("first"))
            ctx.emit(Note("second"))
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
            ctx.emit(Note("pending"))
            await started.wait()
            raise Restart("machine gone")
    assert cancelled.is_set()
    await ctx.drain()


async def test_check_awaits_only_declared_issue_kind() -> None:
    ctx = Context(log=EventLog())
    release = asyncio.Event()

    async def hold(_event: Note) -> None:
        raise Hold("guard")

    async def restart(_event: Note) -> None:
        await release.wait()
        raise Restart("retry")

    async with ctx.scope("work"):
        ctx.observe(Note, hold, raises=Hold)
        ctx.observe(Note, restart, raises=Restart)
        ctx.emit(Note("check"))
        with pytest.raises(ExceptionGroup) as caught:
            await ctx.check(Hold)
        assert isinstance(caught.value.exceptions[0], Hold)
        release.set()
        with pytest.raises(ExceptionGroup) as caught:
            await ctx.check(Restart)
        assert isinstance(caught.value.exceptions[0], Restart)
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
        ctx.emit(Note("buffered"))
        await ctx.check(Hold)
        with pytest.raises(RuntimeError):
            await ctx.drain()
    with pytest.raises(ExceptionGroup) as caught:
        await ctx.drain()
    assert all(isinstance(e, LogFailure) for e in caught.value.exceptions)


@pytest.mark.parametrize("policy", [ReplayPolicy.CACHE, ReplayPolicy.REEXECUTE])
async def test_replay_uses_event_and_emit_time_fingerprints_in_any_order(policy: ReplayPolicy) -> None:
    log = EventLog()
    fingerprint = Fingerprint("fake", "before", policy)
    await log.record(Resolved(Number(1), (fingerprint,), 10))
    await log.record(Resolved(Number(2), (fingerprint,), 20))
    ctx = Context(log=log)
    resource = object()
    current = fingerprint
    executed: list[int] = []

    async def execute(event: Number, obj: object) -> int:
        assert obj is resource
        executed.append(event.value)
        return event.value * 100

    ctx.resource(resource, lambda: current)
    ctx.handle(Number, execute, uses=(resource,))
    async with ctx.scope("replay"):
        second = ctx.emit(Number(2))
        current = Fingerprint("fake", "after", policy)
        assert await second == (20 if policy == ReplayPolicy.CACHE else 200)
        assert await ctx.emit(Number(1)) == 100
        current = fingerprint
        assert await ctx.emit(Number(1)) == (10 if policy == ReplayPolicy.CACHE else 100)
    await ctx.drain()
    assert executed == ([1] if policy == ReplayPolicy.CACHE else [2, 1, 1])
    assert log.resolved()[-1].replayed == (policy == ReplayPolicy.CACHE)
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
        pass

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


async def test_unawaited_executor_failure_is_reported_at_scope_exit() -> None:
    ctx = Context(log=EventLog())

    async def fail(_event: Number) -> int:
        raise Restart("lost effect")

    ctx.handle(Number, fail)
    with pytest.raises(ExceptionGroup) as caught:
        async with ctx.scope("work"):
            ctx.emit(Number(1))
    assert isinstance(caught.value.exceptions[0], Restart)
    await ctx.drain()


async def test_scope_joins_effects_started_by_observers() -> None:
    log = EventLog()
    ctx = Context(log=log)

    async def start_effect(_event: Note) -> None:
        ctx.emit(Number(1))

    async def fail(_event: Number) -> int:
        raise Restart("observer child failed")

    ctx.handle(Number, fail)
    ctx.observe(Note, start_effect)
    with pytest.raises(ExceptionGroup) as caught:
        async with ctx.scope("work"):
            ctx.emit(Note("start"))
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
