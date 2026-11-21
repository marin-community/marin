# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, cast

from taskforge.effects.events import Event, Fingerprint, ReplayPolicy, Resolved, ScopeEntered, ScopeExited
from taskforge.effects.log import EventLog, Missing

type Executor = Callable[..., Awaitable[object]]
type Observer[E: Event[object]] = Callable[[E], Awaitable[None]]


class Resource(Protocol):
    def fingerprint(self) -> Fingerprint: ...


@dataclass(frozen=True, slots=True)
class _Handler:
    etype: type[Event[object]]
    fn: Executor
    resource: Resource | None


class _Scope:
    def __init__(self) -> None:
        self.handlers: tuple[_Handler, ...] = ()
        self.observers: tuple[tuple[type[Event[object]], Observer[Event[object]]], ...] = ()
        self.observed: tuple[asyncio.Future[None], ...] = ()
        """Observer runs this scope's events started; their failures are the scope's issues."""

    def take(self) -> tuple[asyncio.Future[None], ...]:
        runs, self.observed = self.observed, ()
        return runs


async def _failures(take: Callable[[], tuple[asyncio.Future[None], ...]]) -> tuple[BaseException, ...]:
    """Join the runs ``take`` hands over, including runs they start, and return their failures."""
    errors: tuple[BaseException, ...] = ()
    while runs := take():
        results = await asyncio.gather(*runs, return_exceptions=True)
        errors += tuple(result for result in results if isinstance(result, BaseException))
    return errors


def _raise(errors: tuple[BaseException, ...]) -> None:
    if errors:
        raise BaseExceptionGroup("Effect observer issues", errors)


class Context:
    """Execute effects in task-local scopes, record every event, and own observer lifetimes.

    Observer failures are their scope's issues, raised by ``check`` or when it exits cleanly. Recording failures are
    deferred to ``drain``.
    """

    def __init__(self, *, log: EventLog) -> None:
        self._log = log
        self._root = _Scope()
        self._recording: tuple[asyncio.Future[None], ...] = ()
        self._chain: ContextVar[tuple[_Scope, ...]] = ContextVar("forge_scopes", default=(self._root,))

    def handle(self, etype: type[Event[object]], fn: Executor, resource: Resource | None = None) -> None:
        """Execute ``etype`` here as ``fn(event, resource)``, replayed under the resource's fingerprint.

        An inner scope's handler wins. A re-executed resource's executor also receives ``recorded=``, the recorded
        result or ``Missing``, and returns what the caller sees.
        """
        self._chain.get()[-1].handlers += (_Handler(etype, fn, resource),)

    def observe[E: Event[object]](self, etype: type[E], fn: Observer[E]) -> None:
        self._chain.get()[-1].observers += ((etype, cast(Observer[Event[object]], fn)),)

    def notify(self, event: Event[None]) -> None:
        """Record a diagnostic and start its observers without waiting for either."""
        self._publish(event, self._chain.get())

    def _publish(self, event: Event[object], chain: tuple[_Scope, ...]) -> None:
        # Starting eagerly numbers and indexes the record now, so the next emission already replays it.
        self._recording += (asyncio.Task(self._log.record(event), loop=asyncio.get_running_loop(), eager_start=True),)
        for scope in chain:
            for etype, observer in scope.observers:
                if isinstance(event, etype):
                    chain[-1].observed += (asyncio.ensure_future(observer(event)),)

    async def emit[T](self, event: Event[T]) -> T:
        chain = self._chain.get()
        handler = next((h for s in reversed(chain) for h in reversed(s.handlers) if isinstance(event, h.etype)), None)
        uses = () if handler is None or handler.resource is None else (handler.resource,)
        fingerprints = tuple(resource.fingerprint() for resource in uses)
        self._publish(event, chain)
        if handler is None:
            if event.result_type is not type(None):
                raise RuntimeError(f"No executor for {type(event).__qualname__}")
            return cast(T, None)
        value = self._log.lookup(event, fingerprints)
        cached = all(fp.replay == ReplayPolicy.CACHE for fp in fingerprints)
        replayed = cached and not isinstance(value, Missing)
        if not cached:
            value = await handler.fn(event, *uses, recorded=value)
        elif not replayed:
            value = await handler.fn(event, *uses)
        result = cast(T, value)
        self._publish(Resolved(event, fingerprints, result, replayed), chain)
        return result

    async def check(self) -> None:
        """Raise the failures of the observers this scope's events have started."""
        _raise(await _failures(self._chain.get()[-1].take))

    @asynccontextmanager
    async def scope(self, name: str) -> AsyncIterator[None]:
        owner = _Scope()
        chain = (*self._chain.get(), owner)
        token = self._chain.set(chain)
        self._publish(ScopeEntered(name), chain)
        failed = True
        try:
            yield
            _raise(await _failures(owner.take))
            failed = False
        finally:
            try:
                # An exceptional exit cancels its observers and joins them without inspecting their issues.
                if failed:
                    for run in owner.observed:
                        run.cancel()
                self._publish(ScopeExited(name, failed), chain)
                failures = await _failures(owner.take)
                if not failed:
                    _raise(failures)
            finally:
                self._chain.reset(token)

    async def drain(self) -> None:
        """Join every recording and root observer; raise their failures, deferred until now."""
        if self._chain.get() != (self._root,):
            raise RuntimeError("Drain is only valid outside active scopes")

        def take() -> tuple[asyncio.Future[None], ...]:
            runs, self._recording = (*self._root.take(), *self._recording), ()
            return runs

        _raise(await _failures(take))
