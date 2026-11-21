# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, cast

from taskforge.effects.events import Event, Fingerprint, Resolved, ScopeEntered, ScopeExited
from taskforge.effects.log import EventLog, Missing, cached

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
        self.observed: list[asyncio.Future[None]] = []
        """Observer runs this scope's events started; their failures are the scope's issues."""

    def take(self) -> list[asyncio.Future[None]]:
        runs, self.observed = self.observed, []
        return runs


async def _join(take: Callable[[], list[asyncio.Future[None]]], *, propagate: bool = True) -> None:
    """Join the runs ``take`` hands over until it hands over none, including runs they start; raise their failures."""
    errors = []
    while runs := take():
        errors += [result for result in await asyncio.gather(*runs, return_exceptions=True) if result is not None]
    if errors and propagate:
        raise BaseExceptionGroup("Effect observer issues", errors)


class Context:
    """Execute effects in task-local scopes, record every event, and own observer lifetimes.

    Observer failures are their scope's issues, raised by ``check`` or when it exits cleanly. Recording failures are
    deferred to ``drain``.
    """

    def __init__(self, *, log: EventLog) -> None:
        self._log = log
        self._root = _Scope()
        self._recording: list[asyncio.Future[None]] = []
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
        # Starting eagerly numbers and indexes the record now, so the next emission already replays it. A task that
        # publishes as it starts appends its own run first, so each run is appended to the live list once it exists.
        recording = asyncio.Task(self._log.record(event), loop=asyncio.get_running_loop(), eager_start=True)
        self._recording.append(recording)
        for scope in chain:
            for etype, observer in scope.observers:
                if isinstance(event, etype):
                    run = asyncio.ensure_future(observer(event))
                    chain[-1].observed.append(run)

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
        replayed = cached(fingerprints) and not isinstance(value, Missing)
        if not cached(fingerprints):
            value = await handler.fn(event, *uses, recorded=value)
        elif not replayed:
            value = await handler.fn(event, *uses)
        result = cast(T, value)
        self._publish(Resolved(event, fingerprints, result, replayed), chain)
        return result

    async def check(self) -> None:
        """Raise the failures of the observers this scope's events have started."""
        await _join(self._chain.get()[-1].take)

    @asynccontextmanager
    async def scope(self, name: str) -> AsyncIterator[None]:
        owner = _Scope()
        chain = (*self._chain.get(), owner)
        token = self._chain.set(chain)
        self._publish(ScopeEntered(name), chain)
        failed = True
        try:
            yield
            await _join(owner.take)
            failed = False
        finally:
            try:
                # An exceptional exit cancels its observers and joins them without inspecting their issues.
                if failed:
                    for run in owner.observed:
                        run.cancel()
                self._publish(ScopeExited(name, failed), chain)
                await _join(owner.take, propagate=not failed)
            finally:
                self._chain.reset(token)

    async def drain(self) -> None:
        """Join every recording and root observer; raise their failures, deferred until now."""
        if self._chain.get() != (self._root,):
            raise RuntimeError("Drain is only valid outside active scopes")

        def take() -> list[asyncio.Future[None]]:
            runs, self._recording = [*self._root.take(), *self._recording], []
            return runs

        await _join(take)
