# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import Context as Ambient
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from typing import Protocol, cast

from taskforge.effects.events import HANDLED_FAILURE, Event, Fingerprint, Resolved, ScopeEntered, ScopeExited, failure
from taskforge.effects.log import Billing, EventLog, Missing, Record, cached

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
    def __init__(self, bills: Billing | None = None) -> None:
        self.handlers: tuple[_Handler, ...] = ()
        self.observers: tuple[tuple[type[Event[object]], Observer[Event[object]]], ...] = ()
        self.observed: list[asyncio.Future[None]] = []
        """Observer runs this scope's events started; their failures are the scope's issues."""
        self.bills = bills

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

        An inner scope's handler wins. A resource that is not cached (a sandbox) runs on every launch, and a
        ``HANDLED_FAILURE`` its executor raises is recorded as ``Failed``.
        """
        self._chain.get()[-1].handlers += (_Handler(etype, fn, resource),)

    def observe[E: Event[object]](self, etype: type[E], fn: Observer[E]) -> None:
        self._chain.get()[-1].observers += ((etype, cast(Observer[Event[object]], fn)),)

    def detached(self) -> Ambient:
        """The caller's context variables with this context's scope reset to the root, for work no caller scope owns."""
        ambient = copy_context()
        ambient.run(self._chain.set, (self._root,))
        return ambient

    def notify(self, event: Event[None]) -> None:
        """Record a diagnostic and start its observers without waiting for either."""
        self._publish(self._log.append(event), self._chain.get())

    def _publish(self, record: Record, chain: tuple[_Scope, ...]) -> Record:
        # The log numbered and indexed the record at dispatch. A task that publishes as it starts appends its own run
        # first, so each run is appended to the live list once it exists.
        recording = asyncio.Task(self._log.persist(), loop=asyncio.get_running_loop(), eager_start=True)
        self._recording.append(recording)
        for scope in chain:
            for etype, observer in scope.observers:
                if isinstance(record.event, etype):
                    run = asyncio.ensure_future(observer(record.event))
                    chain[-1].observed.append(run)
        return record

    def _handler(self, event: Event[object]) -> tuple[_Handler | None, tuple[Resource, ...], tuple[Fingerprint, ...]]:
        chain = self._chain.get()
        handler = next((h for s in reversed(chain) for h in reversed(s.handlers) if isinstance(event, h.etype)), None)
        uses = () if handler is None or handler.resource is None else (handler.resource,)
        return handler, uses, tuple(resource.fingerprint() for resource in uses)

    def recorded[T](self, event: Event[T]) -> T | None:
        """``event``'s result as ``emit`` would replay it, without executing or recording anything; None when the log
        holds none, so a reader of earlier evidence never buys it again."""
        value = self._log.lookup(event, self._handler(event)[2])
        return None if isinstance(value, Missing) else value

    def receipt(self, event: Event[object]) -> int | None:
        """Return the physical row of the event's latest resolution, if one exists."""
        return self._log.receipt(event, self._handler(event)[2])

    def recorded_at[T](self, receipt: int, event: Event[T]) -> T:
        """Read a physical resolution at its row under the event's current fingerprints."""
        return cast(T, self._log.at(receipt, event, self._handler(event)[2]))

    async def emit[T](self, event: Event[T]) -> T:
        chain = self._chain.get()
        handler, uses, fingerprints = self._handler(event)
        if handler is not None and not cached(fingerprints):
            # TODO: To resume inside a session, cache sandbox outcomes by logical sandbox and operation position,
            # then decide whether reconstructing the machine must verify each recorded outcome.
            request = self._publish(self._log.append(event), chain).sequence
            try:
                value = await handler.fn(event, *uses)
            except HANDLED_FAILURE as error:
                self._publish(self._log.append(failure(error), answers=request), chain)
                raise
            self._publish(self._log.append(Resolved(event, fingerprints, value), answers=request), chain)
            return cast(T, value)
        self._publish(self._log.append(event), chain)
        if handler is None:
            if event.result_type is not type(None):
                raise RuntimeError(f"No executor for {type(event).__qualname__}")
            return cast(T, None)
        value = self._log.lookup(event, fingerprints)
        replayed = not isinstance(value, Missing)
        if not replayed:
            if (invalidated := self._log.invalidate(event, fingerprints)) is not None:
                self._publish(invalidated, chain)  # as the replacement starts
            value = await handler.fn(event, *uses)
        result = cast(T, value)
        billed = None if replayed else next((scope.bills for scope in reversed(chain) if scope.bills), None)
        self._publish(self._log.append(Resolved(event, fingerprints, result, replayed), billed=billed), chain)
        return result

    async def check(self) -> None:
        """Raise the failures of the observers this scope's events have started."""
        await _join(self._chain.get()[-1].take)

    @asynccontextmanager
    async def scope(self, name: str, *, bills: Billing | None = None) -> AsyncIterator[None]:
        """A task-local scope. With ``bills``, a physical cached resolution inside it, unless an inner scope bills
        another role, records that role as its payer."""
        owner = _Scope(bills)
        chain = (*self._chain.get(), owner)
        entered = self._log.append(ScopeEntered(name))
        token = self._chain.set(chain)
        failed = True
        try:
            self._publish(entered, chain)  # its observers start in this scope's context
            yield
            await _join(owner.take)
            failed = False
        finally:
            try:
                # An exceptional exit cancels its observers and joins them without inspecting their issues.
                if failed:
                    for run in owner.observed:
                        run.cancel()
                self._publish(self._log.append(ScopeExited(name, failed)), chain)
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
