# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import cast

from forge.effects.events import Event, Fingerprint, ReplayPolicy, Resolved, ScopeEntered, ScopeExited
from forge.effects.issues import Hold, LogFailure
from forge.effects.log import EventLog, Missing

type Executor[E: Event[object], T] = Callable[..., Awaitable[T]]
type Observer[E: Event[object]] = Callable[[E], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _Observer:
    etype: type[Event[object]]
    fn: Observer[Event[object]]
    raises: type[BaseException]


@dataclass(frozen=True, slots=True)
class _Pending:
    task: asyncio.Task[None]
    raises: type[BaseException]


@dataclass(frozen=True, slots=True)
class _Handler:
    etype: type[Event[object]]
    fn: Executor[Event[object], object]
    uses: tuple[object, ...]


class _Scope:
    def __init__(self) -> None:
        self.handlers: tuple[_Handler, ...] = ()
        self.observers: tuple[_Observer, ...] = ()
        self.resources: tuple[tuple[object, Callable[[], Fingerprint]], ...] = ()
        self.pending: tuple[_Pending, ...] = ()
        self.effects: tuple[Handle[object], ...] = ()


@dataclass(frozen=True, slots=True)
class Handle[T]:
    event: Event[T]
    _task: asyncio.Task[T]
    _consumed: asyncio.Event

    def __await__(self) -> Generator[object, None, T]:
        self._consumed.set()
        return self._task.__await__()


class Context:
    """Execute effects in task-local scopes and own their observer lifetimes."""

    def __init__(self, *, log: EventLog, defer: tuple[type[BaseException], ...] = (LogFailure,)) -> None:
        self._log = log
        self._defer = defer
        self._root = _Scope()
        self._chain: ContextVar[tuple[_Scope, ...]] = ContextVar("forge_scopes", default=(self._root,))

    def resource(self, obj: object, fingerprint: Callable[[], Fingerprint]) -> None:
        self._chain.get()[-1].resources += ((obj, fingerprint),)

    def handle[E: Event[object], T](self, etype: type[E], fn: Executor[E, T], uses: tuple[object, ...] = ()) -> None:
        self._chain.get()[-1].handlers += (_Handler(etype, fn, uses),)

    def observe[E: Event[object]](self, etype: type[E], fn: Observer[E], raises: type[BaseException] = Hold) -> None:
        self._chain.get()[-1].observers += (_Observer(etype, cast(Observer[Event[object]], fn), raises),)

    def _notify(self, event: Event[object], chain: tuple[_Scope, ...]) -> None:
        owner = chain[-1]
        owner.pending += (_Pending(asyncio.create_task(self._log.record(event)), LogFailure),)
        for scope in chain:
            for observer in scope.observers:
                if isinstance(event, observer.etype):
                    owner.pending += (_Pending(asyncio.create_task(self._observe(observer.fn, event)), observer.raises),)

    async def _observe(self, fn: Observer[Event[object]], event: Event[object]) -> None:
        await fn(event)

    def emit[T](self, event: Event[T]) -> Handle[T]:
        chain = self._chain.get()
        handler = next(
            (
                handler
                for scope in reversed(chain)
                for handler in reversed(scope.handlers)
                if isinstance(event, handler.etype)
            ),
            None,
        )
        fingerprints: tuple[Fingerprint, ...] = ()
        if handler is not None:
            for resource in handler.uses:
                fingerprint = next(
                    (fn for scope in reversed(chain) for obj, fn in reversed(scope.resources) if obj is resource), None
                )
                if fingerprint is None:
                    raise RuntimeError("Executor uses an unregistered resource")
                fingerprints += (fingerprint(),)
        self._notify(event, chain)
        task = asyncio.create_task(self._execute(event, handler, fingerprints, chain))
        handle = Handle(event, task, asyncio.Event())
        chain[-1].effects += (cast(Handle[object], handle),)
        return handle

    async def _execute[T](
        self, event: Event[T], handler: _Handler | None, fingerprints: tuple[Fingerprint, ...], chain: tuple[_Scope, ...]
    ) -> T:
        if handler is None:
            if event.result_type is not type(None):
                raise RuntimeError(f"No executor for {type(event).__qualname__}")
            return cast(T, None)
        value = self._log.lookup(event, fingerprints)
        replayed = not isinstance(value, Missing) and all(fp.replay == ReplayPolicy.CACHE for fp in fingerprints)
        if not replayed:
            value = await handler.fn(event, *handler.uses)
        result = cast(T, value)
        self._notify(Resolved(event, fingerprints, result, replayed), chain)
        return result

    async def _check(self, owner: _Scope, types: tuple[type[BaseException], ...]) -> None:
        errors: tuple[BaseException, ...] = ()
        while selected := tuple(
            p for p in owner.pending if (issubclass(p.raises, types) if types else not issubclass(p.raises, self._defer))
        ):
            owner.pending = tuple(p for p in owner.pending if p not in selected)
            results = await asyncio.gather(*(p.task for p in selected), return_exceptions=True)
            errors += tuple(result for result in results if isinstance(result, BaseException))
        if errors:
            raise BaseExceptionGroup("Effect observer issues", errors)

    async def check(self, *types: type[BaseException]) -> None:
        await self._check(self._chain.get()[-1], types)

    async def _settle(self, owner: _Scope) -> None:
        while effects := owner.effects:
            owner.effects = ()
            results = await asyncio.gather(*(effect._task for effect in effects), return_exceptions=True)
            errors = tuple(
                result
                for effect, result in zip(effects, results, strict=True)
                if isinstance(result, BaseException) and not effect._consumed.is_set()
            )
            if errors:
                raise BaseExceptionGroup("Unawaited effect failures", errors)

    async def _finish(self, owner: _Scope, types: tuple[type[BaseException], ...] = ()) -> None:
        while owner.effects or any(
            issubclass(p.raises, types) if types else not issubclass(p.raises, self._defer) for p in owner.pending
        ):
            await self._settle(owner)
            await self._check(owner, types)

    @asynccontextmanager
    async def scope(self, name: str) -> AsyncIterator[None]:
        owner = _Scope()
        chain = (*self._chain.get(), owner)
        token = self._chain.set(chain)
        self._notify(ScopeEntered(name), chain)
        failed = True
        try:
            yield
            await self._finish(owner)
            failed = False
        finally:
            # Exceptional exits join owned work without inspecting observer issues.
            if failed:
                for task in (
                    *(effect._task for effect in owner.effects),
                    *(p.task for p in owner.pending if not issubclass(p.raises, self._defer)),
                ):
                    task.cancel()
                await asyncio.gather(
                    *(effect._task for effect in owner.effects), *(p.task for p in owner.pending), return_exceptions=True
                )
            self._notify(ScopeExited(name, failed), chain)
            remaining = tuple(p for p in owner.pending if not issubclass(p.raises, self._defer))
            try:
                if not failed:
                    await self._finish(owner)
                else:
                    while remaining or owner.effects:
                        effects, owner.effects = owner.effects, ()
                        owner.pending = tuple(p for p in owner.pending if issubclass(p.raises, self._defer))
                        await asyncio.gather(
                            *(p.task for p in remaining), *(e._task for e in effects), return_exceptions=True
                        )
                        remaining = tuple(p for p in owner.pending if not issubclass(p.raises, self._defer))
            finally:
                self._root.pending += tuple(p for p in owner.pending if issubclass(p.raises, self._defer))
                self._chain.reset(token)

    async def drain(self) -> None:
        if self._chain.get() != (self._root,):
            raise RuntimeError("Drain is only valid outside active scopes")
        await self._finish(self._root, (BaseException,))
