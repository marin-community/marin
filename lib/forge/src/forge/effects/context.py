# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Generator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from forge.effects.events import Event, Fingerprint
from forge.effects.issues import Hold, LogFailure
from forge.effects.log import EventLog

type Executor[E: Event[object], T] = Callable[..., Awaitable[T]]
type Observer[E: Event[object]] = Callable[[E], Awaitable[None]]


class Handle[T](Protocol):
    @property
    def event(self) -> Event[T]: ...

    def __await__(self) -> Generator[object, None, T]: ...


class Context(Protocol):
    def __init__(self, *, log: EventLog, defer: tuple[type[BaseException], ...] = (LogFailure,)) -> None: ...

    def resource(self, obj: object, fingerprint: Callable[[], Fingerprint]) -> None: ...

    def handle[E: Event[object], T](self, etype: type[E], fn: Executor[E, T], uses: tuple[object, ...] = ()) -> None: ...

    def observe[E: Event[object]](self, etype: type[E], fn: Observer[E], raises: type[BaseException] = Hold) -> None: ...

    def emit[T](self, event: Event[T]) -> Handle[T]: ...

    def scope(self, name: str) -> AbstractAsyncContextManager[None]: ...

    async def check(self, *types: type[BaseException]) -> None: ...

    async def drain(self) -> None: ...
