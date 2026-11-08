# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recorded identifier allocation; keys distinguish replayable allocation sites."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from forge.effects.events import Event, Fingerprint, ReplayPolicy


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class NewId(Event[str]):
    """A deterministic call-site key; repeated emits of the same key replay the same value."""

    key: str


@dataclass(frozen=True, slots=True)
class Ids:
    source: Callable[[], str] = _new_id

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("ids", "uuid4-v1", ReplayPolicy.CACHE)


async def new_id(_event: NewId, resource: Ids) -> str:
    return resource.source()
