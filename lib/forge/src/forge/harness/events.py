# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from forge.effects.events import Event


@dataclass(frozen=True, slots=True)
class ToolInvoked(Event[None]):
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolReturned(Event[None]):
    call_id: str
    name: str
    content: str
