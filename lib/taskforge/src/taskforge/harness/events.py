# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from taskforge.effects.events import Event


@dataclass(frozen=True, slots=True)
class ToolInvoked(Event[None]):
    kind = "tool.invoked"
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolReturned(Event[None]):
    kind = "tool.returned"
    call_id: str
    name: str
    content: str
