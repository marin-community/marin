# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory calendar tools with a new state for every episode."""

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from taskcompendium.models import FunctionCall, TaskSpec
from taskcompendium.runtime.models import RuntimeEvidence
from taskcompendium.verifiers.runtime import CalendarEvent, CalendarState

INTERFACE = "calendar:v1"


@dataclass
class CalendarEnvironment:
    events: list[CalendarEvent]
    next_id: int = field(default=0, init=False)

    async def step(self, call: FunctionCall) -> str:
        if call.name == "list_events":
            if call.arguments:
                return json.dumps({"error": "list_events takes no arguments"})
            return CalendarState(events=tuple(self.events)).model_dump_json()
        if call.name == "create_event":
            if set(call.arguments) != {"title", "start", "end", "participants"}:
                return json.dumps({"error": "create_event requires title, start, end and participants"})
            while f"created-{self.next_id}" in {event.id for event in self.events}:
                self.next_id += 1
            try:
                event = CalendarEvent.model_validate({**call.arguments, "id": f"created-{self.next_id}"})
            except ValidationError as error:
                return json.dumps({"error": str(error)})
            if event.start >= event.end or not event.participants:
                return json.dumps({"error": "Meeting requires a positive duration and participants"})
            self.events.append(event)
            self.next_id += 1
            return event.model_dump_json()
        if call.name == "delete_event":
            if set(call.arguments) != {"id"}:
                return json.dumps({"error": "delete_event requires id"})
            self.events[:] = [event for event in self.events if event.id != call.arguments["id"]]
            return json.dumps({"deleted": call.arguments["id"]})
        return json.dumps({"error": "Unknown tool"})

    async def evidence(self) -> RuntimeEvidence:
        return RuntimeEvidence({}, CalendarState(events=tuple(self.events)).model_dump_json())

    async def close(self) -> None:
        pass


@dataclass(frozen=True)
class CalendarFactory:
    @property
    def identity(self) -> dict:
        return {"backend": "in-memory-calendar", "interface": INTERFACE, "revision": "1"}

    async def create(self, task: TaskSpec) -> CalendarEnvironment:
        if task.fixture is None or task.fixture.interface != INTERFACE or task.fixture.revision != "1":
            raise ValueError("Unsupported calendar fixture")
        state = CalendarState.model_validate_json(task.fixture.initial_state_json)
        return CalendarEnvironment(list(state.events))
