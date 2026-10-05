# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory calendar tools for the generated mock episode task.

Importing this module does not start an episode. CalendarFactory creates fresh
state when an explicit episode check runs.
"""

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

from taskcompendium.models import AssistantToolCalls, FunctionCall, TaskSpec
from taskcompendium.runtime.controls import Control, tool_turn
from taskcompendium.runtime.models import RuntimeEvidence

INTERFACE = "calendar:v1"


class CalendarEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    title: str
    start: int
    end: int
    participants: tuple[str, ...]


class CalendarState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    events: tuple[CalendarEvent, ...]


class CalendarGoal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str
    participants: tuple[str, ...]
    duration: int
    earliest: int
    latest: int
    original_events: tuple[CalendarEvent, ...]


def calendar_controls(verifier: CalendarGoal) -> tuple[Control, ...]:
    """Choose valid and invalid schedule actions for the calendar episode check."""
    slots = []
    for start in range(verifier.earliest, verifier.latest - verifier.duration + 1):
        if all(
            not set(verifier.participants).intersection(event.participants)
            or start >= event.end
            or event.start >= start + verifier.duration
            for event in verifier.original_events
        ):
            slots.append(start)
    if not slots:
        raise ValueError("Calendar task has no feasible slot")

    def schedule(start: int) -> AssistantToolCalls:
        return tool_turn(
            "create_event",
            {
                "title": verifier.title,
                "participants": list(verifier.participants),
                "start": start,
                "end": start + verifier.duration,
            },
        )

    return (
        Control("noop", (), 0.0),
        Control("reference", (schedule(slots[0]),), 1.0),
        Control("perturbed", (schedule(verifier.latest),), 0.0),
        Control("alternate", (schedule(slots[-1]),), 1.0),
        Control("reset", (schedule(slots[0]),), 1.0),
    )


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
        requirements = task.environment_requirements
        provider = requirements.tool_providers.get("calendar")
        if provider is None or provider.action_interface != INTERFACE:
            raise ValueError("Unsupported calendar fixture")
        if (
            requirements.capabilities
            or requirements.docker_image is not None
            or requirements.working_directory is not None
            or requirements.setup_commands
            or requirements.environment_variables
            or set(requirements.tool_providers) != {"calendar"}
            or task.resources.all
            or task.resources.worker
            or task.resources.oracle
        ):
            raise ValueError("Calendar factory cannot satisfy these environment requirements")
        state = CalendarState.model_validate(provider.initial_state)
        return CalendarEnvironment(list(state.events))
