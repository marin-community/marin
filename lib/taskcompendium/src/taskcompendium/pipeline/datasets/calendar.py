# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated calendar episodes for runtime tests and prototyping.

This mock task is separate from the TaskTrove calendar sources, which request
a final JSON schedule without interactive tools.
"""

import json
from collections.abc import Iterator
from typing import Any

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    ProviderRequirement,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    GeneratedSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)
from taskcompendium.runtime.calendar import INTERFACE
from taskcompendium.verifiers.base import VerifierKind
from taskcompendium.verifiers.runtime import CalendarState, CalendarStateVerifier

TOOLS = (
    FunctionDefinition(
        name="list_events",
        description="Read current calendar events. Times are integer minutes after midnight.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    FunctionDefinition(
        name="create_event",
        description="Add one event to the calendar",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
                "participants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "start", "end", "participants"],
            "additionalProperties": False,
        },
    ),
    FunctionDefinition(
        name="delete_event",
        description="Delete an event by id",
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    if not isinstance(row.data.get("goal"), dict):
        return ImportRejection(reason="malformed_calendar", detail="Goal must be an object")
    try:
        state = CalendarState.model_validate(row.data["initial_state"])
        verifier = CalendarStateVerifier.model_validate_json(
            json.dumps({**row.data["goal"], "original_events": state.model_dump(mode="json")["events"]})
        )
    except (KeyError, ValueError) as error:
        return ImportRejection(reason="malformed_calendar", detail=str(error))
    instruction = (
        f"Schedule exactly one '{verifier.title}' meeting for {', '.join(verifier.participants)}. "
        f"It must last {verifier.duration} minutes and fit within minutes {verifier.earliest} through {verifier.latest} "
        "after midnight. Read the current calendar, avoid overlapping events for any participant, "
        "and preserve every existing event. Any valid slot is acceptable. Use the calendar tools to save it."
    )
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            tool_providers={
                "calendar": ProviderRequirement(action_interface=INTERFACE, initial_state=state.model_dump(mode="json"))
            }
        ),
        interaction_tools=TOOLS,
        answer_type=AnswerType.STATE,
        verifier=VerifierSpec(kind=VerifierKind.CALENDAR_STATE, parameters_json=verifier.model_dump_json()),
        source=row.source,
    )


def generate_rows(limit: int) -> Iterator[dict[str, Any]]:
    for index in range(limit):
        base = 540 + index * 5
        yield {
            "initial_state": {
                "events": [
                    {"id": "busy-alice", "title": "Standup", "start": base, "end": base + 30, "participants": ["Alice"]},
                    {"id": "busy-bob", "title": "Review", "start": base + 45, "end": base + 75, "participants": ["Bob"]},
                ]
            },
            "goal": {
                "title": f"Planning {index}",
                "participants": ["Alice", "Bob"],
                "duration": 30,
                "earliest": base,
                "latest": base + 180,
            },
        }


recipe = DatasetRecipe(
    name="calendar-mock",
    version="calendar-v1",
    source=GeneratedSource("mock/calendar", "1", "default", "train", __name__),
    inputs=RecipeInputs(SourceFiles(("*.jsonl",), SourceFormat.JSONL), ()),
    normalize=normalize,
    rubric=ReviewRubric(
        "calendar-state",
        "1",
        (
            "Check that at least one conflict-free slot exists within the requested window.",
            "Any slot satisfying the constraints is correct; do not demand the control trajectory's slot.",
            "Tools must save the meeting; saying it was scheduled does not change state.",
            "Existing events must remain unchanged, and the goal must specify duration, window and participants.",
        ),
    ),
    intended_use=IntendedUse.TRAIN,
)
