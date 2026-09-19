# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pandas as pd


class CalendarTool:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "csv_data", "processed", "calendar_events.csv")
        self._calendar_events = pd.read_csv(data_path, dtype=str)

    def get_event_information_by_id(self, event_id=None, field=None):
        if not event_id:
            return "Event ID not provided."
        if not field:
            return "Field not provided."

        # Validate event_id format
        if not (isinstance(event_id, str) and len(event_id) == 8 and event_id.isdigit()):
            return "Invalid event ID format. Expected 8-digit ID."

        # Validate field exists
        valid_fields = [
            "event_id",
            "event_name",
            "participant_email",
            "event_start",
            "duration",
        ]
        if field not in valid_fields:
            return f"Invalid field. Available fields are: {', '.join(valid_fields)}"

        event = self._calendar_events[self._calendar_events["event_id"] == event_id].to_dict(orient="records")
        if event:
            if field in event[0]:
                return {field: event[0][field]}
            else:
                return "Field not found."
        else:
            return "Event not found."

    def search_events(self, query="", time_min=None, time_max=None, page=1, page_size=5):
        # Validate time formats if provided
        if time_min:
            try:
                time_min = pd.Timestamp(time_min)
            except ValueError:
                return "Invalid time_min format. Expected YYYY-MM-DD HH:MM:SS."

        if time_max:
            try:
                time_max = pd.Timestamp(time_max)
            except ValueError:
                return "Invalid time_max format. Expected YYYY-MM-DD HH:MM:SS."

        events = self._calendar_events[
            (self._calendar_events["event_name"].str.contains(query, case=False, na=False))
            | (self._calendar_events["participant_email"].str.contains(query, case=False, na=False))
        ].to_dict(orient="records")

        if time_min:
            events = [event for event in events if pd.Timestamp(event["event_start"]) >= time_min]
        if time_max:
            events = [event for event in events if pd.Timestamp(event["event_start"]) <= time_max]

        if events:
            # Sort events by start time (most recent first)
            events.sort(key=lambda x: pd.Timestamp(x["event_start"]), reverse=True)

            # Calculate pagination
            total_events = len(events)
            total_pages = (total_events + page_size - 1) // page_size  # Ceiling division

            # Validate page number
            page = max(1, min(page, total_pages)) if total_pages > 0 else 1

            # Calculate start and end indices for slicing
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_events)

            return {
                "events": events[start_idx:end_idx],
                "pagination": {
                    "total_events": total_events,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                },
            }
        else:
            return "No events found."

    def create_event(self, event_name=None, participant_email=None, event_start=None, duration=None):
        if not event_name:
            return "Event name not provided."
        if not participant_email:
            return "Participant email not provided."
        if not event_start:
            return "Event start not provided."
        if not duration:
            return "Event duration not provided."

        # Validate event_start format
        try:
            pd.Timestamp(event_start)
        except ValueError:
            return "Invalid event_start format. Expected YYYY-MM-DD HH:MM:SS."

        # Validate duration is a positive number
        try:
            duration_val = int(duration)
            if duration_val <= 0:
                return "Duration must be a positive number of minutes."
        except ValueError:
            return "Invalid duration format. Expected a number of minutes."

        participant_email = participant_email.lower()

        event_id = str(int(self._calendar_events["event_id"].max()) + 1).zfill(8)
        new_event = pd.DataFrame(
            {
                "event_id": [event_id],
                "event_name": [event_name],
                "participant_email": [participant_email],
                "event_start": [event_start],
                "duration": [duration],
            }
        )
        self._calendar_events = pd.concat([self._calendar_events, new_event])
        return event_id

    def delete_event(self, event_id=None):
        if not event_id:
            return "Event ID not provided."

        # Validate event_id format
        if not (isinstance(event_id, str) and len(event_id) == 8 and event_id.isdigit()):
            return "Invalid event ID format. Expected 8-digit ID."

        if event_id in self._calendar_events["event_id"].values:
            self._calendar_events = self._calendar_events[self._calendar_events["event_id"] != event_id]
            return "Event deleted successfully."
        else:
            return "Event not found."

    def update_event(self, event_id=None, field=None, new_value=None):
        if not event_id or not field or not new_value:
            return "Event ID, field, or new value not provided."

        # Validate event_id format
        if not (isinstance(event_id, str) and len(event_id) == 8 and event_id.isdigit()):
            return "Invalid event ID format. Expected 8-digit ID."

        # Validate field name
        valid_fields = ["event_name", "participant_email", "event_start", "duration"]
        if field not in valid_fields:
            return f"Invalid field. Available fields are: {', '.join(valid_fields)}"

        # Validate field-specific values
        if field == "event_start":
            try:
                pd.Timestamp(new_value)
            except ValueError:
                return "Invalid event_start format. Expected YYYY-MM-DD HH:MM:SS."
        elif field == "duration":
            try:
                duration_val = int(new_value)
                if duration_val <= 0:
                    return "Duration must be a positive number of minutes."
            except ValueError:
                return "Invalid duration format. Expected a number of minutes."

        if event_id in self._calendar_events["event_id"].values:
            if field == "participant_email":
                new_value = new_value.lower()
            self._calendar_events.loc[self._calendar_events["event_id"] == event_id, field] = new_value
            return "Event updated successfully."
        else:
            return "Event not found."


schema_get_event_information_by_id = {
    "type": "function",
    "name": "calendar_get_event_information_by_id",
    "description": "Returns the event for a given ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "8-digit ID of the event"},
            "field": {
                "type": "string",
                "description": "Field to return. Available fields are: 'event_id', 'event_name', 'participant_email', 'event_start', 'duration'",
            },
        },
        "required": ["event_id", "field"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_search_events = {
    "type": "function",
    "name": "calendar_search_events",
    "description": "Returns the events for a given query with pagination support.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query to search for. Terms will be matched in the event_name and participant_email fields",
            },
            "time_min": {
                "type": "string",
                "description": "Lower bound (inclusive) for an event's end time to filter by. Format: YYYY-MM-DD HH:MM:SS",
            },
            "time_max": {
                "type": "string",
                "description": "Upper bound (inclusive) for an event's start time to filter by. Format: YYYY-MM-DD HH:MM:SS",
            },
            "page": {
                "type": "integer",
                "description": "Page number of results to return",
            },
            "page_size": {
                "type": "integer",
                "description": "Number of events per page",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_create_event = {
    "type": "function",
    "name": "calendar_create_event",
    "description": "Creates a new event.",
    "parameters": {
        "type": "object",
        "properties": {
            "event_name": {"type": "string", "description": "Name of the event"},
            "participant_email": {
                "type": "string",
                "description": "Email of the participant",
            },
            "event_start": {
                "type": "string",
                "description": "Start time of the event. Format: YYYY-MM-DD HH:MM:SS",
            },
            "duration": {
                "type": "string",
                "description": "Duration of the event in minutes",
            },
        },
        "required": ["event_name", "participant_email", "event_start", "duration"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_delete_event = {
    "type": "function",
    "name": "calendar_delete_event",
    "description": "Deletes an event.",
    "parameters": {
        "type": "object",
        "properties": {"event_id": {"type": "string", "description": "8-digit ID of the event"}},
        "required": ["event_id"],
        "additionalProperties": False,
    },
    "strict": False,
}

schema_update_event = {
    "type": "function",
    "name": "calendar_update_event",
    "description": "Updates an event.",
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string", "description": "8-digit ID of the event"},
            "field": {
                "type": "string",
                "description": "Field to update. Available fields are: 'event_name', 'participant_email', 'event_start', 'duration'",
            },
            "new_value": {"type": "string", "description": "New value for the field"},
        },
        "required": ["event_id", "field", "new_value"],
        "additionalProperties": False,
    },
    "strict": False,
}

# List of all calendar tool schemas
calendar_tool_schemas = [
    schema_get_event_information_by_id,
    schema_search_events,
    schema_create_event,
    schema_delete_event,
    schema_update_event,
]
