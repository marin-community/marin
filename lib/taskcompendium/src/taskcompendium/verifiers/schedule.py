# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract final-calendar answers and validate their private grading contract."""

from pydantic import JsonValue, model_validator
from verifyit.modes.grade_schedule import grade_schedule_candidate, normalized_name, parse_time

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier, grade_result
from taskcompendium.submission import extract_answer


class ScheduleAnswerVerifier(Verifier):
    expected_events: dict[str, dict[str, JsonValue]]

    @model_validator(mode="after")
    def validate_expected_events(self) -> "ScheduleAnswerVerifier":
        if not self.expected_events:
            raise ValueError("Expected calendar events are required")
        for key, event in self.expected_events.items():
            int(key)
            duration = event.get("duration")
            if (
                not isinstance(duration, int)
                or isinstance(duration, bool)
                or duration <= 0
                or normalized_name(event.get("event_name")) is None
                or parse_time(event.get("min_time")) is None
                or parse_time(event.get("max_time")) is None
            ):
                raise ValueError(f"Malformed calendar event {key}")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            answer = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        return grade_result(grade_schedule_candidate(self.expected_events, answer))
