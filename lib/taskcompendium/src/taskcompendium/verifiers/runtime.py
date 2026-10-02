# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade captured files and calendar postconditions without inspecting trajectories."""

from pydantic import BaseModel, ConfigDict, ValidationError
from verifyit.modes.grade_calendar import CalendarEvent as CalendarRecord
from verifyit.modes.grade_calendar import score_calendar
from verifyit.modes.grade_nl2bash import score_capture

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier, grade_result
from taskcompendium.runtime.models import RuntimeEvidence


class CaptureOutputVerifier(Verifier):
    output_path: str
    expected_output: str

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(attempt.environment, RuntimeEvidence):
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing runtime evidence")
        data = attempt.environment.files.get(self.output_path)
        if data is None:
            return GradeResult(Outcome.GRADED, 0.0, "Missing capture file")
        reward, errors = score_capture(data.decode(errors="replace"), self.expected_output)
        return GradeResult(Outcome.GRADED, float(reward), "; ".join(errors) or None)


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


def _calendar_record(event: CalendarEvent) -> CalendarRecord:
    return CalendarRecord(event.id, event.title, event.start, event.end, event.participants)


class CalendarStateVerifier(Verifier):
    title: str
    participants: tuple[str, ...]
    duration: int
    earliest: int
    latest: int
    original_events: tuple[CalendarEvent, ...]

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        if not isinstance(attempt.environment, RuntimeEvidence):
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing runtime evidence")
        try:
            state = CalendarState.model_validate_json(attempt.environment.state_json)
        except ValidationError as error:
            return GradeResult(Outcome.INFRA_ERROR, None, str(error))
        return grade_result(
            score_calendar(
                tuple(_calendar_record(event) for event in state.events),
                tuple(_calendar_record(event) for event in self.original_events),
                title=self.title,
                participants=self.participants,
                duration=self.duration,
                earliest=self.earliest,
                latest=self.latest,
            )
        )
