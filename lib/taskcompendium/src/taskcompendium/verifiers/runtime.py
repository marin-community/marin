# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade captured files and calendar postconditions without inspecting trajectories."""

from pydantic import BaseModel, ConfigDict, ValidationError
from verifyit.modes.grade_nl2bash import score_capture

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
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
        originals = {event.id: event for event in self.original_events}
        current = {event.id: event for event in state.events}
        if len(current) != len(state.events) or any(current.get(key) != event for key, event in originals.items()):
            return GradeResult(Outcome.GRADED, 0.0, "Original events changed")
        additions = [event for event in state.events if event.id not in originals]
        if len(additions) != 1:
            return GradeResult(Outcome.GRADED, 0.0, "Exactly one meeting must be added")
        meeting = additions[0]
        valid = (
            meeting.title == self.title
            and set(meeting.participants) == set(self.participants)
            and len(meeting.participants) == len(self.participants)
            and meeting.end - meeting.start == self.duration
            and self.earliest <= meeting.start
            and meeting.end <= self.latest
        )
        conflict = any(
            set(meeting.participants).intersection(event.participants)
            and meeting.start < event.end
            and event.start < meeting.end
            for event in self.original_events
        )
        return GradeResult(Outcome.GRADED, float(valid and not conflict))
