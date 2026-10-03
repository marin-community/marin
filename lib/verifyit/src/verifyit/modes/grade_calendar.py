# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the postcondition of a calendar tool episode."""

from collections.abc import Sequence
from dataclasses import dataclass

from verifyit.grade import Reward, scored


@dataclass(frozen=True)
class CalendarEvent:
    id: str
    title: str
    start: int
    end: int
    participants: tuple[str, ...]


def score_calendar(
    events: Sequence[CalendarEvent],
    original_events: Sequence[CalendarEvent],
    *,
    title: str,
    participants: tuple[str, ...],
    duration: int,
    earliest: int,
    latest: int,
) -> Reward:
    """Accept one valid addition while preserving every original event."""
    originals = {event.id: event for event in original_events}
    current = {event.id: event for event in events}
    if len(current) != len(events) or any(current.get(key) != event for key, event in originals.items()):
        return scored(0.0, error="Original events changed")
    additions = [event for event in events if event.id not in originals]
    if len(additions) != 1:
        return scored(0.0, error="Exactly one meeting must be added")
    meeting = additions[0]
    valid = (
        meeting.title == title
        and set(meeting.participants) == set(participants)
        and len(meeting.participants) == len(participants)
        and meeting.end - meeting.start == duration
        and earliest <= meeting.start
        and meeting.end <= latest
    )
    conflict = any(
        set(meeting.participants).intersection(event.participants)
        and meeting.start < event.end
        and event.start < meeting.end
        for event in original_events
    )
    return scored(float(valid and not conflict))
