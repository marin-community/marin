# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for imported calendar and shell-output scorers."""

from verifyit.grade import Status
from verifyit.modes.grade_calendar import CalendarEvent, score_calendar
from verifyit.modes.grade_nl2bash import score_capture
from verifyit.modes.grade_schedule import score_schedule


def test_schedule_accepts_alternate_slots_but_rejects_overlap():
    expected = {
        "0": {
            "event_name": "Review",
            "duration": 30,
            "min_time": "10:00",
            "max_time": "16:00",
            "constraint": "before 11am",
        },
        "1": {"event_name": "Plan", "duration": 30, "min_time": "10:00", "max_time": "16:00", "constraint": None},
    }
    candidate = [
        {"event_id": 0, "event_name": "Review", "duration": 30, "start_time": "10:15"},
        {"event_id": 1, "event_name": "Plan", "duration": 30, "start_time": "12:00"},
    ]
    accepted = score_schedule(expected, candidate)
    assert accepted.status == Status.SCORED and accepted.reward == 1.0

    overlapping = [candidate[0], {**candidate[1], "start_time": "10:30"}]
    rejected = score_schedule(expected, overlapping)
    assert rejected.status == Status.SCORED and rejected.reward == 0.0
    assert "overlap" in rejected.detail["error"]


def test_calendar_requires_one_nonconflicting_addition_and_preserves_originals():
    original = (
        CalendarEvent("busy-alice", "Standup", 540, 570, ("Alice",)),
        CalendarEvent("busy-bob", "Review", 585, 615, ("Bob",)),
    )
    meeting = CalendarEvent("created-0", "Planning", 630, 660, ("Alice", "Bob"))
    options = {"title": "Planning", "participants": ("Alice", "Bob"), "duration": 30, "earliest": 540, "latest": 720}
    assert score_calendar((*original, meeting), original, **options).reward == 1.0
    assert (
        score_calendar(
            (*original, CalendarEvent("created-0", "Planning", 550, 580, ("Alice", "Bob"))), original, **options
        ).reward
        == 0.0
    )
    changed = (CalendarEvent("busy-alice", "Changed", 540, 570, ("Alice",)), original[1], meeting)
    rejected = score_calendar(changed, original, **options)
    assert rejected.reward == 0.0 and rejected.detail["error"] == "Original events changed"


def test_nl2bash_capture_preserves_duplicate_counts_and_ignores_nonerror_extras():
    expected = "/workspace/a 12 bytes\n/workspace/a 12 bytes\n"
    assert score_capture("a 12\na 12\nextra\n", expected) == (1, [])
    assert score_capture("a 12\n", expected)[0] == 0
    assert score_capture("a 12\na 12\nerror: missing input\n", expected)[0] == 0
