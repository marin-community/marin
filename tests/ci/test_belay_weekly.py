# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime

import pytest

from scripts.ci.belay_weekly import roll_over, week_start, week_title

MONDAY = datetime.date(2026, 10, 19)


class _FakeGitHub:
    repo = "o/r"

    def __init__(self, issues: dict[int, str]):
        self.titles = dict(issues)
        self.open = set(issues)
        self.labels: dict[int, list[str]] = {n: ["belay-weekly"] for n in issues}
        self.comments: dict[int, list[str]] = {}
        self.locked: set[int] = set()

    def open_issues_labeled(self, label: str) -> list[int]:
        return sorted(n for n in self.open if label in self.labels[n])

    def issue(self, number: int) -> dict:
        return {"title": self.titles[number]}

    def create_issue(self, title: str, body: str, labels: list[str]) -> int:
        number = max(self.titles, default=100) + 1
        self.titles[number], self.labels[number] = title, labels
        self.open.add(number)
        self.comments[number] = [body]
        return number

    def lock(self, number: int) -> None:
        self.locked.add(number)

    def post_comment(self, issue: int, body: str) -> None:
        self.comments.setdefault(issue, []).append(body)

    def close(self, number: int) -> None:
        self.open.discard(number)


def _summary(since: datetime.date, until: datetime.date) -> str:
    return f"- X0042 tried {since}..{until}"


def test_rollover_opens_locks_and_links_the_new_week_and_closes_the_old():
    github = _FakeGitHub({9451: "Marin Belay v1: hill-climb the scaling ladder"})
    new = roll_over(github, today=MONDAY + datetime.timedelta(days=2), summary_for=_summary, dry_run=False)
    assert github.titles[new] == "Marin Belay v1 · week of 2026-10-19"
    assert new in github.locked and github.open == {new}
    assert "Previous week: #9451" in github.comments[new][0]
    (closing,) = github.comments[9451]
    assert f"continues in #{new}" in closing and "- X0042 tried 2026-10-12..2026-10-19" in closing


def test_rollover_does_nothing_within_the_same_week():
    github = _FakeGitHub({200: week_title(MONDAY)})
    assert roll_over(github, today=MONDAY + datetime.timedelta(days=6), summary_for=_summary, dry_run=False) == 200
    assert github.open == {200} and github.comments == {}


def test_first_rollover_without_a_previous_issue():
    github = _FakeGitHub({})
    new = roll_over(github, today=MONDAY, summary_for=_summary, dry_run=False)
    assert github.open == {new} and "Previous week" not in github.comments[new][0]


def test_rollover_refuses_two_open_weekly_issues():
    github = _FakeGitHub({200: "a", 201: "b"})
    with pytest.raises(RuntimeError):
        roll_over(github, today=MONDAY, summary_for=_summary, dry_run=False)


def test_week_start_is_monday():
    assert week_start(datetime.date(2026, 10, 25)) == MONDAY and week_start(MONDAY) == MONDAY
