# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print the Slack digest for outstanding Marin review requests."""

import argparse
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from functools import cache
from html import escape
from textwrap import shorten
from zoneinfo import ZoneInfo

REPO = "marin-community/marin"
PACIFIC = ZoneInfo("America/Los_Angeles")
SLACK_LIMIT = 4000


def records(endpoint: str, fields: str) -> list[dict]:
    """Return the jq projection of every record from a repository-relative REST endpoint."""
    separator = "&" if "?" in endpoint else "?"
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{REPO}/{endpoint}{separator}per_page=100",
            "--paginate",
            "--jq",
            f".[] | {fields} | @json",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


@cache
def has_write_access(login: str) -> bool:
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/collaborators/{login}/permission", "--jq", ".permission"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() in {"admin", "maintain", "write"}


def previous_weekday(now: datetime) -> datetime:
    cutoff = now.astimezone(PACIFIC) - timedelta(days=1)
    while cutoff.weekday() >= 5:
        cutoff -= timedelta(days=1)
    return cutoff


def elapsed(start: datetime, now: datetime) -> str:
    hours = max(0, (now - start) // timedelta(hours=1))
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    return f"{hours}h" if hours else "under 1h"


def overdue_since(number: int, reviewers: list[str], cutoff: datetime) -> dict[str, datetime | None]:
    """Return the latest activity for overdue reviewers, or None if the request time is unknown."""
    latest: dict[str, datetime | None] = dict.fromkeys(reviewers)
    requested = set()

    def record(login: str | None, timestamp: datetime) -> None:
        if login is not None and login in latest:
            previous = latest[login]
            if previous is None or timestamp > previous:
                latest[login] = timestamp

    timeline = records(
        f"issues/{number}/timeline",
        "{event, actor: .actor.login, user: .user.login, requested: .requested_reviewer.login,"
        " created_at, submitted_at, updated_at}",
    )
    for event in timeline:
        timestamp = event["updated_at"] or event["submitted_at"] or event["created_at"]
        if not timestamp:
            continue
        when = datetime.fromisoformat(timestamp)
        if event["event"] == "review_requested":
            requested.add(event["requested"])
            record(event["requested"], when)
            record(event["actor"], when)
        if event["event"] in {"commented", "reviewed"}:
            record(event["user"], when)
        if event["event"] in {"labeled", "unlabeled", "closed", "reopened", "merged"}:
            record(event["actor"], when)
    if all(when and when >= cutoff for when in latest.values()):
        return {}
    for comment in records(f"pulls/{number}/comments", "{user: .user.login, updated_at}"):
        record(comment["user"], datetime.fromisoformat(comment["updated_at"]))
    for commit in records(f"pulls/{number}/commits", "{user: .author.login, date: .commit.committer.date}"):
        record(commit["user"], datetime.fromisoformat(commit["date"]))

    return {
        login: when if login in requested else None for login, when in latest.items() if when is None or when < cutoff
    }


def reminder_lines(now: datetime) -> Iterable[str]:
    cutoff = previous_weekday(now)
    stale = now - timedelta(days=7)
    pulls = records(
        "pulls?state=open&sort=updated&direction=desc",
        "{number, title, draft, updated_at, created_at, author: .user.login, author_type: .user.type,"
        ' reviewers: [.requested_reviewers[] | select(.type == "User") | .login]}',
    )
    for pull in pulls:
        updated = datetime.fromisoformat(pull["updated_at"])
        if updated < stale:
            break
        if pull["draft"] or not pull["reviewers"] or datetime.fromisoformat(pull["created_at"]) >= cutoff:
            continue
        if pull["author_type"] != "User" or not has_write_access(pull["author"]):
            continue
        waits = overdue_since(pull["number"], pull["reviewers"], cutoff)
        if waits:
            reviewers = []
            for login in sorted(waits, key=str.casefold):
                since = waits[login]
                age = elapsed(since, now) if since is not None else "request time unavailable"
                reviewers.append(f"{login} ({age})")
            number = pull["number"]
            title = escape(shorten(pull["title"], width=110, placeholder="…"), quote=False)
            yield (
                f"• <https://github.com/{REPO}/pull/{number}|#{number}>  {title}\n"
                f"    Updated {elapsed(updated, now)} ago · Waiting on {', '.join(reviewers)}"
            )


def messages(lines: Iterable[str]) -> list[str]:
    parts = []
    header = "*Marin PRs waiting for review*"
    message = header
    for line in lines:
        if len(f"{message}\n{line}".encode()) > SLACK_LIMIT:
            parts.append(message)
            message = header
        message += f"\n{line}"
    if message != header:
        parts.append(message)
    return parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, type=datetime.fromisoformat, help="Loom scheduled_at with UTC offset")
    now = parser.parse_args().as_of
    if now.tzinfo is None:
        parser.error("--as-of requires a UTC offset")
    print(json.dumps(messages(reminder_lines(now.astimezone(UTC)))))
