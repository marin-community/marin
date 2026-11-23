# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print the Slack digest for outstanding Marin review requests."""

import argparse
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from functools import cache
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


def due_reviewers(number: int, reviewers: list[str], cutoff: datetime) -> list[str]:
    """Return reviewers with neither a new request nor PR activity at or after cutoff."""
    waiting = set(reviewers)
    timeline = records(
        f"issues/{number}/timeline",
        "{event, actor: .actor.login, user: .user.login, requested: .requested_reviewer.login,"
        " created_at, submitted_at, updated_at}",
    )
    for event in timeline:
        timestamp = event["updated_at"] or event["submitted_at"] or event["created_at"]
        if not timestamp or datetime.fromisoformat(timestamp) < cutoff:
            continue
        if event["event"] == "review_requested":
            waiting.discard(event["requested"])
            waiting.discard(event["actor"])
        if event["event"] in {"commented", "reviewed"}:
            waiting.discard(event["user"])
        # A mention or assignment can be recorded in the recipient's name.
        if event["event"] in {"labeled", "unlabeled", "closed", "reopened", "merged"}:
            waiting.discard(event["actor"])
    if waiting:
        since = cutoff.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for comment in records(f"pulls/{number}/comments?since={since}", "{user: .user.login}"):
            waiting.discard(comment["user"])
    if waiting:
        for commit in records(f"pulls/{number}/commits", "{user: .author.login, date: .commit.committer.date}"):
            if datetime.fromisoformat(commit["date"]) >= cutoff:
                waiting.discard(commit["user"])
    return sorted(waiting, key=str.casefold)


def reminder_lines(now: datetime) -> Iterable[str]:
    cutoff = previous_weekday(now)
    stale = now - timedelta(days=7)
    pulls = records(
        "pulls?state=open&sort=updated&direction=desc",
        "{number, draft, updated_at, created_at, author: .user.login, author_type: .user.type,"
        ' reviewers: [.requested_reviewers[] | select(.type == "User") | .login]}',
    )
    for pull in pulls:
        if datetime.fromisoformat(pull["updated_at"]) < stale:
            break
        if pull["draft"] or not pull["reviewers"] or datetime.fromisoformat(pull["created_at"]) >= cutoff:
            continue
        if pull["author_type"] != "User" or not has_write_access(pull["author"]):
            continue
        reviewers = due_reviewers(pull["number"], pull["reviewers"], cutoff)
        if reviewers:
            number = pull["number"]
            yield f"<https://github.com/{REPO}/pull/{number}|#{number}>  " + ", ".join(reviewers)


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
