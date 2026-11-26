# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache public Marin PRs and fill in review metadata in bounded batches."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import zip_longest
from typing import Annotated, TypedDict

import httpx
from fastapi import FastAPI, Header, HTTPException
from marina.applets import AppletServices
from sqlalchemy import Connection, Row, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

REPO_API = "https://api.github.com/repos/marin-community/marin/"
SYNC_LOCK = 0x726576696577
SYNC_SECONDS = 20
GITHUB_TIMEOUT = 4
PAGE_SIZE = 100
BATCH_SIZE = 4
DISCOVERY_INTERVAL = timedelta(minutes=15)
HISTORY_WINDOW = timedelta(days=14)
RETRY_SECONDS = 60
REVIEW_STATES = frozenset({"approved", "changes_requested", "commented", "dismissed"})
logger = logging.getLogger(__name__)


class PullSummary(TypedDict):
    number: int
    title: str
    html_url: str
    created_at: str
    updated_at: str
    draft: bool
    requested_reviewers: list[dict[str, str]]
    requested_teams: list[dict[str, str]]


class ReviewFact(TypedDict):
    login: str
    submitted_at: str
    response_hours: float | None
    response_at: str | None


class PullMetadata(TypedDict):
    number: int
    updated_at: str
    lines: int
    reviews: list[ReviewFact]


class ReviewedPull(TypedDict):
    number: int
    title: str | None
    lines: int
    submitted_at: str


@dataclass
class ReviewerHistory:
    lines: int = 0
    responses: list[float] = field(default_factory=list)
    pulls: list[ReviewedPull] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "reviewed": len(self.pulls),
            "lines_per_week": round(self.lines / (HISTORY_WINDOW / timedelta(weeks=1))),
            "avg_response_hours": sum(self.responses) / len(self.responses) if self.responses else None,
            "response_count": len(self.responses),
            "pulls": sorted(self.pulls, key=lambda pull: pull["submitted_at"], reverse=True),
        }


def migrate(connection: Connection) -> None:
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS queue_snapshot (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            refreshed_at timestamptz,
            retry_at timestamptz NOT NULL DEFAULT now(),
            open_prs jsonb NOT NULL DEFAULT '[]',
            recent_prs jsonb NOT NULL DEFAULT '[]'
        )
    """
        )
    )
    connection.execute(text("INSERT INTO queue_snapshot DEFAULT VALUES ON CONFLICT DO NOTHING"))
    connection.execute(
        text(
            """
        CREATE TABLE IF NOT EXISTS review_metadata (
            number integer PRIMARY KEY,
            updated_at text NOT NULL,
            lines integer NOT NULL,
            reviews jsonb NOT NULL,
            fetched_at timestamptz NOT NULL DEFAULT now()
        )
    """
        )
    )


def pr_summary(pr: dict) -> PullSummary:
    """Keep the fields the queue needs from GitHub's pull list."""
    return {
        "number": pr["number"],
        "title": pr["title"],
        "html_url": pr["html_url"],
        "created_at": pr["created_at"],
        "updated_at": pr["updated_at"],
        "draft": pr["draft"],
        "requested_reviewers": [{"login": user["login"]} for user in pr["requested_reviewers"]],
        "requested_teams": [{"slug": team["slug"]} for team in pr["requested_teams"]],
    }


def review_facts(timeline: list[dict]) -> list[ReviewFact]:
    """Keep one submitted review per person and PR, with matched response time when available."""
    requests = {}
    reviews: dict[str, ReviewFact] = {}
    for event in timeline:
        kind = event["event"]
        reviewer = event.get("requested_reviewer")
        if kind == "review_requested" and reviewer:
            requests[reviewer["login"].lower()] = datetime.fromisoformat(event["created_at"])
        elif kind == "review_request_removed" and reviewer:
            requests.pop(reviewer["login"].lower(), None)
        elif kind == "reviewed" and event.get("state") in REVIEW_STATES and event.get("submitted_at"):
            user = event.get("user")
            if user is None or user["type"] != "User":
                continue
            login = user["login"].lower()
            submitted = datetime.fromisoformat(event["submitted_at"])
            requested = requests.pop(login, None)
            previous = reviews.get(login)
            response_hours = previous["response_hours"] if previous else None
            response_at = previous["response_at"] if previous else None
            if requested is not None and submitted >= requested:
                response_hours = (submitted - requested).total_seconds() / 3600
                response_at = submitted.isoformat()
            reviews[login] = {
                "login": login,
                "submitted_at": submitted.isoformat(),
                "response_hours": response_hours,
                "response_at": response_at,
            }
    return list(reviews.values())


async def github_get(client: httpx.AsyncClient, path: str, **params: str | int) -> httpx.Response:
    response = await client.get(path, params=params)
    response.raise_for_status()
    return response


async def open_prs(client: httpx.AsyncClient) -> list[PullSummary]:
    pulls = {}
    page = 1
    while True:
        response = await github_get(client, "pulls", state="open", per_page=PAGE_SIZE, page=page)
        for pr in response.json():
            pulls[pr["number"]] = pr_summary(pr)
        if "next" not in response.links:
            return list(pulls.values())
        page += 1


async def pr_metadata(client: httpx.AsyncClient, number: int) -> PullMetadata:
    detail = (await github_get(client, f"pulls/{number}")).json()
    timeline = []
    # Only complete timelines can reliably pair a request with its response.
    for page in range(1, 4):
        response = await github_get(client, f"issues/{number}/timeline", per_page=PAGE_SIZE, page=page)
        timeline.extend(response.json())
        if "next" not in response.links:
            break
    else:
        timeline = []
    return {
        "number": number,
        "updated_at": detail["updated_at"],
        "lines": detail["additions"] + detail["deletions"],
        "reviews": review_facts(timeline),
    }


def pending_metadata(snapshot: Row, versions: dict[int, str]) -> list[int]:
    """Return missing or stale metadata IDs in sync order."""
    pending = [pr for pr in snapshot.open_prs if pr["requested_reviewers"] or pr["requested_teams"]]
    # Interleave requests and recent reviews before checking older open PRs.
    prioritized = (pr for pair in zip_longest(pending, snapshot.recent_prs) for pr in pair if pr is not None)
    numbers = dict.fromkeys(
        pr["number"] for pr in (*prioritized, *snapshot.open_prs) if versions.get(pr["number"], "") < pr["updated_at"]
    )
    return list(numbers)


def retry_seconds(error: httpx.HTTPStatusError) -> int:
    headers = error.response.headers
    if error.response.status_code in {403, 429}:
        if headers.get("retry-after", "").isdigit():
            return max(RETRY_SECONDS, int(headers["retry-after"]))
        if headers.get("x-ratelimit-remaining") == "0":
            reset = headers.get("x-ratelimit-reset", "")
            if reset.isdigit():
                return max(RETRY_SECONDS, int(reset) - int(datetime.now(UTC).timestamp()) + 1)
    return RETRY_SECONDS


async def refresh_snapshot(connection: Connection, client: httpx.AsyncClient) -> Row:
    opened = await open_prs(client)
    recent = await github_get(client, "pulls", state="all", sort="updated", direction="desc", per_page=PAGE_SIZE)
    snapshot = connection.execute(
        text(
            """
            UPDATE queue_snapshot SET refreshed_at = clock_timestamp(),
                open_prs = :open_prs, recent_prs = :recent_prs RETURNING *
        """
        ).bindparams(bindparam("open_prs", type_=JSONB), bindparam("recent_prs", type_=JSONB)),
        {"open_prs": opened, "recent_prs": [pr_summary(pr) for pr in recent.json()]},
    ).one()
    connection.execute(text("DELETE FROM review_metadata WHERE fetched_at < now() - interval '30 days'"))
    return snapshot


def save_metadata(connection: Connection, record: PullMetadata) -> None:
    connection.execute(
        text(
            """
            INSERT INTO review_metadata (number, updated_at, lines, reviews)
            VALUES (:number, :updated_at, :lines, :reviews)
            ON CONFLICT (number) DO UPDATE SET updated_at = EXCLUDED.updated_at,
                lines = EXCLUDED.lines, reviews = EXCLUDED.reviews, fetched_at = now()
        """
        ).bindparams(bindparam("reviews", type_=JSONB)),
        dict(record),
    )


async def sync_batch(connection: Connection) -> int:
    """Save a bounded batch and return the retry delay in seconds."""
    snapshot = connection.execute(text("SELECT * FROM queue_snapshot")).one()
    retry_in = (snapshot.retry_at - datetime.now(UTC)).total_seconds()
    if retry_in > 0:
        return int(retry_in) + 1

    delay = RETRY_SECONDS
    try:
        async with (
            asyncio.timeout(SYNC_SECONDS),
            httpx.AsyncClient(
                base_url=REPO_API,
                headers={"Accept": "application/vnd.github+json"},
                timeout=GITHUB_TIMEOUT,
            ) as client,
        ):
            if snapshot.refreshed_at is None or datetime.now(UTC) - snapshot.refreshed_at >= DISCOVERY_INTERVAL:
                snapshot = await refresh_snapshot(connection, client)
            versions = dict(connection.execute(text("SELECT number, updated_at FROM review_metadata")).tuples().all())
            missing = pending_metadata(snapshot, versions)
            for number in missing[:BATCH_SIZE]:
                save_metadata(connection, await pr_metadata(client, number))
            if len(missing) <= BATCH_SIZE:
                delay = max(1, int((snapshot.refreshed_at + DISCOVERY_INTERVAL - datetime.now(UTC)).total_seconds()))
    except httpx.HTTPStatusError as error:
        delay = retry_seconds(error)
        logger.warning("Review queue sync paused after GitHub HTTP %s", error.response.status_code)
    except (TimeoutError, httpx.RequestError) as error:
        logger.warning("Review queue sync paused after %s", type(error).__name__)

    connection.execute(
        text("UPDATE queue_snapshot SET retry_at = clock_timestamp() + :delay"),
        {"delay": timedelta(seconds=delay)},
    )
    return delay


def queue_payload(connection: Connection) -> dict:
    snapshot = connection.execute(text("SELECT * FROM queue_snapshot")).one()
    records = connection.execute(text("SELECT number, updated_at, lines, reviews FROM review_metadata")).all()
    metadata = {row.number: row for row in records}
    for pr in snapshot.open_prs:
        cached = metadata.get(pr["number"])
        pr["lines"] = cached.lines if cached and cached.updated_at >= pr["updated_at"] else None
        pr["reviewed_reviewers"] = [{"login": review["login"]} for review in cached.reviews] if cached else []
    cutoff = datetime.now(UTC) - HISTORY_WINDOW
    titles = {pr["number"]: pr["title"] for pr in snapshot.open_prs + snapshot.recent_prs}
    stats: dict[str, ReviewerHistory] = {}
    for row in records:
        reviews: list[ReviewFact] = row.reviews
        for review in reviews:
            submitted = datetime.fromisoformat(review["submitted_at"])
            if submitted < cutoff:
                continue
            item = stats.setdefault(review["login"], ReviewerHistory())
            item.lines += row.lines
            item.pulls.append(
                {
                    "number": row.number,
                    "title": titles.get(row.number),
                    "lines": row.lines,
                    "submitted_at": review["submitted_at"],
                }
            )
            response_at, hours = review["response_at"], review["response_hours"]
            if response_at and hours is not None and datetime.fromisoformat(response_at) >= cutoff:
                item.responses.append(hours)
    versions = {row.number: row.updated_at for row in records}
    total = len(pending_metadata(snapshot, {}))
    return {
        "refreshed_at": snapshot.refreshed_at,
        "prs": snapshot.open_prs,
        "stats": {login: history.summary() for login, history in stats.items()},
        "sync": {"completed": total - len(pending_metadata(snapshot, versions)), "total": total},
    }


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/queue")
    def queue() -> dict:
        with engine.connect() as connection:
            return queue_payload(connection)

    @api.post("/sync")
    def sync(x_review_queue: Annotated[str, Header()]) -> dict:
        if x_review_queue != "sync":
            raise HTTPException(400, "Invalid sync request")
        with engine.begin() as connection:
            acquired = connection.execute(
                text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": SYNC_LOCK}
            ).scalar_one()
            if not acquired:
                return {"retry_in": 5}
            return {"retry_in": asyncio.run(sync_batch(connection))}

    return api
