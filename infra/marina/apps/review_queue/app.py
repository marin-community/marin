# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache public Marin PRs and fill in review metadata in bounded batches."""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import zip_longest
from typing import Annotated, Literal, NotRequired, TypedDict

import httpx
from fastapi import FastAPI, Header, HTTPException
from marina.applets import AppletServices
from sqlalchemy import Connection, Row, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

REPO_API = "https://api.github.com/repos/marin-community/marin/"
PULLS_PATH = "pulls"
SYNC_LOCK = 0x726576696577
SYNC_SECONDS = 20
GITHUB_TIMEOUT = 4
PAGE_SIZE = 100
BATCH_SIZE = 4
DISCOVERY_INTERVAL = timedelta(minutes=15)
HISTORY_WINDOW = timedelta(days=7)
DISCOVERY_OVERLAP = timedelta(minutes=1)
RETRY_SECONDS = 60
REVIEW_STATES = frozenset({"approved", "changes_requested", "commented", "dismissed"})
READ_SNAPSHOT = text("SELECT * FROM queue_snapshot")
READ_METADATA = text("SELECT number, updated_at, lines, reviews, activity FROM review_metadata")
logger = logging.getLogger(__name__)


class ReviewerRef(TypedDict):
    login: str


class TeamRef(TypedDict):
    slug: str


class PullSummary(TypedDict):
    number: int
    title: str
    html_url: str
    created_at: str
    updated_at: str
    draft: bool
    requested_reviewers: list[ReviewerRef]
    requested_teams: list[TeamRef]
    author: NotRequired[str | None]


class ReviewFact(TypedDict):
    login: str
    submitted_at: str
    response_hours: float | None
    response_at: str | None


ReviewActivity = dict[str, Literal["approved", "feedback"]]


class PullMetadata(TypedDict):
    number: int
    updated_at: str
    lines: int
    reviews: list[ReviewFact]
    activity: ReviewActivity


class ReviewedPull(TypedDict):
    number: int
    title: str | None
    lines: int
    submitted_at: str


class ReviewerSummary(TypedDict):
    reviewed: int
    reviewed_lines: int
    avg_response_hours: float | None
    response_count: int
    pulls: list[ReviewedPull]


class SyncProgress(TypedDict):
    completed: int
    total: int
    history_discovered: bool


class SyncResult(TypedDict):
    retry_in: int


class OpenPull(PullSummary):
    lines: int | None
    reviewed_reviewers: list[ReviewerRef]
    activity: ReviewActivity | None


class QueuePayload(TypedDict):
    refreshed_at: datetime | None
    prs: list[OpenPull]
    stats: dict[str, ReviewerSummary]
    sync: SyncProgress
    history_days: int


@dataclass
class ReviewerHistory:
    lines: int = 0
    responses: list[float] = field(default_factory=list)
    pulls: list[ReviewedPull] = field(default_factory=list)

    def summary(self) -> ReviewerSummary:
        return {
            "reviewed": len(self.pulls),
            "reviewed_lines": self.lines,
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
            open_prs jsonb NOT NULL DEFAULT '[]'
        )
    """
        )
    )
    connection.execute(text("INSERT INTO queue_snapshot DEFAULT VALUES ON CONFLICT DO NOTHING"))
    connection.execute(text("ALTER TABLE queue_snapshot ADD COLUMN IF NOT EXISTS history_since timestamptz"))
    connection.execute(
        text("ALTER TABLE queue_snapshot ADD COLUMN IF NOT EXISTS history_prs jsonb NOT NULL DEFAULT '[]'")
    )
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
    connection.execute(text("ALTER TABLE review_metadata ADD COLUMN IF NOT EXISTS activity jsonb"))


def pr_summary(pr: dict) -> PullSummary:
    """Keep the fields the queue needs from GitHub's pull list."""
    return {
        "number": pr["number"],
        "title": pr["title"],
        "html_url": pr["html_url"],
        "created_at": pr["created_at"],
        "updated_at": pr["updated_at"],
        "draft": pr["draft"],
        "author": pr["user"]["login"] if pr.get("user") else None,
        "requested_reviewers": [{"login": user["login"]} for user in pr["requested_reviewers"]],
        "requested_teams": [{"slug": team["slug"]} for team in pr["requested_teams"]],
    }


def external_login(user: dict | None, author: str | None) -> str | None:
    if user is None or user["type"] != "User":
        return None
    login = user["login"].lower()
    return login if author is None or login != author.lower() else None


def review_facts(timeline: list[dict], author: str | None) -> list[ReviewFact]:
    """Keep each reviewer's latest submission and most recent matched response."""
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
            login = external_login(event.get("user"), author)
            if login is None:
                continue
            submitted = datetime.fromisoformat(event["submitted_at"])
            requested = requests.pop(login, None)
            previous = reviews.get(login)
            response_hours = previous["response_hours"] if previous else None
            matched_submission_at = previous["response_at"] if previous else None
            if requested is not None and submitted >= requested:
                response_hours = (submitted - requested).total_seconds() / 3600
                matched_submission_at = submitted.isoformat()
            reviews[login] = {
                "login": login,
                "submitted_at": submitted.isoformat(),
                "response_hours": response_hours,
                "response_at": matched_submission_at,
            }
    return list(reviews.values())


def review_activity(timeline: list[dict], author: str | None) -> ReviewActivity:
    """Summarize human feedback and current approvals from the PR timeline."""
    activity: ReviewActivity = {}
    for event in timeline:
        kind = event["event"]
        if kind not in {"reviewed", "commented"}:
            continue
        login = external_login(event.get("user"), author)
        if login is None:
            continue
        if kind == "commented":
            activity.setdefault(login, "feedback")
        elif event.get("submitted_at"):
            state = event.get("state")
            if state == "approved":
                activity[login] = "approved"
            elif state == "changes_requested":
                activity[login] = "feedback"
            elif state in {"commented", "dismissed"}:
                activity.setdefault(login, "feedback")
    return activity


async def github_get(client: httpx.AsyncClient, path: str, **params: str | int) -> httpx.Response:
    response = await client.get(path, params=params)
    response.raise_for_status()
    return response


async def github_pages(client: httpx.AsyncClient, path: str, **params: str) -> AsyncIterator[list[dict]]:
    page = 1
    while True:
        response = await github_get(client, path, per_page=PAGE_SIZE, page=page, **params)
        yield response.json()
        if "next" not in response.links:
            return
        page += 1


async def open_prs(client: httpx.AsyncClient) -> list[PullSummary]:
    pulls = {}
    async for page in github_pages(client, PULLS_PATH, state="open"):
        for pr in page:
            pulls[pr["number"]] = pr_summary(pr)
    return list(pulls.values())


async def recent_prs(
    client: httpx.AsyncClient, previous: list[PullSummary], cutoff: datetime, since: datetime
) -> list[PullSummary]:
    """Fetch changed PRs and keep all known PRs inside the rolling window."""
    pulls = {pr["number"]: pr for pr in previous if datetime.fromisoformat(pr["updated_at"]) >= cutoff}
    async for page in github_pages(client, PULLS_PATH, state="all", sort="updated", direction="desc"):
        changed = [pr for pr in page if datetime.fromisoformat(pr["updated_at"]) >= since]
        for pr in changed:
            pulls[pr["number"]] = pr_summary(pr)
        if len(changed) < len(page):
            break
    return sorted(pulls.values(), key=lambda item: item["updated_at"], reverse=True)


async def pr_metadata(client: httpx.AsyncClient, number: int) -> PullMetadata:
    detail = (await github_get(client, f"pulls/{number}")).json()
    timeline = []
    async for page in github_pages(client, f"issues/{number}/timeline"):
        timeline.extend(page)
    author = detail["user"]["login"] if detail.get("user") else None
    return {
        "number": number,
        "updated_at": detail["updated_at"],
        "lines": detail["additions"] + detail["deletions"],
        "reviews": review_facts(timeline, author),
        "activity": review_activity(timeline, author),
    }


def prs_missing_metadata(snapshot: Row, metadata: dict[int, Row]) -> list[int]:
    """Return PR numbers with missing or stale metadata in sync order."""
    opened = [pr for pr in snapshot.open_prs if not pr["draft"]]
    individual = {
        pr["number"]
        for pr in opened
        if pr["requested_reviewers"] or (pr["number"] in metadata and metadata[pr["number"]].reviews)
    }
    assigned = [pr for pr in opened if pr["number"] in individual or pr["requested_teams"]]
    # Interleave active queues with recently updated PRs before checking older open PRs.
    prioritized = (pr for pair in zip_longest(assigned, snapshot.history_prs) for pr in pair if pr is not None)
    numbers = {}
    for pr in (*prioritized, *opened):
        if pr["draft"]:
            continue
        number = pr["number"]
        cached = metadata.get(number)
        if cached is None or cached.updated_at < pr["updated_at"] or (number in individual and cached.activity is None):
            numbers[number] = None
    return list(numbers)


def retry_seconds(error: httpx.HTTPStatusError) -> int:
    headers = error.response.headers
    if error.response.status_code in {403, 429}:
        retry_after = headers.get("retry-after", "")
        if retry_after.isdigit():
            return max(RETRY_SECONDS, int(retry_after))
        if headers.get("x-ratelimit-remaining") == "0":
            reset = headers.get("x-ratelimit-reset", "")
            if reset.isdigit():
                return max(RETRY_SECONDS, int(reset) - int(datetime.now(UTC).timestamp()) + 1)
    return RETRY_SECONDS


async def refresh_snapshot(connection: Connection, client: httpx.AsyncClient, snapshot: Row) -> Row:
    started = datetime.now(UTC)
    cutoff = started - HISTORY_WINDOW
    since = cutoff
    if snapshot.history_since and snapshot.history_since <= cutoff:
        since = max(cutoff, snapshot.history_since + HISTORY_WINDOW - DISCOVERY_OVERLAP)
    opened = await open_prs(client)
    recent = await recent_prs(client, snapshot.history_prs, cutoff, since)
    snapshot = connection.execute(
        text(
            """
            UPDATE queue_snapshot SET refreshed_at = :started,
                open_prs = :open_prs, history_prs = :history_prs, history_since = :cutoff RETURNING *
        """
        ).bindparams(bindparam("open_prs", type_=JSONB), bindparam("history_prs", type_=JSONB)),
        {"open_prs": opened, "history_prs": recent, "started": started, "cutoff": cutoff},
    ).one()
    connection.execute(text("DELETE FROM review_metadata WHERE fetched_at < now() - interval '30 days'"))
    return snapshot


async def sync_prs(connection: Connection, client: httpx.AsyncClient, snapshot: Row) -> int:
    cutoff = datetime.now(UTC) - HISTORY_WINDOW
    if snapshot.history_since is None or not cutoff - DISCOVERY_INTERVAL < snapshot.history_since <= cutoff:
        snapshot = await refresh_snapshot(connection, client, snapshot)
    metadata = {row.number: row for row in connection.execute(READ_METADATA)}
    missing = prs_missing_metadata(snapshot, metadata)
    for number in missing[:BATCH_SIZE]:
        save_metadata(connection, await pr_metadata(client, number))
    if len(missing) > BATCH_SIZE:
        return RETRY_SECONDS
    # The open snapshot timestamp can also change when a retained revision refreshes it.
    next_scan = snapshot.history_since + HISTORY_WINDOW + DISCOVERY_INTERVAL
    return max(1, int((next_scan - datetime.now(UTC)).total_seconds()))


def save_metadata(connection: Connection, record: PullMetadata) -> None:
    connection.execute(
        text(
            """
            INSERT INTO review_metadata (number, updated_at, lines, reviews, activity)
            VALUES (:number, :updated_at, :lines, :reviews, :activity)
            ON CONFLICT (number) DO UPDATE SET updated_at = EXCLUDED.updated_at,
                lines = EXCLUDED.lines, reviews = EXCLUDED.reviews, activity = EXCLUDED.activity, fetched_at = now()
        """
        ).bindparams(bindparam("reviews", type_=JSONB), bindparam("activity", type_=JSONB)),
        dict(record),
    )


async def sync_batch(connection: Connection) -> int:
    """Save a bounded batch and return the retry delay in seconds."""
    snapshot = connection.execute(READ_SNAPSHOT).one()
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
            delay = await sync_prs(connection, client, snapshot)
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


def enriched_open_prs(prs: list[PullSummary], metadata: dict[int, Row]) -> list[OpenPull]:
    opened: list[OpenPull] = []
    for pr in prs:
        if pr["draft"]:
            continue
        cached = metadata.get(pr["number"])
        author = (pr.get("author") or "").lower()
        fresh = cached is not None and cached.updated_at >= pr["updated_at"]
        opened.append(
            {
                **pr,
                "lines": cached.lines if fresh else None,
                "activity": cached.activity if fresh else None,
                "reviewed_reviewers": (
                    [{"login": review["login"]} for review in cached.reviews if review["login"] != author]
                    if cached
                    else []
                ),
            }
        )
    return opened


def reviewer_history(records: list[Row], pulls: dict[int, PullSummary]) -> dict[str, ReviewerSummary]:
    cutoff = datetime.now(UTC) - HISTORY_WINDOW
    stats: dict[str, ReviewerHistory] = {}
    for row in records:
        pr = pulls.get(row.number)
        if pr and pr["draft"]:
            continue
        author = (pr.get("author") or "").lower() if pr else ""
        reviews: list[ReviewFact] = row.reviews
        for review in reviews:
            if review["login"] == author:
                continue
            submitted = datetime.fromisoformat(review["submitted_at"])
            if submitted < cutoff:
                continue
            item = stats.setdefault(review["login"], ReviewerHistory())
            item.lines += row.lines
            item.pulls.append(
                {
                    "number": row.number,
                    "title": pr["title"] if pr else None,
                    "lines": row.lines,
                    "submitted_at": review["submitted_at"],
                }
            )
            response_at, hours = review["response_at"], review["response_hours"]
            if response_at and hours is not None and datetime.fromisoformat(response_at) >= cutoff:
                item.responses.append(hours)
    return {login: history.summary() for login, history in stats.items()}


def queue_payload(connection: Connection) -> QueuePayload:
    snapshot = connection.execute(READ_SNAPSHOT).one()
    records = list(connection.execute(READ_METADATA).all())
    metadata = {row.number: row for row in records}
    pulls = {pr["number"]: pr for pr in snapshot.history_prs + snapshot.open_prs}
    total = len(prs_missing_metadata(snapshot, {}))
    return {
        "refreshed_at": snapshot.refreshed_at,
        "prs": enriched_open_prs(snapshot.open_prs, metadata),
        "stats": reviewer_history(records, pulls),
        "sync": {
            "completed": total - len(prs_missing_metadata(snapshot, metadata)),
            "total": total,
            "history_discovered": snapshot.history_since is not None,
        },
        "history_days": HISTORY_WINDOW.days,
    }


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/queue")
    def queue() -> QueuePayload:
        with engine.connect() as connection:
            return queue_payload(connection)

    @api.post("/sync")
    def sync(x_review_queue: Annotated[str, Header()]) -> SyncResult:
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
