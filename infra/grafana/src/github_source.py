# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GitHub ferry and CI-build status as flat JSON rows.

The bridge fans out over the ferry workflows and precomputes the fields the panels
render (per-tier success rate, run duration; per-commit finalized success rate), so
the Grafana panels stay thin. Results are cached — the same rate-limit shield the
status page relied on — and the token stays server-side.

The repo is public; the token only lifts the REST rate limit (60->5000/hr) and is
required for the GraphQL build query, which GitHub gates even on public repos.
"""

import logging
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

import httpx
from config import BUILD_HISTORY, FERRY_GROUPS, FERRY_RUN_LIMIT, GITHUB_REPO
from errors import UpstreamError
from graphql_source import graphql_data
from nightly import NightlyLaneSnapshot, NightlyRun, project_nightlies
from nightly_config import NIGHTLY_LANES, NightlyLane

logger = logging.getLogger(__name__)

_REST_BASE = "https://api.github.com"
_GRAPHQL_URL = "https://api.github.com/graphql"

# Per-commit rollup states GitHub reports; the build success rate is over finalized ones.
_FINALIZED_STATES = ("SUCCESS", "FAILURE", "ERROR")

# Runs fetched per nightly lane; each lane runs at most once (weekly lanes) to a
# handful of times (daily lanes) within the trailing query window.
_NIGHTLY_RUN_LIMIT = 30

_BUILD_QUERY = """
query MainCommits($owner: String!, $repo: String!, $count: Int!) {
  repository(owner: $owner, name: $repo) {
    ref(qualifiedName: "refs/heads/main") {
      target {
        ... on Commit {
          history(first: $count) {
            nodes {
              oid
              abbreviatedOid
              messageHeadline
              committedDate
              url
              author { user { login avatarUrl(size: 80) } name }
              statusCheckRollup { state }
            }
          }
        }
      }
    }
  }
}
"""


def _iso_to_ms(value: str | None) -> int | None:
    """Parse an ISO-8601 instant to epoch milliseconds (Grafana's time axis)."""
    if not value:
        return None
    return round(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _nightly_query_start(lane: NightlyLane, now: datetime) -> datetime:
    """The earliest run creation time worth fetching for a lane's 7-day matrix.

    Daily lanes only need one extra day of slack for the oldest matrix column;
    weekly (or sparser) lanes need a full extra cadence period so that column's
    occurrence isn't cut off.
    """
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    cadence_days = 1 if len(lane.weekdays) == 7 else 7
    return today - timedelta(days=6 + cadence_days)


def _to_nightly_run(run: dict) -> NightlyRun:
    return NightlyRun(
        id=run["id"],
        status=run["status"],
        conclusion=run.get("conclusion"),
        sha=run["head_sha"],
        created_at=run["created_at"],
        run_started_at=run.get("run_started_at"),
        updated_at=run["updated_at"],
        url=run["html_url"],
    )


class GithubSource:
    """Ferry and build status for the configured repo."""

    def __init__(self, *, token: str | None, timeout: float) -> None:
        headers = {
            "accept": "application/vnd.github+json",
            "x-github-api-version": "2022-11-28",
            "user-agent": "marin-grafana-bridge",
        }
        if token:
            headers["authorization"] = f"Bearer {token}"
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def _get(self, url: str, params: dict | None = None) -> dict:
        try:
            response = self._client.get(url, params=params)
        except httpx.TransportError as err:
            raise UpstreamError("github", f"GET {url} unreachable ({err})", status_code=504) from err
        if response.status_code != 200:
            raise UpstreamError("github", f"GET {url} returned {response.status_code}", status_code=502)
        return response.json()

    def ferries(self) -> list[dict]:
        """One row per recent run across every ferry tier, with per-tier success rate."""
        rows: list[dict] = []
        for group in FERRY_GROUPS:
            for tier in group.tiers:
                runs = self._get(
                    f"{_REST_BASE}/repos/{GITHUB_REPO}/actions/workflows/{tier.file}/runs",
                    params={"per_page": FERRY_RUN_LIMIT, "branch": "main"},
                ).get("workflow_runs", [])
                finished = [r for r in runs if r.get("conclusion")]
                successes = sum(1 for r in finished if r["conclusion"] == "success")
                success_rate = successes / len(finished) if finished else None
                for run in runs:
                    started = run.get("run_started_at") or run.get("created_at")
                    completed = run.get("status") == "completed"
                    duration = None
                    if completed and run.get("run_started_at") and run.get("updated_at"):
                        duration = round((_iso_to_ms(run["updated_at"]) - _iso_to_ms(run["run_started_at"])) / 1000)
                    rows.append(
                        {
                            "group": group.name,
                            "tier": tier.label,
                            "file": tier.file,
                            "run_id": run.get("id"),
                            "conclusion": run.get("conclusion"),
                            "status": run.get("status"),
                            "sha": (run.get("head_sha") or "")[:7],
                            "started_at": _iso_to_ms(started),
                            "duration_seconds": duration,
                            "html_url": run.get("html_url"),
                            "actor": (run.get("actor") or {}).get("login"),
                            "success_rate": success_rate,
                        }
                    )
        return rows

    def builds(self) -> list[dict]:
        """One row per recent commit on main with its CI rollup state and finalized success rate."""
        data = graphql_data(
            self._client,
            source="github",
            url=_GRAPHQL_URL,
            query=_BUILD_QUERY,
            variables={
                "owner": GITHUB_REPO.split("/")[0],
                "repo": GITHUB_REPO.split("/")[1],
                "count": BUILD_HISTORY,
            },
        )
        nodes = ((data.get("repository") or {}).get("ref") or {}).get("target") or {}
        nodes = ((nodes or {}).get("history") or {}).get("nodes") or []

        states = [(node.get("statusCheckRollup") or {}).get("state") or "NONE" for node in nodes]
        finalized = [s for s in states if s in _FINALIZED_STATES]
        success_rate = sum(1 for s in finalized if s == "SUCCESS") / len(finalized) if finalized else None

        rows = []
        for node, state in zip(nodes, states, strict=True):
            author = node.get("author") or {}
            user = author.get("user") or {}
            rows.append(
                {
                    "oid": node.get("oid"),
                    "short_oid": node.get("abbreviatedOid"),
                    "headline": node.get("messageHeadline"),
                    "committed_at": _iso_to_ms(node.get("committedDate")),
                    "author": user.get("login") or author.get("name"),
                    "avatar_url": user.get("avatarUrl"),
                    "url": node.get("url"),
                    "state": state,
                    "success_rate": success_rate,
                }
            )
        return rows

    def _nightly_lane_snapshot(self, lane: NightlyLane, now: datetime) -> NightlyLaneSnapshot:
        """Fetch one lane's recent scheduled runs; a failure becomes an error snapshot, not a raise."""
        url = f"{_REST_BASE}/repos/{lane.repository}/actions/workflows/{quote(lane.workflow_file, safe='')}/runs"
        params = {
            "branch": lane.branch,
            "event": "schedule",
            "per_page": _NIGHTLY_RUN_LIMIT,
            "created": f">={_nightly_query_start(lane, now).strftime('%Y-%m-%dT%H:%M:%S.000Z')}",
        }
        try:
            payload = self._get(url, params=params)
        except UpstreamError as err:
            return NightlyLaneSnapshot(lane_id=lane.id, runs=[], error=str(err))
        runs = [_to_nightly_run(run) for run in payload.get("workflow_runs", [])]
        return NightlyLaneSnapshot(lane_id=lane.id, runs=runs, error=None)

    def nightlies(self, now: datetime | None = None) -> list[dict]:
        """One linked, duration-aware cell per nightly lane and UTC day."""
        effective_now = now or datetime.now(UTC)
        snapshots = [self._nightly_lane_snapshot(lane, effective_now) for lane in NIGHTLY_LANES]
        return project_nightlies(NIGHTLY_LANES, snapshots, effective_now)
