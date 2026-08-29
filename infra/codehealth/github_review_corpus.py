# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded GitHub collection for frozen pull-request review corpora."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import batched
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .review_tables import parse_utc

SCAN_PAGE_SIZE = 20
CONNECTION_PAGE_SIZE = 10
COMMIT_PARENT_PAGE_SIZE = 100
MAX_GITHUB_CHANGED_FILES = 3_000
MAX_GRAPHQL_POINTS = 900
MAX_REST_REQUESTS = 850
HYDRATION_BATCH_SIZE = 20
FINGERPRINT_BATCH_SIZE = 100
REST_RETRY_RESERVE = 150
REST_PAGE_SIZE = 100
DIAGNOSTIC_EVENT_LIMIT = 10

EventKind = Literal["inline_comment", "review", "issue_comment"]


class CorpusModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PullRequestRecord(CorpusModel):
    repository: str
    number: int
    node_id: str
    url: str
    title: str
    body: str
    state: str
    draft: bool
    author: str
    author_type: str
    author_association: str
    created_at: str
    updated_at: str
    closed_at: str | None
    merged_at: str | None
    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    additions: int
    deletions: int
    changed_files: int
    commits: int
    review_comments: int
    issue_comments: int
    commit_shas: tuple[str, ...]
    diff_path: str | None


class ChangedFileRecord(CorpusModel):
    pr_number: int
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int


class CommitRecord(CorpusModel):
    pr_number: int
    sha: str
    author: str | None
    authored_at: str | None
    committed_at: str | None
    message: str
    parents: tuple[str, ...]


class ReviewThreadRecord(CorpusModel):
    pr_number: int
    thread_id: str
    comment_ids: tuple[int, ...]
    is_resolved: bool
    is_outdated: bool
    resolved_by: str | None


class ReviewEventRecord(CorpusModel):
    event_id: str
    kind: EventKind
    database_id: int
    node_id: str | None
    repository: str
    pr_number: int
    pr_author: str
    author: str
    author_type: str
    author_association: str
    body: str
    state: str | None
    created_at: str | None
    updated_at: str | None
    submitted_at: str | None
    source_url: str | None
    review_id: int | None
    thread_id: str | None
    parent_comment_id: int | None
    thread_is_resolved: bool | None
    thread_is_outdated: bool | None
    thread_resolved_by: str | None
    path: str | None
    side: str | None
    line: int | None
    original_line: int | None
    start_side: str | None
    start_line: int | None
    original_start_line: int | None
    commit_id: str | None
    original_commit_id: str | None
    diff_hunk: str | None
    is_bot: bool
    is_agent_marked: bool
    is_human: bool
    in_window: bool


class PullRequestBundle(CorpusModel):
    """Complete PR context; diff is absent only when GitHub refuses an oversized diff."""

    pull_request: PullRequestRecord
    events: tuple[ReviewEventRecord, ...]
    threads: tuple[ReviewThreadRecord, ...]
    files: tuple[ChangedFileRecord, ...]
    commits: tuple[CommitRecord, ...]
    diff: str | None


class GitHubUsage(CorpusModel):
    graphql_requests: int
    graphql_points: int
    rest_requests: int
    projected_rest_requests: int


class CollectionResult(CorpusModel):
    bundles: tuple[PullRequestBundle, ...]
    candidate_pull_requests: int
    usage: GitHubUsage


@dataclass(frozen=True)
class ReviewScope:
    """Frozen review-event window and bot identities used during collection."""

    start: dt.datetime
    end: dt.datetime
    bot_logins: frozenset[str]


@dataclass(frozen=True)
class PullRequestFingerprint:
    """Fields that must remain stable while one pull request is hydrated."""

    updated_at: str
    head_sha: str
    base_sha: str
    changed_files: int
    commits: int
    reviews: int
    review_threads: int
    issue_comments: int


@dataclass(frozen=True)
class PullRequestScan:
    """Relevant scan roots and the total candidate pull-request count."""

    relevant: tuple[dict, ...]
    candidate_count: int


@dataclass(frozen=True)
class ActorIdentity:
    login: str
    actor_type: str


def _is_diff_too_large(response: str) -> bool:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("status") in {406, "406"} and payload.get("errors") == [
        {"resource": "PullRequest", "field": "diff", "code": "too_large"}
    ]


@dataclass
class GitHubClient:
    graphql_requests: int = 0
    graphql_points: int = 0
    rest_requests: int = 0
    projected_rest_requests: int = 0

    def _record_graphql_request(self, cost: int) -> None:
        self.graphql_requests += 1
        self.graphql_points += cost
        if self.graphql_points > MAX_GRAPHQL_POINTS:
            raise RuntimeError(f"GitHub GraphQL collection exceeded the {MAX_GRAPHQL_POINTS}-point safety budget")

    def _record_rest_requests(self, count: int) -> None:
        self.rest_requests += count
        if self.rest_requests > MAX_REST_REQUESTS:
            raise RuntimeError(f"GitHub REST collection exceeded the {MAX_REST_REQUESTS}-request safety budget")

    def graphql(self, query: str, variables: dict[str, object]) -> dict:
        args = ["api", "graphql", "-f", f"query={query}"]
        for name, value in variables.items():
            if value is None:
                continue
            if isinstance(value, list | tuple):
                for item in value:
                    args.extend(["-F", f"{name}[]={item}"])
                continue
            args.extend(["-F", f"{name}={value}"])
        result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise TypeError("GitHub GraphQL response must be an object")
        if payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL returned errors: {payload['errors']}")
        self._record_graphql_request(int((payload.get("data") or {}).get("rateLimit", {}).get("cost", 1)))
        return payload["data"]

    def rest_pages(self, endpoint: str) -> list[dict]:
        result = subprocess.run(
            ["gh", "api", endpoint, "--paginate", "--slurp"],
            check=True,
            capture_output=True,
            text=True,
        )
        pages = json.loads(result.stdout)
        if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
            raise TypeError("GitHub paginated response must be a list of pages")
        self._record_rest_requests(len(pages))
        return [item for page in pages for item in page]

    def rest_text(self, endpoint: str, accept: str) -> str | None:
        """Return text, or None only for GitHub's HTTP 406 PullRequest.diff too_large response."""
        result = subprocess.run(
            ["gh", "api", endpoint, "-H", f"Accept: {accept}"],
            check=False,
            capture_output=True,
            text=True,
        )
        self._record_rest_requests(1)
        if result.returncode:
            if _is_diff_too_large(result.stdout):
                return None
            result.check_returncode()
        return result.stdout

    def usage(self) -> GitHubUsage:
        return GitHubUsage(
            graphql_requests=self.graphql_requests,
            graphql_points=self.graphql_points,
            rest_requests=self.rest_requests,
            projected_rest_requests=self.projected_rest_requests,
        )


PAGE_INFO = "pageInfo { hasNextPage endCursor hasPreviousPage startCursor }"
ACTOR = "author { __typename login } authorAssociation"
ISSUE_COMMENT_FIELDS = f"""
id databaseId body createdAt updatedAt url {ACTOR}
"""
REVIEW_FIELDS = f"""
id databaseId body state createdAt updatedAt submittedAt url {ACTOR}
commit {{ oid }} comments {{ totalCount }}
"""
THREAD_COMMENT_FIELDS = f"""
id databaseId body state createdAt updatedAt url {ACTOR}
diffHunk path line originalLine startLine originalStartLine
commit {{ oid }} originalCommit {{ oid }} replyTo {{ databaseId }}
pullRequestReview {{ databaseId }}
"""
THREAD_FIELDS = f"""
id isResolved isOutdated path line originalLine startLine originalStartLine
diffSide startDiffSide subjectType resolvedBy {{ __typename login }}
comments(first: {CONNECTION_PAGE_SIZE}) {{
  totalCount {PAGE_INFO} nodes {{ {THREAD_COMMENT_FIELDS} }}
}}
"""
COMMIT_FIELDS = f"""
commit {{
  oid message authoredDate committedDate
  author {{ user {{ login }} }}
  parents(first: {COMMIT_PARENT_PAGE_SIZE}) {{ totalCount nodes {{ oid }} }}
}}
"""
FILE_FIELDS = "additions deletions changeType path"
PR_FIELDS = """
id number url title body state isDraft author { __typename login } authorAssociation
createdAt updatedAt closedAt mergedAt baseRefName baseRefOid headRefName headRefOid
additions deletions changedFiles
"""

SCAN_QUERY = f"""
query($owner: String!, $name: String!, $after: String) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{
    pullRequests(
      first: {SCAN_PAGE_SIZE}, after: $after,
      states: [OPEN, CLOSED, MERGED], orderBy: {{field: UPDATED_AT, direction: DESC}}
    ) {{
      totalCount {PAGE_INFO}
      nodes {{
        {PR_FIELDS}
        changedFiles
        commits {{ totalCount }} reviewThreads {{ totalCount }}
        comments(last: {CONNECTION_PAGE_SIZE}) {{
          totalCount {PAGE_INFO} nodes {{ {ISSUE_COMMENT_FIELDS} }}
        }}
        reviews(last: {CONNECTION_PAGE_SIZE}) {{
          totalCount {PAGE_INFO} nodes {{ {REVIEW_FIELDS} }}
        }}
      }}
    }}
  }}
}}
"""

SCAN_COMMENTS_QUERY = f"""
query($owner: String!, $name: String!, $number: Int!, $before: String) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{ pullRequest(number: $number) {{
    comments(last: {CONNECTION_PAGE_SIZE}, before: $before) {{
      totalCount {PAGE_INFO} nodes {{ {ISSUE_COMMENT_FIELDS} }}
    }}
  }} }}
}}
"""

SCAN_REVIEWS_QUERY = f"""
query($owner: String!, $name: String!, $number: Int!, $before: String) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{ pullRequest(number: $number) {{
    reviews(last: {CONNECTION_PAGE_SIZE}, before: $before) {{
      totalCount {PAGE_INFO} nodes {{ {REVIEW_FIELDS} }}
    }}
  }} }}
}}
"""

SEED_PULL_REQUEST_QUERY = f"""
query($owner: String!, $name: String!, $number: Int!) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{ candidate: issueOrPullRequest(number: $number) {{
    __typename
    ... on PullRequest {{
      {PR_FIELDS}
      commits {{ totalCount }} reviewThreads {{ totalCount }}
      comments {{ totalCount }} reviews {{ totalCount }}
    }}
  }} }}
}}
"""

HYDRATION_FIELDS = f"""
{PR_FIELDS}
comments(first: {CONNECTION_PAGE_SIZE}) {{
  totalCount {PAGE_INFO} nodes {{ {ISSUE_COMMENT_FIELDS} }}
}}
reviews(first: {CONNECTION_PAGE_SIZE}) {{
  totalCount {PAGE_INFO} nodes {{ {REVIEW_FIELDS} }}
}}
reviewThreads(first: {CONNECTION_PAGE_SIZE}) {{
  totalCount {PAGE_INFO} nodes {{ {THREAD_FIELDS} }}
}}
commits(first: {CONNECTION_PAGE_SIZE}) {{
  totalCount {PAGE_INFO} nodes {{ {COMMIT_FIELDS} }}
}}
files(first: {CONNECTION_PAGE_SIZE}) {{
  totalCount {PAGE_INFO} nodes {{ {FILE_FIELDS} }}
}}
"""


def _hydration_query(size: int) -> str:
    variables = ", ".join(f"$number{index}: Int!" for index in range(size))
    pulls = "\n".join(
        f"pr{index}: pullRequest(number: $number{index}) {{ {HYDRATION_FIELDS} }}" for index in range(size)
    )
    return f"""
query BatchHydrate($owner: String!, $name: String!, {variables}) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{ {pulls} }}
}}
"""


CONNECTION_FIELDS = MappingProxyType(
    {
        "comments": ISSUE_COMMENT_FIELDS,
        "reviews": REVIEW_FIELDS,
        "reviewThreads": THREAD_FIELDS,
        "commits": COMMIT_FIELDS,
        "files": FILE_FIELDS,
    }
)


def _connections_query(fields: list[str]) -> str:
    variables = ", ".join(f"$number{index}: Int!, $after{index}: String!" for index in range(len(fields)))
    pages = "\n".join(
        f"""pr{index}: pullRequest(number: $number{index}) {{
          {field}(first: {CONNECTION_PAGE_SIZE}, after: $after{index}) {{
            totalCount {PAGE_INFO} nodes {{ {CONNECTION_FIELDS[field]} }}
          }}
        }}"""
        for index, field in enumerate(fields)
    )
    return f"""
query BatchConnections($owner: String!, $name: String!, {variables}) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{ {pages} }}
}}
"""


THREAD_COMMENTS_PAGE_QUERY = f"""
query($thread: ID!, $after: String) {{
  rateLimit {{ cost remaining resetAt }}
  node(id: $thread) {{ ... on PullRequestReviewThread {{
    comments(first: {CONNECTION_PAGE_SIZE}, after: $after) {{
      totalCount {PAGE_INFO} nodes {{ {THREAD_COMMENT_FIELDS} }}
    }}
  }} }}
}}
"""

FINGERPRINT_QUERY = """
query($ids: [ID!]!) {
  rateLimit { cost remaining resetAt }
  nodes(ids: $ids) { ... on PullRequest {
    id updatedAt headRefOid baseRefOid changedFiles
    comments { totalCount } reviews { totalCount } reviewThreads { totalCount } commits { totalCount }
  } }
}
"""


def _repo_parts(repository: str) -> tuple[str, str]:
    parts = repository.split("/", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"repository must be owner/name, got {repository!r}")
    return parts[0], parts[1]


def is_bot(author: dict | None, bot_logins: AbstractSet[str]) -> bool:
    """Return whether a REST or GraphQL author represents automation."""
    author = author or {}
    login = str(author.get("login") or "").lower()
    actor_type = author.get("__typename") or author.get("type")
    return not login or actor_type == "Bot" or login in bot_logins or login.endswith("[bot]")


def _human_event(node: dict, scope: ReviewScope) -> bool:
    body = str(node.get("body") or "")
    return not is_bot(node.get("author"), scope.bot_logins) and not body.lstrip().startswith("🤖")


def _event_time(kind: str, node: dict) -> str | None:
    if kind == "review":
        return node.get("submittedAt") or node.get("createdAt")
    return node.get("createdAt")


def _event_timestamps(kind: str, node: dict) -> tuple[str, ...]:
    if kind == "review":
        values = (node.get("createdAt"), node.get("updatedAt"), node.get("submittedAt"))
    else:
        values = (node.get("createdAt"), node.get("updatedAt"))
    return tuple(value for value in values if value is not None)


def _in_window(value: str | None, scope: ReviewScope) -> bool:
    return value is not None and scope.start <= parse_utc(value) < scope.end


def _page_has_human(
    kind: str,
    nodes: list[dict],
    scope: ReviewScope,
) -> bool:
    return any(
        _human_event(node, scope) and any(_in_window(value, scope) for value in _event_timestamps(kind, node))
        for node in nodes
    )


def _needs_older_page(kind: str, connection: dict, scope: ReviewScope) -> bool:
    if not connection["pageInfo"]["hasPreviousPage"]:
        return False
    nodes = connection["nodes"]
    if not nodes:
        raise RuntimeError(f"GitHub {kind} cursor claims an older page after an empty page")
    oldest = _event_time(kind, nodes[0])
    return oldest is not None and parse_utc(oldest) >= scope.start


def _scan_older(
    client: GitHubClient,
    repository: str,
    number: int,
    kind: Literal["issue_comment", "review"],
    connection: dict,
    scope: ReviewScope,
    *,
    exhaustive: bool,
) -> bool:
    """Return True on an older in-window human event, or False after eligible pages are exhausted."""
    owner, name = _repo_parts(repository)
    seen_cursors: set[str] = set()
    while connection["pageInfo"]["hasPreviousPage"] and (exhaustive or _needs_older_page(kind, connection, scope)):
        cursor = connection["pageInfo"].get("startCursor")
        if not cursor or cursor in seen_cursors:
            raise RuntimeError(f"PR #{number} {kind} pagination cursor did not advance")
        seen_cursors.add(cursor)
        query = SCAN_COMMENTS_QUERY if kind == "issue_comment" else SCAN_REVIEWS_QUERY
        field = "comments" if kind == "issue_comment" else "reviews"
        data = client.graphql(query, {"owner": owner, "name": name, "number": number, "before": cursor})
        pull_request = data["repository"]["pullRequest"]
        if pull_request is None:
            raise RuntimeError(f"GitHub GraphQL could not find PR #{number}")
        connection = pull_request[field]
        if _page_has_human(kind, connection["nodes"], scope):
            return True
    return False


def _rest_seed_prs(
    client: GitHubClient,
    repository: str,
    scope: ReviewScope,
) -> dict[int, dict[tuple[EventKind, int], dict]]:
    since = scope.start.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
    endpoints: dict[EventKind, str] = {
        "inline_comment": (
            f"repos/{repository}/pulls/comments?sort=updated&direction=asc&since={since}&per_page={REST_PAGE_SIZE}"
        ),
        "issue_comment": (
            f"repos/{repository}/issues/comments?sort=updated&direction=asc&since={since}&per_page={REST_PAGE_SIZE}"
        ),
    }
    seeds: dict[int, dict[tuple[EventKind, int], dict]] = {}
    for kind, endpoint in endpoints.items():
        for node in client.rest_pages(endpoint):
            user = node.get("user") or {}
            body = str(node.get("body") or "")
            if is_bot(user, scope.bot_logins) or body.lstrip().startswith("🤖"):
                continue
            timestamps = tuple(value for value in (node.get("created_at"), node.get("updated_at")) if value is not None)
            if not any(_in_window(value, scope) for value in timestamps):
                continue
            source = node.get("pull_request_url") if kind == "inline_comment" else node.get("issue_url")
            if not source:
                continue
            number = int(str(source).rstrip("/").rsplit("/", maxsplit=1)[-1])
            seeds.setdefault(number, {})[(kind, int(node["id"]))] = node
    return seeds


def _scan_pull_requests(
    client: GitHubClient,
    repository: str,
    scope: ReviewScope,
    limit: int | None,
    seed_prs: set[int],
) -> PullRequestScan:
    owner, name = _repo_parts(repository)
    after: str | None = None
    relevant: list[dict] = []
    candidates = 0
    seen_numbers: set[int] = set()
    seen_cursors: set[str] = set()
    reached_old_tail = False
    while not reached_old_tail:
        data = client.graphql(SCAN_QUERY, {"owner": owner, "name": name, "after": after})
        connection = data["repository"]["pullRequests"]
        nodes = connection["nodes"]
        for pull in nodes:
            if parse_utc(pull["updatedAt"]) < scope.start:
                reached_old_tail = True
                break
            number = int(pull["number"])
            if number in seen_numbers:
                raise RuntimeError(f"GitHub pull-request scan returned duplicate PR #{number}")
            seen_numbers.add(number)
            candidates += 1
            issue = pull["comments"]
            reviews = pull["reviews"]
            include = number in seed_prs
            include = include or _page_has_human("issue_comment", issue["nodes"], scope)
            include = include or _page_has_human("review", reviews["nodes"], scope)
            if not include:
                include = _scan_older(
                    client,
                    repository,
                    number,
                    "issue_comment",
                    issue,
                    scope,
                    exhaustive=False,
                )
            if not include:
                # Reviews are ordered by submission, so an old review edited in
                # the window can only be found by exhausting this connection.
                include = _scan_older(
                    client,
                    repository,
                    number,
                    "review",
                    reviews,
                    scope,
                    exhaustive=True,
                )
            if include:
                relevant.append(pull)
            if limit is not None and candidates >= limit:
                reached_old_tail = True
                break
        if reached_old_tail or not connection["pageInfo"]["hasNextPage"]:
            break
        cursor = connection["pageInfo"].get("endCursor")
        if not cursor or cursor in seen_cursors:
            raise RuntimeError("GitHub pull-request scan cursor did not advance")
        seen_cursors.add(cursor)
        after = cursor
    # A limited export is explicitly incomplete. Keep its bound deterministic
    # instead of expanding it with every repository-wide seed after the scan.
    unmatched_seeds = () if limit is not None else sorted(seed_prs - seen_numbers)
    for number in unmatched_seeds:
        data = client.graphql(
            SEED_PULL_REQUEST_QUERY,
            {"owner": owner, "name": name, "number": number},
        )
        pull = data["repository"]["candidate"]
        if pull is None or pull["__typename"] != "PullRequest":
            continue
        relevant.append(pull)
        candidates += 1
    return PullRequestScan(relevant=tuple(relevant), candidate_count=candidates)


@dataclass
class _ConnectionState:
    number: int
    name: str
    expected: int
    nodes: list[dict]
    page_info: dict
    seen_cursors: set[str]


def _validate_connection_nodes(number: int, name: str, expected: int, nodes: list[dict]) -> None:
    ids = [
        node.get("databaseId") or node.get("id") or (node.get("commit") or {}).get("oid") or node.get("path")
        for node in nodes
    ]
    if len(nodes) != expected:
        raise RuntimeError(f"PR #{number} {name} expected {expected} records, fetched {len(nodes)}")
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"PR #{number} {name} contains duplicate records")


def _paginate_pull_connections(client: GitHubClient, repository: str, pulls: list[dict]) -> None:
    owner, name = _repo_parts(repository)
    states = [
        _ConnectionState(
            number=int(pull["number"]),
            name=field,
            expected=int(pull[field]["totalCount"]),
            nodes=list(pull[field]["nodes"]),
            page_info=pull[field]["pageInfo"],
            seen_cursors=set(),
        )
        for pull in pulls
        for field in CONNECTION_FIELDS
    ]
    while active := [state for state in states if state.page_info["hasNextPage"]]:
        for state_batch in batched(active, HYDRATION_BATCH_SIZE):
            batch = list(state_batch)
            variables: dict[str, object] = {"owner": owner, "name": name}
            for index, state in enumerate(batch):
                cursor = state.page_info.get("endCursor")
                if not cursor or cursor in state.seen_cursors:
                    raise RuntimeError(f"PR #{state.number} {state.name} pagination cursor did not advance")
                state.seen_cursors.add(cursor)
                variables[f"number{index}"] = state.number
                variables[f"after{index}"] = cursor
            data = client.graphql(_connections_query([state.name for state in batch]), variables)
            for index, state in enumerate(batch):
                pull_request = data["repository"][f"pr{index}"]
                if pull_request is None:
                    raise RuntimeError(f"GitHub GraphQL could not find PR #{state.number}")
                connection = pull_request[state.name]
                if int(connection["totalCount"]) != state.expected:
                    raise RuntimeError(f"PR #{state.number} {state.name} count changed during pagination")
                state.nodes.extend(connection["nodes"])
                state.page_info = connection["pageInfo"]
    pulls_by_number = {int(pull["number"]): pull for pull in pulls}
    for state in states:
        _validate_connection_nodes(state.number, state.name, state.expected, state.nodes)
        pulls_by_number[state.number][state.name] = {
            "totalCount": state.expected,
            "pageInfo": state.page_info,
            "nodes": state.nodes,
        }


def _thread_comment_nodes(client: GitHubClient, thread: dict) -> list[dict]:
    connection = thread["comments"]
    expected = int(connection["totalCount"])
    nodes = list(connection["nodes"])
    seen_cursors: set[str] = set()
    while connection["pageInfo"]["hasNextPage"]:
        cursor = connection["pageInfo"].get("endCursor")
        if not cursor or cursor in seen_cursors:
            raise RuntimeError(f"review thread {thread['id']} pagination cursor did not advance")
        seen_cursors.add(cursor)
        data = client.graphql(THREAD_COMMENTS_PAGE_QUERY, {"thread": thread["id"], "after": cursor})
        node = data["node"]
        if node is None:
            raise RuntimeError(f"GitHub GraphQL could not find review thread {thread['id']}")
        connection = node["comments"]
        if int(connection["totalCount"]) != expected:
            raise RuntimeError(f"review thread {thread['id']} count changed during pagination")
        nodes.extend(connection["nodes"])
    ids = [node["databaseId"] for node in nodes]
    if len(nodes) != expected:
        raise RuntimeError(f"review thread {thread['id']} expected {expected} comments, fetched {len(nodes)}")
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"review thread {thread['id']} contains duplicate comments")
    return nodes


def _fingerprint(pull: dict) -> PullRequestFingerprint:
    return PullRequestFingerprint(
        updated_at=pull["updatedAt"],
        head_sha=pull["headRefOid"],
        base_sha=pull["baseRefOid"],
        changed_files=int(pull["changedFiles"]),
        commits=int(pull["commits"]["totalCount"]),
        reviews=int(pull["reviews"]["totalCount"]),
        review_threads=int(pull["reviewThreads"]["totalCount"]),
        issue_comments=int(pull["comments"]["totalCount"]),
    )


def _fingerprint_nodes(client: GitHubClient, node_ids: list[str]) -> dict[str, PullRequestFingerprint]:
    fingerprints: dict[str, PullRequestFingerprint] = {}
    for node_id_batch in batched(node_ids, FINGERPRINT_BATCH_SIZE):
        ids = list(node_id_batch)
        data = client.graphql(FINGERPRINT_QUERY, {"ids": ids})
        nodes = data["nodes"]
        if len(nodes) != len(ids) or any(node is None for node in nodes):
            raise RuntimeError("GitHub GraphQL fingerprint recheck returned missing PR nodes")
        returned_ids = [node["id"] for node in nodes]
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(ids):
            raise RuntimeError("GitHub GraphQL fingerprint recheck returned inconsistent PR nodes")
        fingerprints.update((node["id"], _fingerprint(node)) for node in nodes)
    return fingerprints


def _actor_identity(actor: dict | None) -> ActorIdentity:
    actor = actor or {}
    return ActorIdentity(
        login=str(actor.get("login") or "unknown"),
        actor_type=str(actor.get("__typename") or "Unknown"),
    )


def _event_record(
    repository: str,
    pull: dict,
    kind: EventKind,
    node: dict,
    scope: ReviewScope,
    thread: dict | None = None,
) -> ReviewEventRecord:
    author = _actor_identity(node.get("author"))
    pr_author = _actor_identity(pull.get("author"))
    body = str(node.get("body") or "")
    author_is_bot = is_bot(node.get("author"), scope.bot_logins)
    thread = thread or {}
    resolved_by = _actor_identity(thread.get("resolvedBy"))
    review = node.get("pullRequestReview") or {}
    commit = node.get("commit") or {}
    original_commit = node.get("originalCommit") or {}
    database_id = int(node["databaseId"])
    return ReviewEventRecord(
        event_id=f"{repository}:{kind}:{database_id}",
        kind=kind,
        database_id=database_id,
        node_id=node.get("id"),
        repository=repository,
        pr_number=int(pull["number"]),
        pr_author=pr_author.login,
        author=author.login,
        author_type=author.actor_type,
        author_association=str(node.get("authorAssociation") or "UNKNOWN"),
        body=body,
        state=node.get("state"),
        created_at=node.get("createdAt"),
        updated_at=node.get("updatedAt"),
        submitted_at=node.get("submittedAt"),
        source_url=node.get("url"),
        review_id=database_id if kind == "review" else review.get("databaseId"),
        thread_id=thread.get("id"),
        parent_comment_id=(node.get("replyTo") or {}).get("databaseId"),
        thread_is_resolved=thread.get("isResolved"),
        thread_is_outdated=thread.get("isOutdated"),
        thread_resolved_by=None if resolved_by.login == "unknown" else resolved_by.login,
        path=node.get("path") or thread.get("path"),
        side=thread.get("diffSide"),
        line=node.get("line"),
        original_line=node.get("originalLine"),
        start_side=thread.get("startDiffSide"),
        start_line=node.get("startLine"),
        original_start_line=node.get("originalStartLine"),
        commit_id=commit.get("oid"),
        original_commit_id=original_commit.get("oid"),
        diff_hunk=node.get("diffHunk"),
        is_bot=author_is_bot,
        is_agent_marked=body.lstrip().startswith("🤖"),
        is_human=not author_is_bot and not body.lstrip().startswith("🤖"),
        in_window=any(_in_window(value, scope) for value in _event_timestamps(kind, node)),
    )


def _commit_record(pr_number: int, node: dict) -> CommitRecord:
    commit = node["commit"]
    parents = commit["parents"]
    if int(parents["totalCount"]) != len(parents["nodes"]):
        raise RuntimeError(f"commit {commit['oid']} has more than {COMMIT_PARENT_PAGE_SIZE} parents")
    user = ((commit.get("author") or {}).get("user") or {}).get("login")
    return CommitRecord(
        pr_number=pr_number,
        sha=commit["oid"],
        author=user,
        authored_at=commit.get("authoredDate"),
        committed_at=commit.get("committedDate"),
        message=str(commit.get("message") or ""),
        parents=tuple(parent["oid"] for parent in parents["nodes"]),
    )


def _pull_request_record(
    repository: str, pull: dict, commits: list[CommitRecord], review_comments: int
) -> PullRequestRecord:
    author = _actor_identity(pull.get("author"))
    number = int(pull["number"])
    return PullRequestRecord(
        repository=repository,
        number=number,
        node_id=pull["id"],
        url=pull["url"],
        title=pull["title"],
        body=str(pull.get("body") or ""),
        state=pull["state"].lower(),
        draft=bool(pull["isDraft"]),
        author=author.login,
        author_type=author.actor_type,
        author_association=str(pull.get("authorAssociation") or "UNKNOWN"),
        created_at=pull["createdAt"],
        updated_at=pull["updatedAt"],
        closed_at=pull.get("closedAt"),
        merged_at=pull.get("mergedAt"),
        base_ref=pull["baseRefName"],
        base_sha=pull["baseRefOid"],
        head_ref=pull["headRefName"],
        head_sha=pull["headRefOid"],
        additions=int(pull["additions"]),
        deletions=int(pull["deletions"]),
        changed_files=int(pull["changedFiles"]),
        commits=len(commits),
        review_comments=review_comments,
        issue_comments=int(pull["comments"]["totalCount"]),
        commit_shas=tuple(commit.sha for commit in commits),
        diff_path=f"diffs/{number}.diff",
    )


def _changed_file(pr_number: int, item: dict) -> ChangedFileRecord:
    additions = int(item["additions"])
    deletions = int(item["deletions"])
    return ChangedFileRecord(
        pr_number=pr_number,
        filename=item["path"],
        status=str(item["changeType"]).lower(),
        additions=additions,
        deletions=deletions,
        changes=additions + deletions,
    )


def _hydration_roots(client: GitHubClient, repository: str, numbers: list[int]) -> list[dict]:
    owner, name = _repo_parts(repository)
    roots: list[dict] = []
    for number_batch in batched(numbers, HYDRATION_BATCH_SIZE):
        batch = list(number_batch)
        variables: dict[str, object] = {"owner": owner, "name": name}
        variables.update((f"number{index}", number) for index, number in enumerate(batch))
        data = client.graphql(_hydration_query(len(batch)), variables)
        repository_data = data["repository"]
        for index, number in enumerate(batch):
            pull = repository_data[f"pr{index}"]
            if pull is None:
                raise RuntimeError(f"GitHub GraphQL could not find PR #{number}")
            if int(pull["number"]) != number:
                raise RuntimeError(f"GitHub GraphQL returned PR #{pull['number']} for requested PR #{number}")
            roots.append(pull)
    return roots


def _hydrate_graphql(
    client: GitHubClient,
    repository: str,
    pull: dict,
    scope: ReviewScope,
) -> tuple[PullRequestBundle, PullRequestFingerprint]:
    number = int(pull["number"])
    issue_nodes = pull["comments"]["nodes"]
    review_nodes = pull["reviews"]["nodes"]
    thread_nodes = pull["reviewThreads"]["nodes"]
    commit_nodes = pull["commits"]["nodes"]
    file_nodes = pull["files"]["nodes"]

    thread_comments: list[tuple[dict, dict]] = []
    thread_records: list[ReviewThreadRecord] = []
    for thread in thread_nodes:
        comments = _thread_comment_nodes(client, thread)
        thread_comments.extend((thread, comment) for comment in comments)
        resolved_by = _actor_identity(thread.get("resolvedBy"))
        thread_records.append(
            ReviewThreadRecord(
                pr_number=number,
                thread_id=thread["id"],
                comment_ids=tuple(int(comment["databaseId"]) for comment in comments),
                is_resolved=bool(thread["isResolved"]),
                is_outdated=bool(thread["isOutdated"]),
                resolved_by=None if resolved_by.login == "unknown" else resolved_by.login,
            )
        )

    comments_by_review: dict[int, int] = {}
    for _, comment in thread_comments:
        review = comment.get("pullRequestReview")
        if not review or review.get("databaseId") is None:
            raise RuntimeError(f"PR #{number} thread comment {comment['databaseId']} has no review")
        review_id = int(review["databaseId"])
        comments_by_review[review_id] = comments_by_review.get(review_id, 0) + 1
    for review in review_nodes:
        expected = int(review["comments"]["totalCount"])
        actual = comments_by_review.get(int(review["databaseId"]), 0)
        if expected != actual:
            raise RuntimeError(
                f"PR #{number} review {review['databaseId']} expected {expected} comments, fetched {actual}"
            )

    events = [
        *(_event_record(repository, pull, "issue_comment", node, scope) for node in issue_nodes),
        *(_event_record(repository, pull, "review", node, scope) for node in review_nodes),
        *(
            _event_record(repository, pull, "inline_comment", comment, scope, thread)
            for thread, comment in thread_comments
        ),
    ]
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError(f"PR #{number} contains duplicate review event ids")
    commits = [_commit_record(number, node) for node in commit_nodes]
    bundle = PullRequestBundle(
        pull_request=_pull_request_record(repository, pull, commits, len(thread_comments)),
        events=tuple(sorted(events, key=lambda event: (event.kind, event.database_id))),
        threads=tuple(sorted(thread_records, key=lambda thread: thread.thread_id)),
        files=tuple(sorted((_changed_file(number, item) for item in file_nodes), key=lambda item: item.filename)),
        commits=tuple(commits),
        diff="",
    )
    return bundle, _fingerprint(pull)


def _with_diff(client: GitHubClient, repository: str, bundle: PullRequestBundle) -> PullRequestBundle:
    endpoint = f"repos/{repository}/pulls/{bundle.pull_request.number}"
    diff = client.rest_text(endpoint, "application/vnd.github.diff")
    pull_request = bundle.pull_request
    if diff is None:
        pull_request = pull_request.model_copy(update={"diff_path": None})
    return bundle.model_copy(update={"pull_request": pull_request, "diff": diff})


def _hydrate_pull_requests(
    client: GitHubClient,
    repository: str,
    numbers: list[int],
    scope: ReviewScope,
) -> dict[int, tuple[PullRequestBundle, PullRequestFingerprint]]:
    snapshots: dict[int, tuple[PullRequestBundle, PullRequestFingerprint]] = {}
    pulls = _hydration_roots(client, repository, numbers)
    _paginate_pull_connections(client, repository, pulls)
    for pull in pulls:
        bundle, fingerprint = _hydrate_graphql(client, repository, pull, scope)
        number = bundle.pull_request.number
        if number in snapshots:
            raise RuntimeError(f"GitHub hydration returned duplicate PR #{number}")
        snapshots[number] = (_with_diff(client, repository, bundle), fingerprint)
    if set(snapshots) != set(numbers):
        raise RuntimeError("GitHub hydration returned an incomplete pull-request set")
    return snapshots


def _changed_pull_request_numbers(
    client: GitHubClient,
    snapshots: dict[int, tuple[PullRequestBundle, PullRequestFingerprint]],
) -> list[int]:
    node_ids = [snapshots[number][0].pull_request.node_id for number in sorted(snapshots)]
    current = _fingerprint_nodes(client, node_ids)
    return [
        number
        for number in sorted(snapshots)
        if snapshots[number][1] != current[snapshots[number][0].pull_request.node_id]
    ]


def collect_corpus(
    repository: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    bot_logins: set[str],
    limit: int | None = None,
    client: GitHubClient | None = None,
) -> CollectionResult:
    """Collect matching PRs, or a bounded and intentionally incomplete probe when limit is set."""
    client = client or GitHubClient()
    scope = ReviewScope(start=start, end=end, bot_logins=frozenset(bot_logins))
    seeds = _rest_seed_prs(client, repository, scope)
    scan = _scan_pull_requests(
        client,
        repository,
        scope,
        limit,
        set(seeds),
    )
    oversized = sorted(
        int(pull["number"]) for pull in scan.relevant if int(pull["changedFiles"]) > MAX_GITHUB_CHANGED_FILES
    )
    if oversized:
        raise RuntimeError(f"PR #{oversized[0]} exceeds GitHub's {MAX_GITHUB_CHANGED_FILES:,}-file API cap")
    numbers = [int(pull["number"]) for pull in scan.relevant]
    projected_rest = client.rest_requests + len(numbers) + REST_RETRY_RESERVE
    client.projected_rest_requests = projected_rest
    if projected_rest > MAX_REST_REQUESTS:
        raise RuntimeError(
            f"GitHub context collection projects {projected_rest} REST requests, above the "
            f"{MAX_REST_REQUESTS}-request safety budget"
        )
    snapshots = _hydrate_pull_requests(client, repository, numbers, scope)
    changed_numbers = _changed_pull_request_numbers(client, snapshots)
    if len(changed_numbers) > REST_RETRY_RESERVE:
        raise RuntimeError(
            f"GitHub changed {len(changed_numbers)} PRs during collection, above the "
            f"{REST_RETRY_RESERVE}-request retry reserve"
        )
    if changed_numbers:
        retries = _hydrate_pull_requests(client, repository, changed_numbers, scope)
        changed_twice = _changed_pull_request_numbers(client, retries)
        if changed_twice:
            raise RuntimeError(f"PR #{changed_twice[0]} changed during both collection attempts")
        snapshots.update(retries)
    bundles = [snapshots[number][0] for number in numbers]
    for bundle in bundles:
        expected = seeds.get(bundle.pull_request.number, {})
        actual = {(event.kind, event.database_id): event for event in bundle.events}
        missing = sorted(set(expected) - set(actual))
        if missing:
            raise RuntimeError(
                f"PR #{bundle.pull_request.number} is missing seeded review events: "
                f"{missing[:DIAGNOSTIC_EVENT_LIMIT]}"
            )
        changed = [
            key
            for key, seed in expected.items()
            if actual[key].body != str(seed.get("body") or "") or actual[key].updated_at != seed.get("updated_at")
        ]
        if changed:
            raise RuntimeError(
                f"PR #{bundle.pull_request.number} changed between REST seeding and GraphQL hydration: "
                f"{changed[:DIAGNOSTIC_EVENT_LIMIT]}"
            )
    return CollectionResult(
        bundles=tuple(sorted(bundles, key=lambda bundle: bundle.pull_request.number)),
        candidate_pull_requests=scan.candidate_count,
        usage=client.usage(),
    )
