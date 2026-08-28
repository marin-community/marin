# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog row types for the code-health review tables.

Four append-only namespaces under ``codehealth.autolint``:

- ``invocations`` — one row per ``pre-commit.py --review`` / ``/code-review`` /
  ``/review-pr`` run. Rows with ``finding_count = 0`` are kept; they are the
  "tool ran and had no objection" signal.
- ``findings`` — one row per finding, joined to its run on ``invocation_id``.
- ``human_comments`` — one row per classified human review comment.
- ``pr_review_outcomes`` — one row per PR, rolling up the two above.

Re-running the aggregator appends a fresh row per PR to both comment tables, so
a reader must take the highest ``seq`` per natural key. ``review_quality.py`` exposes
that as ``LATEST_HUMAN_COMMENTS_SQL`` and ``LATEST_PR_OUTCOMES_SQL``.

Every row type declares a ``key_column``. With none declared the server looks
for a column named ``timestamp_ms``; none of these have one, so registration
fails.
"""

import datetime as dt
import logging
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from finelog.client import FlushResult, LogClient, StoragePolicy
from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client

RowT = TypeVar("RowT")
logger = logging.getLogger("codehealth.review_tables")

DEFAULT_DEPLOYMENT = "marin"
DEFAULT_REPOSITORY = "marin-community/marin"
DEFAULT_BOT_LOGINS = frozenset(
    {
        "github-actions",
        "dependabot",
        "claude",
        "claude-review",
        "loom-oa-dev",
        "renovate",
        "weaverbot",
    }
)

NAMESPACE_PREFIX = "codehealth.autolint"

INVOCATIONS_NAMESPACE = f"{NAMESPACE_PREFIX}.invocations"
FINDINGS_NAMESPACE = f"{NAMESPACE_PREFIX}.findings"
HUMAN_COMMENTS_NAMESPACE = f"{NAMESPACE_PREFIX}.human_comments"
PR_REVIEW_OUTCOMES_NAMESPACE = f"{NAMESPACE_PREFIX}.pr_review_outcomes"

# A few thousand rows a month, all of it worth keeping, so the policy caps
# bytes and never ages a row out.
TABLE_MAX_BYTES = 512 * 1024 * 1024
STORAGE_POLICY = StoragePolicy(max_bytes=TABLE_MAX_BYTES)

DEFAULT_FLUSH_TIMEOUT = 60.0
# Bound each send so a large import arrives as several segments rather than one
# oversized request.
WRITE_CHUNK = 500


@dataclass(frozen=True)
class Invocation:
    """One review-automation run."""

    key_column: ClassVar[str] = "invocation_id"

    ts: dt.datetime
    invocation_id: str
    tool: str
    variant: str | None
    trigger: str | None
    agent_cli: str | None
    git_branch: str | None
    merge_base_sha: str | None
    head_sha: str | None
    pr_number: int | None
    marin_user: str | None
    lint_catalog_sha: str | None
    diff_files: int | None
    diff_added_lines: int | None
    diff_removed_lines: int | None
    finding_count: int | None
    elapsed: float | None
    agent_exit_code: int | None
    timed_out: bool | None


@dataclass(frozen=True)
class Finding:
    """One finding emitted by a run, denormalized so it can be queried alone."""

    key_column: ClassVar[str] = "invocation_id"

    ts: dt.datetime
    invocation_id: str
    tool: str
    pr_number: int | None
    git_branch: str | None
    head_sha: str | None
    marin_user: str | None
    file: str | None
    line: int | None
    code: str | None
    confidence: float | None
    message: str | None


@dataclass(frozen=True)
class HumanComment:
    """One human review comment and its classification.

    ``comment_class`` avoids the name ``class``, which is a Python keyword and
    needs quoting in every SQL reference.
    """

    key_column: ClassVar[str] = "pr_number"

    ts: dt.datetime
    pr_number: int
    pr_title: str | None
    merged_at: dt.datetime | None
    author: str | None
    comment_id: int
    comment_type: str
    file: str | None
    line: int | None
    body: str | None
    source_url: str | None
    context_hash: str | None
    context: str | None
    comment_class: str | None
    catchable_strict: bool | None
    catchable_generous: bool | None
    confidence: float | None
    reason: str | None


@dataclass(frozen=True)
class PrReviewOutcome:
    """Per-PR rollup of human comments against the bot's own findings."""

    key_column: ClassVar[str] = "pr_number"

    ts: dt.datetime
    pr_number: int
    pr_title: str | None
    merged_at: dt.datetime | None
    author: str | None
    head_sha: str | None
    base_sha: str | None
    total_human_comments: int | None
    by_class_json: str | None
    catchable_strict_count: int | None
    catchable_generous_count: int | None
    bot_findings_count: int | None
    overlap_count: int | None


def parse_utc(text: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp, stamping UTC when it carries no offset."""
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


@contextmanager
def open_tables_client(deployment: str, finelog_url: str | None = None) -> Iterator[LogClient]:
    """Yield a client for these tables.

    ``finelog_url`` connects to that address and skips deployment resolution
    and its IAP handshake.
    """
    if finelog_url:
        with closing(LogClient.connect(finelog_url)) as client:
            yield client
        return
    with open_client(load_finelog_config(deployment), deployment) as client:
        yield client


def row_count(client: LogClient, namespace: str) -> int | None:
    """Current row count for ``namespace``, or None if it is not registered."""
    for info in client.list_namespaces():
        if info.namespace == namespace:
            return info.row_count
    return None


def query_rows(client: LogClient, sql: str, namespace: str) -> list[dict]:
    """Run a query, treating an unregistered namespace as an empty table."""
    if row_count(client, namespace) is None:
        logger.info("namespace %s does not exist yet; treating as empty", namespace)
        return []
    return [
        {
            key: value.replace(tzinfo=dt.UTC) if isinstance(value, dt.datetime) and value.tzinfo is None else value
            for key, value in row.items()
        }
        for row in client.query(sql).to_pylist()
    ]


def append_rows(
    client: LogClient,
    namespace: str,
    row_type: type[RowT],
    rows: Sequence[RowT],
    *,
    flush_timeout: float = DEFAULT_FLUSH_TIMEOUT,
) -> int:
    """Append ``rows``, raising unless the server recorded them.

    Returns:
        The number of rows written.

    Raises:
        RuntimeError: the rows were dropped or the flush did not drain in time.
    """
    if not rows:
        return 0
    table = client.get_table(namespace, row_type, storage_policy=STORAGE_POLICY)
    for start in range(0, len(rows), WRITE_CHUNK):
        table.write(rows[start : start + WRITE_CHUNK])
        result = table.flush(timeout=flush_timeout)
        if result is not FlushResult.SUCCEEDED:
            raise RuntimeError(f"{namespace}: {len(rows)} rows were not recorded ({result})")
    return len(rows)
