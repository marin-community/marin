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

Both comment tables are rewritten per PR by re-running the aggregator, so a
reader takes the highest ``seq`` per natural key rather than assuming one row.
``review.py`` exposes that as ``LATEST_HUMAN_COMMENTS_SQL`` and
``LATEST_PR_OUTCOMES_SQL``.

Every row type declares a ``key_column``. The server's fallback for an
undeclared key is a column literally named ``timestamp_ms``, which none of
these have, and it rejects the registration rather than defaulting to the
declared timestamp.
"""

import datetime as dt
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from finelog.client import FlushResult, LogClient, StoragePolicy
from finelog.deploy.connect import open_named_client

RowT = TypeVar("RowT")

DEFAULT_DEPLOYMENT = "marin"

NAMESPACE_PREFIX = "codehealth.autolint"

INVOCATIONS_NAMESPACE = f"{NAMESPACE_PREFIX}.invocations"
FINDINGS_NAMESPACE = f"{NAMESPACE_PREFIX}.findings"
HUMAN_COMMENTS_NAMESPACE = f"{NAMESPACE_PREFIX}.human_comments"
PR_REVIEW_OUTCOMES_NAMESPACE = f"{NAMESPACE_PREFIX}.pr_review_outcomes"

# These tables are small (low thousands of rows a month) and their value is
# historical, so cap bytes rather than age and let compaction keep everything.
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


@contextmanager
def open_tables_client(deployment: str, finelog_url: str | None = None) -> Iterator[LogClient]:
    """Yield a client for these tables.

    ``finelog_url`` bypasses deployment resolution and connects to an address
    directly, which is how tests reach an embedded server.
    """
    if finelog_url:
        with closing(LogClient.connect(finelog_url)) as client:
            yield client
        return
    with open_named_client(deployment) as client:
        yield client


def row_count(client: LogClient, namespace: str) -> int | None:
    """Current row count for ``namespace``, or None if it is not registered."""
    for info in client.list_namespaces():
        if info.namespace == namespace:
            return info.row_count
    return None


def append_rows(
    client: LogClient,
    namespace: str,
    row_type: type[RowT],
    rows: Sequence[RowT],
    *,
    flush_timeout: float = DEFAULT_FLUSH_TIMEOUT,
) -> int:
    """Append ``rows`` and confirm the server stored them.

    ``Table.flush`` reports only that the client queue drained, so this
    compares the namespace row count across the write. Concurrent writers can
    only inflate the delta, making the check a floor.

    Returns:
        The namespace row count after the write.

    Raises:
        RuntimeError: the flush timed out, the namespace was never registered,
            or fewer rows landed than were written.
    """
    if not rows:
        return row_count(client, namespace) or 0

    before = row_count(client, namespace) or 0
    table = client.get_table(namespace, row_type, storage_policy=STORAGE_POLICY)
    for start in range(0, len(rows), WRITE_CHUNK):
        table.write(rows[start : start + WRITE_CHUNK])
        result = table.flush(timeout=flush_timeout)
        if result is not FlushResult.SUCCEEDED:
            raise RuntimeError(f"{namespace}: flush did not complete within {flush_timeout:.0f}s: {result}")

    after = row_count(client, namespace)
    if after is None:
        raise RuntimeError(f"{namespace}: not registered after writing {len(rows)} rows; the server rejected the schema")
    if after - before < len(rows):
        raise RuntimeError(f"{namespace}: wrote {len(rows)} rows but only {after - before} landed ({before} -> {after})")
    return after
