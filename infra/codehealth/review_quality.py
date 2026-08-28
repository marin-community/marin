# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregator and reporter for the code-health stats dashboard.

Two subcommands:
  - `aggregate` (designed to run as a daily GHA cron) classifies reviewer
    comments and appends them to the `codehealth.autolint` Finelog namespaces.
  - `report` reads those accumulated tables back and renders a markdown digest
    (summary table, by-class breakdown, weekly trend, top PRs, examples). It
    also folds in the linter-fed `invocations`/`findings` tables (written by
    `infra/codehealth/log_stats.py` on every review-bot run) as a "Review
    automation activity" section — runs, runtime, and the catalog rules fired.
    Published as a gist by default.

`aggregate`, for each PR merged in the last N days:
  1. Pull review/inline/issue comments via the `gh` CLI.
  2. Drop bot comments and agent-authored `🤖` replies.
  3. Classify each human comment with a pluggable classifier — a sandboxed
     Codex session (`codex exec`) by default. Comments from all PRs are pooled, batched, and the batches
     classified in parallel so a many-PR run is not a long serial trickle of
     one request per comment. Two independent "could automation have caught
     this?" signals:
       - catchable_strict   — a deterministic linter / type checker / ml-*
                              catalog rule could mechanically flag it.
       - catchable_generous — a modern LLM running on the diff alone would
                              plausibly flag it with high confidence.
     Strict ⊆ generous by construction (the prompt enforces it).
  4. Pull the bot's own findings for the PR's head_sha from the `findings`
     namespace written by `infra/codehealth/log_stats.py`.
  5. Append two tables alongside the invocations and findings that
     `infra/codehealth/log_stats.py` writes:
       - human_comments      — one row per classified comment.
       - pr_review_outcomes  — one row per PR (rollup).

The tables are append-only, and a rolling window re-emits rows for PRs seen on
an earlier run, so both reads take the most recent row per natural key by the
server-assigned `seq`.

Requires a logged-in `codex` CLI and Finelog access
(`uv run iris --cluster marin login`), plus `gh auth login` for the GitHub
side. Designed to run as a daily GHA cron.
"""

import datetime as dt
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import click
from finelog.client import LogClient
from pydantic import BaseModel, Field, TypeAdapter

# Sibling module: the row types for every table this tool reads and writes.
from .review_tables import (
    DEFAULT_DEPLOYMENT,
    FINDINGS_NAMESPACE,
    HUMAN_COMMENTS_NAMESPACE,
    INVOCATIONS_NAMESPACE,
    NAMESPACE_PREFIX,
    PR_REVIEW_OUTCOMES_NAMESPACE,
    HumanComment,
    PrReviewOutcome,
    append_rows,
    open_tables_client,
    parse_utc,
    row_count,
)

logger = logging.getLogger("codehealth.review_quality")

DEFAULT_REPO = "marin-community/marin"
# Balance classification quality and cost for bounded, schema-constrained
# comment batches.
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_AGENT_COMMAND = "codex exec"
DEFAULT_BATCH_SIZE = 20
# One headless Codex subprocess per batch, so concurrency caps simultaneous
# processes and subscription rate pressure.
DEFAULT_CONCURRENCY = 4
MAX_COMMENT_CONTEXT = 6_000
MAX_FILE_PATCH = 1_500

# ---------------------------------------------------------------------------
# Comment classifier: schema + prompt
# ---------------------------------------------------------------------------


CommentClass = Literal["bug", "lint", "structure", "test", "doc", "design", "approval", "ack", "other"]


class CommentClassification(BaseModel):
    klass: CommentClass = Field(alias="class")
    catchable_strict: bool
    catchable_generous: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    model_config = {"populate_by_name": True}


@dataclass
class CommentToClassify:
    """One comment handed to a classifier. `id` is a caller-assigned marker the
    classifier echoes back so batched results can be matched to their input."""

    id: int
    file: str | None
    line: int | None
    body: str
    context: str | None


class BatchedClassification(CommentClassification):
    """A `CommentClassification` plus the `id` marker echoed back in a batch."""

    id: int


# A classifier turns a complete batch into classifications keyed by `id`.
# Pluggable so the backend can change without touching batching orchestration.
Classifier = Callable[[list[CommentToClassify]], dict[int, CommentClassification]]


CLASSIFIER_SYSTEM = """\
You triage human PR review comments to measure how much reviewer effort our
review automation is missing. You receive a batch of comments, each delimited
by a `=== COMMENT id=N ===` marker. Classify each comment independently and
return a JSON array holding exactly one object per comment — never merge,
split, drop, or reorder comments.

Fields:

  id — echo back, unchanged, the integer N from this comment's
       `=== COMMENT id=N ===` marker so the result can be matched to its input.

  class — what is the comment about? Pick exactly one:
    bug       — flags a logic error, missing await, wrong type, null deref
    lint      — code style / formatting / naming (ruff/black/pyrefly territory)
    structure — architecture: use dataclass, dead code, _utils suffix,
                StrEnum, separate I/O from compute, etc. (matches the
                marin `ml-*` catalog).
    test      — missing or broken tests
    doc       — missing/wrong docstring, comment, or markdown
    design    — open question, proposed alternative, architectural pushback
    approval  — LGTM-style, "ship it", explicit approval
    ack       — acknowledgment, status update, brief thanks, emoji-only
    other     — none of the above

  catchable_strict — bool. TRUE only if a deterministic tool could mechanically
                     flag this from the diff alone, with no judgment:
                     ruff, black, pyrefly, a regex/AST rule like the
                     marin `infra/lint/` ml-* catalog.
                     Examples TRUE: unused import; missing return type;
                     `_utils.py` filename; `TYPE_CHECKING:` guard; local
                     import that isn't an optional-dep guard.
                     Examples FALSE: "this conditional looks inverted"
                     (requires logic judgment); "does this work on TPU?"
                     (requires runtime knowledge).

  catchable_generous — bool. TRUE if a modern LLM running on the diff alone
                       (no broader repo context, no runtime data) would
                       plausibly flag this with high confidence. By
                       construction TRUE whenever catchable_strict is TRUE.
                       Examples TRUE: "missing await on this coroutine";
                       "this should be a dataclass not a dict"; "you shadow
                       the outer `state` variable here"; "off-by-one — should
                       this be `<=`?"; "this raises on empty input".
                       Examples FALSE: "I'd prefer to land this after the
                       migration"; "does this still meet the ferry latency
                       budget?"; "@alice can you take a look?"; "👍"; "ship it".

  confidence - your confidence in the two booleans, 0.0-1.0. Below 0.7 means
               you are unsure; the consumer will treat low-confidence rows
               as noise.

  reason — one sentence explaining the catchable verdicts. Be concrete.
           If catchable, name the rule or check; if not, name what context
           a human would need.

Hard rules:
  - Return one object per input comment, each echoing the comment's `id`.
  - If catchable_strict is TRUE, catchable_generous must be TRUE.
  - approval / ack comments are never catchable.
  - When a comment is multi-issue, classify by the most material issue.
  - Prefer FALSE when uncertain — false positives erode trust in the metric.
"""


_BATCH_ADAPTER = TypeAdapter(list[BatchedClassification])


def _format_batch(items: list[CommentToClassify]) -> str:
    """Render a batch as marker-delimited blocks the classifier can split."""
    blocks = []
    for it in items:
        where = f"File: {it.file}\nLine: {it.line}" if it.file else "Comment scope: top-level PR comment"
        context = f"\nDiff context:\n{it.context.strip()}" if it.context else ""
        blocks.append(f"=== COMMENT id={it.id} ===\n{where}\nBody:\n{it.body.strip()}{context}")
    return "\n\n".join(blocks)


# Env markers that would bind a spawned classifier to its parent agent session.
# Each batch runs as a fresh, isolated Codex session.
AGENT_STRIPPED_ENV = (
    "CODEX_THREAD_ID",
    "LOOM_SESSION_ID",
    "LOOM_TOKEN",
    "WEAVER_BRANCH",
)

# Per-batch wall-clock ceiling for one headless classification call.
CLASSIFIER_TIMEOUT = 300


def _headless_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in AGENT_STRIPPED_ENV}


def _classification_schema() -> dict:
    """Codex structured-output schema for one batch result."""
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "class": {"type": "string", "enum": list(get_args(CommentClass))},
            "catchable_strict": {"type": "boolean"},
            "catchable_generous": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "required": ["id", "class", "catchable_strict", "catchable_generous", "confidence", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": item}},
        "required": ["results"],
        "additionalProperties": False,
    }


def _parse_codex_batch(output: str) -> list[BatchedClassification] | None:
    """Parse one schema-constrained Codex final response."""
    try:
        envelope = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(envelope, dict) or "results" not in envelope:
        return None
    try:
        return _BATCH_ADAPTER.validate_python(envelope["results"])
    except ValueError:
        return None


def make_codex_classifier(
    model: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    agent_command: str = DEFAULT_AGENT_COMMAND,
) -> Classifier:
    """Build a sandboxed, schema-constrained Codex classifier."""
    env = _headless_env()
    base_command = shlex.split(agent_command)

    def classify(items: list[CommentToClassify]) -> dict[int, CommentClassification]:
        if not items:
            return {}
        prompt = f"{CLASSIFIER_SYSTEM}\n\n{_format_batch(items)}"
        with tempfile.TemporaryDirectory() as directory:
            schema_path = Path(directory) / "schema.json"
            output_path = Path(directory) / "result.json"
            schema_path.write_text(json.dumps(_classification_schema()))
            cmd = [
                *base_command,
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--ignore-rules",
                "--model",
                model,
                "--config",
                f'model_reasoning_effort="{reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-",
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=CLASSIFIER_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                logger.warning("Codex classification timed out for batch of %d", len(items))
                return {}
            output = output_path.read_text() if output_path.exists() else ""
        parsed = _parse_codex_batch(output) if proc.returncode == 0 else None
        if parsed is None:
            logger.warning(
                "Codex classification failed for batch of %d (exit=%s): %s",
                len(items),
                proc.returncode,
                (proc.stderr or proc.stdout or output).strip()[:200],
            )
            return {}
        wanted = {it.id for it in items}
        out: dict[int, CommentClassification] = {}
        for c in parsed:
            if c.id not in wanted:
                continue
            # Enforce the strict ⊆ generous invariant even if the model slipped.
            if c.catchable_strict and not c.catchable_generous:
                c.catchable_generous = True
            out[c.id] = c
        missing = len(wanted) - len(out)
        if missing:
            logger.warning("Codex omitted %d of %d comments in a batch", missing, len(items))
        return out

    return classify


def classify_comments(
    classifier: Classifier, items: list[CommentToClassify], batch_size: int, concurrency: int
) -> dict[int, CommentClassification]:
    """Classify every comment, batched into groups of `batch_size` and run
    `concurrency` batches at a time. Returns a map from comment `id` to its
    classification.

    Raises:
        RuntimeError: One or more comments were omitted by a classifier batch.
    """
    if not items:
        return {}
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results: dict[int, CommentClassification] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(classifier, batch) for batch in batches]
        for fut in as_completed(futures):
            results.update(fut.result())
    missing = sorted({item.id for item in items} - set(results))
    if missing:
        raise RuntimeError(f"classifier omitted {len(missing)} comment(s): {missing[:10]}")
    return results


# ---------------------------------------------------------------------------
# GitHub via gh CLI
# ---------------------------------------------------------------------------


@dataclass
class Comment:
    pr_number: int
    pr_title: str
    merged_at: str
    pr_author: str
    head_sha: str
    base_sha: str
    comment_id: int
    comment_type: str  # "inline" | "review_body" | "issue"
    author: str
    is_bot: bool
    file: str | None
    line: int | None
    body: str
    context: str | None


def _gh_json(args: list[str]) -> object:
    r = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def _gh_paginated(args: list[str]) -> list:
    """Fetch a paginated gh API endpoint as a single flat list of items.

    `gh api --paginate` alone emits one JSON document per page concatenated,
    which `json.loads` cannot parse. Pairing it with `--slurp` wraps the
    pages in an outer array `[[page1_items], [page2_items], ...]`; we flatten
    so callers don't have to care how many pages came back.
    """
    pages = _gh_json([*args, "--paginate", "--slurp"])
    if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
        raise TypeError("gh paginated response must be a list of pages")
    out: list = []
    for page in pages:
        out.extend(page)
    return out


def _optional_github_timestamp(value: str | None) -> dt.datetime | None:
    """Parse a GitHub timestamp that an unmerged PR leaves unset."""
    return parse_utc(value) if value else None


def _pr_from_rest_pull(pull: dict) -> dict:
    user = pull.get("user") or {}
    return {
        "number": pull["number"],
        "title": pull["title"],
        "mergedAt": pull["merged_at"],
        "author": {"login": user.get("login"), "type": user.get("type")},
        "headRefOid": (pull.get("head") or {})["sha"],
        "baseRefOid": (pull.get("base") or {})["sha"],
    }


def list_merged_prs(repo: str, days: int, limit: int | None) -> list[dict]:
    since = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    prs: list[dict] = []
    page = 1

    while limit is None or len(prs) < limit:
        pulls = _gh_json(
            [
                "api",
                f"repos/{repo}/pulls?state=closed&sort=updated&direction=desc&per_page=100&page={page}",
            ]
        )
        if not pulls:
            break
        assert isinstance(pulls, list)

        for pull in pulls:
            merged_at = pull.get("merged_at")
            if merged_at is None:
                continue
            if parse_utc(merged_at) < since:
                continue
            prs.append(_pr_from_rest_pull(pull))
            if limit is not None and len(prs) >= limit:
                break

        last_updated_at = pulls[-1].get("updated_at")
        if last_updated_at is not None and parse_utc(last_updated_at) < since:
            break
        page += 1

    return prs


def _is_bot(author: dict | None, bot_logins: set[str]) -> bool:
    if not author:
        return True
    if author.get("type") == "Bot":
        return True
    login = (author.get("login") or "").lower()
    if login in bot_logins:
        return True
    if login.endswith("[bot]"):
        return True
    return False


def _is_reviewer_comment(comment: Comment) -> bool:
    return not comment.is_bot and not comment.body.lstrip().startswith("🤖")


def _pull_request_context(files: list[dict]) -> str | None:
    """Render bounded changed-file patches for a top-level review comment."""
    sections: list[str] = []
    size = 0
    for file in files:
        path = file.get("filename")
        if not path:
            continue
        header = (
            f"File: {path} ({file.get('status', 'modified')}, "
            f"+{file.get('additions', 0)}/-{file.get('deletions', 0)})"
        )
        patch = (file.get("patch") or "")[:MAX_FILE_PATCH]
        section = f"{header}\n{patch}" if patch else header
        separator_size = 2 if sections else 0
        remaining = MAX_COMMENT_CONTEXT - size - separator_size
        if remaining <= 0:
            break
        sections.append(section[:remaining])
        size += separator_size + len(sections[-1])
    return "\n\n".join(sections) or None


def fetch_pr_comments(repo: str, pr: dict, bot_logins: set[str]) -> list[Comment]:
    n = pr["number"]
    title = pr["title"]
    merged_at = pr["mergedAt"]
    author = (pr.get("author") or {}).get("login") or "unknown"
    head_sha = pr["headRefOid"]
    base_sha = pr["baseRefOid"]

    out: list[Comment] = []
    files = _gh_paginated(["api", f"repos/{repo}/pulls/{n}/files"])
    pull_request_context = _pull_request_context(files)

    # Inline review comments (anchored to file:line)
    inline = _gh_paginated(["api", f"repos/{repo}/pulls/{n}/comments"])
    for c in inline:
        u = c.get("user") or {}
        out.append(
            Comment(
                pr_number=n,
                pr_title=title,
                merged_at=merged_at,
                pr_author=author,
                head_sha=head_sha,
                base_sha=base_sha,
                comment_id=c["id"],
                comment_type="inline",
                author=u.get("login") or "unknown",
                is_bot=_is_bot(u, bot_logins),
                file=c.get("path"),
                line=c.get("line") or c.get("original_line"),
                body=c.get("body") or "",
                context=(c.get("diff_hunk") or "")[:MAX_COMMENT_CONTEXT] or None,
            )
        )

    # Review summary bodies (state + free text)
    reviews = _gh_paginated(["api", f"repos/{repo}/pulls/{n}/reviews"])
    for r in reviews:
        body = r.get("body") or ""
        if not body.strip():
            continue
        u = r.get("user") or {}
        out.append(
            Comment(
                pr_number=n,
                pr_title=title,
                merged_at=merged_at,
                pr_author=author,
                head_sha=head_sha,
                base_sha=base_sha,
                comment_id=r["id"],
                comment_type="review_body",
                author=u.get("login") or "unknown",
                is_bot=_is_bot(u, bot_logins),
                file=None,
                line=None,
                body=body,
                context=pull_request_context,
            )
        )

    # Top-level PR comments (issue thread)
    issue_comments = _gh_paginated(["api", f"repos/{repo}/issues/{n}/comments"])
    for c in issue_comments:
        u = c.get("user") or {}
        out.append(
            Comment(
                pr_number=n,
                pr_title=title,
                merged_at=merged_at,
                pr_author=author,
                head_sha=head_sha,
                base_sha=base_sha,
                comment_id=c["id"],
                comment_type="issue",
                author=u.get("login") or "unknown",
                is_bot=_is_bot(u, bot_logins),
                file=None,
                line=None,
                body=c.get("body") or "",
                context=pull_request_context,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Finelog: load prior rows + append new ones
# ---------------------------------------------------------------------------


# The comment tables are append-only and a rolling window re-emits a PR's rows,
# so every read collapses to the most recent row per natural key. `seq` is the
# server-assigned per-row counter; ordering on it breaks ties within a run.
LATEST_HUMAN_COMMENTS_SQL = f"""
    SELECT * FROM (
        SELECT *, row_number() OVER (
            PARTITION BY pr_number, comment_type, comment_id ORDER BY seq DESC
        ) AS recency
        FROM "{HUMAN_COMMENTS_NAMESPACE}"
    ) WHERE recency = 1
"""

LATEST_PR_OUTCOMES_SQL = f"""
    SELECT * FROM (
        SELECT *, row_number() OVER (
            PARTITION BY pr_number ORDER BY seq DESC
        ) AS recency
        FROM "{PR_REVIEW_OUTCOMES_NAMESPACE}"
    ) WHERE recency = 1
"""

_SHA_PATTERN = re.compile(r"\A[0-9a-fA-F]{7,40}\Z")


def _as_utc(value):
    """Stamp UTC on Finelog's naive timestamps; pass everything else through."""
    if isinstance(value, dt.datetime) and value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value


def query_rows(client: LogClient, sql: str, namespace: str) -> list[dict]:
    """Run `sql` over `namespace` and return rows with tz-aware timestamps.

    An unregistered namespace is the normal state before the first write and
    reads as empty. A query error against a registered namespace propagates, so
    a schema mismatch is never rendered as an empty report.
    """
    if row_count(client, namespace) is None:
        logger.info("namespace %s does not exist yet; treating as empty", namespace)
        return []
    return [{k: _as_utc(v) for k, v in row.items()} for row in client.query(sql).to_pylist()]


def load_findings_for_shas(client: LogClient, shas: set[str]) -> dict[str, list[dict]]:
    """Fetch the bot's own findings for `shas`, grouped by head_sha."""
    by_sha: dict[str, list[dict]] = {sha: [] for sha in shas}
    safe = sorted(sha for sha in shas if _SHA_PATTERN.match(sha))
    if not safe:
        return by_sha
    in_list = ", ".join(f"'{sha}'" for sha in safe)
    rows = query_rows(
        client,
        f"SELECT head_sha, file, line, code, confidence, message "
        f'FROM "{FINDINGS_NAMESPACE}" WHERE head_sha IN ({in_list})',
        FINDINGS_NAMESPACE,
    )
    for r in rows:
        if r["head_sha"] in by_sha:
            by_sha[r["head_sha"]].append(
                {
                    "file": r["file"],
                    "line": r["line"],
                    "code": r["code"],
                    "confidence": r["confidence"],
                    "message": r["message"],
                }
            )
    return by_sha


def build_classification_cache(human_rows: list[dict]) -> dict[tuple[str, int], tuple[str, CommentClassification]]:
    """Index already-classified comments so unchanged comments can skip
    re-classification. Keyed on (comment_type, comment_id) — GitHub's inline /
    review / issue comment ids live in separate spaces, so the type disambiguates
    them. Maps to (stored body, classification); the body is kept so an edited
    comment (same id, new text) is re-classified rather than served stale. The
    cache is model-agnostic — re-run with `--refresh` after changing `--model`,
    since the table does not record it."""
    cache: dict[tuple[str, int], tuple[str, CommentClassification]] = {}
    for row in human_rows:
        cls = CommentClassification(
            **{"class": row["comment_class"]},
            catchable_strict=bool(row["catchable_strict"]),
            catchable_generous=bool(row["catchable_generous"]),
            confidence=float(row["confidence"]),
            reason=row["reason"] or "",
        )
        cache[(row["comment_type"], int(row["comment_id"]))] = (row["body"] or "", cls)
    return cache


def resolve_classifications(
    comments: list[Comment],
    cache: dict[tuple[str, int], tuple[str, CommentClassification]],
    classifier: Classifier,
    batch_size: int,
    concurrency: int,
) -> list[CommentClassification | None]:
    """Classify `comments`, returning one verdict per comment aligned with the
    input. A comment is reused from `cache` when the same (comment_type,
    comment_id) was seen before with identical (truncated) text; the rest are
    sent to `classifier` in parallel batches."""
    final: list[CommentClassification | None] = [None] * len(comments)
    pending: list[tuple[int, Comment]] = []
    for i, c in enumerate(comments):
        cached = cache.get((c.comment_type, c.comment_id))
        if cached and cached[0] == c.body[:500]:
            final[i] = cached[1]
        else:
            pending.append((i, c))
    logger.info(
        "%d human comments: %d cached, %d to classify in batches of %d, %d in parallel",
        len(comments),
        len(comments) - len(pending),
        len(pending),
        batch_size,
        concurrency,
    )
    items = [
        CommentToClassify(id=j, file=c.file, line=c.line, body=c.body, context=c.context)
        for j, (_, c) in enumerate(pending)
    ]
    fresh = classify_comments(classifier, items, batch_size, concurrency)
    for j, (i, _) in enumerate(pending):
        final[i] = fresh.get(j)
    return final


def overlap_count(bot_findings: list[dict], human_comments: list[Comment], window: int = 5) -> int:
    """Bot finding and human inline comment on same file within ±`window` lines."""
    if not bot_findings:
        return 0
    by_file: dict[str, list[int]] = {}
    for f in bot_findings:
        if f["file"] and f["line"] is not None:
            by_file.setdefault(f["file"], []).append(int(f["line"]))
    n = 0
    for c in human_comments:
        if c.comment_type != "inline" or not c.file or c.line is None:
            continue
        for ln in by_file.get(c.file, []):
            if abs(ln - int(c.line)) <= window:
                n += 1
                break
    return n


# ---------------------------------------------------------------------------
# Report: render the accumulated tables into a shareable markdown digest
# ---------------------------------------------------------------------------


def _in_window(row: dict, key: str, start: dt.datetime) -> bool:
    """True when the timestamp at `row[key]` is on/after `start`. Rows with no
    timestamp fall outside the window. Shared by the merged-PR tables (keyed on
    `merged_at`) and the automation tables (`ts`)."""
    ts = row.get(key)
    return ts is not None and ts >= start


def _group_by_isoweek(rows: list[dict], ts_key: str) -> dict[tuple[int, int], list[dict]]:
    """Bucket rows by the ISO (year, week) of `row[ts_key]`. Rows with a missing
    timestamp are dropped; callers sort the keys for display."""
    weeks: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        ts = row.get(ts_key)
        if not ts:
            continue
        y, w, _ = ts.isocalendar()
        weeks.setdefault((y, w), []).append(row)
    return weeks


def _pct(n: int, d: int) -> str:
    return f"{round(100 * n / d)}%" if d else "—"


def _to_int(value: object) -> int:
    """Coerce a possibly-null cell to int; missing/garbage counts as 0."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float | None:
    """Coerce a possibly-null cell to float, or None when absent/garbage."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _cell(value: object, maxlen: int | None = None) -> str:
    """Render a value as a single safe markdown table cell."""
    text = " ".join(str(value).split()).replace("|", "\\|")
    if maxlen and len(text) > maxlen:
        text = text[: maxlen - 1] + "…"
    return text


def _md_table(headers: list[str], aligns: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _pr_link(repo: str, number: object) -> str:
    return f"[#{number}](https://github.com/{repo}/pull/{number})"


def _where(file: object, line: object) -> str:
    """`file:line` for an inline comment, the file alone if unlocated, else —."""
    if not file:
        return "—"
    return f"{file}:{line}" if line not in (None, "") else str(file)


def build_automation_section(invocations: list[dict], findings: list[dict], start: dt.datetime, days: int) -> str:
    """Render review-automation activity from the linter-fed `invocations` and
    `findings` tables (written by `infra/codehealth/log_stats.py` on every
    `pre-commit.py --review` / `/code-review` / `/review-pr` run). Pure: takes
    already-loaded rows, returns markdown, filtered to runs whose `ts` falls
    on/after `start`.

    This is what the review bot itself did — runs, runtime, and the catalog
    rules it fired — distinct from the human-comment catchability analysis,
    which measures what reviewers caught that the bot could have."""

    runs = [r for r in invocations if _in_window(r, "ts", start)]
    finds = [f for f in findings if _in_window(f, "ts", start)]

    heading = "## Review automation activity"
    blurb = (
        "What the review bot itself did over the window — every "
        "`pre-commit.py --review`, `/code-review`, and `/review-pr` run logs to "
        "Finelog. Distinct from the catchability analysis above: that measures human "
        "comments the bot could have caught; this is the bot's own output."
    )
    if not runs:
        return f"{heading}\n\n{blurb}\n\n_No review-bot runs recorded in the last {days} days._"

    n_runs = len(runs)
    with_findings = sum(1 for r in runs if _to_int(r.get("finding_count")) > 0)
    failed = sum(1 for r in runs if _to_int(r.get("agent_exit_code")) != 0 or bool(r.get("timed_out")))
    total_findings = sum(_to_int(r.get("finding_count")) for r in runs)
    elapsed = sorted(v for r in runs if (v := _to_float(r.get("elapsed"))) is not None)
    median_elapsed = elapsed[len(elapsed) // 2] if elapsed else None

    summary = "### Activity\n\n" + _md_table(
        ["Metric", "Value"],
        ["---", "---:"],
        [
            ["Review runs", n_runs],
            ["— produced findings", f"{with_findings} ({_pct(with_findings, n_runs)})"],
            ["— silent (no findings)", f"{n_runs - with_findings} ({_pct(n_runs - with_findings, n_runs)})"],
            ["— failed or timed out", failed],
            ["Findings emitted", total_findings],
            ["Findings per run (mean)", f"{total_findings / n_runs:.1f}"],
            ["Median runtime", f"{median_elapsed:.0f}s" if median_elapsed is not None else "—"],
        ],
    )

    # Most-fired catalog rules: the `code` on each finding (e.g.
    # `ml-exception-swallow`). Answers "what does the reviewer flag most?"
    by_code: dict[str, list] = {}
    for f in finds:
        code = str(f.get("code") or "(uncoded)")
        e = by_code.setdefault(code, [0, 0.0, ""])
        e[0] += 1
        conf = _to_float(f.get("confidence"))
        if conf is not None:
            e[1] += conf
        if not e[2]:
            e[2] = str(f.get("message") or "")
    code_rows = [
        [_cell(code), n, _pct(n, len(finds)), f"{conf_sum / n:.2f}", _cell(example, 80)]
        for code, (n, conf_sum, example) in sorted(by_code.items(), key=lambda kv: kv[1][0], reverse=True)[:15]
    ]
    codes_section = "### Most frequent findings\n\n" + (
        _md_table(
            ["Catalog code", "Count", "% of findings", "Mean conf.", "Example"],
            ["---", "---:", "---:", "---:", "---"],
            code_rows,
        )
        if code_rows
        else "_No findings emitted in this window._"
    )

    # Weekly adoption: runs + findings keyed by ISO week of the run timestamp.
    week_rows = []
    for (y, w), group in sorted(_group_by_isoweek(runs, "ts").items()):
        found = sum(_to_int(r.get("finding_count")) for r in group)
        with_finds = sum(1 for r in group if _to_int(r.get("finding_count")) > 0)
        week_rows.append([f"{y}-W{w:02d}", len(group), with_finds, found])
    trend = "### Weekly trend\n\n" + _md_table(
        ["Week", "Runs", "With findings", "Findings"],
        ["---", "---:", "---:", "---:"],
        week_rows,
    )

    return "\n\n".join([heading, blurb, summary, codes_section, trend])


def build_report(
    outcomes: list[dict],
    comments: list[dict],
    invocations: list[dict],
    findings: list[dict],
    repo: str,
    start: dt.datetime,
    now: dt.datetime,
    days: int,
) -> str:
    """Render the per-PR outcome rows and classified comments into a markdown
    digest. Pure: takes already-loaded rows, returns markdown. Rows are filtered
    to PRs merged on/after `start`."""

    outcomes = [d for d in outcomes if _in_window(d, "merged_at", start)]
    comments = [d for d in comments if _in_window(d, "merged_at", start)]

    header = (
        f"# Marin code-health review report\n\n"
        f"**Window:** {start.date()} → {now.date()} ({days} days)  \n"
        f"**Generated:** {now.replace(microsecond=0).isoformat()}"
    )

    # Two distinct lenses, kept in separate sections below so they are not
    # conflated: what humans flagged (and whether a bot could have), versus what
    # the bot itself flagged.
    overview = (
        "Two lenses on review quality:\n\n"
        "- **Human review feedback** — comments people left on merged PRs, each classified by "
        "whether an automated review *could* have caught it (**strict** = a deterministic "
        "linter/type-checker would; **generous** = an LLM reading the diff would). This is the "
        "gap automation still leaves.\n"
        "- **Review automation activity** — what the review bot actually flagged when it ran. "
        "This is what automation already does."
    )

    # The automation section reads its own (`invocations`/`findings`) tables and
    # filters by run timestamp, so it is independent of whether any PR merged.
    automation_section = build_automation_section(invocations, findings, start, days)

    if not outcomes:
        note = f"No PRs merged in the last {days} days were found in the review-stats tables."
        return "\n\n".join([header, overview, note, automation_section])

    n_prs = len(outcomes)
    reviewed = sum(1 for d in outcomes if int(d["total_human_comments"]) > 0)
    total = sum(int(d["total_human_comments"]) for d in outcomes)
    strict = sum(int(d["catchable_strict_count"]) for d in outcomes)
    generous = sum(int(d["catchable_generous_count"]) for d in outcomes)
    overlap = sum(int(d["overlap_count"]) for d in outcomes)

    narrative = (
        "## Human review feedback\n\n"
        "What human reviewers flagged on merged PRs, and how much of it an automated review could "
        "have caught.\n\n"
        f"Over the last {days} days, **{n_prs}** PRs merged; **{reviewed}** ({_pct(reviewed, n_prs)}) drew human "
        f"review comments. Of **{total}** human comments, **{strict}** ({_pct(strict, total)}) were strictly "
        f"catchable by a deterministic tool and **{generous}** ({_pct(generous, total)}) generously catchable by an "
        f"LLM reading the diff alone. Our review bot independently flagged **{overlap}** of the spots humans "
        "commented on — see *Review automation activity* below for everything it caught."
    )

    summary = "### Summary\n\n" + _md_table(
        ["Metric", "Value"],
        ["---", "---:"],
        [
            ["PRs merged", n_prs],
            ["PRs with human review comments", f"{reviewed} ({_pct(reviewed, n_prs)})"],
            ["Human review comments", total],
            ["— strictly catchable (deterministic tool)", f"{strict} ({_pct(strict, total)})"],
            ["— generously catchable (LLM on the diff)", f"{generous} ({_pct(generous, total)})"],
            ["— independently flagged by the bot", f"{overlap} ({_pct(overlap, total)})"],
        ],
    )

    # By-class breakdown comes from the per-comment table, which carries the
    # per-comment class + catchable flags the per-PR rollup does not.
    by_class: dict[str, list[int]] = {}
    for c in comments:
        e = by_class.setdefault(str(c["comment_class"]), [0, 0, 0])
        e[0] += 1
        e[1] += 1 if c["catchable_strict"] else 0
        e[2] += 1 if c["catchable_generous"] else 0
    class_rows = [
        [cls, n, _pct(n, total), f"{s} ({_pct(s, n)})", f"{g} ({_pct(g, n)})"]
        for cls, (n, s, g) in sorted(by_class.items(), key=lambda kv: kv[1][0], reverse=True)
    ]
    by_class_section = "### By comment class\n\n" + (
        _md_table(
            ["Class", "Comments", "% of all", "Strict", "Generous"], ["---", "---:", "---:", "---:", "---:"], class_rows
        )
        if class_rows
        else "_No classified human comments in this window._"
    )

    # Weekly trend, keyed by ISO week of the merge date.
    week_rows = []
    for (y, w), group in sorted(_group_by_isoweek(outcomes, "merged_at").items()):
        cmts = sum(int(d["total_human_comments"]) for d in group)
        st = sum(int(d["catchable_strict_count"]) for d in group)
        gen = sum(int(d["catchable_generous_count"]) for d in group)
        week_rows.append([f"{y}-W{w:02d}", len(group), cmts, st, gen, _pct(gen, cmts)])
    trend_section = "### Weekly trend\n\n" + _md_table(
        ["Week", "PRs", "Comments", "Strict", "Generous", "Generous %"],
        ["---", "---:", "---:", "---:", "---:", "---:"],
        week_rows,
    )

    # Top PRs by how much catchable feedback they drew — where automation would
    # have helped reviewers most.
    top = sorted(
        (d for d in outcomes if int(d["total_human_comments"]) > 0),
        key=lambda d: (int(d["catchable_generous_count"]), int(d["total_human_comments"])),
        reverse=True,
    )[:10]
    top_rows = [
        [
            _pr_link(repo, d["pr_number"]),
            _cell(d["pr_title"], 60),
            int(d["total_human_comments"]),
            int(d["catchable_strict_count"]),
            int(d["catchable_generous_count"]),
            int(d["bot_findings_count"]),
            int(d["overlap_count"]),
        ]
        for d in top
    ]
    top_section = "### PRs with the most catchable feedback\n\n" + (
        _md_table(
            ["PR", "Title", "Comments", "Strict", "Generous", "Bot findings", "Overlap"],
            ["---", "---", "---:", "---:", "---:", "---:", "---:"],
            top_rows,
        )
        if top_rows
        else "_No human review comments in this window._"
    )

    # Every comment an automated check could have caught — the full "automation
    # should have caught this" list, not a sample (strict ⊆ generous, so the
    # catchable_strict/generous flags together cover all flagged comments).
    # Volume is low (~tens/month), so it is intentionally not truncated.
    flagged = sorted(
        (c for c in comments if c["catchable_strict"] or c["catchable_generous"]),
        key=lambda c: (not c["catchable_strict"], -float(c["confidence"])),
    )
    flagged_rows = [
        [
            _pr_link(repo, c["pr_number"]),
            _cell(_where(c.get("file"), c.get("line"))),
            "strict" if c["catchable_strict"] else "generous",
            _cell(c["comment_class"]),
            f"{float(c['confidence']):.2f}",
            _cell(c["body"], 120),
            _cell(c["reason"], 80),
        ]
        for c in flagged
    ]
    flagged_section = f"### Catchable comments ({len(flagged_rows)})\n\n" + (
        "Every human comment an automated check could plausibly have caught, "
        "strict (deterministic) first. **strict** = a linter/type-check could "
        "flag it; **generous** = an LLM reading the diff could.\n\n"
        + _md_table(
            ["PR", "Where", "Tier", "Class", "Conf.", "Comment", "Why catchable"],
            ["---", "---", "---", "---", "---:", "---", "---"],
            flagged_rows,
        )
        if flagged_rows
        else "_No catchable comments in this window._"
    )

    return "\n\n".join(
        [
            header,
            overview,
            # Lens 1 — human review feedback (narrative carries the "## Human
            # review feedback" banner; the rest are its ### subsections).
            narrative,
            summary,
            by_class_section,
            trend_section,
            top_section,
            flagged_section,
            # Lens 2 — what the review bot itself did.
            automation_section,
        ]
    )


def publish_gist(markdown: str, desc: str, public: bool, filename: str) -> str:
    """Write `markdown` to a temp file and create a gist via `gh`. Returns the
    gist URL `gh` prints. Gists default to secret unless `public` is set."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / filename
        path.write_text(markdown)
        args = ["gist", "create", str(path), "--desc", desc]
        if public:
            args.append("--public")
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Code-health review stats: classify reviewer comments, and report on them."""


@cli.command()
@click.option("--repo", default=DEFAULT_REPO, show_default=True)
@click.option("--days", type=int, default=1, show_default=True, help="Look back N days of merged PRs")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Max PRs to process; omit for all PRs")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Codex model id for the classifier")
@click.option(
    "--reasoning-effort",
    default=DEFAULT_REASONING_EFFORT,
    show_default=True,
    type=click.Choice(["none", "low", "medium", "high", "xhigh", "max"]),
    help="Codex reasoning effort for the classifier",
)
@click.option(
    "--agent-command",
    default=DEFAULT_AGENT_COMMAND,
    show_default=True,
    help="Headless agent invocation for classification (reads its prompt on stdin)",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Comments classified per model request",
)
@click.option(
    "--concurrency",
    type=int,
    default=DEFAULT_CONCURRENCY,
    show_default=True,
    help="Batches classified in parallel",
)
@click.option(
    "--bot-logins",
    default="github-actions,dependabot,claude,claude-review,loom-oa-dev,renovate,weaverbot",
    show_default=True,
    help="Comma-separated bot logins to skip (lowercase)",
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Re-classify every comment, ignoring the cache (use after changing --model)",
)
@click.option("--dry-run", is_flag=True, help="Skip the Finelog write; print the rollup")
@click.option("--deployment", default=DEFAULT_DEPLOYMENT, show_default=True, help="Finelog deployment to use")
def aggregate(
    repo: str,
    days: int,
    limit: int,
    model: str,
    reasoning_effort: str,
    agent_command: str,
    batch_size: int,
    concurrency: int,
    bot_logins: str,
    refresh: bool,
    dry_run: bool,
    deployment: str,
) -> None:
    """Classify reviewer comments on recently-merged PRs and append to Finelog.

    Comments already classified in the `human_comments` table are reused by
    `comment_id` (unless their text changed), so a daily run over a rolling
    window only sends genuinely-new comments to the model. Pass `--refresh` to
    re-classify everything."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bot_login_set = {x.strip().lower() for x in bot_logins.split(",") if x.strip()}

    agent_binary = agent_command.split()[0]
    if not shutil.which(agent_binary):
        logger.error("classifier agent %r not found on PATH (need a logged-in `codex` CLI)", agent_binary)
        sys.exit(2)

    classifier = make_codex_classifier(model, reasoning_effort, agent_command)

    logger.info("Listing PRs merged in last %d day(s) in %s", days, repo)
    prs = list_merged_prs(repo, days, limit)
    logger.info("Found %d merged PRs", len(prs))

    aggregator_ts = dt.datetime.now(dt.UTC)

    human_rows: list[HumanComment] = []
    pr_rows: list[PrReviewOutcome] = []
    all_shas: set[str] = set()
    per_pr_comments: dict[int, list[Comment]] = {}

    for pr in prs:
        comments = fetch_pr_comments(repo, pr, bot_login_set)
        per_pr_comments[pr["number"]] = comments
        all_shas.add(pr["headRefOid"])

    # Pull the bot's findings for these PRs plus the already-classified
    # comments, which seed the classification cache below. A dry run needs
    # neither, so it never opens a client.
    if dry_run:
        findings_by_sha = {sha: [] for sha in all_shas}
        existing_human_rows: list[dict] = []
    else:
        with open_tables_client(deployment) as client:
            findings_by_sha = load_findings_for_shas(client, all_shas)
            existing_human_rows = query_rows(client, LATEST_HUMAN_COMMENTS_SQL, HUMAN_COMMENTS_NAMESPACE)
    cache = {} if refresh else build_classification_cache(existing_human_rows)

    # Flatten every human comment across all PRs. Reuse a cached classification
    # when the same comment_id was classified before with identical text;
    # otherwise queue it for the model. Only the queued ones are batched and
    # classified in parallel, so an overlapping daily window stays cheap.
    human_by_pr = {
        pr["number"]: [c for c in per_pr_comments.get(pr["number"], []) if _is_reviewer_comment(c)] for pr in prs
    }
    flat_comments = [c for pr in prs for c in human_by_pr[pr["number"]]]
    final_cls = resolve_classifications(flat_comments, cache, classifier, batch_size, concurrency)

    # Regroup by PR, preserving flat-list ordering.
    idx = 0
    classified_by_pr: dict[int, list[tuple[Comment, CommentClassification | None]]] = {}
    for pr in prs:
        pairs: list[tuple[Comment, CommentClassification | None]] = []
        for c in human_by_pr[pr["number"]]:
            pairs.append((c, final_cls[idx]))
            idx += 1
        classified_by_pr[pr["number"]] = pairs

    # Build rows.
    for pr in prs:
        n = pr["number"]
        human = human_by_pr[n]
        by_class: dict[str, int] = {}
        strict_cnt = generous_cnt = 0

        for c, cls in classified_by_pr[n]:
            if cls is None:
                continue
            by_class[cls.klass] = by_class.get(cls.klass, 0) + 1
            if cls.catchable_strict:
                strict_cnt += 1
            if cls.catchable_generous:
                generous_cnt += 1
            human_rows.append(
                HumanComment(
                    ts=aggregator_ts,
                    pr_number=n,
                    pr_title=pr["title"],
                    merged_at=_optional_github_timestamp(pr["mergedAt"]),
                    author=c.author,
                    comment_id=c.comment_id,
                    comment_type=c.comment_type,
                    file=c.file,
                    line=c.line,
                    body=c.body[:500],
                    comment_class=cls.klass,
                    catchable_strict=cls.catchable_strict,
                    catchable_generous=cls.catchable_generous,
                    confidence=cls.confidence,
                    reason=cls.reason,
                )
            )

        bot_findings = findings_by_sha.get(pr["headRefOid"], [])
        pr_rows.append(
            PrReviewOutcome(
                ts=aggregator_ts,
                pr_number=n,
                pr_title=pr["title"],
                merged_at=_optional_github_timestamp(pr["mergedAt"]),
                author=(pr.get("author") or {}).get("login") or "unknown",
                head_sha=pr["headRefOid"],
                base_sha=pr["baseRefOid"],
                total_human_comments=len(human),
                by_class_json=json.dumps(by_class),
                catchable_strict_count=strict_cnt,
                catchable_generous_count=generous_cnt,
                bot_findings_count=len(bot_findings),
                overlap_count=overlap_count(bot_findings, human),
            )
        )
        logger.info(
            "PR #%s: %d human comments, strict=%d generous=%d bot_findings=%d",
            n,
            len(human),
            strict_cnt,
            generous_cnt,
            len(bot_findings),
        )

    if dry_run:
        payload = {
            "pr_rollups": [asdict(r) for r in pr_rows],
            "human_comments": [asdict(r) for r in human_rows[:20]],
        }
        click.echo(json.dumps(payload, default=str, indent=2))
        return

    # Rows for a PR seen on an earlier run are appended again with their cached
    # verdicts; LATEST_*_SQL is what collapses them on read.
    with open_tables_client(deployment) as client:
        append_rows(client, HUMAN_COMMENTS_NAMESPACE, HumanComment, human_rows)
        append_rows(client, PR_REVIEW_OUTCOMES_NAMESPACE, PrReviewOutcome, pr_rows)
    logger.info(
        "Appended %d PR rollups and %d classified human comments to %s",
        len(pr_rows),
        len(human_rows),
        NAMESPACE_PREFIX,
    )


@cli.command()
@click.option("--repo", default=DEFAULT_REPO, show_default=True, help="Repo used to build PR links")
@click.option("--days", type=int, default=30, show_default=True, help="Report window: PRs merged in last N days")
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="Also write the markdown report here")
@click.option("--public", is_flag=True, help="Create a public gist (default: secret)")
@click.option("--no-gist", is_flag=True, help="Print the report to stdout instead of creating a gist")
@click.option("--deployment", default=DEFAULT_DEPLOYMENT, show_default=True, help="Finelog deployment to use")
def report(repo: str, days: int, out: str | None, public: bool, no_gist: bool, deployment: str) -> None:
    """Render the accumulated review stats into a markdown digest and gist it.

    Reads the four `codehealth.autolint` namespaces: `pr_review_outcomes` and
    `human_comments` (the catchability analysis) plus the linter-fed
    `invocations` and `findings` (the review bot's own activity)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    now = dt.datetime.now(dt.UTC)
    start = now - dt.timedelta(days=days)

    with open_tables_client(deployment) as client:
        outcomes = query_rows(client, LATEST_PR_OUTCOMES_SQL, PR_REVIEW_OUTCOMES_NAMESPACE)
        comments = query_rows(client, LATEST_HUMAN_COMMENTS_SQL, HUMAN_COMMENTS_NAMESPACE)
        invocations = query_rows(client, f'SELECT * FROM "{INVOCATIONS_NAMESPACE}"', INVOCATIONS_NAMESPACE)
        findings = query_rows(client, f'SELECT * FROM "{FINDINGS_NAMESPACE}"', FINDINGS_NAMESPACE)

    markdown = build_report(outcomes, comments, invocations, findings, repo=repo, start=start, now=now, days=days)

    if out:
        Path(out).write_text(markdown)
        logger.info("Wrote report to %s", out)

    if no_gist:
        click.echo(markdown)
        return

    url = publish_gist(
        markdown,
        desc=f"Marin code-health review — last {days} days ({now.date()})",
        public=public,
        filename="marin-code-health-report.md",
    )
    logger.info("Published gist: %s", url)
    click.echo(url)


if __name__ == "__main__":
    cli()
