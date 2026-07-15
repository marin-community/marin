#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.0",
#     "pydantic>=2.0",
#     "wandb>=0.18",
# ]
# ///

"""Aggregator and reporter for the code-health stats dashboard.

Two subcommands:
  - `aggregate` (designed to run as a daily GHA cron) classifies reviewer
    comments and appends them to the shared `review-stats` W&B run.
  - `report` reads those accumulated tables back and renders a markdown digest
    (summary table, by-class breakdown, weekly trend, top PRs, examples). It
    also folds in the linter-fed `invocations`/`findings` tables (written by
    `infra/codehealth/log_stats.py` on every review-bot run) as a "Review
    automation activity" section — runs, runtime, and the catalog rules fired.
    Published as a gist by default.

`aggregate`, for each PR merged in the last N days:
  1. Pull review/inline/issue comments via the `gh` CLI.
  2. Drop bot comments (only humans should be in the denominator).
  3. Classify each human comment with a pluggable classifier — a headless
     Claude Code session (`claude -p`) by default, the same agent that runs the
     lint review. Comments from all PRs are pooled, batched, and the batches
     classified in parallel so a many-PR run is not a long serial trickle of
     one request per comment. Two independent "could automation have caught
     this?" signals:
       - catchable_strict   — a deterministic linter / type checker / ml-*
                              catalog rule could mechanically flag it.
       - catchable_generous — a modern LLM running on the diff alone would
                              plausibly flag it with high confidence.
     Strict ⊆ generous by construction (the prompt enforces it).
  4. Pull the bot's own findings for the PR's head_sha from the existing
     `findings` artifact written by `infra/codehealth/log_stats.py`.
  5. Append two flat tables to the shared `review-stats` W&B run (same run
     that `infra/codehealth/log_stats.py` writes invocations + findings to):
       - human_comments      — one row per classified comment.
       - pr_review_outcomes  — one row per PR (rollup).

Same append-to-artifact pattern as `infra/codehealth/log_stats.py`. Concurrent
writers race; loser's row is dropped (acceptable at our scale).

Requires a logged-in `claude` CLI (subscription auth) and W&B auth, plus
`gh auth login` for the GitHub side. Designed to run as a daily GHA cron.
"""

import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import click
import wandb

# Sibling standalone script: the canonical schema for the linter-fed tables this
# report reads back. Importing keeps the column layouts in one place rather than
# re-declaring them here and risking drift.
from log_stats import FINDING_COLUMNS, INVOCATION_COLUMNS
from pydantic import BaseModel, Field, TypeAdapter

logger = logging.getLogger("codehealth.review")

WANDB_PROJECT = "marin-review-stats"
WANDB_RUN_ID = "review-stats"
DEFAULT_REPO = "marin-community/marin"
# Classifier backend: a headless Claude Code session (`claude -p`). The model is
# a CLI alias (`sonnet`, `opus`, `haiku`) or a full model id. `sonnet` balances
# classification quality against cost for the per-comment judgment calls.
DEFAULT_MODEL = "sonnet"
DEFAULT_BATCH_SIZE = 20
# One headless `claude` subprocess per batch, so concurrency caps simultaneous
# processes (and subscription rate pressure) — lower than an HTTP-API backend.
DEFAULT_CONCURRENCY = 4

HUMAN_COMMENT_COLUMNS = [
    "ts",
    "pr_number",
    "pr_title",
    "merged_at",
    "author",
    "comment_id",
    "comment_type",  # inline | review_body | issue
    "file",
    "line",
    "body",
    "class",
    "catchable_strict",
    "catchable_generous",
    "confidence",
    "reason",
]

PR_OUTCOME_COLUMNS = [
    "ts",
    "pr_number",
    "pr_title",
    "merged_at",
    "author",
    "head_sha",
    "base_sha",
    "total_human_comments",
    "by_class_json",
    "catchable_strict_count",
    "catchable_generous_count",
    "bot_findings_count",
    "overlap_count",
]


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


class BatchedClassification(CommentClassification):
    """A `CommentClassification` plus the `id` marker echoed back in a batch."""

    id: int


# A classifier turns a batch of comments into classifications keyed by their
# `id` marker. Comments it cannot classify are simply absent from the result.
# Pluggable so the backend (currently a headless `claude -p` session) can be
# swapped without touching the batching/parallelism orchestration.
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
        blocks.append(f"=== COMMENT id={it.id} ===\n{where}\nBody:\n{it.body.strip()}")
    return "\n\n".join(blocks)


# Env markers that would bind a spawned `claude` to its parent Claude Code
# session or to metered API billing. Stripped before exec (mirrors the lint
# review in infra/linter.py) so each batch runs as a fresh, isolated session on
# Claude subscription auth rather than nesting under the caller's transcript.
CLAUDE_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_SSE_PORT",
)

# Per-batch wall-clock ceiling for one headless classification call.
CLAUDE_TIMEOUT = 300


def _headless_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in CLAUDE_STRIPPED_ENV}


def _classification_schema() -> dict:
    """Anthropic structured-output JSON schema for one batch result.

    The API uses this schema as a tool `input_schema`, whose root must be an
    object (a top-level array is rejected), so the per-comment classifications
    are wrapped in a `results` array. Field set mirrors `BatchedClassification`.
    """
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
    }
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": item}},
        "required": ["results"],
    }


def _parse_claude_batch(stdout: str) -> list[BatchedClassification] | None:
    """Extract the classifications from a `claude -p --output-format json`
    envelope. Schema-validated structured output lands in the top-level
    `structured_output` field (`result` carries only the model's prose).
    Returns None on any error so the caller drops the batch.
    """
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(envelope, dict) or envelope.get("is_error"):
        return None
    structured = envelope.get("structured_output")
    if not isinstance(structured, dict) or "results" not in structured:
        return None
    try:
        return _BATCH_ADAPTER.validate_python(structured["results"])
    except ValueError:
        return None


def make_claude_classifier(model: str, agent_command: str = "claude -p") -> Classifier:
    """Build a `Classifier` that shells out to a headless Claude Code session
    per batch — the same agent the lint review uses, run as a fresh isolated
    session on Claude subscription auth. A failed or malformed call yields no
    classifications for that batch; those comments are dropped from the metric,
    the same degradation as any classifier backend.
    """
    env = _headless_env()
    cmd = [
        *agent_command.split(),
        "--output-format",
        "json",
        "--tools",
        "",
        "--model",
        model,
        "--json-schema",
        json.dumps(_classification_schema()),
    ]

    def classify(items: list[CommentToClassify]) -> dict[int, CommentClassification]:
        if not items:
            return {}
        prompt = f"{CLASSIFIER_SYSTEM}\n\n{_format_batch(items)}"
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=CLAUDE_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("claude classify timed out for batch of %d", len(items))
            return {}
        parsed = _parse_claude_batch(proc.stdout)
        if parsed is None:
            logger.warning(
                "claude classify failed for batch of %d (exit=%s): %s",
                len(items),
                proc.returncode,
                (proc.stderr or proc.stdout or "").strip()[:200],
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
            logger.warning("claude omitted %d of %d comments in a batch", missing, len(items))
        return out

    return classify


def classify_comments(
    classifier: Classifier, items: list[CommentToClassify], batch_size: int, concurrency: int
) -> dict[int, CommentClassification]:
    """Classify every comment, batched into groups of `batch_size` and run
    `concurrency` batches at a time. Returns a map from comment `id` to its
    classification; ids absent from the map could not be classified."""
    if not items:
        return {}
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results: dict[int, CommentClassification] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(classifier, batch) for batch in batches]
        for fut in as_completed(futures):
            results.update(fut.result())
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
    out: list = []
    for page in pages:
        out.extend(page)
    return out


def _parse_github_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


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
            if _parse_github_timestamp(merged_at) < since:
                continue
            prs.append(_pr_from_rest_pull(pull))
            if limit is not None and len(prs) >= limit:
                break

        last_updated_at = pulls[-1].get("updated_at")
        if last_updated_at is not None and _parse_github_timestamp(last_updated_at) < since:
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


def fetch_pr_comments(repo: str, pr: dict, bot_logins: set[str]) -> list[Comment]:
    n = pr["number"]
    title = pr["title"]
    merged_at = pr["mergedAt"]
    author = (pr.get("author") or {}).get("login") or "unknown"
    head_sha = pr["headRefOid"]
    base_sha = pr["baseRefOid"]

    out: list[Comment] = []

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
            )
        )

    return out


# ---------------------------------------------------------------------------
# W&B: load bot findings + append new rows
# ---------------------------------------------------------------------------


def _load_existing_rows(wandb, log_key: str) -> list[list]:
    try:
        art = wandb.use_artifact(f"run-{WANDB_RUN_ID}-{log_key}:latest")
        table = art.get(log_key)
        return [list(row) for row in table.data]
    except Exception:
        return []


def _rows_to_dicts(columns: list[str], rows: list[list]) -> list[dict]:
    """Zip the flat W&B table rows back into dicts keyed by column name."""
    return [dict(zip(columns, r, strict=False)) for r in rows]


def build_classification_cache(human_rows: list[list]) -> dict[tuple[str, int], tuple[str, CommentClassification]]:
    """Index already-classified comments so unchanged comments can skip
    re-classification. Keyed on (comment_type, comment_id) — GitHub's inline /
    review / issue comment ids live in separate spaces, so the type disambiguates
    them. Maps to (stored body, classification); the body is kept so an edited
    comment (same id, new text) is re-classified rather than served stale. The
    cache is model-agnostic — re-run with `--refresh` after changing `--model`,
    since the table does not record it."""
    cache: dict[tuple[str, int], tuple[str, CommentClassification]] = {}
    for row in _rows_to_dicts(HUMAN_COMMENT_COLUMNS, human_rows):
        cls = CommentClassification(
            **{"class": row["class"]},
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
    items = [CommentToClassify(id=j, file=c.file, line=c.line, body=c.body) for j, (_, c) in enumerate(pending)]
    fresh = classify_comments(classifier, items, batch_size, concurrency)
    for j, (i, _) in enumerate(pending):
        final[i] = fresh.get(j)
    return final


def load_findings_for_shas(wandb, shas: set[str]) -> dict[str, list[dict]]:
    """Fetch bot findings rows for the given head_shas from the shared run.

    Returns sha -> list of finding dicts (file, line, code, confidence, message).
    """
    by_sha: dict[str, list[dict]] = {sha: [] for sha in shas}
    for r in _rows_to_dicts(FINDING_COLUMNS, _load_existing_rows(wandb, "findings")):
        sha = r["head_sha"]
        if sha in by_sha:
            by_sha[sha].append(
                {
                    "file": r["file"],
                    "line": r["line"],
                    "code": r["code"],
                    "confidence": r["confidence"],
                    "message": r["message"],
                }
            )
    return by_sha


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
# Report: render the accumulated W&B tables into a shareable markdown digest
# ---------------------------------------------------------------------------


def _parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _in_window(row: dict, key: str, start: dt.datetime) -> bool:
    """True when the timestamp at `row[key]` is on/after `start`. Rows with a
    missing or unparseable timestamp fall outside the window. Shared by the
    merged-PR tables (keyed on `merged_at`) and the automation tables (`ts`)."""
    ts = row.get(key)
    if not ts:
        return False
    try:
        return _parse_ts(ts) >= start
    except ValueError:
        return False


def _group_by_isoweek(rows: list[dict], ts_key: str) -> dict[tuple[int, int], list[dict]]:
    """Bucket rows by the ISO (year, week) of `row[ts_key]`. Rows with a missing
    or unparseable timestamp are dropped; callers sort the keys for display."""
    weeks: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        ts = row.get(ts_key)
        if not ts:
            continue
        try:
            y, w, _ = _parse_ts(ts).isocalendar()
        except ValueError:
            continue
        weeks.setdefault((y, w), []).append(row)
    return weeks


def _pct(n: int, d: int) -> str:
    return f"{round(100 * n / d)}%" if d else "—"


def _to_int(value: object) -> int:
    """Coerce a possibly-null W&B cell to int; missing/garbage counts as 0."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float | None:
    """Coerce a possibly-null W&B cell to float, or None when absent/garbage."""
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
        "W&B. Distinct from the catchability analysis above: that measures human "
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
        e = by_class.setdefault(str(c["class"]), [0, 0, 0])
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
            _cell(c["class"]),
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
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Claude model alias or id for the classifier")
@click.option(
    "--agent-command",
    default="claude -p",
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
    default="github-actions,dependabot,claude,claude-review,renovate",
    show_default=True,
    help="Comma-separated bot logins to skip (lowercase)",
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Re-classify every comment, ignoring the cache (use after changing --model)",
)
@click.option("--dry-run", is_flag=True, help="Skip W&B upload; print rollup")
def aggregate(
    repo: str,
    days: int,
    limit: int,
    model: str,
    agent_command: str,
    batch_size: int,
    concurrency: int,
    bot_logins: str,
    refresh: bool,
    dry_run: bool,
) -> None:
    """Classify reviewer comments on recently-merged PRs and append to W&B.

    Comments already classified in the W&B `human_comments` table are reused by
    `comment_id` (unless their text changed), so a daily run over a rolling
    window only sends genuinely-new comments to the model. Pass `--refresh` to
    re-classify everything."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bot_login_set = {x.strip().lower() for x in bot_logins.split(",") if x.strip()}

    agent_binary = agent_command.split()[0]
    if not shutil.which(agent_binary):
        logger.error("classifier agent %r not found on PATH (need a logged-in `claude` CLI)", agent_binary)
        sys.exit(2)

    classifier = make_claude_classifier(model, agent_command)

    logger.info("Listing PRs merged in last %d day(s) in %s", days, repo)
    prs = list_merged_prs(repo, days, limit)
    logger.info("Found %d merged PRs", len(prs))

    aggregator_ts = dt.datetime.now(dt.UTC).isoformat()

    human_rows: list[list] = []
    pr_rows: list[list] = []
    all_shas: set[str] = set()
    per_pr_comments: dict[int, list[Comment]] = {}

    for pr in prs:
        try:
            comments = fetch_pr_comments(repo, pr, bot_login_set)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to fetch comments for PR #%s: %s", pr["number"], e)
            continue
        per_pr_comments[pr["number"]] = comments
        all_shas.add(pr["headRefOid"])

    # Pull bot findings + the existing human_comments table from the shared run
    # (skip in dry-run). The human_comments rows serve double duty: the
    # classification cache below, and the replace-by-PR merge at the end.
    if dry_run:
        findings_by_sha = {sha: [] for sha in all_shas}
        existing_human_rows: list[list] = []
    else:
        wandb.init(
            project=WANDB_PROJECT,
            id=WANDB_RUN_ID,
            resume="allow",
            settings=wandb.Settings(silent=True, _disable_stats=True, _disable_meta=True),
        )
        findings_by_sha = load_findings_for_shas(wandb, all_shas)
        existing_human_rows = _load_existing_rows(wandb, "human_comments")
    cache = {} if refresh else build_classification_cache(existing_human_rows)

    # Flatten every human comment across all PRs. Reuse a cached classification
    # when the same comment_id was classified before with identical text;
    # otherwise queue it for the model. Only the queued ones are batched and
    # classified in parallel, so an overlapping daily window stays cheap.
    human_by_pr = {pr["number"]: [c for c in per_pr_comments.get(pr["number"], []) if not c.is_bot] for pr in prs}
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
                [
                    aggregator_ts,
                    n,
                    pr["title"],
                    pr["mergedAt"],
                    c.author,
                    c.comment_id,
                    c.comment_type,
                    c.file,
                    c.line,
                    c.body[:500],
                    cls.klass,
                    cls.catchable_strict,
                    cls.catchable_generous,
                    cls.confidence,
                    cls.reason,
                ]
            )

        bot_findings = findings_by_sha.get(pr["headRefOid"], [])
        pr_rows.append(
            [
                aggregator_ts,
                n,
                pr["title"],
                pr["mergedAt"],
                (pr.get("author") or {}).get("login") or "unknown",
                pr["headRefOid"],
                pr["baseRefOid"],
                len(human),
                json.dumps(by_class),
                strict_cnt,
                generous_cnt,
                len(bot_findings),
                overlap_count(bot_findings, human),
            ]
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
        click.echo(json.dumps({"pr_rollups": pr_rows, "human_comments": human_rows[:20]}, default=str, indent=2))
        return

    # Replace-by-PR: a daily cron over a rolling window re-emits rows for PRs
    # we've seen before. Drop existing rows whose pr_number is in this batch,
    # then append the fresh ones — the new rows are the source of truth for
    # those PRs. (Cached comments are re-emitted with their stored verdict, so
    # the dropped rows are reconstructed identically unless the text changed.)
    refreshed_prs = {pr["number"] for pr in prs}
    pr_col_idx = HUMAN_COMMENT_COLUMNS.index("pr_number")
    out_pr_col_idx = PR_OUTCOME_COLUMNS.index("pr_number")

    existing_humans = [r for r in existing_human_rows if r[pr_col_idx] not in refreshed_prs]
    existing_humans.extend(human_rows)
    wandb.log({"human_comments": wandb.Table(columns=HUMAN_COMMENT_COLUMNS, data=existing_humans)})

    existing_outcomes = [
        r for r in _load_existing_rows(wandb, "pr_review_outcomes") if r[out_pr_col_idx] not in refreshed_prs
    ]
    existing_outcomes.extend(pr_rows)
    wandb.log({"pr_review_outcomes": wandb.Table(columns=PR_OUTCOME_COLUMNS, data=existing_outcomes)})

    wandb.finish(quiet=True)
    logger.info(
        "Logged %d PR rollups and %d classified human comments to %s/%s",
        len(pr_rows),
        len(human_rows),
        WANDB_PROJECT,
        WANDB_RUN_ID,
    )


@cli.command()
@click.option("--repo", default=DEFAULT_REPO, show_default=True, help="Repo used to build PR links")
@click.option("--days", type=int, default=30, show_default=True, help="Report window: PRs merged in last N days")
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="Also write the markdown report here")
@click.option("--public", is_flag=True, help="Create a public gist (default: secret)")
@click.option("--no-gist", is_flag=True, help="Print the report to stdout instead of creating a gist")
def report(repo: str, days: int, out: str | None, public: bool, no_gist: bool) -> None:
    """Render the accumulated review stats into a markdown digest and gist it.

    Reads four tables from the shared `review-stats` run: `pr_review_outcomes`
    and `human_comments` (the catchability analysis) plus the linter-fed
    `invocations` and `findings` (the review bot's own activity)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    now = dt.datetime.now(dt.UTC)
    start = now - dt.timedelta(days=days)

    wandb.init(
        project=WANDB_PROJECT,
        id=WANDB_RUN_ID,
        resume="allow",
        settings=wandb.Settings(silent=True, _disable_stats=True, _disable_meta=True),
    )
    outcomes = _rows_to_dicts(PR_OUTCOME_COLUMNS, _load_existing_rows(wandb, "pr_review_outcomes"))
    comments = _rows_to_dicts(HUMAN_COMMENT_COLUMNS, _load_existing_rows(wandb, "human_comments"))
    invocations = _rows_to_dicts(INVOCATION_COLUMNS, _load_existing_rows(wandb, "invocations"))
    findings = _rows_to_dicts(FINDING_COLUMNS, _load_existing_rows(wandb, "findings"))
    wandb.finish(quiet=True)

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
