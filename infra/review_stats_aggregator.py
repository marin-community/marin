#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai>=1.0",
#     "pydantic>=2.0",
#     "wandb>=0.18",
# ]
# ///

"""Daily aggregator for the review-automation stats dashboard.

For each PR merged in the last N days:
  1. Pull review/inline/issue comments via the `gh` CLI.
  2. Drop bot comments (only humans should be in the denominator).
  3. Classify each human comment with Gemini 3.5 Flash. Two independent
     "could automation have caught this?" signals:
       - catchable_strict   — a deterministic linter / type checker / ml-*
                              catalog rule could mechanically flag it.
       - catchable_generous — a modern LLM running on the diff alone would
                              plausibly flag it with high confidence.
     Strict ⊆ generous by construction (the prompt enforces it).
  4. Pull the bot's own findings for the PR's head_sha from the existing
     `findings` artifact written by `infra/review_stats.py`.
  5. Append two flat tables to the shared `review-stats` W&B run (same run
     that `infra/review_stats.py` writes invocations + findings to):
       - human_comments      — one row per classified comment.
       - pr_review_outcomes  — one row per PR (rollup).

Same append-to-artifact pattern as `infra/review_stats.py`. Concurrent
writers race; loser's row is dropped (acceptable at our scale).

Requires GEMINI_API_KEY and W&B auth. Invoke via `gh auth login` for the
GitHub side. Designed to run as a daily GHA cron.

Usage:
    infra/review_stats_aggregator.py [--days 1] [--limit N] [--repo OWNER/REPO]
                                     [--model gemini-3.5-flash] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger("review_stats_aggregator")

WANDB_PROJECT = "marin-review-stats"
WANDB_RUN_ID = "review-stats"
DEFAULT_REPO = "marin-community/marin"
DEFAULT_MODEL = "gemini-3.5-flash"

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
# Gemini classifier schema + prompt
# ---------------------------------------------------------------------------


CommentClass = Literal["bug", "lint", "structure", "test", "doc", "design", "approval", "ack", "other"]


class CommentClassification(BaseModel):
    klass: CommentClass = Field(alias="class")
    catchable_strict: bool
    catchable_generous: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    model_config = {"populate_by_name": True}


CLASSIFIER_SYSTEM = """\
You triage human PR review comments to measure how much reviewer effort our
review automation is missing. For each comment, emit a single JSON object.

Fields:

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
                     ruff, black, pyrefly/mypy, a regex/AST rule like the
                     marin `infra/lint.md` ml-* catalog.
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
  - If catchable_strict is TRUE, catchable_generous must be TRUE.
  - approval / ack comments are never catchable.
  - When a comment is multi-issue, classify by the most material issue.
  - Prefer FALSE when uncertain — false positives erode trust in the metric.
"""


def classify_comment(
    client: genai.Client, model: str, file: str | None, line: int | None, body: str
) -> CommentClassification | None:
    """Return classification, or None if the API call fails."""
    where = f"File: {file}\nLine: {line}\n" if file else "Comment scope: top-level PR comment\n"
    prompt = f"{where}Body:\n{body.strip()}"
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFIER_SYSTEM,
                response_mime_type="application/json",
                response_schema=CommentClassification,
                temperature=0.0,
            ),
        )
    except Exception as e:
        logger.warning("Gemini call failed: %s", e)
        return None
    try:
        c = CommentClassification.model_validate_json(resp.text)
    except Exception as e:
        logger.warning("Gemini returned unparseable JSON: %s | text=%r", e, resp.text[:200])
        return None
    # Enforce the strict ⊆ generous invariant even if the model slipped.
    if c.catchable_strict and not c.catchable_generous:
        c.catchable_generous = True
    return c


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


def list_merged_prs(repo: str, days: int, limit: int) -> list[dict]:
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    search = f"is:pr is:merged merged:>={since} repo:{repo}"
    data = _gh_json(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "merged",
            "--search",
            search,
            "--limit",
            str(limit),
            "--json",
            "number,title,mergedAt,author,headRefOid,baseRefOid",
        ]
    )
    # gh returns a list of {number, title, mergedAt, author{login}, headRefOid, baseRefOid}
    return data  # type: ignore[return-value]


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
    inline = _gh_json(["api", f"repos/{repo}/pulls/{n}/comments", "--paginate"])
    for c in inline:  # type: ignore[union-attr]
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
    reviews = _gh_json(["api", f"repos/{repo}/pulls/{n}/reviews", "--paginate"])
    for r in reviews:  # type: ignore[union-attr]
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
    issue_comments = _gh_json(["api", f"repos/{repo}/issues/{n}/comments", "--paginate"])
    for c in issue_comments:  # type: ignore[union-attr]
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


def load_findings_for_shas(wandb, shas: set[str]) -> dict[str, list[dict]]:
    """Fetch bot findings rows for the given head_shas from the shared run.

    Returns sha -> list of finding dicts (file, line, code, confidence, message).
    """
    by_sha: dict[str, list[dict]] = {sha: [] for sha in shas}
    # FINDING_COLUMNS layout from review_stats.py:
    #   ts, invocation_id, tool, pr_number, git_branch, head_sha, marin_user,
    #   file, line, code, confidence, message
    for r in _load_existing_rows(wandb, "findings"):
        sha = r[5]
        if sha in by_sha:
            by_sha[sha].append({"file": r[7], "line": r[8], "code": r[9], "confidence": r[10], "message": r[11]})
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
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--days", type=int, default=1, help="Look back N days of merged PRs")
    parser.add_argument("--limit", type=int, default=100, help="Max PRs to process")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--bot-logins",
        default="github-actions,dependabot,claude,claude-review,renovate",
        help="Comma-separated bot logins to skip (lowercase)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip W&B upload; print rollup")
    args = parser.parse_args()

    bot_logins = {x.strip().lower() for x in args.bot_logins.split(",") if x.strip()}

    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set")
        return 2

    client = genai.Client()

    logger.info("Listing PRs merged in last %d day(s) in %s", args.days, args.repo)
    prs = list_merged_prs(args.repo, args.days, args.limit)
    logger.info("Found %d merged PRs", len(prs))

    aggregator_ts = dt.datetime.now(dt.timezone.utc).isoformat()

    human_rows: list[list] = []
    pr_rows: list[list] = []
    all_shas: set[str] = set()
    per_pr_comments: dict[int, list[Comment]] = {}

    for pr in prs:
        try:
            comments = fetch_pr_comments(args.repo, pr, bot_logins)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to fetch comments for PR #%s: %s", pr["number"], e)
            continue
        per_pr_comments[pr["number"]] = comments
        all_shas.add(pr["headRefOid"])

    # Pull bot findings for these shas from the shared run.
    if not args.dry_run:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            id=WANDB_RUN_ID,
            resume="allow",
            settings=wandb.Settings(silent=True, _disable_stats=True, _disable_meta=True),
        )
        findings_by_sha = load_findings_for_shas(wandb, all_shas)
    else:
        wandb = None  # type: ignore[assignment]
        findings_by_sha = {sha: [] for sha in all_shas}

    # Classify + build rows.
    for pr in prs:
        n = pr["number"]
        comments = per_pr_comments.get(n, [])
        human = [c for c in comments if not c.is_bot]
        by_class: dict[str, int] = {}
        strict_cnt = generous_cnt = 0

        for c in human:
            cls = classify_comment(client, args.model, c.file, c.line, c.body)
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

    if args.dry_run:
        print(json.dumps({"pr_rollups": pr_rows, "human_comments": human_rows[:20]}, default=str, indent=2))
        return 0

    # Replace-by-PR: a daily cron over a rolling window re-classifies PRs we've
    # seen before. Drop existing rows whose pr_number is in this batch, then
    # append the fresh ones — the new rows are the source of truth for those PRs.
    refreshed_prs = {pr["number"] for pr in prs}
    pr_col_idx = HUMAN_COMMENT_COLUMNS.index("pr_number")
    out_pr_col_idx = PR_OUTCOME_COLUMNS.index("pr_number")

    existing_humans = [r for r in _load_existing_rows(wandb, "human_comments") if r[pr_col_idx] not in refreshed_prs]
    existing_humans.extend(human_rows)
    wandb.log(  # type: ignore[union-attr]
        {"human_comments": wandb.Table(columns=HUMAN_COMMENT_COLUMNS, data=existing_humans)}
    )

    existing_outcomes = [
        r for r in _load_existing_rows(wandb, "pr_review_outcomes") if r[out_pr_col_idx] not in refreshed_prs
    ]
    existing_outcomes.extend(pr_rows)
    wandb.log(  # type: ignore[union-attr]
        {"pr_review_outcomes": wandb.Table(columns=PR_OUTCOME_COLUMNS, data=existing_outcomes)}
    )

    wandb.finish(quiet=True)  # type: ignore[union-attr]
    logger.info(
        "Logged %d PR rollups and %d classified human comments to %s/%s",
        len(pr_rows),
        len(human_rows),
        WANDB_PROJECT,
        WANDB_RUN_ID,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
