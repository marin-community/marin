#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["PyGithub>=2.3.0", "openai>=1.0.0"]
# ///
"""Nightly scrub that maintains newcomer-friendly TL;DR blocks on experiment issues."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass

from github import Github, GithubException
from github.Issue import Issue
from github.Repository import Repository
from openai import OpenAI

LOGGER = logging.getLogger(__name__)

EXPERIMENT_LABEL = "experiment"
TLDR_LABEL = "tldr"
DOCUMENTATION_LABEL = "documentation"
AGENT_GENERATED_LABEL = "agent-generated"
SUMMARY_START = "<!-- experiment-tldr:start -->"
SUMMARY_END = "<!-- experiment-tldr:end -->"
DOC_ISSUE_MARKER_PREFIX = "<!-- experiment-tldr:doc-issue="
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_COMMENT_COUNT = 15
MAX_SUMMARY_WORDS = 100
TLDR_LABEL_COLOR = "1d76db"
TLDR_LABEL_DESCRIPTION = "Experiment issue includes a newcomer-friendly TL;DR and enough supporting context."
BODY_BLOCK_RE = re.compile(
    rf"\n?{re.escape(SUMMARY_START)}.*?{re.escape(SUMMARY_END)}\n?",
    re.DOTALL,
)
DOC_ISSUE_MARKER_RE = re.compile(r"<!-- experiment-tldr:doc-issue=(\d+) -->")
URL_RE = re.compile(r"https?://[^\s>)\]]+")


@dataclass(frozen=True)
class IssueAnalysis:
    """Structured LLM output for one experiment issue."""

    summary: str
    documentation_sufficient: bool
    relevant_links: list[str]
    needs_doc_issue: bool
    doc_issue_title: str | None
    doc_issue_body: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="marin-community/marin", help="GitHub repo in owner/name form.")
    parser.add_argument("--issue-number", type=int, help="Only scrub a single issue number.")
    parser.add_argument("--max-issues", type=int, default=20, help="Maximum issues to inspect in one run.")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Refresh issues that already have a managed TL;DR block instead of skipping them.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="Model used for summarization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute changes without mutating GitHub issues or labels.",
    )
    return parser.parse_args()


def ensure_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def extract_urls(text: str) -> list[str]:
    return dedupe_preserving_order(URL_RE.findall(text or ""))


def parse_analysis(raw_text: str) -> IssueAnalysis:
    content = raw_text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    payload = json.loads(content)
    return IssueAnalysis(
        summary=str(payload["summary"]).strip(),
        documentation_sufficient=bool(payload["documentation_sufficient"]),
        relevant_links=[str(link).strip() for link in payload.get("relevant_links", []) if str(link).strip()],
        needs_doc_issue=bool(payload.get("needs_doc_issue", False)),
        doc_issue_title=_optional_string(payload.get("doc_issue_title")),
        doc_issue_body=_optional_string(payload.get("doc_issue_body")),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def sanitize_analysis(analysis: IssueAnalysis, allowed_links: list[str]) -> IssueAnalysis:
    if len(words(analysis.summary)) > MAX_SUMMARY_WORDS:
        raise ValueError(f"Summary exceeds {MAX_SUMMARY_WORDS} words: {analysis.summary!r}")

    allowed = set(allowed_links)
    relevant_links = [link for link in analysis.relevant_links if link in allowed]
    relevant_links = dedupe_preserving_order(relevant_links)

    needs_doc_issue = analysis.needs_doc_issue and not analysis.documentation_sufficient
    doc_issue_title = analysis.doc_issue_title if needs_doc_issue else None
    doc_issue_body = analysis.doc_issue_body if needs_doc_issue else None
    if needs_doc_issue and (not doc_issue_title or not doc_issue_body):
        raise ValueError("Documentation gap issues require both a title and a body.")

    return IssueAnalysis(
        summary=analysis.summary,
        documentation_sufficient=analysis.documentation_sufficient,
        relevant_links=relevant_links,
        needs_doc_issue=needs_doc_issue,
        doc_issue_title=doc_issue_title,
        doc_issue_body=doc_issue_body,
    )


def issue_has_managed_block(body: str | None) -> bool:
    return bool(body and SUMMARY_START in body and SUMMARY_END in body)


def extract_existing_doc_issue_number(body: str | None) -> int | None:
    if not body:
        return None
    match = DOC_ISSUE_MARKER_RE.search(body)
    if match is None:
        return None
    return int(match.group(1))


def render_tldr_block(analysis: IssueAnalysis, *, doc_issue_number: int | None) -> str:
    lines = [SUMMARY_START, "## TL;DR", analysis.summary]
    if analysis.relevant_links:
        lines.append("")
        lines.append("### Helpful links")
        for link in analysis.relevant_links:
            lines.append(f"- {link}")

    lines.append("")
    lines.append("### Documentation status")
    if analysis.documentation_sufficient:
        lines.append("- Sufficiently documented for a newcomer who knows the field but not Marin.")
    elif doc_issue_number is not None:
        lines.append(f"- More context is still needed. Follow-up: #{doc_issue_number}.")
        lines.append(f"{DOC_ISSUE_MARKER_PREFIX}{doc_issue_number} -->")
    else:
        lines.append("- More context is still needed.")
    lines.append(SUMMARY_END)
    return "\n".join(lines)


def upsert_tldr_block(body: str | None, block: str) -> str:
    if not body:
        return block
    if BODY_BLOCK_RE.search(body):
        updated = BODY_BLOCK_RE.sub(f"\n{block}\n", body).strip()
        return updated
    return f"{body.rstrip()}\n\n{block}"


def collect_issue_context(issue: Issue) -> tuple[str, list[str]]:
    comments = list(issue.get_comments())[:MAX_COMMENT_COUNT]
    comment_blocks = []
    candidate_links = extract_urls(issue.body or "")

    for comment in comments:
        candidate_links.extend(extract_urls(comment.body or ""))
        comment_author = json.dumps(comment.user.login)
        created_at = json.dumps(comment.created_at.isoformat())
        comment_blocks.append(
            "\n".join(
                [
                    f"<comment author={comment_author} created_at={created_at}>",
                    comment.body or "",
                    "</comment>",
                ]
            )
        )

    context = "\n".join(
        [
            f"<issue number={issue.number} state={json.dumps(issue.state)}>",
            f"<title>{issue.title}</title>",
            f"<labels>{','.join(label.name for label in issue.labels)}</labels>",
            "<body>",
            issue.body or "",
            "</body>",
            "<comments>",
            "\n".join(comment_blocks),
            "</comments>",
            "</issue>",
        ]
    )
    return context, dedupe_preserving_order(candidate_links)


def analyze_issue(client: OpenAI, *, issue: Issue, model: str) -> IssueAnalysis:
    issue_context, candidate_links = collect_issue_context(issue)
    prompt = "\n".join(
        [
            "You maintain GitHub experiment issues for the Marin project.",
            (
                "Write a TL;DR aimed at a smart college junior or new research assistant "
                "who understands the field but not this codebase."
            ),
            f"The TL;DR must be at most {MAX_SUMMARY_WORDS} words.",
            "Decide whether the issue plus its linked material is sufficiently documented for that audience.",
            "If more context is needed, draft a documentation request issue.",
            "Only choose relevant_links from the candidate URLs provided below.",
            (
                'Return JSON with keys: "summary", "documentation_sufficient", '
                '"relevant_links", "needs_doc_issue", "doc_issue_title", "doc_issue_body".'
            ),
            "Do not wrap the JSON in markdown fences.",
            "",
            "<candidate_urls>",
            json.dumps(candidate_links, indent=2),
            "</candidate_urls>",
            "",
            issue_context,
        ]
    )
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a precise assistant that returns compact JSON for GitHub issue maintenance.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content or ""
    analysis = parse_analysis(raw)
    return sanitize_analysis(analysis, candidate_links)


def ensure_label(repo: Repository, name: str, *, color: str, description: str) -> None:
    try:
        repo.get_label(name)
    except GithubException as exc:
        if exc.status != 404:
            raise
        LOGGER.info("Creating missing label '%s'", name)
        repo.create_label(name=name, color=color, description=description)


def find_existing_doc_issue(repo: Repository, *, title: str) -> Issue | None:
    try:
        issues = repo.get_issues(state="open", labels=[repo.get_label(DOCUMENTATION_LABEL)])
    except GithubException as exc:
        if exc.status == 404:
            issues = repo.get_issues(state="open")
        else:
            raise

    for issue in issues:
        if issue.pull_request is None and issue.title == title:
            return issue
    return None


def ensure_doc_issue(repo: Repository, source_issue: Issue, analysis: IssueAnalysis, *, dry_run: bool) -> int | None:
    if not analysis.needs_doc_issue or not analysis.doc_issue_title or not analysis.doc_issue_body:
        return None

    existing_number = extract_existing_doc_issue_number(source_issue.body)
    if existing_number is not None:
        return existing_number

    existing = find_existing_doc_issue(repo, title=analysis.doc_issue_title)
    if existing is not None:
        return existing.number

    if dry_run:
        LOGGER.info("Would create documentation gap issue for #%s: %s", source_issue.number, analysis.doc_issue_title)
        return None

    doc_issue_body = f"{analysis.doc_issue_body.rstrip()}\n\nRefs #{source_issue.number}"
    ensure_label(
        repo,
        DOCUMENTATION_LABEL,
        color="0075ca",
        description="Documentation improvements and requests.",
    )
    ensure_label(
        repo,
        AGENT_GENERATED_LABEL,
        color="5319e7",
        description="Created automatically by an agent.",
    )
    created = repo.create_issue(
        title=analysis.doc_issue_title,
        body=doc_issue_body,
        labels=[DOCUMENTATION_LABEL, AGENT_GENERATED_LABEL],
    )
    LOGGER.info("Created documentation gap issue #%s for experiment issue #%s", created.number, source_issue.number)
    return created.number


def sync_tldr_label(issue: Issue, *, documentation_sufficient: bool, dry_run: bool) -> None:
    existing_labels = {label.name for label in issue.labels}
    if documentation_sufficient and TLDR_LABEL not in existing_labels:
        if dry_run:
            LOGGER.info("Would add '%s' label to issue #%s", TLDR_LABEL, issue.number)
        else:
            issue.add_to_labels(TLDR_LABEL)
    if not documentation_sufficient and TLDR_LABEL in existing_labels:
        if dry_run:
            LOGGER.info("Would remove '%s' label from issue #%s", TLDR_LABEL, issue.number)
        else:
            issue.remove_from_labels(TLDR_LABEL)


def issue_candidates(repo: Repository, args: argparse.Namespace) -> list[Issue]:
    if args.issue_number is not None:
        issue = repo.get_issue(args.issue_number)
        return [issue]

    candidates: list[Issue] = []
    for issue in repo.get_issues(
        state="all", labels=[repo.get_label(EXPERIMENT_LABEL)], sort="updated", direction="desc"
    ):
        if issue.pull_request is not None:
            continue
        if not args.refresh_existing and issue_has_managed_block(issue.body):
            continue
        candidates.append(issue)
        if len(candidates) >= args.max_issues:
            break
    return candidates


def process_issue(repo: Repository, client: OpenAI, issue: Issue, args: argparse.Namespace) -> None:
    LOGGER.info("Processing issue #%s: %s", issue.number, issue.title)
    analysis = analyze_issue(client, issue=issue, model=args.model)
    doc_issue_number = ensure_doc_issue(repo, issue, analysis, dry_run=args.dry_run)
    block = render_tldr_block(analysis, doc_issue_number=doc_issue_number)
    updated_body = upsert_tldr_block(issue.body, block)

    if updated_body != (issue.body or ""):
        if args.dry_run:
            LOGGER.info("Would update issue body for #%s", issue.number)
        else:
            issue.edit(body=updated_body)

    sync_tldr_label(issue, documentation_sufficient=analysis.documentation_sufficient, dry_run=args.dry_run)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    github_token = ensure_env("GITHUB_TOKEN")
    openai_api_key = ensure_env("OPENAI_API_KEY")

    github = Github(github_token)
    repo = github.get_repo(args.repo)
    ensure_label(repo, TLDR_LABEL, color=TLDR_LABEL_COLOR, description=TLDR_LABEL_DESCRIPTION)

    client = OpenAI(api_key=openai_api_key)
    candidates = issue_candidates(repo, args)
    LOGGER.info("Found %s issue(s) to inspect", len(candidates))

    for issue in candidates:
        try:
            process_issue(repo, client, issue, args)
        except Exception:
            LOGGER.exception("Failed to scrub issue #%s", issue.number)


if __name__ == "__main__":
    main()
