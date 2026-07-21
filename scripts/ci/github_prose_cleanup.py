#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and publish agent-edited GitHub descriptions."""

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

CLEANUP_FOOTER_MARKER = "<!-- marin-prose-cleanup -->"
GITHUB_BODY_SIZE_LIMIT = 65_536
UPDATED_DESCRIPTION_FOOTER_RESERVE = 512

_ARCHIVE_COMMENT_PREFIX = "🤖 Archived the description before automated prose cleanup."
_ARCHIVE_MARKER_PREFIX = "marin-prose-cleanup-archive"
_CLEANUP_FOOTER_RE = re.compile(
    r"\n{2,}---\n" r"\[Original description\]\([^\n)]+\)\s+" + re.escape(CLEANUP_FOOTER_MARKER) + r"\s*$"
)
_FENCE_START_RE = re.compile(r"^(?: {0,3})(?P<fence>`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+[^`\n]*?`+)")
_ATX_HEADING_RE = re.compile(
    r"^[ \t]{0,3}#{1,6}(?:[ \t]+(?P<heading_text>[^\n]*?))?[ \t]*#*[ \t]*(?:\n|$)", re.MULTILINE
)
_HTML_HEADING_LINE_RE = re.compile(
    r"^[ \t]*<h[1-6](?:\s+[^>\n]*)?>(?P<heading_text>[^\n]*?)</h[1-6]\s*>[ \t]*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
_STANDALONE_BOLD_LABEL_RE = re.compile(
    r"^[ \t]*(?:\*\*(?P<asterisk_heading>[^*\n]+)\*\*|__(?P<underscore_heading>[^_\n]+)__)[ \t]*(?:\n|$)",
    re.MULTILINE,
)
_BLOCK_HTML_CLOSE_RE = re.compile(r"</(?:h[1-6]|p|div|center)\s*>", re.IGNORECASE)
_BLOCK_HTML_OPEN_RE = re.compile(r"<(?:h[1-6]|p|div|center)(?:\s+[^>]*)?>", re.IGNORECASE)
_INLINE_HTML_RE = re.compile(r"</?(?:b|strong|em|i|u|font|span)(?:\s+[^>]*)?>", re.IGNORECASE)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_DECORATIVE_EMOJI_RE = re.compile(r"[🚀🔥✨🎉💥🤯🌟]+")
_HYPE_ONLY_LINE_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:[*_]{1,2})?"
    r"(?:oh my god|omg|wow|amazing|incredible|awesome|boom|let['\u2019]?s go|let['\u2019]?s dive in|"
    r"here['\u2019]?s the exciting part|this changes everything|this is huge|this is a game[- ]changer|"
    r"this is where things get interesting|this is the part many people miss)"
    r"[!.,:; \t]*(?:[*_]{1,2})?[!.,:; \t]*(?:[🚀🔥✨🎉💥🤯🌟]+)?[ \t]*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
_HYPE_PREFIX_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?:(?:[-*+]|\d+\.)[ \t]+|>[ \t]*)?)"
    r"(?:[*_]{1,2})?(?:oh my god|omg|wow)(?:[*_]{1,2})?[!.,:; \t]*"
    r"(?:[\u2014\u2013-][ \t]*)?(?P<first>[a-z])",
    re.IGNORECASE | re.MULTILINE,
)
_FRAMING_OPENER_RE = re.compile(
    r"(?P<prefix>^[ \t]*(?:(?:[-*+]|\d+\.)[ \t]+|>[ \t]*)?|(?<=[.!?])[ \t]+)"
    r"(?:it is worth noting that|what this means is(?: that)?|the key takeaway(?: here)? is(?: that)?|"
    r"at its core,?|importantly,?|notably,?|crucially,?|interestingly,?|remarkably,?|"
    r"the main change(?: here)? is that|what changed is that|the current state is that)"
    r"[ \t]+(?P<first>[a-z])",
    re.IGNORECASE | re.MULTILINE,
)
_NOT_JUST_RE = re.compile(
    r"\bnot\s+(?:just|only|merely)\s+(?P<first>[^,.;\n]+?)\s*,?\s+" r"but(?:\s+also)?\s+(?P<second>[^.;\n]+)",
    re.IGNORECASE,
)
_THIS_NOT_THAT_RE = re.compile(
    r"\bthis is not\s+(?P<first>[^,.;\n]+?)\s*,?\s+but\s+(?P<second>[^.;\n]+)",
    re.IGNORECASE,
)
_BOLD_RE = re.compile(r"\*\*(?P<asterisk_content>[^*\n]+)\*\*|__(?P<underscore_content>[^_\n]+)__")
_CHECKBOX_RE = re.compile(r"^(?P<prefix>[ \t]*(?:[-*+]|\d+\.)[ \t]+)\[[ xX]\][ \t]+", re.MULTILINE)
_FILLER_OPENER_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?:(?:[-*+]|\d+\.)[ \t]+|>[ \t]*)?)"
    r"(?:this (?:pull request|pr|change)\s+|in this (?:pull request|pr|change),?[ \t]+)"
    r"(?P<first>[a-z])",
    re.IGNORECASE | re.MULTILINE,
)
_GENERIC_HEADING_RE = re.compile(
    r"^(?:summary|what|what changed|changes|key pieces|implementation|details|testing|tests|test plan|"
    r"validation|verification|performance|results|status|context|background|compatibility|reproduction|"
    r"reproduce it|completion criteria|acceptance criteria|problem|fix|solution|paired with)$|"
    r"^(?:parity verdict|why this\b)",
    re.IGNORECASE,
)
_PATH_HEADING_RE = re.compile(r"^`?[\w./-]+\.(?:py|md|rst|toml|ya?ml|json|ts|tsx|js|rs|go|sh)`?$", re.IGNORECASE)


class GithubItemKind(StrEnum):
    """GitHub description types supported by the cleanup workflow."""

    ISSUE = "issue"
    PULL_REQUEST = "pull_request"


@dataclass(frozen=True)
class BodyCleanup:
    """Proposed description text and whether it differs from the input."""

    cleaned_body: str
    changed: bool


@dataclass(frozen=True)
class CleanupResult:
    """Cleanup and archive payload consumed by the GitHub Actions wrapper."""

    kind: GithubItemKind
    number: int
    changed: bool
    cleaned_body: str
    original_body_hash: str
    archive_comment_body: str = ""
    archive_marker: str = ""
    skip_reason: str | None = None


def _capitalize_replacement(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}{match.group('first').upper()}"


def _heading_replacement(heading_text: str) -> str:
    heading_text = heading_text.strip()
    normalized = heading_text.strip("#*_` :.!?").strip()
    if not normalized or _GENERIC_HEADING_RE.search(normalized) or _PATH_HEADING_RE.fullmatch(normalized):
        return ""
    return f"{heading_text}\n"


def _replace_atx_or_html_heading(match: re.Match[str]) -> str:
    return _heading_replacement(match.group("heading_text") or "")


def _replace_bold_heading(match: re.Match[str]) -> str:
    heading_text = match.group("asterisk_heading") or match.group("underscore_heading") or ""
    return _heading_replacement(heading_text)


def _replace_not_just(match: re.Match[str]) -> str:
    return f"{match.group('first').strip()} and {match.group('second').strip()}"


def _replace_this_not_that(match: re.Match[str]) -> str:
    first = match.group("first").strip()
    second = match.group("second").strip()
    return f"This is {second}. It is not {first}"


def _bold_span_is_structural(line: str, match: re.Match[str]) -> bool:
    content = (match.group("asterisk_content") or match.group("underscore_content")).strip()
    before = line[: match.start()]
    after = line[match.end() :]
    line_is_label = not before.strip() and not after.strip() and len(content.split()) <= 4
    list_label = bool(re.fullmatch(r"[ \t]*(?:(?:[-*+]|\d+\.)[ \t]+)?", before)) and (
        content.endswith(":") or after.lstrip().startswith(":")
    )
    return line_is_label or list_label


def _strip_decorative_bold(text: str) -> str:
    def clean_line(line: str) -> str:
        def replace(match: re.Match[str]) -> str:
            if _bold_span_is_structural(line, match):
                return match.group(0)
            return match.group("asterisk_content") or match.group("underscore_content")

        return _BOLD_RE.sub(replace, line)

    return "".join(clean_line(line) for line in text.splitlines(keepends=True))


def _clean_plain_text(text: str) -> str:
    cleaned = _ATX_HEADING_RE.sub(_replace_atx_or_html_heading, text)
    cleaned = _HTML_HEADING_LINE_RE.sub(_replace_atx_or_html_heading, cleaned)
    cleaned = _STANDALONE_BOLD_LABEL_RE.sub(_replace_bold_heading, cleaned)
    cleaned = _HTML_BREAK_RE.sub("\n", cleaned)
    cleaned = _BLOCK_HTML_CLOSE_RE.sub("", cleaned)
    cleaned = _BLOCK_HTML_OPEN_RE.sub("", cleaned)
    cleaned = _INLINE_HTML_RE.sub("", cleaned)
    cleaned = _DECORATIVE_EMOJI_RE.sub("", cleaned)
    cleaned = _HYPE_ONLY_LINE_RE.sub("", cleaned)
    cleaned = _HYPE_PREFIX_RE.sub(_capitalize_replacement, cleaned)
    cleaned = _FRAMING_OPENER_RE.sub(_capitalize_replacement, cleaned)
    cleaned = _NOT_JUST_RE.sub(_replace_not_just, cleaned)
    cleaned = _THIS_NOT_THAT_RE.sub(_replace_this_not_that, cleaned)
    cleaned = _CHECKBOX_RE.sub(r"\g<prefix>", cleaned)
    cleaned = _FILLER_OPENER_RE.sub(_capitalize_replacement, cleaned)
    return _strip_decorative_bold(cleaned)


def _clean_outside_inline_code(text: str) -> str:
    parts = _INLINE_CODE_RE.split(text)
    cleaned_parts = [part if index % 2 else _clean_plain_text(part) for index, part in enumerate(parts)]
    cleaned = "".join(cleaned_parts)
    if cleaned == text:
        return text
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _clean_outside_fenced_code(body: str) -> str:
    output: list[str] = []
    prose: list[str] = []
    fence_character: str | None = None
    fence_length = 0

    def flush_prose() -> None:
        if prose:
            output.append(_clean_outside_inline_code("".join(prose)))
            prose.clear()

    for line in body.splitlines(keepends=True):
        fence_match = _FENCE_START_RE.match(line)
        if fence_character is None:
            if fence_match is None:
                prose.append(line)
                continue
            flush_prose()
            fence = fence_match.group("fence")
            fence_character = fence[0]
            fence_length = len(fence)
            output.append(line)
            continue

        output.append(line)
        if fence_match is None:
            continue
        fence = fence_match.group("fence")
        if fence[0] == fence_character and len(fence) >= fence_length and not line[fence_match.end() :].strip():
            fence_character = None
            fence_length = 0

    flush_prose()
    return "".join(output)


def _body_without_cleanup_footer(body: str) -> str:
    return _CLEANUP_FOOTER_RE.sub("", body).rstrip()


def cleanup_github_body(body: str) -> BodyCleanup:
    """Apply mechanical safeguards to an agent-edited GitHub body."""
    body_without_footer = _body_without_cleanup_footer(body)
    cleaned_body = _clean_outside_fenced_code(body_without_footer)
    if cleaned_body == body_without_footer:
        return BodyCleanup(cleaned_body=body, changed=False)
    return BodyCleanup(cleaned_body=cleaned_body.strip(), changed=True)


def _archive_fence(original_body: str) -> str:
    def longest_run(character: str) -> int:
        return max((len(run) for run in re.findall(re.escape(character) + r"+", original_body)), default=0)

    backtick_length = max(3, longest_run("`") + 1)
    tilde_length = max(3, longest_run("~") + 1)
    if backtick_length <= tilde_length:
        return "`" * backtick_length
    return "~" * tilde_length


def _archive_comment(original_body: str, archive_marker: str) -> str:
    fence = _archive_fence(original_body)
    return (
        f"{_ARCHIVE_COMMENT_PREFIX}\n"
        f"{archive_marker}\n\n"
        "<details><summary>Original description</summary>\n\n"
        f"{fence}markdown\n{original_body}\n{fence}\n\n"
        "</details>"
    )


def _event_item(event: dict[str, Any]) -> tuple[GithubItemKind, dict[str, Any]]:
    pull_request = event.get("pull_request")
    if isinstance(pull_request, dict):
        return GithubItemKind.PULL_REQUEST, pull_request
    issue = event.get("issue")
    if isinstance(issue, dict):
        return GithubItemKind.ISSUE, issue
    raise ValueError("Event does not contain an issue or pull request")


def cleanup_result_from_event(event: dict[str, Any], rewritten_body: str) -> CleanupResult:
    """Build an archive payload from an event and an agent-edited body."""
    kind, item = _event_item(event)
    number = int(item["number"])
    original_body_value = item.get("body")
    original_body = original_body_value if isinstance(original_body_value, str) else ""
    body_hash = hashlib.sha256(original_body.encode("utf-8")).hexdigest()
    candidate_body = _body_without_cleanup_footer(rewritten_body).strip()
    if original_body.strip() and not candidate_body:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
            skip_reason="rewrite_empty",
        )

    body_cleanup = cleanup_github_body(candidate_body)
    original_without_footer = _body_without_cleanup_footer(original_body).strip()
    if body_cleanup.cleaned_body == original_without_footer:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
        )

    archive_marker = f"<!-- {_ARCHIVE_MARKER_PREFIX}:{body_hash} -->"
    archive_comment_body = _archive_comment(original_body, archive_marker)
    if len(archive_comment_body.encode("utf-8")) > GITHUB_BODY_SIZE_LIMIT:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
            skip_reason="archive_too_large",
        )
    if len(body_cleanup.cleaned_body.encode("utf-8")) > GITHUB_BODY_SIZE_LIMIT - UPDATED_DESCRIPTION_FOOTER_RESERVE:
        return CleanupResult(
            kind=kind,
            number=number,
            changed=False,
            cleaned_body=original_body,
            original_body_hash=body_hash,
            skip_reason="description_too_large",
        )

    return CleanupResult(
        kind=kind,
        number=number,
        changed=True,
        cleaned_body=body_cleanup.cleaned_body,
        original_body_hash=body_hash,
        archive_comment_body=archive_comment_body,
        archive_marker=archive_marker,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True, help="GitHub event JSON")
    parser.add_argument("--output", type=Path, required=True, help="Cleanup result JSON")
    parser.add_argument("--rewritten-body-env", required=True, help="Environment variable containing the agent edit")
    parser.add_argument("--github-output", type=Path, help="GitHub Actions output file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    event = json.loads(args.event.read_text(encoding="utf-8"))
    result = cleanup_result_from_event(event, os.environ[args.rewritten_body_env])
    args.output.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as github_output:
            github_output.write(f"changed={str(result.changed).lower()}\n")
            github_output.write(f"result={args.output}\n")
    status = "cleanup required" if result.changed else result.skip_reason or "no cleanup needed"
    print(f"{result.kind} #{result.number}: {status}")


if __name__ == "__main__":
    main()
