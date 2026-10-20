#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Read and update a goals Google Doc and its GitHub project board for the `pm` Loom profile.

`write-status` replaces everything below the doc's `Status` heading with a Markdown subset:
paragraphs, `#`-`###` headings, `- ` items nested by two spaces, `**bold**`, `[text](url)`.
GitHub calls use the token in `BOARD_GITHUB_TOKEN`, which needs organization project write
access. Docs calls impersonate a service account because the VM metadata token lacks the
documents scope.
"""

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

GITHUB_API = "https://api.github.com"
DOCS_API = "https://docs.googleapis.com/v1/documents"
DOCS_SCOPE = "https://www.googleapis.com/auth/documents"
STATUS_HEADING = "Status"
SECTION_HEADING_STYLES = frozenset({"TITLE", "HEADING_1", "HEADING_2"})
LIST_PRESET = "NUMBERED_DECIMAL_ALPHA_ROMAN"
MAX_LIST_LEVEL = 2
MAX_HEADING_LEVEL = 3
BOARD_TOKEN_ENV = "BOARD_GITHUB_TOKEN"
LINK_COLOR = {"color": {"rgbColor": {"red": 0.06666667, "green": 0.33333334, "blue": 0.8}}}
NO_GOAL = "No goal"
INLINE_RE = re.compile(r"\*\*(?P<bold>.+?)\*\*|\[(?P<text>[^\]]+)\]\((?P<url>[^)\s]+)\)")
LIST_ITEM_RE = re.compile(r"^(?P<indent> *)- (?P<text>.+)$")
HEADING_RE = re.compile(r"^(?P<marks>#+) (?P<text>.+)$")
ISSUE_REF_RE = re.compile(r"^(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)#(?P<number>\d+)$")


@dataclass(frozen=True)
class Run:
    text: str
    bold: bool = False
    url: str | None = None


@dataclass(frozen=True)
class Block:
    """One paragraph; `level` is the list nesting level and `heading` the heading level, if any."""

    runs: tuple[Run, ...]
    level: int | None = None
    heading: int | None = None


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def parse_inline(text: str) -> tuple[Run, ...]:
    runs = []
    position = 0
    for match in INLINE_RE.finditer(text):
        if match.start() > position:
            runs.append(Run(text[position : match.start()]))
        if match["bold"] is not None:
            runs.append(Run(match["bold"], bold=True))
        else:
            runs.append(Run(match["text"], url=match["url"]))
        position = match.end()
    if position < len(text):
        runs.append(Run(text[position:]))
    return tuple(runs)


def parse_status(markdown: str) -> tuple[Block, ...]:
    """Parse the Markdown subset accepted for the content below the Status heading."""
    blocks = []
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        if not line.strip():
            continue
        if "\t" in line:
            raise ValueError(f"line {line_number}: use spaces, not tabs")
        heading = HEADING_RE.match(line)
        if heading is not None:
            level = len(heading["marks"])
            if level > MAX_HEADING_LEVEL:
                raise ValueError(f"line {line_number}: use at most {MAX_HEADING_LEVEL} '#' for a heading")
            if heading["text"].strip() == STATUS_HEADING:
                raise ValueError(f"line {line_number}: the document already has the {STATUS_HEADING!r} heading")
            blocks.append(Block(parse_inline(heading["text"]), heading=level))
            continue
        item = LIST_ITEM_RE.match(line)
        if item is None:
            if line.startswith(" ") or line.startswith("#"):
                raise ValueError(f"line {line_number}: expected a paragraph, a heading, or a list item: {line!r}")
            blocks.append(Block(parse_inline(line)))
            continue
        indent = len(item["indent"])
        if indent % 2 or indent // 2 > MAX_LIST_LEVEL:
            raise ValueError(f"line {line_number}: nest list items by 2 spaces, at most {MAX_LIST_LEVEL} levels")
        level = indent // 2
        previous_level = blocks[-1].level if blocks and blocks[-1].level is not None else -1
        if level > previous_level + 1:
            raise ValueError(f"line {line_number}: list item skips a nesting level")
        blocks.append(Block(parse_inline(item["text"]), level=level))
    if not blocks:
        raise ValueError("status is empty")
    return tuple(blocks)


def _paragraph_text(paragraph: dict) -> str:
    return "".join(element.get("textRun", {}).get("content", "") for element in paragraph["elements"])


def status_region(body: dict) -> tuple[int, int]:
    """Return the body index range from the end of the Status heading to the end of the document."""
    content = body["content"]
    headings = [
        element
        for element in content
        if "paragraph" in element
        and element["paragraph"]["paragraphStyle"].get("namedStyleType") in SECTION_HEADING_STYLES
        and _paragraph_text(element["paragraph"]).strip() == STATUS_HEADING
    ]
    if len(headings) != 1:
        raise ValueError(f"expected one {STATUS_HEADING!r} heading, found {len(headings)}")
    start, end = headings[0]["endIndex"], content[-1]["endIndex"]
    if start >= end:
        raise ValueError(f"add one paragraph under the {STATUS_HEADING!r} heading first")
    return start, end


def _range(start: int, end: int) -> dict:
    return {"range": {"startIndex": start, "endIndex": end}}


def status_requests(body: dict, blocks: tuple[Block, ...]) -> list[dict]:
    """Build Docs API batchUpdate requests that replace everything below the Status heading."""
    start, body_end = status_region(body)
    lines = ["\t" * (block.level or 0) + "".join(run.text for run in block.runs) for block in blocks]
    # The body's final newline cannot be deleted; it ends the last inserted paragraph.
    text = "\n".join(lines)
    delete_end = body_end - 1
    end = start + utf16_len(text)

    requests: list[dict] = []
    if delete_end > start:
        requests.append({"deleteContentRange": _range(start, delete_end)})
    requests += [
        {"insertText": {"location": {"index": start}, "text": text}},
        {"deleteParagraphBullets": _range(start, end)},
        {
            "updateParagraphStyle": {
                **_range(start, end),
                "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                "fields": "namedStyleType,alignment,indentStart,indentFirstLine",
            }
        },
        {
            "updateTextStyle": {
                **_range(start, end),
                "textStyle": {},
                "fields": "bold,italic,underline,strikethrough,link,foregroundColor,backgroundColor",
            }
        },
    ]

    list_runs: list[tuple[int, int]] = []
    position = start
    for block in blocks:
        line_start = position
        if block.heading is not None:
            requests.append(
                {
                    "updateParagraphStyle": {
                        **_range(line_start, line_start + 1),
                        "paragraphStyle": {"namedStyleType": f"HEADING_{block.heading}"},
                        "fields": "namedStyleType",
                    }
                }
            )
        position += utf16_len("\t" * (block.level or 0))
        for run in block.runs:
            run_end = position + utf16_len(run.text)
            if run.bold:
                requests.append(
                    {"updateTextStyle": {**_range(position, run_end), "textStyle": {"bold": True}, "fields": "bold"}}
                )
            if run.url is not None:
                requests.append(
                    {
                        "updateTextStyle": {
                            **_range(position, run_end),
                            "textStyle": {"link": {"url": run.url}, "underline": True, "foregroundColor": LINK_COLOR},
                            "fields": "link,underline,foregroundColor",
                        }
                    }
                )
            position = run_end
        position += 1
        if block.level is None:
            continue
        if list_runs and list_runs[-1][1] == line_start:
            list_runs[-1] = (list_runs[-1][0], position)
        else:
            list_runs.append((line_start, position))

    # Creating bullets strips the leading tabs, which shifts later indices, so work backwards.
    for list_start, list_end in reversed(list_runs):
        requests.append(
            {"createParagraphBullets": {**_range(list_start, min(list_end, end)), "bulletPreset": LIST_PRESET}}
        )
    return requests


def _render_element(element: dict) -> str:
    if "textRun" in element:
        run = element["textRun"]
        text = run["content"].rstrip("\n")
        style = run.get("textStyle", {})
        if not text.strip():
            return text
        if "link" in style and "url" in style["link"]:
            text = f"[{text}]({style['link']['url']})"
        if style.get("bold"):
            text = f"**{text}**"
        return text
    if "person" in element:
        properties = element["person"]["personProperties"]
        return f"@{properties.get('name') or properties['email']}"
    if "dateElement" in element:
        return element["dateElement"]["dateElementProperties"]["displayText"]
    if "richLink" in element:
        properties = element["richLink"]["richLinkProperties"]
        return f"[{properties.get('title', properties['uri'])}]({properties['uri']})"
    return ""


def render_document(body: dict) -> str:
    """Render a document body as Markdown; the content below Status round-trips through `parse_status`."""
    lines = []
    for element in body["content"]:
        if "table" in element:
            lines.append("[table]")
            continue
        if "paragraph" not in element:
            continue
        paragraph = element["paragraph"]
        text = "".join(_render_element(child) for child in paragraph["elements"]).strip()
        style = paragraph["paragraphStyle"].get("namedStyleType", "NORMAL_TEXT")
        if not text:
            continue
        if style.startswith("HEADING_"):
            lines.append(f"\n{'#' * int(style.removeprefix('HEADING_'))} {text}\n")
        elif style == "TITLE":
            lines.append(f"\n# {text}\n")
        elif "bullet" in paragraph:
            lines.append(f"{'  ' * paragraph['bullet'].get('nestingLevel', 0)}- {text}")
        else:
            lines.append(text)
    return "\n".join(lines).strip() + "\n"


def _open(url: str, token: str, method: str = "GET", payload: dict | None = None) -> http.client.HTTPResponse:
    # Google APIs ignore the GitHub version header, so GitHub and Docs calls share it.
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method))


def _call(url: str, token: str, method: str = "GET", payload: dict | None = None) -> dict:
    with _open(url, token, method, payload) as response:
        return json.load(response)


def _github_get(path: str, token: str) -> list[dict] | dict:
    """GET a GitHub REST resource, following pagination for list results."""
    url: str | None = f"{GITHUB_API}{path}"
    pages: list[dict] = []
    while url is not None:
        with _open(url, token) as response:
            data = json.load(response)
            links = response.headers.get("Link", "")
        if isinstance(data, dict):
            return data
        pages += data
        match = re.search(r'<([^>]+)>; rel="next"', links)
        url = match[1] if match else None
    return pages


def _field_value(item: dict, name: str) -> dict | list | None:
    return next((field["value"] for field in item["fields"] if field["name"] == name), None)


def render_board_item(item: dict) -> str:
    content = item["content"]
    status = _field_value(item, "Status")
    status_name = status["name"]["raw"] if isinstance(status, dict) else "No status"
    if item["content_type"] == "DraftIssue":
        return f"- [{status_name}] draft: {content['title']} (updated {content['updated_at'][:10]})"
    repository = content["repository_url"].rsplit("/", 1)[1]
    details = [content["state"]]
    assignees = _field_value(item, "Assignees") or []
    details.append("assignees: " + (", ".join(user["login"] for user in assignees) or "none"))
    pulls = _field_value(item, "Linked pull requests") or []
    if pulls:
        details.append("PRs: " + ", ".join(f"#{pull['number']}" for pull in pulls))
    sub_issues = content.get("sub_issues_summary") or {}
    if sub_issues.get("total"):
        details.append(f"sub-issues {sub_issues['completed']}/{sub_issues['total']}")
    details.append(f"updated {content['updated_at'][:10]}")
    return (
        f"- [{status_name}] [{repository}#{content['number']}]({content['html_url']}) "
        f"{content['title']} ({'; '.join(details)})"
    )


def render_board(items: Iterable[dict], goal_options: Iterable[str]) -> str:
    """Render board items as Markdown grouped by the Goal field, in the field's option order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        goal = _field_value(item, "Goal")
        groups[goal["name"]["raw"] if isinstance(goal, dict) else NO_GOAL].append(render_board_item(item))
    sections = []
    for goal in [*goal_options, NO_GOAL]:
        entries = groups.pop(goal, [])
        if entries or goal != NO_GOAL:
            sections.append(f"## {goal} ({len(entries)})\n\n" + ("\n".join(sorted(entries)) or "(no items)"))
    assert not groups, f"items reference unknown goals: {sorted(groups)}"
    return "\n\n".join(sections) + "\n"


def _project_fields(org: str, project: int, token: str) -> list[dict]:
    fields = _github_get(f"/orgs/{org}/projectsV2/{project}/fields", token)
    assert isinstance(fields, list)
    return fields


def _project_items(org: str, project: int, token: str, fields: list[dict]) -> list[dict]:
    field_ids = ",".join(str(field["id"]) for field in fields)
    items = _github_get(f"/orgs/{org}/projectsV2/{project}/items?per_page=100&fields={field_ids}", token)
    assert isinstance(items, list)
    return items


def _named_field(fields: list[dict], name: str) -> dict:
    field = next((field for field in fields if field["name"] == name), None)
    if field is None:
        raise ValueError(f"the project has no {name!r} field")
    return field


def board_markdown(org: str, project: int, token: str) -> str:
    fields = _project_fields(org, project, token)
    goal_field = _named_field(fields, "Goal")
    items = _project_items(org, project, token, fields)
    project_info = _github_get(f"/orgs/{org}/projectsV2/{project}", token)
    assert isinstance(project_info, dict)
    url = f"https://github.com/orgs/{org}/projects/{project}"
    header = f"# {project_info['title']} ({len(items)} items)\n\n{url}\n\n"
    return header + render_board(items, [option["name"]["raw"] for option in goal_field["options"]])


def board_field_updates(fields: list[dict], values: dict[str, str]) -> list[dict]:
    updates = []
    for name, value in values.items():
        field = _named_field(fields, name)
        options = {option["name"]["raw"]: option["id"] for option in field["options"]}
        if value not in options:
            raise ValueError(f"{name} has no option {value!r}; choose one of {sorted(options)}")
        updates.append({"id": field["id"], "value": options[value]})
    return updates


def find_board_item(items: Iterable[dict], owner: str, repo: str, number: int) -> dict | None:
    repository_url = f"{GITHUB_API}/repos/{owner}/{repo}".lower()
    for item in items:
        content = item["content"]
        if (
            item["content_type"] == "Issue"
            and content["repository_url"].lower() == repository_url
            and content["number"] == number
        ):
            return item
    return None


def board_set(org: str, project: int, token: str, issue: str, values: dict[str, str]) -> str:
    """Add `issue` to the board if it is missing, then set its fields; return what changed."""
    match = ISSUE_REF_RE.fullmatch(issue)
    if match is None:
        raise ValueError(f"expected an issue as OWNER/REPO#NUMBER, got {issue!r}")
    if not values:
        raise ValueError("set at least one of --goal and --status")
    owner, repo, number = match["owner"], match["repo"], int(match["number"])
    fields = _project_fields(org, project, token)
    updates = board_field_updates(fields, values)
    base = f"/orgs/{org}/projectsV2/{project}/items"
    item = find_board_item(_project_items(org, project, token, fields), owner, repo, number)
    action = "updated"
    if item is None:
        item = _call(
            f"{GITHUB_API}{base}", token, "POST", {"type": "Issue", "owner": owner, "repo": repo, "number": number}
        )
        action = "added"
    _call(f"{GITHUB_API}{base}/{item['id']}", token, "PATCH", {"fields": updates})
    return f"{action} {issue}: " + ", ".join(f"{name} = {value}" for name, value in values.items())


def _docs_token(service_account: str) -> str:
    return subprocess.run(
        [
            "gcloud",
            "auth",
            "print-access-token",
            f"--impersonate-service-account={service_account}",
            f"--scopes={DOCS_SCOPE}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="Print the project board grouped by its Goal field.")
    board_set_command = commands.add_parser("board-set", help="Add an issue to the board and set its fields.")
    for command in (board, board_set_command):
        command.add_argument("--org", required=True)
        command.add_argument("--project", type=int, required=True)
    board_set_command.add_argument("--issue", required=True, help="OWNER/REPO#NUMBER")
    board_set_command.add_argument("--goal", help="Goal option name.")
    board_set_command.add_argument("--status", help="Status option name.")
    for name, help_text in (
        ("read", "Print the document as Markdown."),
        ("write-status", "Replace everything below the Status heading with a Markdown file ('-' reads stdin)."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--doc", required=True, help="Google Doc ID.")
        command.add_argument("--impersonate", required=True, help="Service account with edit access to the doc.")
    commands.choices["write-status"].add_argument("status", type=argparse.FileType("r"))
    args = parser.parse_args()

    if args.command in ("board", "board-set"):
        token = os.environ[BOARD_TOKEN_ENV]
    if args.command == "board":
        sys.stdout.write(board_markdown(args.org, args.project, token))
        return
    if args.command == "board-set":
        values = {name: value for name, value in (("Goal", args.goal), ("Status", args.status)) if value is not None}
        print(board_set(args.org, args.project, token, args.issue, values))
        return
    token = _docs_token(args.impersonate)
    document = _call(f"{DOCS_API}/{args.doc}", token)
    if args.command == "read":
        sys.stdout.write(f"# {document['title']}\n\n" + render_document(document["body"]))
        return
    requests = status_requests(document["body"], parse_status(args.status.read()))
    _call(
        f"{DOCS_API}/{args.doc}:batchUpdate",
        token,
        "POST",
        {"requests": requests, "writeControl": {"requiredRevisionId": document["revisionId"]}},
    )
    print(f"updated the content below the Status heading of {document['title']!r}")


if __name__ == "__main__":
    main()
