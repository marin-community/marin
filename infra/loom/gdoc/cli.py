# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Read, write and comment on a Google Doc from a Loom session.

Run it from the root of a marin checkout with the system Python; see `google_api` for credentials:

    python3 -m infra.loom.gdoc read DOC_ID [--tab TAB]             # the whole doc, or one tab
    python3 -m infra.loom.gdoc append DOC_ID "Markdown" [--tab TAB]   # or `-` to read it from stdin
    python3 -m infra.loom.gdoc prepend DOC_ID - [--section HEADING] [--tab TAB]   # newest first, below the heading
    python3 -m infra.loom.gdoc replace-section DOC_ID "Heading" - [--tab TAB]   # preserves unchanged paragraphs
    python3 -m infra.loom.gdoc length DOC_ID [--exclude-section HEADING] [--tab TAB]   # words and estimated pages
    python3 -m infra.loom.gdoc comment DOC_ID "A comment on the doc."
    python3 -m infra.loom.gdoc comments DOC_ID                     # every comment with its id, status and replies
    python3 -m infra.loom.gdoc reply DOC_ID COMMENT_ID "Text" [--resolve]
    python3 -m infra.loom.gdoc agent-status DOC_ID "input needed"  # set the Agent status chip in the doc's header
"""

import argparse
import sys
from enum import StrEnum

from infra.loom.gdoc.docs_format import (
    DiagramStore,
    Tab,
    append_markdown,
    batch_update,
    find_tab,
    paragraph_text,
    prepend_markdown,
    read_document,
    replace_section,
    tab_length,
    tab_text,
)
from infra.loom.gdoc.google_api import (
    DRIVE_API,
    GoogleApi,
    access_token,
    gcloud_account,
    is_service_account,
)

MARKDOWN_MIME_TYPE = "text/markdown"
# `me` marks a comment or reply that the caller's account wrote, so an agent's own.
COMMENT_FIELDS = (
    "comments(id,createdTime,modifiedTime,content,resolved,quotedFileContent(value),author(displayName,me),"
    "replies(id,createdTime,content,action,author(displayName,me))),nextPageToken"
)
# The Agent status line is in the header: the first paragraphs of the first tab.
HEADER_PARAGRAPHS = 30
AGENT_STATUS_LABEL = "agent status"


class AgentStatus(StrEnum):
    """The options of the Agent status chip in a doc's header, which the agent keeps current."""

    UNKNOWN = "unknown"
    WORKING = "working"
    INPUT_NEEDED = "input needed"
    GOAL_ACHIEVED = "goal achieved"


def doc_markdown(api: GoogleApi, doc_id: str) -> str:
    # The export is UTF-8, but its text/markdown content type has no charset.
    return api.request("GET", f"{DRIVE_API}/files/{doc_id}/export", params={"mimeType": MARKDOWN_MIME_TYPE}).decode()


def add_comment(api: GoogleApi, doc_id: str, text: str) -> None:
    api.post(f"{DRIVE_API}/files/{doc_id}/comments", {"content": text}, params={"fields": "id"})


def reply_to_comment(api: GoogleApi, doc_id: str, comment_id: str, text: str, *, resolve: bool) -> None:
    body = {"content": text, **({"action": "resolve"} if resolve else {})}
    api.post(f"{DRIVE_API}/files/{doc_id}/comments/{comment_id}/replies", body, params={"fields": "id"})


def doc_comments(api: GoogleApi, doc_id: str) -> list[dict]:
    comments: list[dict] = []
    params = {"fields": COMMENT_FIELDS, "pageSize": 100}
    while True:
        body = api.get(f"{DRIVE_API}/files/{doc_id}/comments", params=params)
        comments.extend(body.get("comments", []))
        if "nextPageToken" not in body:
            return comments
        params["pageToken"] = body["nextPageToken"]


def is_resolved(comment: dict) -> bool:
    # Drive leaves `resolved` out of a comment that is not resolved.
    return comment.get("resolved", False)


def describe_comment(comment: dict) -> str:
    status = "resolved" if is_resolved(comment) else "open"
    lines = [f"- [{comment['id']}] ({status}) {comment['author']['displayName']}: {comment['content']}"]
    if quoted := comment.get("quotedFileContent", {}).get("value"):
        lines.append(f'  on the text: "{quoted}"')
    for reply in comment.get("replies", []):
        action = f" ({reply['action']}d)" if reply.get("action") else ""
        # A reply that only resolves or reopens the comment can come without content.
        lines.append(f"  - reply from {reply['author']['displayName']}{action}: {reply.get('content', '')}")
    return "\n".join(lines)


def agent_status_request(tab: Tab, status: AgentStatus) -> dict:
    """The Docs API request that sets the Agent status chip in the tab's header to status."""
    for element in tab.body[:HEADER_PARAGRAPHS]:
        if "paragraph" not in element or paragraph_text(element).partition(":")[0].strip().lower() != AGENT_STATUS_LABEL:
            continue
        chips = [part["dropdown"] for part in element["paragraph"]["elements"] if "dropdown" in part]
        if not chips:
            break
        definition = tab.dropdown_definitions[chips[0]["dropdownProperties"]["dropdownDefinitionId"]]
        options = {
            option["displayValue"].strip().lower(): option["optionId"]
            for option in definition["dropdownDefinitionProperties"]["options"]
        }
        if status not in options:
            raise ValueError(f"the Agent status chip has no option {status.value!r}; its options are {sorted(options)}")
        return {
            "updateDropdownProperties": {
                "tabId": tab.id,
                "dropdownId": chips[0]["dropdownId"],
                "dropdownProperties": {"selectedOptionId": options[status]},
                "fields": "selectedOptionId",
            }
        }
    raise ValueError("the doc's header has no Agent status chip; a doc made from a Loom doc template has one")


def set_agent_status(api: GoogleApi, doc_id: str, status: AgentStatus) -> None:
    document = read_document(api, doc_id)
    batch_update(api, document, [agent_status_request(find_tab(document, None), status)])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m infra.loom.gdoc", description="Read, write and comment on a Google Doc."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read", help="Print the doc as markdown, or one tab with --tab.")
    read.add_argument("doc_id")
    read.add_argument("--tab", help="Tab title; the whole doc when not given.")
    length = commands.add_parser("length", help="Print a tab's words, images and estimated pages.")
    length.add_argument("doc_id")
    length.add_argument("--tab", help="Tab title; the first tab when not given.")
    length.add_argument("--exclude-section", help="Heading of a section not to count.")
    comments = commands.add_parser(
        "comments", help="Print every comment with its id, status, the text it is on, and its replies."
    )
    comments.add_argument("doc_id")
    for name, help_text in [
        ("append", "Add markdown at the end of a tab, as Docs formatting. `-` reads it from stdin."),
        (
            "prepend",
            "Add markdown at the top of a section (--section) or of a tab, below the heading. `-` reads stdin.",
        ),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("doc_id")
        command.add_argument("text")
        command.add_argument("--tab", help="Tab title; the first tab when not given.")
    commands.choices["prepend"].add_argument("--section", help="Heading of the section to add to.")
    comment = commands.add_parser("comment", help="Add a comment on the whole doc.")
    comment.add_argument("doc_id")
    comment.add_argument("text")
    replace = commands.add_parser(
        "replace-section",
        help="Replace a section with markdown, preserving unchanged paragraphs. `-` reads stdin.",
    )
    replace.add_argument("doc_id")
    replace.add_argument("heading")
    replace.add_argument("text")
    replace.add_argument("--tab", help="Tab title; the first tab when not given.")
    reply = commands.add_parser("reply", help="Reply to a comment, and resolve it with --resolve.")
    reply.add_argument("doc_id")
    reply.add_argument("comment_id")
    reply.add_argument("text")
    reply.add_argument("--resolve", action="store_true")
    agent_status = commands.add_parser("agent-status", help="Set the Agent status chip in the doc's header.")
    agent_status.add_argument("doc_id")
    agent_status.add_argument("status", type=AgentStatus, choices=list(AgentStatus))
    return parser


def main() -> None:
    args = _parser().parse_args()
    account = gcloud_account()
    api = GoogleApi(access_token(account))
    diagrams = DiagramStore(account) if is_service_account(account) else None
    text = sys.stdin.read() if getattr(args, "text", None) == "-" else getattr(args, "text", None)
    if args.command == "read":
        if args.tab:
            print(tab_text(find_tab(read_document(api, args.doc_id), args.tab)))
        else:
            print(doc_markdown(api, args.doc_id))
    elif args.command == "length":
        tab = find_tab(read_document(api, args.doc_id), args.tab)
        words, images, pages = tab_length(tab, args.exclude_section)
        print(f"tab {tab.title!r}: {words} words, {images} images, about {pages:.1f} pages")
    elif args.command == "comments":
        comments = sorted(doc_comments(api, args.doc_id), key=lambda c: (is_resolved(c), c["modifiedTime"]))
        print("\n".join(describe_comment(comment) for comment in comments) or "no comments")
    elif args.command == "append":
        append_markdown(api, args.doc_id, text, diagrams, args.tab)
        print(f"appended {len(text)} characters of markdown to {args.tab or 'the first tab'}")
    elif args.command == "prepend":
        prepend_markdown(api, args.doc_id, text, diagrams, args.tab, args.section)
        where = f"section {args.section!r}" if args.section else (args.tab or "the first tab")
        print(f"prepended {len(text)} characters of markdown to {where}")
    elif args.command == "replace-section":
        replace_section(api, args.doc_id, args.tab, args.heading, text, diagrams)
        print(f"replaced the section {args.heading!r} in {args.tab or 'the first tab'}")
    elif args.command == "agent-status":
        set_agent_status(api, args.doc_id, args.status)
        print(f"set the agent status to {args.status.value}")
    elif args.command == "reply":
        reply_to_comment(api, args.doc_id, args.comment_id, text, resolve=args.resolve)
        print(f"replied to comment {args.comment_id}" + (" and resolved it" if args.resolve else ""))
    else:
        add_comment(api, args.doc_id, text)
        print(f"commented on {args.doc_id}")
