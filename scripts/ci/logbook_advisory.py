# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect bounded public research evidence for a merged contribution PR."""

import base64
import json
import os
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

LOGBOOKS = "marin-community/marin-logbooks"
API = f"https://api.github.com/repos/{LOGBOOKS}"
WEB = f"https://github.com/{LOGBOOKS}"
MAX_RESPONSE_BYTES = 1_048_576
MAX_EVIDENCE_CHARS = 4_000
MAX_TITLE_CHARS = 500
RECORD_NAME = re.compile(r"[1-9][0-9]{0,9}-[a-z0-9-]+\.md")
NOTEBOOK_LINK = re.compile(
    rf"{re.escape(WEB)}/blob/([0-9a-f]{{40}})/(logbooks/[1-9][0-9]{{0,9}}-[a-z0-9-]+/notebook\.md)"
)


def github(path: str) -> Any:
    response_request = Request(f"{API}/{path}", headers={"Accept": "application/vnd.github+json"})
    with urlopen(response_request, timeout=15) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError(f"Logbooks API response exceeded the size limit: {path}")
    return json.loads(body)


def excerpt(url: str, content: str | None) -> dict:
    text = content or ""
    truncated = len(text) > MAX_EVIDENCE_CHARS
    if truncated:
        half = MAX_EVIDENCE_CHARS // 2
        text = f"{text[:half]}\n\n[Middle omitted]\n\n{text[-half:]}"
    return {"url": url, "text": text, "truncated": truncated}


def evidence(pr: dict, issue_number: str) -> dict:
    """Read the issue and any matching index record from a fixed public repository."""
    issue_url = f"{WEB}/issues/{issue_number}"
    issue = github(f"issues/{issue_number}")
    if issue.get("pull_request"):
        raise ValueError("Research logbook link points to a pull request")
    report = {
        "contribution": {
            **excerpt(f"https://github.com/marin-community/marin/pull/{pr['number']}", pr.get("body")),
            "title": pr["title"][:MAX_TITLE_CHARS],
            "merge_commit": pr["merge_commit_sha"],
        },
        "issue": {
            **excerpt(issue_url, issue.get("body")),
            "title": issue["title"][:MAX_TITLE_CHARS],
            "state": issue["state"],
        },
    }
    revision = github("commits/main")["sha"]
    records = [
        entry
        for entry in github(f"contents/index?ref={revision}")
        if RECORD_NAME.fullmatch(entry["name"]) and entry["name"].startswith(f"{issue_number}-")
    ]
    if len(records) != 1:
        return report | {"record_status": "missing" if not records else "ambiguous"}

    record_path = f"index/{records[0]['name']}"
    record_blob = github(f"contents/{record_path}?ref={revision}")
    record = base64.b64decode(record_blob["content"]).decode("utf-8")
    report["record"] = excerpt(f"{WEB}/blob/{revision}/{record_path}", record)
    if f"]({issue_url})" not in record:
        return report | {"record_status": "issue link mismatch"}
    notebooks = set(NOTEBOOK_LINK.findall(record))
    if len(notebooks) != 1:
        return report | {"record_status": "missing or ambiguous pinned notebook"}
    notebook_revision, notebook_path = notebooks.pop()
    if notebook_path != f"logbooks/{Path(record_path).stem}/notebook.md":
        return report | {"record_status": "pinned notebook identifies another task"}
    try:
        notebook_blob = github(f"contents/{notebook_path}?ref={notebook_revision}")
    except HTTPError as error:
        if error.code == HTTPStatus.NOT_FOUND:
            return report | {"record_status": "pinned notebook unavailable"}
        raise
    notebook = base64.b64decode(notebook_blob["content"]).decode("utf-8")
    return report | {
        "record_status": "matched",
        "notebook": excerpt(f"{WEB}/blob/{notebook_revision}/{notebook_path}", notebook),
    }


def main() -> None:
    issue = os.environ["LOGBOOK_ISSUE"]
    if not re.fullmatch(r"[1-9][0-9]{0,9}", issue):
        raise ValueError("Expected a positive logbooks issue number")
    with Path(os.environ["CONTRIBUTION_FILE"]).open(encoding="utf-8") as contribution_file:
        pr = json.load(contribution_file)
    if pr["merged"] is not True or pr["base"]["ref"] != "main":
        raise ValueError("Expected a pull request merged to main")
    report = evidence(pr, issue)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print(f"json={json.dumps(report, ensure_ascii=False, separators=(',', ':'))}", file=output)


if __name__ == "__main__":
    main()
