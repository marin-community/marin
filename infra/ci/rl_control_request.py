# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve an authorized control refresh to a merged commit."""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RefreshRequest:
    pull: int
    commit: str
    dry_run: bool


def _allowed_comment(comment: dict, allowed: set[str]) -> bool:
    return comment["body"] == "/rl-controls" and comment["user"]["login"].casefold() in allowed


def _github(*args: str):
    return json.loads(subprocess.check_output(["gh", "api", *args], text=True))


def resolve_request(event_name: str, event: dict, repository: str, allowed: set[str]) -> RefreshRequest | None:
    dry_run = event_name == "workflow_dispatch"
    if event_name == "issue_comment":
        issue = event["issue"]
        if "pull_request" not in issue or not _allowed_comment(event["comment"], allowed):
            return None
        number = int(issue["number"])
    elif event_name == "pull_request_target":
        if not event["pull_request"]["merged"]:
            return None
        number = int(event["pull_request"]["number"])
        pages = _github(f"repos/{repository}/issues/{number}/comments?per_page=100", "--paginate", "--slurp")
        if not any(_allowed_comment(comment, allowed) for page in pages for comment in page):
            return None
    elif dry_run:
        number = int(event["inputs"]["pull_request"])
    else:
        return None

    pull = _github(f"repos/{repository}/pulls/{number}")
    if pull["base"]["ref"] != "main" or pull["base"]["repo"]["full_name"] != repository:
        return None
    if not pull["merged"]:
        print(f"PR #{number} awaits its merge into main.")
        return None
    commit = pull.get("merge_commit_sha") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"PR #{number} has no valid merge commit")
    return RefreshRequest(number, commit, dry_run)


def main() -> None:
    event_name = os.environ["GITHUB_EVENT_NAME"]
    allowed = {login.casefold() for login in os.environ.get("RL_CONTROLS_ALLOWED_LOGINS", "").split()}
    if event_name != "workflow_dispatch" and (not allowed or not os.environ.get("RL_CONTROLS_BOT_LOGIN")):
        print("Control refresh requires configured requester and bot identities.")
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    request = resolve_request(event_name, event, os.environ["GITHUB_REPOSITORY"], allowed)
    if request is not None:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            print(f"pull={request.pull}\ncommit={request.commit}\ndry_run={str(request.dry_run).lower()}", file=output)


if __name__ == "__main__":
    main()
