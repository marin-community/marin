#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import requests
import argparse
import shlex


def extract_marinbot_command(body: str) -> Optional[Tuple[str, List[str]]]:
    """Extract @marinbot command from comment body."""
    for raw in re.split(r"\r?\n", body):
        s = raw.strip()
        if s.startswith("@marinbot"):
            # Remove @marinbot prefix and split into tokens
            rest = s[len("@marinbot") :].strip()
            tokens = shlex.split(rest)
            return s, tokens
    return None


def post_issue_comment(session: requests.Session, owner: str, repo: str, issue_number: int, body: str) -> None:
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        body = f"{body}\n\n[View run](https://github.com/{owner}/{repo}/actions/runs/{run_id})"
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    resp = session.post(url, json={"body": body})
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to post comment: {resp.status_code} {resp.text}")


def load_authorized(session: requests.Session, owner: str, repo: str, default_branch: str) -> List[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/marinbot.json?ref={default_branch}"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    content_b64 = data.get("content")
    encoding = data.get("encoding")
    if not content_b64 or encoding != "base64":
        raise RuntimeError("Unexpected content response for marinbot.json")
    decoded = base64.b64decode(content_b64).decode("utf-8")
    cfg = json.loads(decoded)
    authorized = cfg.get("authorized")
    if isinstance(authorized, list) and all(isinstance(x, str) for x in authorized):
        return authorized
    raise RuntimeError(f"Unexpected authorized response for marinbot.json: {str(authorized)}")


def validate_pull_request(session: requests.Session, owner: str, repo: str, payload: Dict[str, object]) -> None:
    issue = payload.get("issue") if isinstance(payload, dict) else None
    if not (isinstance(issue, dict) and issue.get("pull_request")):
        if isinstance(issue, dict) and isinstance(issue.get("number"), int):
            post_issue_comment(session, owner, repo, int(issue["number"]), "❌ Only works on pull requests")
        raise RuntimeError("not pull request comment")


def validate_authorized(
    session: requests.Session, owner: str, repo: str, issue_number: int, actor: str, authorized: List[str]
) -> None:
    if actor not in authorized:
        post_issue_comment(session, owner, repo, issue_number, f"❌ @{actor} is not authorized")
        raise RuntimeError(f"@{actor} is not authorized.")


def get_repo_info(session: requests.Session, owner: str, repo: str) -> Dict[str, object]:
    resp = session.get(f"https://api.github.com/repos/{owner}/{repo}")
    resp.raise_for_status()
    return resp.json()


def get_pr(session: requests.Session, owner: str, repo: str, pr_number: int) -> Dict[str, object]:
    resp = session.get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}")
    resp.raise_for_status()
    return resp.json()


def write_outputs(mapping: Dict[str, object]) -> None:
    github_output = os.environ["GITHUB_OUTPUT"]
    lines = [f"{key}={value}" for key, value in mapping.items()]
    with open(github_output, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def handle_stop(
    session: requests.Session,
    payload: Dict[str, object],
    owner: str,
    repo: str,
    actor: str,
    args: argparse.Namespace,
    full_command: str,
) -> Dict[str, str]:
    issue = payload["issue"]
    assert isinstance(issue, dict)
    issue_number = int(issue["number"])  # type: ignore[call-arg]

    repo_info = get_repo_info(session, owner, repo)
    default_branch = str(repo_info.get("default_branch"))
    authorized = load_authorized(session, owner, repo, default_branch)
    validate_authorized(session, owner, repo, issue_number, actor, authorized)

    if not args.cluster:
        post_issue_comment(
            session,
            owner,
            repo,
            issue_number,
            "❌ Missing --cluster. Use: `@marinbot stop --cluster <path> <job_id>`",
        )
        raise RuntimeError(f"missing cluster")

    if not args.job_id:
        post_issue_comment(
            session,
            owner,
            repo,
            issue_number,
            "❌ Missing job ID. Use: `@marinbot stop --cluster <path> <job_id>`",
        )
        raise RuntimeError(f"missing job id")

    pr_number = issue_number
    pr = get_pr(session, owner, repo, pr_number)

    result = {
        "pr_number": str(pr_number),
        "head_ref": str(pr["head"]["ref"]),
        "sha": str(pr["head"]["sha"]),
        "cluster_path": args.cluster,
        "job_id": args.job_id,
        "actor": actor,
    }

    write_outputs(result)
    return result


def handle_ray_run(
    session: requests.Session,
    payload: Dict[str, object],
    owner: str,
    repo: str,
    actor: str,
    args: argparse.Namespace,
    ray_args: List[str],
    full_command: str,
) -> Dict[str, str]:
    issue = payload["issue"]
    assert isinstance(issue, dict)
    issue_number = int(issue["number"])  # type: ignore[call-arg]

    repo_info = get_repo_info(session, owner, repo)
    default_branch = str(repo_info.get("default_branch"))
    authorized = load_authorized(session, owner, repo, default_branch)
    validate_authorized(session, owner, repo, issue_number, actor, authorized)

    if not args.cluster:
        post_issue_comment(
            session,
            owner,
            repo,
            issue_number,
            "❌ Missing --cluster. Use: `@marinbot ray_run --cluster <path> <module>`",
        )
        raise RuntimeError(f"missing cluster")

    if not args.module:
        post_issue_comment(
            session,
            owner,
            repo,
            issue_number,
            "❌ Missing module. Use: `@marinbot ray_run --cluster <path> <module>`",
        )
        raise RuntimeError(f"missing module")

    pr_number = issue_number
    pr = get_pr(session, owner, repo, pr_number)

    result = {
        "pr_number": str(pr_number),
        "head_ref": str(pr["head"]["ref"]),
        "sha": str(pr["head"]["sha"]),
        "module": args.module,
        "cluster_path": args.cluster,
        "ray_args": " ".join(ray_args),
        "full_command": full_command,
        "actor": actor,
        "wait_for_job_id": "true",  # Signal to wait for job ID in logs
    }

    write_outputs(result)

    post_issue_comment(session, owner, repo, issue_number, f"🚀 Starting: `{full_command}`")

    return result


def parse_command(tokens: List[str]) -> Tuple[Optional[argparse.Namespace], str, List[str]]:
    """Parse command tokens and return parsed args, command type, and remaining args."""
    if not tokens:
        return None, "unknown", []

    parser = argparse.ArgumentParser(prog="marinbot", add_help=False)
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Stop subcommand
    stop_parser = subparsers.add_parser("stop", add_help=False)
    stop_parser.add_argument("--cluster", required=True, help="Cluster path")
    stop_parser.add_argument("job_id", help="Job ID to stop")

    # Ray run subcommand - only parse --cluster and module, everything else is ray args
    ray_run_parser = subparsers.add_parser("ray_run", add_help=False)
    ray_run_parser.add_argument("--cluster", required=True, help="Cluster path")

    # Special handling for ray_run to extract module and ray args
    if tokens[0] == "ray_run":
        # Use parse_known_args to capture --cluster and leave the rest
        try:
            args, remaining = parser.parse_known_args(tokens)
            if remaining:
                # Last item is the module, everything else before it is ray args
                args.module = remaining[-1]
                ray_args = remaining[:-1]
                return args, "ray_run", ray_args
            else:
                # No module provided
                args.module = None
                return args, "ray_run", []
        except (SystemExit, argparse.ArgumentError):
            return None, "ray_run", []
    else:
        # For other commands, use regular parse_args
        try:
            args = parser.parse_args(tokens)
            return args, args.command if args else "unknown", []
        except (SystemExit, argparse.ArgumentError):
            return None, "unknown", []


def main() -> None:
    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    body = str(payload["comment"]["body"])
    sys.stdout.write(f"body: {body}\n")

    github_output = os.environ["GITHUB_OUTPUT"]
    token = os.environ["GITHUB_TOKEN"]
    try:
        owner = str(payload["repository"]["owner"]["login"])
    except KeyError:
        try:
            owner = str(payload["organization"]["login"])
        except KeyError:
            raise RuntimeError("Could not determine owner from event payload")
    repo = str(payload["repository"]["name"])

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "marinbot-python",
        }
    )

    extracted = extract_marinbot_command(body)
    if not extracted:
        write_outputs({"command": "unknown"})
        raise RuntimeError("No @marinbot command found")

    full_command, tokens = extracted
    args, command, ray_args = parse_command(tokens)
    write_outputs({"command": command})
    validate_pull_request(session, owner, repo, payload)
    actor = str(payload["comment"]["user"]["login"])

    if command == "stop" and args:
        handle_stop(session, payload, owner, repo, actor, args, full_command)
    elif command == "ray_run" and args:
        handle_ray_run(session, payload, owner, repo, actor, args, ray_args, full_command)
    else:
        issue = payload.get("issue")
        if isinstance(issue, dict):
            issue_number = int(issue.get("number", 0))
            if issue_number:
                post_issue_comment(
                    session,
                    owner,
                    repo,
                    issue_number,
                    f"❌ Unknown or invalid command. Supported commands: `@marinbot stop`, `@marinbot ray_run`",
                )
        raise RuntimeError(f"Unknown or invalid command: {command}")


if __name__ == "__main__":
    main()
