#!/usr/bin/env python3

from __future__ import annotations
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import requests
import argparse
import shlex


def extract_marinbot_command(body: str) -> Optional[Tuple[str, List[str]]]:
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


# https://docs.github.com/en/webhooks/webhook-events-and-payloads#issue_comment
def parse_payload(payload: Dict[str, object]) -> Tuple[int, str, str, str, str]:
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        raise RuntimeError("No issue in event payload")
    number_value = issue.get("number")
    if not isinstance(number_value, int):
        raise RuntimeError("Issue number missing or invalid in payload")

    body = str(payload["comment"]["body"])
    repo = str(payload["repository"]["name"])
    actor = str(payload["comment"]["user"]["login"])

    try:
        owner = str(payload["repository"]["owner"]["login"])
    except KeyError:
        try:
            owner = str(payload["organization"]["login"])
        except KeyError:
            raise RuntimeError("Could not determine owner from event payload")

    return number_value, body, repo, actor, owner


def load_authorized(config_path: str = "marinbot.json") -> List[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

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
    issue_number: int,
    owner: str,
    repo: str,
    cluster_path: str,
    job_id: str,
) -> Dict[str, str]:
    pr = get_pr(session, owner, repo, issue_number)

    result = {
        "pr_number": str(issue_number),
        "sha": str(pr["head"]["sha"]),
        "cluster_path": cluster_path,
        "job_id": job_id,
    }

    write_outputs(result)
    return result


def handle_ray_run(
    session: requests.Session,
    issue_number: int,
    owner: str,
    repo: str,
    cluster_path: str,
    module: Optional[str],
    is_dry_run: bool,
    ray_run_args: List[str],
    full_command: str,
) -> Dict[str, str]:
    pr = get_pr(session, owner, repo, issue_number)

    result = {
        "pr_number": str(issue_number),
        "sha": str(pr["head"]["sha"]),
        "module": module,
        "cluster_path": cluster_path,
        "ray_run_args": shlex.join(ray_run_args),
        "full_command": full_command,
        "dry_run": "1" if is_dry_run else "0",
    }

    if is_dry_run:
        post_issue_comment(
            session,
            owner,
            repo,
            issue_number,
            "🧪 Running a dry run. See the 'Execute ray_run' step for the command output.",
        )

    write_outputs(result)
    return result


def parse_command(tokens: List[str]) -> Tuple[Optional[argparse.Namespace], str, List[str], argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="marinbot", description="Marinbot PR command helper")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Stop subcommand
    stop_parser = subparsers.add_parser("stop", help="Stop a Ray job by ID")
    stop_parser.add_argument("--cluster", required=True, help="Cluster path")
    stop_parser.add_argument("job_id", help="Job ID to stop")

    # Ray run subcommand - only parse --cluster and module, everything else is ray args
    ray_run_parser = subparsers.add_parser("ray_run", help="Submit a Ray job using marin.run.ray_run")
    ray_run_parser.add_argument("--cluster", required=True, help="Cluster path")
    ray_run_parser.add_argument("--dry-run", action="store_true", help="Only print the command; do not execute")

    # Help subcommand
    subparsers.add_parser("help", help="Show help message")

    if tokens[0] == "ray_run":
        # @marinbot ray_run [ray_run_args] [--dry-run] --cluster <path> <module>
        args, remaining = parser.parse_known_args(tokens)
        if remaining:
            # Last item is the module, everything else before it is ray args
            # Unfortunately we can't use add_argument("module") and then have
            # ray_run_args be remaining because for this command line:
            #   --env_vars foo "bar bazz" --cluster infra/jyc.yaml experiments.tutorials.train_tiny_model_cpu
            # ... it will incorrectly grab "foo" as the module positional argument.
            # TODO show [module] in the help message.
            args.module = remaining[-1]
            ray_run_args = remaining[:-1]
            return args, "ray_run", ray_run_args, parser
        else:
            raise RuntimeError("expected module")
    elif tokens[0] == "help":
        # Just return command type; main() will print help
        args = parser.parse_args(["help"])  # produce a namespace with command
        return args, "help", [], parser
    else:
        args = parser.parse_args(tokens)
        return args, args.command if args else "unknown", [], parser


def main() -> None:
    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    issue_number, body, repo, actor, owner = parse_payload(payload)
    sys.stdout.write(f"body: {body}\n")

    session = requests.Session()
    token = os.environ["GITHUB_TOKEN"]
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "marinbot-python",
        }
    )

    authorized = load_authorized()
    validate_authorized(session, owner, repo, issue_number, actor, authorized)

    extracted = extract_marinbot_command(body)
    if not extracted:
        write_outputs({"command": "unknown"})
        raise RuntimeError("No @marinbot command found")

    full_command, tokens = extracted
    args, command, ray_run_args, parser = parse_command(tokens)
    write_outputs({"command": command})
    validate_pull_request(session, owner, repo, payload)

    if command == "stop" and args:
        handle_stop(session, issue_number, owner, repo, args.cluster, args.job_id)
    elif command == "ray_run" and args:
        handle_ray_run(
            session,
            issue_number,
            owner,
            repo,
            args.cluster,
            args.module,
            bool(getattr(args, "dry_run", False)),
            ray_run_args,
            full_command,
        )
    else:
        assert isinstance(parser, argparse.ArgumentParser)
        subparser_helps = []
        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for _choice, subparser in action.choices.items():
                subparser_helps.append(subparser.format_help())

        comment = f"""
```text
{parser.format_help()}
subcommands:

{"\n".join(subparser_helps)}
```
"""
        post_issue_comment(
            session,
            owner,
            repo,
            issue_number,
            comment,
        )
        return


if __name__ == "__main__":
    main()
