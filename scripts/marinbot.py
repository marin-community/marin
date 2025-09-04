#!/usr/bin/env python3

import json
import os
import re
import sys
import argparse
from typing import Dict, List, Optional, Tuple
from argparse import ArgumentParser

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
        body = f"{body}\n\n[View logs](https://github.com/{owner}/{repo}/actions/runs/{run_id})"
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

    body = str(payload["comment"]["body"])  # pyright: ignore[reportIndexIssue]
    repo = str(payload["repository"]["name"])  # pyright: ignore[reportIndexIssue]
    actor = str(payload["comment"]["user"]["login"])  # pyright: ignore[reportIndexIssue]

    try:
        owner = str(payload["repository"]["owner"]["login"])  # pyright: ignore[reportIndexIssue]
    except KeyError:
        try:
            owner = str(payload["organization"]["login"])  # pyright: ignore[reportIndexIssue]
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


def write_outputs(mapping: Dict[str, object]) -> None:
    github_output = os.environ["GITHUB_OUTPUT"]
    lines = [f"{key}={value}" for key, value in mapping.items()]
    with open(github_output, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def handle_stop(
    cluster_path: str,
    job_id: str,
):
    write_outputs(
        {
            "cluster_path": cluster_path,
            "job_id": job_id,
        }
    )


def handle_ray_run(
    session: requests.Session,
    issue_number: int,
    owner: str,
    repo: str,
    cluster_path: str,
    commit: str,
    module: str,
    is_dry_run: bool,
    ray_run_args: List[str],
):
    if is_dry_run:
        post_issue_comment(
            session,
            owner,
            repo,
            issue_number,
            "🧪 Running a dry run. See the 'Execute ray_run' step for the command output.",
        )

    write_outputs(
        {
            "sha": commit,
            "module": module,
            "cluster_path": cluster_path,
            "ray_run_args": shlex.join(ray_run_args),
            "dry_run": "1" if is_dry_run else "0",
        }
    )


def handle_help(session: requests.Session, owner: str, repo: str, issue_number: int, parser: ArgumentParser):
    assert isinstance(parser, ArgumentParser)
    subparser_helps = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for _choice, subparser in action.choices.items():
            subparser_helps.append(subparser.format_help())

    newline = "\n"
    comment = f"""
```text
{parser.format_help()}
subcommands:

{newline.join(subparser_helps)}
```
"""
    post_issue_comment(
        session,
        owner,
        repo,
        issue_number,
        comment,
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="marinbot", description="Marinbot PR command helper")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop a Ray job by ID")
    stop_parser.add_argument("--cluster", required=True, help="Cluster path")
    stop_parser.add_argument("job_id", help="Job ID to stop")

    # ray_run
    ray_run_parser = subparsers.add_parser(
        "ray_run",
        help="Submit a Ray job using marin.run.ray_run",
        usage="%(prog)s ray_run [ray_run_args] [--dry-run] --cluster <path> <commit> <module>",
    )
    ray_run_parser.add_argument("--cluster", required=True, help="Cluster path")
    ray_run_parser.add_argument("--dry-run", action="store_true", help="Only print the command; do not execute")

    # help
    subparsers.add_parser("help", help="Show help message")

    return parser


def parse_command(parser: ArgumentParser, tokens: List[str]) -> Tuple[Optional[argparse.Namespace], str, List[str]]:
    if tokens[0] == "ray_run":
        # @marinbot ray_run [ray_run_args] [--dry-run] --cluster <path> <commit> <module>
        args, remaining = parser.parse_known_args(tokens)
        if len(remaining) >= 2:
            # Unfortunately we can't use `add_argument` for `commit` and `module` and
            # then grab `ray_run_args` from `remaining` because argparse will
            # grab the first arguments for the positionl parameters instead of
            # the last. So this command line:
            #   --env_vars foo "bar bazz" --cluster infra/jyc.yaml experiments.tutorials.train_tiny_model_cpu
            # ... would assign "foo" to `commit` and "bar bazz" to `module`.
            args.commit = remaining[-2]
            if not re.fullmatch(r"[0-9a-fA-F]{40}", args.commit):
                raise RuntimeError("expected full 40-character git commit hash")
            args.module = remaining[-1]
            ray_run_args = remaining[:-2]
            return args, "ray_run", ray_run_args
        else:
            raise RuntimeError("expected commit and module")
    elif tokens[0] == "help":
        # Just return command type; main() will print help
        args = parser.parse_args(["help"])  # produce a namespace with command
        return args, "help", []
    else:
        args = parser.parse_args(tokens)
        if not args.command:
            raise RuntimeError("expected command")
        return args, args.command, []


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
        raise RuntimeError("no @marinbot command found")

    full_command, tokens = extracted
    write_outputs({"pr_number": str(issue_number), "full_command": full_command})
    parser = build_parser()

    try:
        args, command, ray_run_args = parse_command(parser, tokens)
    except Exception as e:
        handle_help(session, owner, repo, issue_number, parser)
        raise e

    write_outputs({"command": command})
    validate_pull_request(session, owner, repo, payload)

    if command == "stop" and args:
        handle_stop(args.cluster, args.job_id)
    elif command == "ray_run" and args:
        handle_ray_run(
            session,
            issue_number,
            owner,
            repo,
            args.cluster,
            args.commit,
            args.module,
            bool(getattr(args, "dry_run", False)),
            ray_run_args,
        )
    else:
        handle_help(session, owner, repo, issue_number, parser)


if __name__ == "__main__":
    main()
