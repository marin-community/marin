#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Roll the Marin Belay agent log over to a new weekly issue.

The Belay agent posts its batch plans and results into one locked issue per week, the open issue labeled
``belay-weekly`` (the intake bot copies approved suggestions there too). Run on Mondays, this opens and locks the new
week's issue, posts a closing summary on the previous one (the week's experiments from the registry's
``summary`` command) with a link forward, and closes it. It does nothing if the open issue is already this week's.
"""

import argparse
import datetime
import os
import subprocess
import sys
import urllib.error

from scripts.ci.belay_intake import WEEKLY_LABEL, GitHub

BELAY_LABEL = "belay"
AGENT_LABEL = "agent-generated"
TITLE_PREFIX = "Marin Belay v1 · week of "
PAGE_URL = "https://claude.ai/artifact/2ZfzxTKZBMF3HJ77xeSQb5"
INTAKE_ISSUE = 9514


def week_start(day: datetime.date) -> datetime.date:
    return day - datetime.timedelta(days=day.weekday())


def week_title(monday: datetime.date) -> str:
    return f"{TITLE_PREFIX}{monday.isoformat()}"


def week_body(monday: datetime.date, previous: int | None, repo: str) -> str:
    tree = f"https://github.com/{repo}/tree"
    lines = [
        f"🤖 Marin Belay agent log for the week of {monday.isoformat()}: batch plans and results, and the intake "
        f"suggestions approved this week. Locked; propose ideas on the intake issue, #{INTAKE_ISSUE}.",
        "",
        f"- Page: {PAGE_URL}",
        f"- Experiment registry: {tree}/marin_belay_v1_registry/experiments/belay",
        f"- Current checkpoint: {tree}/marin_belay_v1_ckpt1",
    ]
    if previous is not None:
        lines.append(f"- Previous week: #{previous}")
    return "\n".join(lines) + "\n"


def closing_body(next_issue: int, summary: str) -> str:
    return f"🤖 This week is closed; the log continues in #{next_issue}.\n\n{summary.strip()}\n"


def registry_summary(registry_checkout: str, since: datetime.date, until: datetime.date) -> str:
    """The registry's markdown summary of experiments tried or decided in [since, until)."""
    return subprocess.run(
        [sys.executable, "-m", "experiments.belay.registry", "summary", "--since", str(since), "--until", str(until)],
        cwd=registry_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def roll_over(github: GitHub, *, today: datetime.date, summary_for, dry_run: bool) -> int | None:
    """Open this week's issue unless it exists; return its number (None in a dry run that would create it).
    ``summary_for(since, until)`` gives the closing summary of the previous week."""
    monday = week_start(today)
    open_issues = github.open_issues_labeled(WEEKLY_LABEL)
    if len(open_issues) > 1:
        raise RuntimeError(f"more than one open {WEEKLY_LABEL} issue: {sorted(open_issues)}")
    previous = open_issues[0] if open_issues else None
    if previous is not None and github.issue(previous)["title"] == week_title(monday):
        print(f"#{previous} is already this week's issue")
        return previous
    print(f"open {week_title(monday)!r}" + (f", close #{previous}" if previous else ""))
    if dry_run:
        return None
    labels = [BELAY_LABEL, WEEKLY_LABEL, AGENT_LABEL]
    new = github.create_issue(week_title(monday), week_body(monday, previous, github.repo), labels)
    github.lock(new)
    if previous is not None:
        since = monday - datetime.timedelta(days=7)
        github.post_comment(previous, closing_body(new, summary_for(since, monday)))
        github.close(previous)
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-checkout", required=True, help="a checkout of the marin_belay_v1_registry branch")
    parser.add_argument("--today", type=datetime.date.fromisoformat, default=datetime.datetime.now(datetime.UTC).date())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    github = GitHub(os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"])
    try:
        issue = roll_over(
            github,
            today=args.today,
            summary_for=lambda since, until: registry_summary(args.registry_checkout, since, until),
            dry_run=args.dry_run,
        )
    except urllib.error.HTTPError as e:
        raise SystemExit(f"GitHub API error {e.code}: {e.read().decode(errors='replace')}") from e
    print(f"current weekly issue: {issue}", file=sys.stderr)


if __name__ == "__main__":
    main()
