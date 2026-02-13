#!/usr/bin/env -S uv run
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# dependencies = ["PyGithub>=2.3.0"]
# ///
"""
This script iterates through all issues in the marin-community/marin GitHub repository and replaces all
wandb.ai/stanford-mercury links in the issue body with wandb.ai/marin-community. By default, it updates the issues.
If --dry-run is passed, it prints the substitutions it would make instead of editing the issues.

Usage:
    ./scripts/pm/replace_wandb_links.py [--dry-run]

Requires:
    export GITHUB_TOKEN=your_token
"""

import os
import re
import argparse
from github import Github


def find_and_replace_links(text):
    """
    Find all wandb.ai/stanford-mercury links and replace with wandb.ai/marin-community.
    Returns (new_text, [(old_link, new_link), ...])
    """
    pattern = r"https://wandb\.ai/stanford-mercury([/\w\-?&=%.#]*)"
    replacements = []

    def replacer(match):
        old_link = match.group(0)
        new_link = f"https://wandb.ai/marin-community{match.group(1)}"
        replacements.append((old_link, new_link))
        return new_link

    new_text = re.sub(pattern, replacer, text)
    return new_text, replacements


def main():
    parser = argparse.ArgumentParser(description="Replace wandb.ai/stanford-mercury links in GitHub issues.")
    parser.add_argument("--dry-run", action="store_true", help="Print substitutions instead of editing issues")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set.")
        exit(1)
    g = Github(token)
    repo = g.get_repo("marin-community/marin")

    print("Fetching all issues (open and closed)...")
    issues = repo.get_issues(state="all")
    count = 0
    for issue in issues:
        body = issue.body or ""
        new_body, replacements = find_and_replace_links(body)
        if replacements:
            count += 1
            print(f"Issue #{issue.number}: {issue.title}")
            for old, new in replacements:
                print(f"  Replace: {old}\n      With: {new}")
            if not args.dry_run:
                issue.edit(body=new_body)
                print("  -> Issue updated.")
            else:
                print("  (dry run, not updated)")
    if count == 0:
        print("No wandb.ai/stanford-mercury links found in any issue.")
    else:
        print(f"Processed {count} issues with wandb.ai/stanford-mercury links.")


if __name__ == "__main__":
    main()
