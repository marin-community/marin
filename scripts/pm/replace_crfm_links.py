#!/usr/bin/env -S uv run
# /// script
# dependencies = ["PyGithub>=2.3.0"]
# ///
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script iterates through all issues in the marin-community/marin GitHub repository and replaces all
crfm.stanford.edu/marin/data_browser/ links in the issue body with marin.community/data-browser/.
By default, it updates the issues. If --dry-run is passed, it prints the substitutions it would make
instead of editing the issues.

Usage:
    ./scripts/pm/replace_crfm_links.py [--dry-run]

Requires:
    export GITHUB_TOKEN=your_token
"""

import os
import re
import argparse
from github import Github


def find_and_replace_links(text):
    """
    Find all crfm.stanford.edu/marin/data_browser/ links and replace with marin.community/data-browser/.
    Returns (new_text, [(old_link, new_link), ...])
    """
    pattern = r"crfm\.stanford\.edu/marin/data_browser/([^\s<>\"']*)"
    replacements = []

    def replacer(match):
        old_link = match.group(0)
        path = match.group(1)
        new_link = f"marin.community/data-browser/{path}"
        replacements.append((old_link, new_link))
        return new_link

    new_text = re.sub(pattern, replacer, text)
    return new_text, replacements


def main():
    parser = argparse.ArgumentParser(description="Replace CRFM Stanford links in GitHub issues.")
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
        print("No crfm.stanford.edu/marin/data_browser/ links found in any issue.")
    else:
        print(f"Processed {count} issues with CRFM Stanford links.")


if __name__ == "__main__":
    main()
