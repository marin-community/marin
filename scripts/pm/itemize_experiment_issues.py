#!/usr/bin/env -S uv run
# /// script
# dependencies = ["PyGithub>=2.3.0", "wandb"]
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
This script updates the `docs/reports/index.md` file with new experiment GitHub issues.

Usage:
    ./scripts/pm/itemize_experiment_issues.py
"""

import os
from pathlib import Path
import re
from github import Github
from urllib.parse import urlparse
import wandb


def clean_title(title):
    """Clean up the title by removing 'Experiment:' prefix."""
    if title.startswith("Experiment:"):
        return title[len("Experiment:") :].strip()
    return title


def get_run_name(wandb_url):
    """Extract run name from WandB URL using API."""
    # Parse the URL to get entity, project, and run ID
    # URL format: https://wandb.ai/entity/project/runs/run_id
    parsed = urlparse(wandb_url)
    path = parsed.path.strip("/")
    parts = path.split("/")

    if len(parts) < 3:
        return "WandB Run"

    entity = parts[0]
    project = parts[1]
    run_id = parts[3]  # parts[2] is 'runs'

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        return f"WandB Run: {run.name}"
    except Exception as e:
        print(f"Warning: Could not get run name for {wandb_url}: {e}")
        return "WandB Run"


def get_existing_reports():
    """Read existing reports from experiments.md if it exists."""
    reports_path = Path("docs/reports/index.md")
    if not reports_path.exists():
        return set()

    with open(reports_path, "r") as f:
        content = f.read()

    # Extract report URLs using regex
    report_urls = set(
        # TODO: move this once we replace
        re.findall(r"https://(?:marin\.community/data-browser/|wandb\.ai/[^)]+|api\.wandb\.ai/links/[^)]+)", content)
    )
    return report_urls


def get_github_issues():
    """Get all issues with experiments label and extract experiment links."""
    # Initialize GitHub API
    g = Github(os.environ.get("GITHUB_TOKEN"))
    repo = g.get_repo("marin-community/marin")

    # Get all issues with experiments label
    issues = repo.get_issues(labels=["experiment"], state="all")

    experiment_links = []
    for issue in issues:
        # Look for experiment links in the description
        description = issue.body or ""

        # Find all experiment URLs
        urls = {
            "wandb": re.findall(r"https://(?:wandb\.ai/[^\s)]+|api\.wandb\.ai/links/[^\s)]+)", description),
            "data_browser": re.findall(r"https://marin\.community/data-browser/[^\s)]+", description),
        }

        if any(urls.values()):  # If we found any experiment links
            badge_url = f"https://img.shields.io/github/issues/detail/state/marin-community/marin/{issue.number}"
            experiment_links.append(
                {"title": clean_title(issue.title), "issue_num": issue.number, "badge_url": badge_url, "urls": urls}
            )

    return experiment_links


def update_reports_md(new_reports):
    """Update experiments.md with new reports in the Uncategorized section."""
    reports_path = Path("docs/reports/index.md")

    # print absolute path of reports_path
    print(reports_path.absolute())

    # Create directory if it doesn't exist
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing content if file exists
    existing_content = ""
    if reports_path.exists():
        with open(reports_path, "r") as f:
            existing_content = f.read()

    # Prepare new content
    new_content = existing_content
    for report in sorted(new_reports, key=lambda x: x["issue_num"]):
        # Escape title for markdown
        title = report["title"].replace("[", r"\[").replace("]", r"\]")

        # Check if any of the URLs are already in the content
        if any(url in existing_content for urls in report["urls"].values() for url in urls):
            continue

        # Add the main experiment entry
        new_content += f"- {title}\n"

        # Add GitHub issue link
        new_content += f"    - [GitHub Issue #{report['issue_num']}](https://github.com/marin-community/marin/issues/{report['issue_num']}) [![#{report['issue_num']}]({report['badge_url']})](https://github.com/marin-community/marin/issues/{report['issue_num']})\n"

        # Add WandB links if present
        for url in report["urls"]["wandb"]:
            if "/runs/" in url:
                link_text = get_run_name(url)
            elif "api.wandb.ai/links" in url:
                link_text = "WandB Report"
            else:
                link_text = "WandB Report"
            new_content += f"    - [{link_text}]({url})\n"

        # Add Data Browser links if present
        for url in report["urls"]["data_browser"]:
            new_content += f"    - [Data Browser]({url})\n"

    # Write new content
    with open(reports_path, "w") as f:
        f.write(new_content)


def main():
    # Get existing reports
    existing_reports = get_existing_reports()

    # Get GitHub issues with experiment links
    github_reports = get_github_issues()

    # Find new reports
    new_reports = []
    for report in github_reports:
        # Check if any of the URLs are already in existing_reports
        if not any(url in existing_reports for urls in report["urls"].values() for url in urls):
            new_reports.append(report)

    if new_reports:
        print(f"Found {len(new_reports)} new reports to add")
        if len(new_reports) == 1:
            print(f"Adding new report: {new_reports[0]['title']}")
        update_reports_md(new_reports)
        print("Updated experiments.md successfully")
    else:
        print("No new reports to add")


if __name__ == "__main__":
    main()
