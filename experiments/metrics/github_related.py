import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests


@dataclass
class GithubApiConfig:
    github_token: str
    time_since: str
    repo_name: str = "marin"
    repo_owner: str = "stanford-crfm"
    headers: dict = None

    def __post_init__(self):
        self.headers = {"Authorization": f"Bearer {self.github_token}", "Accept": "application/vnd.github+json"}


def get_average_duration_for_all_workflows(config: GithubApiConfig) -> dict[str, float]:
    """Fetch all workflows and calculate the average duration (in min) in the time duration for each workflow."""

    workflow_url = f"https://api.github.com/repos/{config.repo_owner}/{config.repo_name}/actions/workflows"
    response = requests.get(workflow_url, headers=config.headers)
    response.raise_for_status()
    workflows = response.json()["workflows"]

    average_times = {}
    for workflow in workflows:
        workflow_id = workflow["id"]
        workflow_name = workflow["name"]
        average_duration = get_average_duration(config, workflow_id)
        if average_duration is not None:
            average_times[workflow_name] = average_duration

    return average_times


def get_average_duration(config: GithubApiConfig, workflow_id: int) -> float | None:
    """Fetch workflow run and calculate the average duration for a specific workflow."""

    url = f"https://api.github.com/repos/{config.repo_owner}/{config.repo_name}/actions/workflows/{workflow_id}/runs"
    total_duration = 0
    count = 0
    page = 1

    while True:
        # Request runs page by page
        params = {
            "created": f">={config.time_since}",  # Filter for runs created in the past week
            "per_page": 100,
            "page": page,
        }

        response = requests.get(url, headers=config.headers, params=params)
        response.raise_for_status()
        data = response.json()

        # Stop if there are no more runs on this page
        if not data["workflow_runs"]:
            break

        for run in data["workflow_runs"]:
            start_time = datetime.fromisoformat(run["created_at"][:-1])
            end_time = datetime.fromisoformat(run["updated_at"][:-1])
            duration = (end_time - start_time).total_seconds()
            total_duration += duration
            count += 1

        # Move to the next page
        page += 1

    if count > 0:
        average_duration = total_duration / count / 60  # Convert seconds to minutes
        return average_duration
    else:
        return None


@dataclass
class GithubIssueConfig(GithubApiConfig):
    label: str = ""


def get_closed_issues_with_label(config: GithubIssueConfig) -> int:
    """Fetch issues closed with the label as label."""
    closed_issues_count = 0
    page = 1
    issues_url = f"https://api.github.com/repos/{config.repo_owner}/{config.repo_name}/issues"
    while True:
        # Request issues page by page
        params = {"state": "closed", "labels": config.label, "since": config.time_since, "per_page": 100, "page": page}

        response = requests.get(issues_url, headers=config.headers, params=params)
        response.raise_for_status()
        issues = response.json()

        # Stop if there are no more issues on this page
        if not issues:
            break

        # Count the closed issues
        for issue in issues:
            if "closed_at" in issue and datetime.fromisoformat(issue["closed_at"][:-1]) >= datetime.fromisoformat(
                config.time_since
            ):
                closed_issues_count += 1

        # Move to the next page
        page += 1

    return closed_issues_count


# Run the main function
if __name__ == "__main__":
    config = GithubApiConfig(os.getenv("GITHUB_TOKEN"), time_since=(datetime.now() - timedelta(days=7)).isoformat())
    print(get_average_duration_for_all_workflows(config))
    print(
        get_closed_issues_with_label(
            GithubIssueConfig(
                github_token=os.getenv("GITHUB_TOKEN"),
                time_since=(datetime.now() - timedelta(days=7)).isoformat(),
                label="infrastructure",
            )
        )
    )
