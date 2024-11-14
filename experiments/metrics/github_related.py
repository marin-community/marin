import json
import os
from dataclasses import dataclass
from datetime import datetime

import fsspec
import requests


@dataclass
class GITHUB_API_CONFIG:
    GITHUB_TOKEN: str
    output_path: str
    TIME_SINCE: str
    REPO_NAME: str = "marin"
    REPO_OWNER: str = "stanford-crfm"
    headers: dict = None

    def __post_init__(self):
        self.headers = {"Authorization": f"Bearer {self.GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}


def get_average_duration_for_all_workflows(config: GITHUB_API_CONFIG) -> dict[str, float]:
    """Fetch all workflows and calculate the average duration (in min) in the time duration for each workflow."""

    workflow_url = f"https://api.github.com/repos/{config.REPO_OWNER}/{config.REPO_NAME}/actions/workflows"
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

    with fsspec.open(os.path.join(config.output_path, "metric.json"), "w") as f:
        print(json.dumps({"Workflow Times per workflow": average_times}), file=f)

    return average_times


def get_average_duration(config: GITHUB_API_CONFIG, workflow_id: int) -> float | None:
    """Fetch workflow run and calculate the average duration for a specific workflow."""

    url = f"https://api.github.com/repos/{config.REPO_OWNER}/{config.REPO_NAME}/actions/workflows/{workflow_id}/runs"
    total_duration = 0
    count = 0
    page = 1

    while True:
        # Request runs page by page
        params = {
            "created": f">={config.TIME_SINCE}",  # Filter for runs created in the past week
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
class GITHUB_ISSUE_CONFIG(GITHUB_API_CONFIG):
    LABEL: str = ""


def get_closed_issues_with_label(config: GITHUB_ISSUE_CONFIG) -> int:
    """Fetch issues closed with the label as label."""
    closed_issues_count = 0
    page = 1
    issues_url = f"https://api.github.com/repos/{config.REPO_OWNER}/{config.REPO_NAME}/issues"
    while True:
        # Request issues page by page
        params = {"state": "closed", "labels": config.LABEL, "since": config.TIME_SINCE, "per_page": 100, "page": page}

        response = requests.get(issues_url, headers=config.headers, params=params)
        response.raise_for_status()
        issues = response.json()

        # Stop if there are no more issues on this page
        if not issues:
            break

        # Count the closed issues
        for issue in issues:
            if "closed_at" in issue and datetime.fromisoformat(issue["closed_at"][:-1]) >= datetime.fromisoformat(
                config.TIME_SINCE
            ):
                closed_issues_count += 1

        # Move to the next page
        page += 1

    with fsspec.open(os.path.join(config.output_path, "metric.json"), "w") as f:
        print(json.dumps({f"Closed Issues with label {config.LABEL}": closed_issues_count}), file=f)

    return closed_issues_count


# Run the main function
if __name__ == "__main__":
    config = GITHUB_API_CONFIG(os.getenv("GITHUB_TOKEN"))
    print(get_average_duration_for_all_workflows(config))
    print(get_closed_issues_with_label(config, "infrastructure"))
