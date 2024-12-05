import os
from datetime import datetime, timedelta

from experiments.metrics.gcp_related import NumRestartConfig, get_gcp_restart_events
from experiments.metrics.github_related import (
    GithubApiConfig,
    GithubIssueConfig,
    get_average_duration_for_all_workflows,
    get_closed_issues_with_label,
)


def main():

    final_metrics = {}
    final_metrics["Workflow Times per workflow"] = get_average_duration_for_all_workflows(
        GithubApiConfig(
            github_token=os.getenv("GITHUB_TOKEN"), time_since=(datetime.now() - timedelta(days=7)).isoformat()
        )
    )

    label = "experiments"
    final_metrics[f"Closed Issues with label {label}"] = get_closed_issues_with_label(
        GithubIssueConfig(
            github_token=os.getenv("GITHUB_TOKEN"),
            time_since=(datetime.now() - timedelta(days=7)).isoformat(),
            label=label,
        ),
    )

    events = get_gcp_restart_events(
        NumRestartConfig(time_since=((datetime.now() - timedelta(days=7)).isoformat("T") + "Z"))
    )
    final_metrics["Number of Ray cluster restart"] = len(events)
    final_metrics["Ray restart events"] = events
    return final_metrics


if __name__ == "__main__":
    print(main())
