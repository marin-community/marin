import os

from experiments.metrics.gcp_related import GCP_API_CONFIG, get_number_of_restarts
from experiments.metrics.github_related import (
    GITHUB_API_CONFIG,
    get_average_duration_for_all_workflows,
    get_closed_issues_with_label,
    post_comment_on_issue,
)

if __name__ == "__main__":
    comment = "# Metrics calculation for the past week."
    comment += f"\n1. Number of restarts for ray cluster: {get_number_of_restarts(GCP_API_CONFIG())}"

    config = GITHUB_API_CONFIG(os.getenv("GITHUB_TOKEN"))
    comment += f"\n2. Average duration for all workflows: \n " f"{get_average_duration_for_all_workflows(config)}"
    comment += (
        f"\n3. Number of closed issues with label 'experiments': "
        f"{get_closed_issues_with_label(config, 'experiments')}"
    )
    post_comment_on_issue(config, 547, comment)
