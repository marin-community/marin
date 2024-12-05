import argparse
import json
import os
from datetime import datetime, timedelta

import fsspec

from experiments.metrics.gcp_related import NumRestartConfig, get_gcp_restart_events
from experiments.metrics.github_related import (
    GithubApiConfig,
    GithubIssueConfig,
    get_average_duration_for_all_workflows,
    get_closed_issues_with_label,
)
from experiments.metrics.wandb_related import WandbMetricsConfig, calculate_wandb_metrics


def main(save_path: str) -> dict:

    final_metrics = {
        "Workflow Times per workflow": get_average_duration_for_all_workflows(
            GithubApiConfig(
                github_token=os.getenv("GITHUB_TOKEN"), time_since=(datetime.now() - timedelta(days=7)).isoformat()
            )
        )
    }

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

    experiment_metrics = calculate_wandb_metrics(
        WandbMetricsConfig(num_days=7, entity="stanford-mercury", project="marin")
    )
    for key, value in experiment_metrics.items():
        final_metrics[key] = value
    today = datetime.now().strftime("%Y%m%d")

    # Log final metrics json to GCP path
    save_path = os.path.join(save_path, today, "metrics.json")
    with fsspec.open(save_path, "w") as f:
        print(json.dumps(final_metrics), file=f)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get metrics.")
    parser.add_argument("--save_path", help="Save path for the metrics", default="gs://marin-us-central2/metrics")
    args = parser.parse_args()
    print(main(args.save_path))
