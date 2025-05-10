import argparse
import json
import logging
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
from infra.github_wandb_metrics import log_data_to_wandb

logger = logging.getLogger(__name__)


def compute_metrics(save_path: str) -> dict:

    final_metrics = {}

    final_metrics = {
        "Workflow Times": get_average_duration_for_all_workflows(
            GithubApiConfig(
                github_token=os.getenv("GITHUB_TOKEN"), time_since=(datetime.now() - timedelta(days=7)).isoformat()
            )
        )
    }

    logger.info("Workflow Times: %s", final_metrics["Workflow Times"])

    label = "experiments"
    final_metrics[f"Closed Issues with label {label}"] = get_closed_issues_with_label(
        GithubIssueConfig(
            github_token=os.getenv("GITHUB_TOKEN"),
            time_since=(datetime.now() - timedelta(days=7)).isoformat(),
            label=label,
        ),
    )

    logger.info("Closed Issues with label %s: %s", label, final_metrics[f"Closed Issues with label {label}"])

    events = get_gcp_restart_events(
        NumRestartConfig(time_since=((datetime.now() - timedelta(days=7)).isoformat("T") + "Z"))
    )
    final_metrics["Number of Ray cluster restarts"] = len(events)
    final_metrics["Ray restart events"] = events

    logger.info("Number of Ray cluster restarts: %s", final_metrics["Number of Ray cluster restarts"])

    # get all runs; num_days=-1 means all runs
    experiment_metrics = calculate_wandb_metrics(
        WandbMetricsConfig(num_days=None, entity="stanford-mercury", project="marin")
    )

    logger.info("Experiment metrics: %s", experiment_metrics)

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
    data = compute_metrics(args.save_path)
    print(data)
    log_data_to_wandb(data)
