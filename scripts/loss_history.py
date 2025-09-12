# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import wandb


DEFAULT_METRIC = "train/loss"


def fetch_metric_for_sha(sha, metric_name=DEFAULT_METRIC, steps=range(0, 101, 10)):
    """
    Fetches the metrics for a given git sha using the wandb API.
    """
    # Setting up the wandb API
    api = wandb.Api()

    # Search for the run tagged with this git sha
    runs = api.runs("levanter", {"config.git_commit": sha})

    metric_values = {}

    if runs:
        # Get the history of metrics for this run
        metrics = runs[0].scan_history(keys=["_step", metric_name], min_step=min(steps), max_step=max(steps))

        for row in metrics:
            step = row["_step"]
            loss = row[metric_name]
            if step not in steps:
                continue
            metric_values[int(step)] = float(loss)

    return metric_values


def visualize_git_tree_with_metric(metric_name=DEFAULT_METRIC):
    """
    Fetches the git log, and associates each commit with the metric.
    """
    # Getting the git log with shas
    result = subprocess.run(["git", "log", "--pretty=format:%H %s"], stdout=subprocess.PIPE)
    log_lines = result.stdout.decode("utf-8").strip().split("\n")

    for line in log_lines:
        sha, message = line.split(" ", 1)
        metric_values = fetch_metric_for_sha(sha, metric_name)

        metrics_str = ", ".join(f"{step}: {value}" for step, value in metric_values.items())
        print(f"{sha} - {message} -> {metrics_str}")


if __name__ == "__main__":
    visualize_git_tree_with_metric()
