from datetime import datetime, timedelta, timezone
from typing import Any

import wandb


def get_wandb_run_metrics(
    run_id: str, metrics=None, entity: str = "stanford-mercury", project: str = "marin"
) -> dict[str, Any]:
    """
    Retrieves key metrics for a specific WandB run.

    Args:
    - run_id (str): The ID of the WandB run.
    - entity (str): The WandB entity (user or organization).
    - project (str): The name of the WandB project.

    Returns:
    - dict: A dictionary containing relevant metrics for the run.
    """
    if metrics is None:
        metrics = ["eval/paloma/c4_en/bpb", "throughput/total_gflops", "_runtime"]

    # Initialize the WandB API
    api = wandb.Api()

    try:
        # Fetch the specified run
        run = api.run(f"{entity}/{project}/{run_id}")

        assert run is not None, f"Run {run_id} not found."

        metrics_dict = {}
        for metric in metrics:
            # Retrieve the metric value for the run
            value = run.summary.get(metric, None)
            metrics_dict[metric] = value

        return metrics_dict

    except wandb.errors.CommError as e:
        print("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        print(f"An unexpected error occurred when trying to retrieve run data: {e}")
        return None


def get_flops_usage_over_period(num_days=7, entity="stanford-mercury", project="marin") -> float:
    """
    Calculates the total FLOPs used across all runs in the past week (or given time window).

    Args:
    - num_days (int): The number of days to look back for FLOPs usage.
    - entity (str): The WandB entity (user or organization).
    - project (str): The name of the WandB project.

    Returns:
    - float: The total FLOPs used in the past week.
    """
    # Initialize the WandB API
    api = wandb.Api()

    # Define the time window (one week ago from now)
    one_week_ago = datetime.now(timezone.utc) - timedelta(days=num_days)

    # Initialize total FLOP usage
    total_flops = 0.0

    try:
        # Fetch all runs for the project
        runs = api.runs(f"{entity}/{project}")

        # Filter and sum FLOP usage for runs created within the past week
        num_runs = 0
        for run in runs:
            run_created_at = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S%z")

            # if run was created in the past week
            if run_created_at >= one_week_ago:

                gflops = run.summary.get("throughput/total_gflops", 0.0)
                total_flops += gflops * 1e9
                num_runs += 1

        print(f"Total runs in the past week: {num_runs}")

        return total_flops

    except wandb.errors.CommError as e:
        print("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        print(f"An unexpected error occurred when trying to get FLOPs info from WandB: {e}")
        return None


if __name__ == "__main__":
    run_metrics = get_wandb_run_metrics("exp446-fineweb-edu-1.4b-9e4be7")
    print(run_metrics)

    flops_usage = get_flops_usage_over_period()
    print(f"FLOPs used in the past week: {flops_usage} flops or {flops_usage / 1e9} GFLOPs")
