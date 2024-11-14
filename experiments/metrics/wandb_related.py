import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import fsspec
import wandb


@dataclass
class WANDB_METRICS_CONFIG:

    entity: str
    project: str
    num_days: int  # number of days before today to get metrics for
    output_path: str


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
            if value is None:
                print(f"Metric '{metric}' not found for run {run_id}")
            metrics_dict[metric] = value

        print("Run metrics: ", metrics_dict)

        return metrics_dict

    except wandb.errors.CommError as e:
        print("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        print(f"An unexpected error occurred when trying to retrieve run data: {e}")
        return None


def get_all_runs_over_period(num_days=7, entity="stanford-mercury", project="marin") -> dict[str, Any]:
    """
    Retrieves all runs created within the past week (or given time window).
    """
    # Initialize the WandB API
    api = wandb.Api()

    # Define the time window (one week ago from now)
    time_window = datetime.now(timezone.utc) - timedelta(days=num_days)

    # Initialize the list of runs
    runs_list = []

    try:

        # Check if project exists first
        try:
            api.project(entity, project)
        except wandb.errors.CommError:
            print(f"Project '{project}' not found in entity '{entity}'")
            return None

        # Fetch all runs for the project
        runs = api.runs(f"{entity}/{project}")

        print(runs)

        # Filter runs by creation date
        runs_list = [run for run in runs if datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S%z") >= time_window]

        print(f"Successfully retrieved {len(runs_list)} runs from the past {num_days} days")

        return runs_list

    except wandb.errors.CommError as e:
        print("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        print(f"An unexpected error occurred when trying to get runs from WandB: {e}")
        return None


def count_params_for_run(run_id: str, entity="stanford-mercury", project="marin") -> int:
    """
    Retrieves the number of parameters for a specific WandB run.
    """
    from experiments.llama import compute_num_parameters

    # Initialize the WandB API
    api = wandb.Api()

    try:
        # Fetch the specified run
        run = api.run(f"{entity}/{project}/{run_id}")

        assert run is not None, f"Run {run_id} not found."

        # Calculate the number of parameters for the run
        tokenizer = run.config.get("data", {}).get("tokenizer")
        if tokenizer == "EleutherAI/gpt-neox-20b":
            vocab_size = 50_257
        elif tokenizer == "meta-llama/Meta-Llama-3.1-8B":
            vocab_size = 128_256
        elif tokenizer == "meta-llama/Llama-2-7b":
            vocab_size = 32_000

        model_config = run.config.get("model", {})

        num_parameters = compute_num_parameters(config=model_config, vocab_size=vocab_size)

        print(f"Number of parameters for run {run_id}: {num_parameters}")

        return num_parameters

    except wandb.errors.CommError as e:
        print("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        print(f"An unexpected error occurred when trying to retrieve run data: {e}")
        return None


def calculate_wandb_metrics(config: WANDB_METRICS_CONFIG) -> tuple[dict[str, Any], str]:
    """
    Calculate and upload metrics to GCS for the past num_days.

    Returns:
    - tuple: a metrics dict and the GCS path to the metrics file.
    """

    # get all runs from past num_days
    runs = get_all_runs_over_period(num_days=config.num_days, entity=config.entity, project=config.project)

    # get metrics for each run
    run_metrics = {}
    for run in runs:
        run_id = run.id
        run_metrics[run_id] = get_wandb_run_metrics(run_id)

        # get parameter count for the run and add to metrics
        run_metrics[run_id]["num_parameters"] = count_params_for_run(run_id)

    # get the model with best bpb eval for each group in 1b parameter scale
    best_bpb_1b, best_bpb_7b = None, None
    best_bpb1b_run_id, best_bpb7b_run_id = None, None
    for run_id, metrics in run_metrics.items():
        if metrics["eval/paloma/c4_en/bpb"] is not None:
            num_parameters = metrics["num_parameters"]

            # if num_parameters is below 2 billion and bpb is better than current best, update best_bpb
            if num_parameters < 2_000_000_000 and (
                best_bpb_1b is None or metrics["eval/paloma/c4_en/bpb"] < best_bpb_1b
            ):
                best_bpb_1b = metrics["eval/paloma/c4_en/bpb"]
                best_bpb1b_run_id = run_id

            # if num_parameters is above 6 billion and bpb is better than current best, update best_bpb
            if num_parameters > 6_000_000_000 and (
                best_bpb_7b is None or metrics["eval/paloma/c4_en/bpb"] < best_bpb_7b
            ):
                best_bpb_7b = metrics["eval/paloma/c4_en/bpb"]
                best_bpb7b_run_id = run_id

    metrics = {
        "num_runs": len(run_metrics),
        "best_c4_en_bpb": {
            "1b": {
                "run_id": best_bpb1b_run_id,
                "run_metrics": run_metrics[best_bpb1b_run_id] if best_bpb1b_run_id else None,
            },
            "7b": {
                "run_id": best_bpb7b_run_id,
                "run_metrics": run_metrics[best_bpb7b_run_id] if best_bpb7b_run_id else None,
            },
        },
        "total_gflops_across_runs": sum(
            [
                metrics["throughput/total_gflops"]
                for metrics in run_metrics.values()
                if metrics["throughput/total_gflops"] is not None
            ]
        ),
    }

    with fsspec.open(os.path.join(config.output_path, "metric.json"), "w") as f:
        print(json.dumps(metrics), file=f)


if __name__ == "__main__":
    config = WANDB_METRICS_CONFIG(entity="stanford-mercury", project="marin", num_days=7, output_path=".")
    calculate_wandb_metrics(config)
