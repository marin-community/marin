import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import wandb

logger = logging.getLogger(__name__)


@dataclass
class WandbMetricsConfig:
    entity: str
    project: str
    num_days: int  # number of days before today to get metrics for


def get_wandb_run_metrics(
    run_id: str, metrics=None, entity: str = "stanford-mercury", project: str = "marin"
) -> dict[str, Any] | None:
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
                logger.info(f"Metric '{metric}' not found for run {run_id}")
            metrics_dict[metric] = value

        logger.info("Run metrics: ", metrics_dict)

        return metrics_dict

    except wandb.errors.CommError as e:
        logger.error("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred when trying to retrieve run data: {e}")
        return None


def get_all_runs_over_period(num_days=7, entity="stanford-mercury", project="marin") -> list[dict[str, Any]] | None:
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
            logger.error(f"Project '{project}' not found in entity '{entity}'")
            return None

        # Fetch all runs for the project
        runs = api.runs(f"{entity}/{project}")

        # Filter runs by creation date
        runs_list = [run for run in runs if datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S%z") >= time_window]

        logger.info(f"Successfully retrieved {len(runs_list)} runs from the past {num_days} days")

        return runs_list

    except wandb.errors.CommError as e:
        logger.error("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred when trying to get runs from WandB: {e}")
        return None


def get_vocab_size_for_tokenizer(tokenizer: str) -> int | None:

    logger.info(f"Tokenizer:{tokenizer}")
    if tokenizer == "EleutherAI/gpt-neox-20b":
        vocab_size = 50_257
    elif tokenizer == "meta-llama/Meta-Llama-3.1-8B":
        vocab_size = 128_256
    elif tokenizer == "meta-llama/Llama-2-7b":
        vocab_size = 32_000
    elif tokenizer == "gpt2":
        vocab_size = 50_257
    else:
        logger.error(f"Unknown tokenizer: {tokenizer}")
        return None

    logger.info(f"Vocab size:  {vocab_size}")
    return vocab_size


def count_params_for_run(run_id: str, entity="stanford-mercury", project="marin") -> int | None:
    """
    Retrieves the number of parameters for a specific WandB run.
    """
    from experiments.llama import LlamaConfig, compute_num_parameters

    # Initialize the WandB API
    api = wandb.Api()

    try:
        # Fetch the specified run
        run = api.run(f"{entity}/{project}/{run_id}")

        assert run is not None, f"Run {run_id} not found."

        tokenizer = run.config.get("data", {}).get("tokenizer", None)
        vocab_size = get_vocab_size_for_tokenizer(tokenizer)
        if not vocab_size:
            return None
        model_dict = run.config.get("model", {})
        logger.info("Model dict: ", model_dict)

        llama_config = LlamaConfig(
            hidden_dim=model_dict.get("hidden_dim"),
            num_heads=model_dict.get("num_heads"),
            num_kv_heads=model_dict.get("num_kv_heads"),
            intermediate_dim=model_dict.get("intermediate_dim"),
            num_layers=model_dict.get("num_layers"),
        )

        # Calculate the number of parameters for the run
        # TODO: actually can this just be replaced by getting the parameter_count from the run?
        # there seems be some discrepancy in the parameter count between the two methods
        num_parameters = compute_num_parameters(llama_config, vocab_size=vocab_size)
        logger.info(f"Number of parameters for run {run_id}: {num_parameters}\n")

        return num_parameters

    except wandb.errors.CommError as e:
        logger.error("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred when trying to retrieve run data: {e}")
        return None


def calculate_wandb_metrics(config: WandbMetricsConfig) -> dict[str, Any]:
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
        num_params = count_params_for_run(run_id)
        if num_params:
            run_metrics[run_id]["num_parameters"] = num_params

    # Define parameter scale thresholds (exclusive upper bounds)
    parameter_scales = {
        "<1B": 1_000_000_000,  # for runs in the millions of params (300M, etc.)
        "<2B": 2_000_000_000,  # for 1B scale models or smaller
        "<9B": 9_000_000_000,  # for 7 or 8B scale models or smaller
        "<23B": 23_000_000_000,  # for 22B scale models or smaller
    }

    # track best BPB for each scale
    best_bpb_per_scale = {scale: {"bpb": None, "run_id": None} for scale in parameter_scales}

    # get the model with best bpb eval for each group in 1b parameter scale
    # best_bpb_1b, best_bpb_7b = None, None
    # best_bpb1b_run_id, best_bpb7b_run_id = None, None
    for run_id, metrics in run_metrics.items():
        if metrics["eval/paloma/c4_en/bpb"] is not None:
            num_parameters = metrics["num_parameters"]
            for scale_label, threshold in parameter_scales.items():
                if num_parameters < threshold:
                    current_best_bpb = best_bpb_per_scale[scale_label]["bpb"]
                    if current_best_bpb is None or metrics["eval/paloma/c4_en/bpb"] < current_best_bpb:
                        best_bpb_per_scale[scale_label] = {
                            "bpb": metrics["eval/paloma/c4_en/bpb"],
                            "run_id": run_id,
                        }

    metrics = {
        "num_runs": len(run_metrics),
        "best_c4_en_bpb": {
            scale: {
                "run_id": data["run_id"],
                "run_metrics": run_metrics[data["run_id"]] if data["run_id"] else None,
            }
            for scale, data in best_bpb_per_scale.items()
        },
        "total_gflops_across_runs": sum(
            [
                metrics["throughput/total_gflops"]
                for metrics in run_metrics.values()
                if metrics["throughput/total_gflops"] is not None
            ]
        ),
    }

    logger.info(metrics)
    return metrics


if __name__ == "__main__":

    config = WandbMetricsConfig(entity="stanford-mercury", project="marin", num_days=7)

    calculate_wandb_metrics(config)
