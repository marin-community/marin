import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import wandb
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

logger = logging.getLogger(__name__)


@dataclass
class WandbMetricsConfig:
    entity: str
    project: str

    # restrict aggregation to this many days before today
    num_days: int | None  # use None for specifying no time window (i.e, all runs are counted)


def get_wandb_run_metrics(
    run_id: str, metrics=None, entity: str = WANDB_ENTITY, project: str = WANDB_PROJECT
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
        metrics = [
            "eval/paloma/c4_en/bpb",
            "lm_eval/averages/macro_avg_acc",
            "throughput/total_gflops",
            "_runtime",
            "parameter_count",
        ]

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

            # if the metric is not found or is not a valid value, set it to None
            # A metric being None means we skip that run in the aggregation
            if value is not None:
                if isinstance(value, str) and value.lower() == "nan":
                    # happens when evals fail or haven't been run, typically for quickstart/test runs
                    logger.info(f"Metric '{metric}' for run {run_id} is 'NaN'. Setting to None.")
                    value = None
                else:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        logger.info(f"Metric '{metric}' for run {run_id} is not a float: {value}. Setting to None.")
                        value = None
            else:
                logger.info(f"Metric '{metric}' not found for run {run_id}.")
            metrics_dict[metric] = value

        logger.info("Run metrics: ", metrics_dict)

        return metrics_dict

    except wandb.errors.CommError as e:
        logger.error("Failed to retrieve run data:", e)
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred when trying to retrieve run data: {e}")
        return None


def get_all_runs_over_period(
    num_days: int | None = None,
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
) -> list[dict[str, Any]] | None:
    """
    Retrieves all runs created within the past week (or given time window).
    If num_days is None, it will retrieve all runs for the project.
    """
    # Initialize the WandB API
    api = wandb.Api()

    try:

        # Check if project exists first
        try:
            api.project(entity, project)
        except wandb.errors.CommError:
            logger.error(f"Project '{project}' not found in entity '{entity}'")
            return None

        # Fetch all runs for the project
        runs = api.runs(f"{entity}/{project}")

        logger.info(f"Found {len(runs)} runs in project {project}")

        # Filter runs by update date
        if num_days is not None:
            time_window = datetime.now(timezone.utc) - timedelta(days=num_days)
            filtered_runs = [
                run for run in runs if datetime.strptime(run.updated_at, "%Y-%m-%dT%H:%M:%S%z") >= time_window
            ]
            logger.info(f"Successfully retrieved {len(filtered_runs)} runs updated in the past {num_days} days")
        else:
            filtered_runs = runs
            logger.info(f"Successfully retrieved {len(filtered_runs)} runs since the beginning of time")

        return filtered_runs

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


def count_params_for_run(run_id: str, entity=WANDB_ENTITY, project=WANDB_PROJECT) -> int | None:
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

        tokenizer = run.train_config.get("data", {}).get("tokenizer", None)
        vocab_size = get_vocab_size_for_tokenizer(tokenizer)
        if not vocab_size:
            return None
        model_dict = run.train_config.get("model", {})
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

    logger.info("Calculating metrics for WandB runs...")
    logger.info(f"Entity: {config.entity}, Project: {config.project}, Num days: {config.num_days}")

    # get all runs from past num_days
    runs = get_all_runs_over_period(num_days=config.num_days, entity=config.entity, project=config.project)

    # get metrics for each run
    run_metrics = {}
    for run in runs:
        run_id = run.id
        run_metrics[run_id] = get_wandb_run_metrics(run_id)

    # Define parameter scale thresholds (exclusive upper bounds)
    parameter_scales = {
        "<1B": 1_000_000_000,  # for runs in the millions of params (300M, etc.)
        "<2B": 2_000_000_000,  # for 1B scale models or smaller
        "<9B": 9_000_000_000,  # for 7 or 8B scale models or smaller
        "<23B": 23_000_000_000,  # for 22B scale models or smaller
    }

    # track best metrics for each scale
    best_metrics_per_scale = {
        scale: {"bpb": {"value": None, "run_id": None}, "macro_avg_acc": {"value": None, "run_id": None}}
        for scale in parameter_scales
    }

    # Initialize final metrics dictionary
    final_metrics = {}

    for run_id, metrics in run_metrics.items():
        if metrics["parameter_count"] is None:
            continue

        try:
            num_parameters = float(metrics["parameter_count"])
        except (ValueError, TypeError):
            logger.info(f"Invalid parameter count for run {run_id}: {metrics['parameter_count']}. Skipping.")
            continue

        # Check BPB metric
        if metrics["eval/paloma/c4_en/bpb"] is not None:
            try:
                bpb = float(metrics["eval/paloma/c4_en/bpb"])
                for scale_label, threshold in parameter_scales.items():
                    if num_parameters < threshold:
                        current_best_bpb = best_metrics_per_scale[scale_label]["bpb"]["value"]
                        if current_best_bpb is None or bpb < current_best_bpb:
                            best_metrics_per_scale[scale_label]["bpb"] = {"value": bpb, "run_id": run_id}
            except (ValueError, TypeError):
                logger.info(f"Invalid BPB for run {run_id}: {metrics['eval/paloma/c4_en/bpb']}. Skipping BPB metric.")

        # Check macro_avg_acc metric
        if metrics.get("lm_eval/averages/macro_avg_acc") is not None:
            try:
                macro_avg_acc = float(metrics["lm_eval/averages/macro_avg_acc"])
                for scale_label, threshold in parameter_scales.items():
                    if num_parameters < threshold:
                        current_best_acc = best_metrics_per_scale[scale_label]["macro_avg_acc"]["value"]
                        if (
                            current_best_acc is None or macro_avg_acc > current_best_acc
                        ):  # Note: higher is better for accuracy
                            best_metrics_per_scale[scale_label]["macro_avg_acc"] = {
                                "value": macro_avg_acc,
                                "run_id": run_id,
                            }
            except (ValueError, TypeError):
                logger.info(
                    f"""Invalid macro_avg_acc for run {run_id}:
                    {metrics['lm_eval/averages/macro_avg_acc']}. Skipping accuracy metric."""
                )
    # Add best metrics to final metrics
    for scale, data in best_metrics_per_scale.items():
        # Add BPB metrics
        if data["bpb"]["value"] is not None:
            final_metrics[f"Best BPB for {scale}"] = data["bpb"]["value"]
            final_metrics[f"Best BPB Run ID for {scale}"] = data["bpb"]["run_id"]

        # Add macro_avg_acc metrics
        if data["macro_avg_acc"]["value"] is not None:
            final_metrics[f"Best Macro Avg Acc for {scale}"] = data["macro_avg_acc"]["value"]
            final_metrics[f"Best Macro Avg Acc Run ID for {scale}"] = data["macro_avg_acc"]["run_id"]

    # calculate total gflops across all runs
    total_gflops = sum(
        [
            metrics["throughput/total_gflops"]
            for metrics in run_metrics.values()
            if metrics["throughput/total_gflops"] is not None
        ]
    )

    # Add additional metrics
    final_metrics.update(
        {
            "num_runs": len(run_metrics),
            "best_c4_en_bpb": {
                scale: {
                    "run_id": data["bpb"]["run_id"],
                    "run_metrics": run_metrics[data["bpb"]["run_id"]] if data["bpb"]["run_id"] else None,
                }
                for scale, data in best_metrics_per_scale.items()
            },
            "best_macro_avg_acc": {
                scale: {
                    "run_id": data["macro_avg_acc"]["run_id"],
                    "run_metrics": (
                        run_metrics[data["macro_avg_acc"]["run_id"]] if data["macro_avg_acc"]["run_id"] else None
                    ),
                }
                for scale, data in best_metrics_per_scale.items()
            },
            "total_gflops_across_runs": total_gflops,
            "total_petaflops_across_runs": total_gflops / 1e6,
        }
    )

    logger.info(final_metrics)
    return final_metrics


if __name__ == "__main__":

    config = WandbMetricsConfig(entity=WANDB_ENTITY, project=WANDB_PROJECT, num_days=7)

    calculate_wandb_metrics(config)
