import logging
import os
from typing import Any

import wandb

logger = logging.getLogger(__name__)

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "marin-community")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "marin")
WANDB_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"


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
        metrics = []

    # initialize the WandB API
    api = wandb.Api()
    try:
        # fetch the specified run
        run = api.run(f"{entity}/{project}/{run_id}")

        assert run is not None, f"Run {run_id} not found."

        metrics_dict = {}
        for metric in metrics:
            # retrieve the metric value for the run
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
