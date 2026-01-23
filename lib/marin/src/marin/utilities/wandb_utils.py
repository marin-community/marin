# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from datetime import datetime, timezone
from typing import Any

import wandb

logger = logging.getLogger(__name__)

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "marin-community")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "marin")
WANDB_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"


def init_wandb(
    *,
    run_name: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
) -> wandb.sdk.wandb_run.Run | None:
    """
    Initialize a wandb run if WANDB_API_KEY is set.

    Args:
        run_name: Name for the run. If not provided, uses WANDB_RUN_NAME env var
            or generates a timestamp-based name.
        tags: Optional list of tags to apply to the run.
        config: Optional config dict to log with the run.
        entity: Wandb entity (defaults to WANDB_ENTITY).
        project: Wandb project (defaults to WANDB_PROJECT).

    Returns:
        The wandb run object if initialized, None if WANDB_API_KEY is not set.
    """
    if "WANDB_API_KEY" not in os.environ:
        return None

    if not run_name:
        run_name = os.environ.get("WANDB_RUN_NAME")
    if not run_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_name = f"run-{timestamp}"

    return wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        tags=tags,
        config=config,
    )


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
