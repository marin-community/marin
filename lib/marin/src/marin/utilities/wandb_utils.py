# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime, timezone
from typing import Any

import wandb

logger = logging.getLogger(__name__)

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "marin-community")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "marin")


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
    if "WANDB_API_KEY" not in os.environ or not os.environ["WANDB_API_KEY"]:
        logger.info("WANDB_API_KEY not set, skipping wandb initialization.")
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
