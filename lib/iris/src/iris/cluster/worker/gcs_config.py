# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS configuration for Iris worker log persistence."""

import logging
import os

logger = logging.getLogger(__name__)


def get_iris_log_prefix() -> str | None:
    """Get the GCS prefix for Iris worker logs.

    Checks IRIS_WORKER_PREFIX environment variable first. If not set,
    attempts to infer from the VM's GCS region using the metadata API.
    Falls back to local storage if not running on GCP.

    Returns:
        GCS prefix (e.g., "gs://marin-tmp-us-central2/ttl=30d/iris-logs"),
        local path if not on GCP, or None if disabled.
    """
    # Explicit configuration takes precedence
    prefix = os.environ.get("IRIS_WORKER_PREFIX")
    if prefix:
        logger.info("Using IRIS_WORKER_PREFIX=%s", prefix)
        return prefix

    # Try to infer from GCP metadata
    try:
        from marin.utilities.gcs_utils import get_vm_region

        region = get_vm_region()
        inferred_prefix = f"gs://marin-tmp-{region}/ttl=30d/iris-logs"
        logger.info("Inferred IRIS_WORKER_PREFIX from region %s: %s", region, inferred_prefix)
        return inferred_prefix
    except Exception as e:
        logger.debug("Could not infer IRIS_WORKER_PREFIX from GCP metadata: %s", e)

    # Not on GCP or can't detect - disable GCS logging
    logger.info("IRIS_WORKER_PREFIX not set and region detection failed; GCS log persistence disabled")
    return None
