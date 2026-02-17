# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS configuration for Iris worker log persistence."""

import logging
import os

logger = logging.getLogger(__name__)


def get_iris_log_prefix() -> str | None:
    """Get GCS prefix for Iris worker logs.

    Returns the IRIS_WORKER_PREFIX environment variable if set, otherwise
    infers from GCP region metadata. Returns None if not on GCP or env var not set.

    Returns:
        GCS prefix like "gs://marin-tmp-us-central2/ttl=30d/iris-logs" or None
    """
    # Explicit configuration takes precedence
    prefix = os.environ.get("IRIS_WORKER_PREFIX")
    if prefix:
        return prefix

    # Fallback: infer from VM region
    try:
        from marin.utilities.gcs_utils import get_vm_region

        region = get_vm_region()
        inferred_prefix = f"gs://marin-tmp-{region}/ttl=30d/iris-logs"
        logger.info(f"Inferred IRIS_WORKER_PREFIX from region: {inferred_prefix}")
        return inferred_prefix
    except (ImportError, ValueError) as e:
        logger.debug(f"Could not infer IRIS_WORKER_PREFIX: {e}")
        return None
