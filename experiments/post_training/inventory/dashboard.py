# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register Quick RL attempts for the Marina dashboard's terminal importer."""

import hashlib
import logging
from collections.abc import Callable

import yaml
from iris.cluster.client.job_info import get_job_info
from marin.execution.artifact import Artifact
from marin.rl.skyrl import SkyRLRunConfig
from rigging.filesystem.buckets import MissingCredentials
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from rigging.telemetry.serialization import json_bytes

DASHBOARD_INBOX = "rl-data/quick-loop-dashboard/v1"
logger = logging.getLogger(__name__)


def run_dashboard_attempt(
    config: SkyRLRunConfig, *, run: Callable[[SkyRLRunConfig], Artifact | None]
) -> Artifact | None:
    """Run the training step after a best-effort registration with the dashboard."""
    info = get_job_info()
    if info is not None:
        launch = yaml.safe_load(config.launch_config_yaml)
        provenance = Provenance.capture()
        entry = {
            "run_id": config.run_id,
            "attempt_id": config.attempt_id,
            "coordinator_job_id": str(info.job_id),
            "terminal_manifest_uri": config.output.terminal_manifest_uri,
            "expected_step": launch["skyrl"]["trainer"]["max_steps"],
            "source_revision": provenance.tree_hash or provenance.base_commit or "unknown",
            "data_identity": launch["inputs"]["train_data"][0]["identity"],
            "validation_identity": launch["inputs"]["validation_data"][0]["identity"],
            "initializer_identity": config.model.identity,
        }
        identity = hashlib.sha256(json_bytes([config.run_id, config.attempt_id])).hexdigest()
        path = StoragePath(marin_prefix()) / DASHBOARD_INBOX / f"{identity}.json"
        try:
            path.write_text(json_bytes(entry).decode())
        except (OSError, MissingCredentials):
            logger.exception("Could not register Quick RL attempt %s for dashboard publication", config.attempt_id)
    return run(config)
