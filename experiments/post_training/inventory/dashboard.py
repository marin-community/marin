# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register Quick RL attempts for the Marina dashboard's terminal importer."""

import hashlib
import json
import logging

import yaml
from iris.cluster.client.job_info import get_job_info
from marin.rl.skyrl import SkyRLRun, SkyRLRunConfig, run_skyrl
from rigging.filesystem.buckets import MissingCredentials
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

DASHBOARD_INBOX = "rl-data/quick-loop-dashboard/v1"
logger = logging.getLogger(__name__)


def run_dashboard_attempt(config: SkyRLRunConfig) -> SkyRLRun:
    """Publish the attempt locator before training, so terminal failures are discoverable."""
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
        identity = hashlib.sha256(json.dumps([config.run_id, config.attempt_id]).encode()).hexdigest()
        path = StoragePath(marin_prefix()) / DASHBOARD_INBOX / f"{identity}.json"
        try:
            path.write_text(json.dumps(entry, allow_nan=False))
        except (OSError, MissingCredentials):
            logger.exception("Could not register Quick RL attempt %s for dashboard publication", config.attempt_id)
    return run_skyrl(config)
