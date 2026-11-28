# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive ingest execution identity from the checkout and storage deployment."""

import subprocess
from pathlib import Path

from experiments.post_training.task_curation.atlas_deployment import region_deployment
from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.release_stages.plan_derivation import _freeze_artifact_version


def derive_execution(config_path: Path, repo_root: Path, worker_argv: tuple[str, ...]) -> IngestExecution:
    """Resolve the code revision, calendar version and regional artifact root."""
    policy = load_ingest_config(config_path)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    deployment = region_deployment(policy.cluster)
    return IngestExecution(
        policy=policy,
        source_revision=revision,
        artifact_version=_freeze_artifact_version(repo_root, revision),
        artifact_root=f"{deployment.region_prefix}ingest/{policy.digest()}/{revision}",
        worker_argv=worker_argv,
    )
