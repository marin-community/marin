# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Node-local paths and cleanup for an Iris GPU autotune snapshot."""

import hashlib
import logging
import pathlib
import shutil

from iris.cluster.client.job_info import JobInfo
from iris.cluster.runtime.env import SCRATCH_CACHE_PATH

logger = logging.getLogger(__name__)

AUTOTUNE_CACHE_SUBDIR = "xla/per-fusion-autotune"


def autotune_job_dir(job_info: JobInfo, cache_path: str = SCRATCH_CACHE_PATH) -> pathlib.Path:
    """Return the node-local directory owned by one task attempt."""
    attempt = f"{job_info.task_id.to_wire()}:{job_info.attempt_id}:{job_info.attempt_uid or ''}"
    attempt_key = hashlib.sha256(attempt.encode()).hexdigest()[:24]
    return pathlib.Path(cache_path, AUTOTUNE_CACHE_SUBDIR, "jobs", attempt_key)


def cleanup_shared_autotune_cache(job_info: JobInfo, cache_path: str = SCRATCH_CACHE_PATH) -> None:
    """Remove an attempt's local snapshot after its processes have exited."""
    job_dir = autotune_job_dir(job_info, cache_path)
    if not job_dir.exists():
        return
    try:
        for path in job_dir.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        shutil.rmtree(job_dir)
    except OSError as exc:
        logger.warning("Could not remove XLA autotune snapshot %s: %s", job_dir, exc)
