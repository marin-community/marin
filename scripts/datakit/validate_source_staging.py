# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every Datakit source's pre-normalize step terminated SUCCESS.

For each :class:`marin.datakit.sources.DatakitSource`, the first pre-normalize
step (``normalize_steps[:-1][0]``) is the staged raw dump the ferry expects
to exist on GCS. The staging region is pinned via ``MARIN_PREFIX`` so
``step.output_path`` resolves to ``gs://marin-us-central1/...`` regardless of
the caller's environment. Enforced daily as a parallel lane of the
datakit-smoke workflow.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from marin.datakit.sources import all_sources
from marin.execution.executor_step_status import STATUS_SUCCESS, StatusFile
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

STAGING_PREFIX = "gs://marin-us-central1"
MAX_WORKERS = 16
WORKER_ID = "datakit-smoke-sources-check"


def _check(output_path: str) -> tuple[str, str]:
    """Return (output_path, status) where status is ``SUCCESS`` or a failure token."""
    status = StatusFile(output_path, worker_id=WORKER_ID).status
    return output_path, status or "MISSING"


def main() -> None:
    configure_logging()
    # Pin the staging region before building the registry — StepSpec caches
    # output_path on first access, so MARIN_PREFIX must be set before
    # all_sources() materializes any chain.
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    sources = all_sources()
    unique_paths = sorted({s.normalize_steps[:-1][0].output_path for s in sources.values()})
    logger.info("Verifying %d unique pre-normalize paths under %s", len(unique_paths), STAGING_PREFIX)

    bad: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for output_path, status in pool.map(_check, unique_paths):
            if status == STATUS_SUCCESS:
                logger.debug("OK: %s", output_path)
            else:
                logger.error("%s: %s", status, output_path)
                bad.append((output_path, status))

    if bad:
        raise SystemExit(f"{len(bad)}/{len(unique_paths)} pre-normalize paths not SUCCESS under {STAGING_PREFIX}")
    logger.info("All %d pre-normalize paths report SUCCESS under %s", len(unique_paths), STAGING_PREFIX)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
