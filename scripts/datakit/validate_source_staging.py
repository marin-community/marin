# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every Datakit source's staged dump terminated SUCCESS.

Each :class:`marin.datakit.sources.DatakitSource` with a non-empty ``staged_path``
must resolve to a GCS prefix under ``gs://marin-us-central1`` whose
``.executor_status`` file (plain text or legacy JSON-lines) reports ``SUCCESS`` —
otherwise the ferry's verify-only download step is pointing at a partial or
missing dump. Enforced daily as a parallel lane of the datakit-smoke workflow.
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from marin.datakit.sources import all_sources
from marin.execution.executor_step_status import STATUS_SUCCESS, StatusFile
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

BUCKET = "gs://marin-us-central1"
MAX_WORKERS = 16
WORKER_ID = "datakit-smoke-sources-check"


def _check(staged_path: str) -> tuple[str, str]:
    """Return (output_path, status) where status is ``SUCCESS`` or a failure token."""
    output_path = f"{BUCKET}/{staged_path}"
    status = StatusFile(output_path, worker_id=WORKER_ID).status
    return output_path, status or "MISSING"


def main() -> None:
    configure_logging()
    sources = all_sources()
    unique_paths = sorted({s.staged_path for s in sources.values() if s.staged_path})
    logger.info("Verifying %d unique staged paths under %s", len(unique_paths), BUCKET)

    bad: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for output_path, status in pool.map(_check, unique_paths):
            if status == STATUS_SUCCESS:
                logger.debug("OK: %s", output_path)
            else:
                logger.error("%s: %s", status, output_path)
                bad.append((output_path, status))

    if bad:
        raise SystemExit(f"{len(bad)}/{len(unique_paths)} staged paths not SUCCESS under {BUCKET}")
    logger.info("All %d staged paths report SUCCESS under %s", len(unique_paths), BUCKET)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
