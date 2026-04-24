# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every Datakit source's staged and normalized outputs terminated SUCCESS.

For each :class:`marin.datakit.sources.DatakitSource`, two independent
``.executor_status`` checks run against GCS:

* **pre-normalize** — the first step in ``normalize_steps`` (the raw
  staged dump the ferry expects to already exist upstream).
* **normalized** — ``source.normalized.output_path`` (the terminal
  normalize step's output — what downstream sample/tokenize consumes).

The staging region is pinned via ``MARIN_PREFIX`` so all ``output_path``s
resolve to ``gs://marin-us-central1/...`` regardless of the caller's
environment. Enforced daily as a parallel lane of the datakit-smoke
workflow.
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


def _validate(label: str, paths: list[str]) -> list[tuple[str, str]]:
    """Probe every path in parallel; log + return anything not SUCCESS."""
    logger.info("Verifying %d unique %s paths under %s", len(paths), label, STAGING_PREFIX)
    bad: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for output_path, status in pool.map(_check, paths):
            if status == STATUS_SUCCESS:
                logger.debug("OK: %s", output_path)
            else:
                logger.error("%s %s: %s", label, status, output_path)
                bad.append((output_path, status))
    if not bad:
        logger.info("All %d %s paths report SUCCESS", len(paths), label)
    return bad


def main() -> None:
    configure_logging()
    # Pin the staging region before building the registry — StepSpec caches
    # output_path on first access, so MARIN_PREFIX must be set before
    # all_sources() materializes any chain.
    os.environ["MARIN_PREFIX"] = STAGING_PREFIX

    sources = all_sources()
    pre_normalize_paths = sorted({s.normalize_steps[:-1][0].output_path for s in sources.values()})
    normalized_paths = sorted({s.normalized.output_path for s in sources.values()})

    bad_pre = _validate("pre-normalize", pre_normalize_paths)
    bad_norm = _validate("normalized", normalized_paths)

    if bad_pre or bad_norm:
        raise SystemExit(
            f"{len(bad_pre)}/{len(pre_normalize_paths)} pre-normalize and "
            f"{len(bad_norm)}/{len(normalized_paths)} normalized paths not SUCCESS under {STAGING_PREFIX}"
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
