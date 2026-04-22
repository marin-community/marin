# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every Datakit source's staged_path exists under gs://marin-us-central1.

Each :class:`marin.datakit.sources.DatakitSource` with a non-empty ``staged_path``
must resolve to a GCS prefix with at least one object — otherwise the ferry's
verify-only download step will 404 at runtime. Enforced daily as a parallel
lane of the datakit-smoke workflow.
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from marin.datakit.sources import all_sources
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

BUCKET = "gs://marin-us-central1"
MAX_WORKERS = 16


def _check(full_path: str) -> tuple[str, bool]:
    fs, _ = url_to_fs(full_path)
    try:
        children = fs.ls(full_path, detail=False)
    except FileNotFoundError:
        return full_path, False
    return full_path, bool(children)


def main() -> None:
    configure_logging()
    sources = all_sources()
    unique_paths = sorted({s.staged_path for s in sources.values() if s.staged_path})
    logger.info("Verifying %d unique staged paths under %s", len(unique_paths), BUCKET)
    urls = [f"{BUCKET}/{p}" for p in unique_paths]

    missing: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for full, exists in pool.map(_check, urls):
            if exists:
                logger.debug("OK: %s", full)
            else:
                logger.error("MISSING: %s", full)
                missing.append(full)

    if missing:
        raise SystemExit(f"{len(missing)}/{len(unique_paths)} staged paths missing under {BUCKET}")
    logger.info("All %d staged paths present under %s", len(unique_paths), BUCKET)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
