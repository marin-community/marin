# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit Testbed ferry: fan-in download → normalize → noop_dedup → consolidate → tokenize.

Runs the canonical Datakit Testbed source set (see
``marin.datakit.sources.all_sources``) through the Marin
pipeline. v0 baseline uses ``noop_dedup`` — a metadata-only stand-in for real
fuzzy-dup marking — so every document passes through consolidate unchanged,
matching the RFC's "deliberately trivial" no-dedup baseline. The DAG shape is
identical to the future dedup variant, so enabling real dedup later is a
one-line swap in ``dag.py``.

Output paths land under ``$MARIN_PREFIX/datakit-testbed/$TESTBED_RUN_ID/...``.
``MARIN_PREFIX`` defaults to a region-local temp bucket with 1-day TTL if unset,
which is fine for shakeout; production runs should pin ``MARIN_PREFIX`` to
``gs://marin-us-central1/...`` to match the pre-staged raw dumps.
"""

import json
import logging
import os

from rigging.filesystem import (
    check_path_in_region,
    marin_temp_bucket,
    region_from_metadata,
    url_to_fs,
)
from rigging.log_setup import configure_logging
from rigging.timing import log_time

from marin.execution.step_runner import StepRunner

from experiments.datakit_testbed.sampler import build_testbed_steps
from marin.datakit.sources import all_sources

from experiments.datakit_testbed.settings import TESTBED_STAGING_REGION

logger = logging.getLogger(__name__)


def _write_status(status: str, marin_prefix: str) -> None:
    """Write ferry run status to FERRY_STATUS_PATH if set."""
    status_path = os.environ.get("FERRY_STATUS_PATH")
    if not status_path:
        return
    payload = json.dumps({"status": status, "marin_prefix": marin_prefix})
    fs, _ = url_to_fs(status_path)
    with fs.open(status_path, "w") as f:
        f.write(payload)
    logger.info("Wrote ferry status to %s", status_path)


def _guard_staged_regions(marin_prefix: str) -> None:
    """Fail fast if the running region doesn't match the staged dumps' region.

    The testbed set pins every pre-staged dump to ``TESTBED_STAGING_REGION``
    (us-central1). Cross-region reads would blow the cost budget per repo
    policy, so we refuse to proceed if the ferry is launched elsewhere.
    """
    region = region_from_metadata()
    if region is None:
        logger.info("No GCE metadata region; skipping region guard.")
        return
    if not region.startswith(TESTBED_STAGING_REGION):
        raise RuntimeError(
            f"Ferry running in region {region!r} but staged raw dumps live in "
            f"{TESTBED_STAGING_REGION!r}. Refusing to do cross-region reads."
        )
    # Spot-check each pre-staged path is reachable under the chosen prefix.
    seen: set[str] = set()
    for src in all_sources().values():
        if src.staged_path is None or src.staged_path in seen:
            continue
        seen.add(src.staged_path)
        absolute = f"{marin_prefix.rstrip('/')}/{src.staged_path}"
        check_path_in_region(src.name, absolute, region)


def main() -> None:
    configure_logging()
    if not os.environ.get("MARIN_PREFIX"):
        os.environ["MARIN_PREFIX"] = marin_temp_bucket(ttl_days=1)

    marin_prefix = os.environ["MARIN_PREFIX"]
    logger.info("MARIN_PREFIX defaulted to %s", marin_prefix)
    run_id = os.environ["TESTBED_RUN_ID"]

    _guard_staged_regions(marin_prefix)

    _write_status("running", marin_prefix)
    with log_time("Datakit testbed ferry total wall time"):
        dag = build_testbed_steps(run_id)
        logger.info(
            "Running %d steps across %d tokenized sources",
            len(dag.all_steps),
            len(dag.tokenized_by_source),
        )
        StepRunner().run(dag.all_steps)
    _write_status("succeeded", marin_prefix)


if __name__ == "__main__":
    main()
