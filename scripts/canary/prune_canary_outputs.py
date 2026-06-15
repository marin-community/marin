# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prune stale canary ferry outputs from object storage.

The canary ferry (`marin-canary-ferry-coreweave.yaml`) writes one output
directory per run under ``MARIN_PREFIX/canary/``. Nothing reads these once a
run's regression gate (``validate_canary_metrics.py``) finishes — the gate
resolves the *current* run by version hash — so they accumulate indefinitely.
This deletes run directories under that subdir whose most recent object is
older than ``--max-age-days``.

Storage access mirrors the canary workflow: ``MARIN_PREFIX`` plus the standard
``AWS_*`` env (including ``AWS_ENDPOINT_URL`` for R2) is honored by ``url_to_fs``.
"""

import argparse
import datetime
import logging
import os

from rigging.filesystem import url_to_fs

from experiments.ferries.canary_paths import CANARY_OUTPUT_SUBDIR

logger = logging.getLogger(__name__)


def _latest_mtime(fs, directory: str) -> datetime.datetime | None:
    """Return the most recent object mtime under ``directory``, or None if empty."""
    entries = fs.find(directory, detail=True)
    mtimes = [info["LastModified"] for info in entries.values() if info.get("LastModified") is not None]
    return max(mtimes) if mtimes else None


def find_stale_dirs(fs, canary_root: str, cutoff: datetime.datetime) -> list[str]:
    """Return canary run directories whose newest object predates ``cutoff``."""
    if not fs.exists(canary_root):
        return []
    stale = []
    for entry in fs.ls(canary_root, detail=False):
        latest = _latest_mtime(fs, entry)
        if latest is None:
            # Empty directory marker — safe to drop.
            stale.append(entry)
        elif latest < cutoff:
            stale.append(entry)
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=os.environ.get("MARIN_PREFIX"),
        help="Storage root containing canary outputs (defaults to $MARIN_PREFIX).",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=14,
        help="Delete canary outputs whose newest object is older than this many days.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without deleting.",
    )
    args = parser.parse_args()

    if not args.prefix:
        raise ValueError("No prefix given; set --prefix or MARIN_PREFIX.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.max_age_days)
    fs, path = url_to_fs(args.prefix)
    canary_root = os.path.join(path, CANARY_OUTPUT_SUBDIR)

    stale = find_stale_dirs(fs, canary_root, cutoff)
    logger.info(
        "Found %d canary dirs older than %d days (cutoff %s)",
        len(stale),
        args.max_age_days,
        cutoff.isoformat(),
    )
    for directory in stale:
        if args.dry_run:
            logger.info("  would delete: %s", directory)
        else:
            fs.rm(directory, recursive=True)
            logger.info("  deleted: %s", directory)

    if args.dry_run:
        logger.info("Dry run — nothing deleted.")
    else:
        logger.info("Pruned %d canary dirs.", len(stale))


if __name__ == "__main__":
    main()
