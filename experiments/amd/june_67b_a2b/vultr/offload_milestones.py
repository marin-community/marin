# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy a run's milestone checkpoints from the cluster filesystem to object storage, and free their local space.

Milestones are the permanent checkpoints Levanter writes under ``<run dir>/checkpoints`` (``keep: [{every: N}]``);
temporary checkpoints go to ``<run dir>/checkpoints-temp`` and Levanter prunes them itself. Each milestone is copied
with ``verified_copy_prefix`` to ``<destination>/step-<N>``, which publishes a completion manifest only after every
object is checked. With ``--delete-local``, a milestone is deleted from the cluster once its manifest exists and a
newer complete checkpoint is on the cluster, so the run can always resume locally. The metadata file goes first, so
a deletion cut short leaves a directory that checkpoint discovery skips.

It imports no JAX, so it runs from a small venv with ``rigging`` on the login node beside a training job::

    python -m experiments.amd.june_67b_a2b.vultr.offload_milestones --run-dir $WORK/runs/<run> \\
        --destination s3://marin-us-east-02a/marin/snowball-amd/<run>/checkpoints --delete-local --watch 600
"""

import argparse
import dataclasses
import json
import logging
import shutil
import time
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.fsutil.verified_copy import COMPLETION_MANIFEST, verified_copy_prefix

logger = logging.getLogger(__name__)

# Levanter's CHECKPOINT_METADATA_FILENAME; a checkpoint directory is complete once its save wrote this file.
METADATA_FILENAME = "metadata.json"
PERMANENT_SUBDIR = "checkpoints"
TEMPORARY_SUBDIR = "checkpoints-temp"


@dataclasses.dataclass(frozen=True)
class LocalCheckpoint:
    path: Path
    step: int
    is_temporary: bool


def complete_checkpoints(run_dir: Path) -> list[LocalCheckpoint]:
    """The run's complete checkpoints in both subdirectories, oldest step first."""
    found = []
    for subdir in (PERMANENT_SUBDIR, TEMPORARY_SUBDIR):
        for path in sorted((run_dir / subdir).glob("step-*")):
            metadata_path = path / METADATA_FILENAME
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text())
            found.append(LocalCheckpoint(path, int(metadata["step"]), bool(metadata.get("is_temporary", False))))
    return sorted(found, key=lambda c: (c.step, c.is_temporary))


def is_offloaded(destination: str) -> bool:
    fs, path = filesystem_for(f"{destination}/{COMPLETION_MANIFEST}")
    return fs.exists(path)


def delete_local(checkpoint: LocalCheckpoint) -> None:
    (checkpoint.path / METADATA_FILENAME).unlink()
    shutil.rmtree(checkpoint.path)


def offload_once(run_dir: Path, destination: str, *, delete_after_offload: bool, workers: int) -> list[int]:
    """Copy every local milestone without a manifest at the destination, then delete the offloaded milestones that
    have a newer complete checkpoint beside them when ``delete_after_offload``. Returns the steps copied."""
    checkpoints = complete_checkpoints(run_dir)
    copied = []
    for checkpoint in checkpoints:
        if checkpoint.is_temporary:
            continue
        target = f"{destination.rstrip('/')}/step-{checkpoint.step}"
        if not is_offloaded(target):
            start = time.monotonic()
            result = verified_copy_prefix(str(checkpoint.path), target, workers=workers)
            elapsed = time.monotonic() - start
            logger.info(
                "offloaded step %d: %d files, %.1f GiB in %.0f s (%.2f GiB/s) to %s",
                checkpoint.step,
                result.total_files,
                result.total_bytes / 2**30,
                elapsed,
                result.total_bytes / 2**30 / max(elapsed, 1e-9),
                target,
            )
            copied.append(checkpoint.step)
        newest_step = max(c.step for c in complete_checkpoints(run_dir))
        if delete_after_offload and checkpoint.step < newest_step:
            logger.info("deleting offloaded step %d from %s (newest local step %d)", checkpoint.step, run_dir, newest_step)
            delete_local(checkpoint)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Holds checkpoints/ and checkpoints-temp/.")
    parser.add_argument("--destination", required=True, help="Object-store prefix; milestones go to step-<N> below.")
    parser.add_argument("--delete-local", action="store_true", help="Delete offloaded milestones that are not newest.")
    parser.add_argument("--watch", type=float, default=0, help="Repeat every this many seconds (0: once).")
    parser.add_argument("--workers", type=int, default=32, help="Objects copied at once.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    while True:
        offload_once(args.run_dir, args.destination, delete_after_offload=args.delete_local, workers=args.workers)
        if args.watch <= 0:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
