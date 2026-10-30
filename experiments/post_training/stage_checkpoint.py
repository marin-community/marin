# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a model checkpoint into the S3 prefix a cluster job reads.

A job must read its weights in-region, so a checkpoint that lives on the Hub or in
another bucket is copied once into the target cluster's bucket:

    python -m experiments.post_training.stage_checkpoint \\
        --source 'hf://open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92@fdc3c3c5...' \\
        --output s3://marin-us-east-02a/marin/users/benfeuer/opd-interference/models/b-step92

A Hub source downloads to a local directory named after the output prefix's last
segment and streams to the prefix; a prefix source copies object-by-object. The copy
finishes with a byte-total and object-count check against the source, so a truncated
copy raises instead of looking complete.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import fsspec
from huggingface_hub import snapshot_download
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.transfer import copy


@dataclass(frozen=True)
class CheckpointSummary:
    """What a staged checkpoint holds, for the job log."""

    output_prefix: str
    objects: int
    bytes: float


def _prefix_summary(prefix: str) -> tuple[int, int]:
    """(object count, total bytes) under a prefix."""
    filesystem, path = fsspec.core.url_to_fs(prefix)
    if not filesystem.exists(path):
        raise FileNotFoundError(f"staged prefix {prefix} does not exist")
    total_bytes = 0
    count = 0
    for info in filesystem.find(path, detail=True).values():
        if info.get("type") == "directory":
            continue
        count += 1
        total_bytes += int(info.get("size") or 0)
    return count, total_bytes


def stage_from_hub(repo_id: str, revision: str, output_prefix: str, *, local_dir: Path) -> CheckpointSummary:
    """Download a pinned HF checkpoint and copy it to ``output_prefix``, then verify.

    The copy targets the output prefix's parent, so the local directory's own
    name (which ``copy`` appends) becomes the output prefix's last segment; a
    ``local_dir`` named differently would land under the wrong name and raises.
    """
    if local_dir.name != output_prefix.rstrip("/").split("/")[-1]:
        raise ValueError(
            f"--local-dir {local_dir.name!r} must be named after the output prefix's last segment "
            f"{output_prefix.rstrip('/').split('/')[-1]!r}"
        )
    parent = output_prefix.rstrip("/").rsplit("/", 1)[0]
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(local_dir), max_workers=16)
    shutil.rmtree(local_dir / ".cache", ignore_errors=True)  # download metadata, not checkpoint files
    copy(str(local_dir), parent + "/", recursive=True)
    count, total = _prefix_summary(output_prefix)
    local_count, local_total = _prefix_summary(str(local_dir))
    if (count, total) != (local_count, local_total):
        raise ValueError(
            f"staged copy of {local_dir} is incomplete: {count} objects/{total} bytes on {output_prefix} "
            f"against {local_count}/{local_total} locally"
        )
    return CheckpointSummary(output_prefix=output_prefix, objects=count, bytes=float(total))


def stage_from_prefix(source_prefix: str, output_prefix: str) -> CheckpointSummary:
    """Copy one object-storage prefix to another (an exact rename) and verify the totals."""
    copy(source_prefix.rstrip("/"), output_prefix.rstrip("/"), recursive=True)
    count, total = _prefix_summary(output_prefix)
    source_count, source_total = _prefix_summary(source_prefix)
    if (count, total) != (source_count, source_total):
        raise ValueError(
            f"staged copy of {source_prefix} is incomplete: {count} objects/{total} bytes on {output_prefix} "
            f"against {source_count}/{source_total} at the source"
        )
    return CheckpointSummary(output_prefix=output_prefix, objects=count, bytes=float(total))


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="hf://repo@revision, or an s3:// source prefix")
    parser.add_argument("--output", required=True, help="destination s3:// prefix")
    parser.add_argument("--local-dir", type=Path, default=None, help="scratch directory for a Hub download")
    args = parser.parse_args()

    if args.source.startswith("hf://"):
        rest = args.source.removeprefix("hf://")
        repo_id, _, revision = rest.partition("@")
        if not revision:
            raise ValueError(f"--source {args.source} must pin a revision as hf://repo@revision")
        local_dir = args.local_dir or Path("/tmp") / "stage" / args.output.rstrip("/").split("/")[-1]
        summary = stage_from_hub(repo_id, revision, args.output, local_dir=local_dir)
    else:
        summary = stage_from_prefix(args.source, args.output)
    print(f"staged {summary.objects} objects ({summary.bytes:.0f} bytes) at {summary.output_prefix}")


if __name__ == "__main__":
    main()
