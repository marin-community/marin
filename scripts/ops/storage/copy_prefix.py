#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy every object below one CoreWeave prefix to another, as an Iris job.

The operator's machine submits a coordinator job through the ``marin`` federation hub,
which hands it off to the CoreWeave peer cluster in the source's region, or in the
destination's with ``--run-near destination``. The script streams the job's logs.
The coordinator lists both prefixes, plans each source file whose
destination copy is missing or has a different size, packs the plan into shards of
about ``TARGET_SHARD_BYTES``, and runs a Zephyr pipeline whose workers stream each
shard's files. It then lists the destination again and fails if any source file is
missing or has a different size there. Rerunning the same command resumes the copy:
files already at the destination with the source's size are skipped.

CoreWeave rejects ``CopyObject`` between buckets in different regions with
``InvalidRegion``, so workers read and rewrite every byte. This bulk path is for
immutable datakit outputs. It compares files by size only: a rerun can skip changed
source content when its size is unchanged. It writes no completion manifest. Use
``fsutil verified-copy`` when content verification or a completion marker is required.

Only CoreWeave URLs are accepted. Hugging Face buckets and R2 are not supported yet.

Usage:
    ./scripts/ops/storage/copy_prefix.py \\
        --from s3://marin-us-east-02a/marin/datakit/cluster/domain/v1/harrier-all-sources-10m \\
        --to s3://marin-us-east-08a/marin/datakit/cluster/domain/v1/harrier-all-sources-10m
    ./scripts/ops/storage/copy_prefix.py --from ... --to ... --dry-run
    ./scripts/ops/storage/copy_prefix.py --from ... --to ... --max-workers 5
    ./scripts/ops/storage/copy_prefix.py --from ... --to ... --run-near destination

Iris refuses CoreWeave jobs without the submitter's object-storage key, so export
``CW_KEY_ID`` and ``CW_KEY_SECRET`` before running. If the connection drops, the job keeps
running; follow it with ``uv run iris --cluster=marin job logs -f <job id>``.
"""

import logging
import sys
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import click
from fray.types import ResourceConfig
from iris.cli.connect import open_iris_client
from iris.cluster.config import load_config
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import CopyAction, TransferLocation, execute_copies
from rigging.fsutil.listing import metadata_listing_pages
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
# CoreWeave Iris clusters are named after their region: US-EAST-02A runs as cw-us-east-02a.
COREWEAVE_CLUSTER_PREFIX = "cw-"
# Jobs go through the federation hub, which hands them off to the named CoreWeave peer.
HUB_CLUSTER = "marin"
HUB_CONFIG = REPO_ROOT / "lib" / "iris" / "config" / f"{HUB_CLUSTER}.yaml"
DEFAULT_MAX_WORKERS = 100
# One shard is the unit Zephyr retries; at about 40 MB per file this is ~800 files.
TARGET_SHARD_BYTES = 32 * 1024**3
COPY_THREADS_PER_WORKER = 8
WORKER_RESOURCES = ResourceConfig(cpu=4, ram="16g", disk="10g")
# The coordinator holds both listings in memory: about 1M paths fit well below this.
COORDINATOR_RESOURCES = ResourceSpec(cpu=4, memory="16GB", disk="5GB")
DIRECTORY_TYPE = "directory"
SUPPORTED_STORES = frozenset({StoreType.COREWEAVE})


class RunNear(StrEnum):
    """Which side of the copy the job's cluster shares a region with."""

    SOURCE = "source"
    DESTINATION = "destination"


@dataclass(frozen=True)
class CopyShard:
    """Files that one Zephyr task copies, as paths relative to both roots."""

    source_url: str
    destination_url: str
    relative_paths: tuple[str, ...]


def storage_prefix(url: str) -> str:
    """Return *url* without a trailing slash, or raise if its bucket is not a supported store."""
    parsed = StoragePath(url)
    if parsed.scheme != "s3":
        raise click.BadParameter(f"{url}: only s3:// CoreWeave URLs are supported")
    spec = data_buckets().get(parsed.bucket)
    if spec is None or spec.store not in SUPPORTED_STORES:
        raise click.BadParameter(f"{url}: bucket {parsed.bucket} is not a declared CoreWeave bucket")
    if not parsed.key:
        raise click.BadParameter(f"{url}: copying a whole bucket is not supported; name a prefix")
    return f"s3://{parsed.bucket}/{parsed.key}"


def prefixes_overlap(left: str, right: str) -> bool:
    """Whether one storage prefix equals or contains the other."""
    for inner, outer in ((left, right), (right, left)):
        try:
            StoragePath(inner).relative_to(StoragePath(outer))
        except ValueError:
            continue
        return True
    return False


def cluster_for(url: str) -> str:
    """Return the hub peer cluster in the region of *url*'s bucket."""
    spec = data_buckets()[StoragePath(url).bucket]
    cluster = COREWEAVE_CLUSTER_PREFIX + spec.signing_region.lower()
    if cluster not in load_config(HUB_CONFIG).peers:
        raise click.ClickException(f"the {HUB_CLUSTER} hub has no peer {cluster} in the region of {url}")
    return cluster


def file_sizes(url: str) -> dict[str, int]:
    """Map each file below *url* to its size, keyed by path relative to *url*."""
    root = StoragePath(url)
    sizes = {}
    for page in metadata_listing_pages(url):
        for entry in page.entries:
            if entry["type"] == DIRECTORY_TYPE:
                continue
            # S3 listings name objects as "<bucket>/<key>".
            sizes[StoragePath(f"{root.scheme}://{entry['name']}").relative_to(root)] = entry["size"]
    return sizes


def pending_files(source: Mapping[str, int], destination: Mapping[str, int]) -> dict[str, int]:
    """Return the source files whose destination copy is missing or has a different size."""
    return {path: size for path, size in source.items() if destination.get(path) != size}


def pack_shards(files: Mapping[str, int], target_bytes: int) -> list[tuple[str, ...]]:
    """Group files, in path order, into shards that each hold about *target_bytes*."""
    shards: list[tuple[str, ...]] = []
    current: list[str] = []
    current_bytes = 0
    for path in sorted(files):
        current.append(path)
        current_bytes += files[path]
        if current_bytes >= target_bytes:
            shards.append(tuple(current))
            current, current_bytes = [], 0
    if current:
        shards.append(tuple(current))
    return shards


def _copy_shard(shard: CopyShard) -> int:
    """Zephyr task: copy one shard's files and return the number of files copied."""
    source_fs, source_root = filesystem_for(shard.source_url)
    destination_fs, destination_root = filesystem_for(shard.destination_url)
    source = TransferLocation(shard.source_url, source_fs, source_root)
    destination = TransferLocation(shard.destination_url, destination_fs, destination_root)

    def copy_one(relative_path: str) -> None:
        action = CopyAction(source.child(relative_path), destination.child(relative_path))
        execute_copies((action,))
        counters.pipeline.update_counter("files_copied", 1)

    with ThreadPoolExecutor(max_workers=COPY_THREADS_PER_WORKER) as pool:
        # list() re-raises the first failed copy so Zephyr retries the shard.
        list(pool.map(copy_one, shard.relative_paths))
    return len(shard.relative_paths)


def _report(message: str) -> None:
    # The job's stdout reaches the submitter's terminal; the worker's logging setup filters INFO.
    print(message, flush=True)


def _format_bytes(size: int) -> str:
    return f"{size / 1e12:.2f} TB" if size >= 1e12 else f"{size / 1e9:.2f} GB"


def _summary(label: str, files: Iterable[int]) -> str:
    sizes = list(files)
    return f"{label}: {len(sizes)} files, {_format_bytes(sum(sizes))}"


def _plan(source_url: str, destination_url: str) -> tuple[dict[str, int], dict[str, int]]:
    """List both prefixes, report the plan, and return the source files and those still to copy."""
    source = file_sizes(source_url)
    if not source:
        raise ValueError(f"{source_url} holds no files")
    pending = pending_files(source, file_sizes(destination_url))
    _report(_summary("source", source.values()))
    _report(_summary("already at destination", (source[p] for p in source.keys() - pending.keys())))
    _report(_summary("to copy", pending.values()))
    return source, pending


def _copy_job(source_url: str, destination_url: str, max_workers: int) -> None:
    """Iris-side entrypoint: plan, copy, and verify."""
    source, pending = _plan(source_url, destination_url)
    if not pending:
        return

    shards = [CopyShard(source_url, destination_url, paths) for paths in pack_shards(pending, TARGET_SHARD_BYTES)]
    _report(f"copying in {len(shards)} shards on up to {max_workers} workers")
    ctx = ZephyrContext(name="copy-prefix", resources=WORKER_RESOURCES, max_workers=max_workers)
    ctx.execute(Dataset.from_list(shards).map(_copy_shard))

    remaining = pending_files(source, file_sizes(destination_url))
    if remaining:
        sample = ", ".join(sorted(remaining)[:10])
        raise RuntimeError(f"{len(remaining)} files are missing or differ in size at {destination_url}: {sample}")
    _report(f"verified {len(source)} files at {destination_url}")


@click.command()
@click.option("--from", "source_url", required=True, help="Source prefix, e.g. s3://marin-us-east-02a/marin/x.")
@click.option("--to", "destination_url", required=True, help="Destination prefix; files keep their relative paths.")
@click.option(
    "--run-near",
    type=click.Choice(list(RunNear), case_sensitive=False),
    default=RunNear.SOURCE,
    show_default=True,
    help="Run on the CoreWeave cluster in the source's or the destination's region.",
)
@click.option(
    "--max-workers",
    default=DEFAULT_MAX_WORKERS,
    show_default=True,
    type=click.IntRange(min=1),
    help=f"Zephyr workers; each copies {COPY_THREADS_PER_WORKER} files at a time.",
)
@click.option("--dry-run", is_flag=True, help="List both prefixes from this machine and report what would be copied.")
def main(source_url: str, destination_url: str, run_near: str, max_workers: int, dry_run: bool) -> None:
    """Copy a CoreWeave prefix to another CoreWeave prefix on an Iris cluster."""
    # Iris reports submission progress and streams the job's log lines through logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    source = storage_prefix(source_url)
    destination = storage_prefix(destination_url)
    if prefixes_overlap(source, destination):
        raise click.BadParameter("source and destination prefixes overlap")
    if dry_run:
        # Listing is metadata only, so it runs here and avoids Iris dropping a short job's final output.
        _plan(source, destination)
        return
    cluster = cluster_for(source if RunNear(run_near) == RunNear.SOURCE else destination)

    with open_iris_client(cluster_name=HUB_CLUSTER, workspace=REPO_ROOT) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(_copy_job, source, destination, max_workers),
            name=f"copy-prefix-{datetime.now(UTC):%Y%m%d-%H%M%S}",
            resources=COORDINATOR_RESOURCES,
            environment=EnvironmentSpec(env_vars={}),
            constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=cluster)],
        )
        print(f"Submitted {job.job_id} through {HUB_CLUSTER} to {cluster}", file=sys.stderr)
        job.wait(stream_logs=True, timeout=float("inf"))


if __name__ == "__main__":
    main()
