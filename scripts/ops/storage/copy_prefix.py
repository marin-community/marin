#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy and verify every object below one CoreWeave prefix to another, as an Iris job.

The operator's machine submits a coordinator job through the ``marin`` federation hub,
which hands it off to the CoreWeave peer cluster in the source's region, or in the
destination's with ``--run-near destination``. The script streams the job's logs.

The coordinator lists the source and packs its files, in key order, into shards of about
``TARGET_SHARD_BYTES``, and splits any shard with more than ``MAX_SHARD_FILES`` files. A
shard's ID hashes each file's key, size, and source ETag. Zephyr
workers copy each shard with ``rigging.fsutil.verified_copy``: an upload in fixed parts
publishes only when complete, and its size and ETag are checked against the streamed
bytes; smaller objects are read back and hashed. Each source ETag is checked again after
its copy. A worker writes the shard's record under ``<destination>.copy-status/shards/``
only after every file in the shard passes.

Resume is per shard and all-or-nothing. A shard counts as copied when its record exists
and every destination object still has the recorded size and ETag; a rerun copies every
file of every other shard again. A changed source file changes its shard's ID, so that
shard is copied again. After the copy, the coordinator runs the same check on every shard
and writes ``<destination>.copy-status/complete.json`` last.

CoreWeave rejects ``CopyObject`` between buckets in different regions with
``InvalidRegion``, so workers read and rewrite every byte. Only CoreWeave URLs are
accepted. Hugging Face buckets and R2 are not supported yet. ``fsutil verified-copy``
gives the same per-object checks from one process, with a staging prefix.

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

import hashlib
import json
import logging
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import click
from fray.types import ResourceConfig
from iris.cli.connect import open_iris_client
from iris.cluster.config import load_config
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from rigging.filesystem.buckets import S3UploadPolicy, filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.listing import metadata_listing_pages
from rigging.fsutil.render import format_size
from rigging.fsutil.verified_copy import ETAG_IDENTITY_PREFIX, copy_verified_object, object_etag, source_identity
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
# CoreWeave Iris clusters are named after their region: US-EAST-02A runs as cw-us-east-02a.
COREWEAVE_CLUSTER_PREFIX = "cw-"
# Jobs go through the federation hub, which hands them off to the named CoreWeave peer.
HUB_CLUSTER = "marin"
HUB_CONFIG = REPO_ROOT / "lib" / "iris" / "config" / f"{HUB_CLUSTER}.yaml"
# Sized to stay inside the CPU node pool so the copy does not spill onto GPU nodes.
DEFAULT_MAX_WORKERS = 50
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="10g")
COPY_THREADS_PER_WORKER = 8
# One shard is the unit of retry and resume. A byte-packed shard over the file cap is split, so a
# run of tiny files cannot pile into one shard that a single worker copies file by file.
TARGET_SHARD_BYTES = 32 * 1024**3
MAX_SHARD_FILES = 2_000
# The coordinator holds the source listing and the destination listing in memory.
COORDINATOR_RESOURCES = ResourceSpec(cpu=4, memory="16GB", disk="5GB")
RECORD_READ_THREADS = 64
PROGRESS_INTERVAL = 60
DIRECTORY_TYPE = "directory"
STATUS_SUFFIX = ".copy-status"
SHARD_RECORDS = "shards"
SHARD_RECORD_SUFFIX = ".json"
COMPLETION_RECORD = "complete.json"


class RunNear(StrEnum):
    """Which side of the copy the job's cluster shares a region with."""

    SOURCE = "source"
    DESTINATION = "destination"


@dataclass(frozen=True, order=True)
class SourceFile:
    """One source object: its key relative to the source prefix, size, and version identity."""

    path: str
    size: int
    identity: str


@dataclass(frozen=True)
class CopyShard:
    """Files that one Zephyr task copies and verifies together."""

    shard_id: str
    source_url: str
    destination_url: str
    files: tuple[SourceFile, ...]


@dataclass(frozen=True)
class CopiedObject:
    """A verified destination object, as recorded in its shard's record."""

    path: str
    size: int
    source_identity: str
    destination_identity: str


def storage_prefix(url: str) -> str:
    """Return *url* without a trailing slash, or raise if its bucket is not a supported store."""
    parsed = StoragePath(url)
    if parsed.scheme != "s3":
        raise click.BadParameter(f"{url}: only s3:// CoreWeave URLs are supported")
    spec = data_buckets().get(parsed.bucket)
    if spec is None or spec.store != StoreType.COREWEAVE:
        raise click.BadParameter(f"{url}: bucket {parsed.bucket} is not a declared CoreWeave bucket")
    if not parsed.key:
        raise click.BadParameter(f"{url}: copying a whole bucket is not supported; name a prefix")
    # Storage routing parses every URL through StoragePath, which drops empty key segments, so
    # such a prefix would silently address a different one.
    raw_key = url.split("://", 1)[1].partition("/")[2]
    if raw_key.removesuffix("/") != parsed.key:
        raise click.BadParameter(f"{url}: prefixes with empty path segments (//) are not supported")
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


def status_url(destination_url: str) -> str:
    return destination_url + STATUS_SUFFIX


def shard_records_url(destination_url: str) -> str:
    return f"{status_url(destination_url)}/{SHARD_RECORDS}"


def shard_record_url(destination_url: str, shard_id: str) -> str:
    return f"{shard_records_url(destination_url)}/{shard_id}{SHARD_RECORD_SUFFIX}"


def list_objects(url: str) -> dict[str, dict[str, Any]]:
    """Map each object below *url* to its listing entry, keyed by its exact key relative to *url*."""
    _, root = filesystem_for(url)
    prefix = f"{root}/"
    objects = {}
    for page in metadata_listing_pages(url):
        for entry in page.entries:
            if entry["type"] == DIRECTORY_TYPE:
                continue
            name = entry["name"]
            if not name.startswith(prefix):
                raise ValueError(f"listing of {url} returned {name}")
            # Object keys are opaque: StoragePath would merge "a//b" into "a/b" and collide distinct keys.
            objects[name[len(prefix) :]] = entry
    return objects


def source_files(url: str) -> list[SourceFile]:
    """List the objects below *url* with the size and identity each copy is checked against."""
    files = []
    for path, entry in list_objects(url).items():
        identity = source_identity(entry)
        if identity is None:
            raise ValueError(f"{url}/{path} has no version identity")
        files.append(SourceFile(path, int(entry["size"]), identity))
    return files


def shard_id(files: Sequence[SourceFile]) -> str:
    """Name a shard by its files' keys, sizes, and identities, so any source change renames it."""
    encoded = json.dumps([[f.path, f.size, f.identity] for f in files]).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def pack_shards(files: Sequence[SourceFile], target_bytes: int, max_files: int) -> list[tuple[SourceFile, ...]]:
    """Group files, in key order, into shards of about *target_bytes*, then split any over *max_files*.

    Splitting after byte packing leaves every shard under the cap exactly as byte packing forms it,
    so its ID, and any record of it, does not depend on *max_files*.
    """
    by_bytes: list[list[SourceFile]] = []
    current: list[SourceFile] = []
    current_bytes = 0
    for file in sorted(files):
        current.append(file)
        current_bytes += file.size
        if current_bytes >= target_bytes:
            by_bytes.append(current)
            current, current_bytes = [], 0
    if current:
        by_bytes.append(current)
    return [tuple(group[start : start + max_files]) for group in by_bytes for start in range(0, len(group), max_files)]


def recorded_shard_ids(destination_url: str) -> set[str]:
    """IDs of the shards whose record exists below the destination's status prefix."""
    records = list_objects(shard_records_url(destination_url))
    return {path.removesuffix(SHARD_RECORD_SUFFIX) for path in records if path.endswith(SHARD_RECORD_SUFFIX)}


def _write_json(url: str, value: Any) -> None:
    filesystem, path = filesystem_for(url)
    with filesystem.open(path, "w") as file:
        json.dump(value, file)


def _read_json(url: str) -> Any:
    filesystem, path = filesystem_for(url)
    with filesystem.open(path, "r") as file:
        return json.load(file)


def _copy_shard(shard: CopyShard) -> int:
    """Zephyr task: copy and verify every file of one shard, then write its record."""
    source_fs, source_root = filesystem_for(shard.source_url)
    destination_fs, destination_root = filesystem_for(shard.destination_url, s3_upload_policy=S3UploadPolicy.FIXED_PARTS)

    shard_bytes = sum(file.size for file in shard.files)
    _report(
        f"shard {shard.shard_id}: {len(shard.files)} files, {format_size(shard_bytes)}, "
        f"{shard.source_url} -> {shard.destination_url}"
    )

    def copy_one(file: SourceFile) -> CopiedObject:
        started = time.monotonic()
        verified = copy_verified_object(
            source_fs,
            f"{source_root}/{file.path}",
            destination_fs,
            f"{destination_root}/{file.path}",
            expected_size=file.size,
            expected_identity=file.identity,
        )
        elapsed = time.monotonic() - started
        _report(
            f"shard {shard.shard_id}: verified {file.path} ({format_size(file.size)}) "
            f"in {elapsed:.0f}s, {file.size / max(elapsed, 1e-3) / 1e6:.0f} MB/s"
        )
        counters.pipeline.update_counter("files_copied", 1)
        counters.pipeline.update_counter("bytes_copied", file.size)
        return CopiedObject(file.path, file.size, file.identity, verified.destination_identity)

    with ThreadPoolExecutor(max_workers=COPY_THREADS_PER_WORKER) as pool:
        # list() re-raises the first failure, so Zephyr retries the whole shard and no record is written.
        copied = list(pool.map(copy_one, shard.files))
    _write_json(shard_record_url(shard.destination_url, shard.shard_id), [asdict(c) for c in copied])
    _report(f"shard {shard.shard_id}: recorded {len(copied)} verified files")
    return len(copied)


def _report(message: str) -> None:
    # The job's stdout reaches the submitter's terminal; the worker's logging setup filters INFO.
    print(message, flush=True)


def _shard_bytes(shards: Sequence[CopyShard]) -> int:
    return sum(file.size for shard in shards for file in shard.files)


def shard_is_copied(shard: CopyShard, record: Sequence[CopiedObject] | None, destination: Mapping[str, Any]) -> bool:
    """Whether *record* covers every file of *shard* and each destination object still matches it."""
    if record is None:
        return False
    recorded = {item.path: item for item in record}
    for file in shard.files:
        item = recorded.get(file.path)
        entry = destination.get(file.path)
        if (
            item is None
            or entry is None
            or item.source_identity != file.identity
            or int(entry["size"]) != file.size
            or f"{ETAG_IDENTITY_PREFIX}{object_etag(entry)}" != item.destination_identity
        ):
            return False
    return True


def _pending_shards(shards: Sequence[CopyShard], destination_url: str) -> list[CopyShard]:
    """Shards without a record, or with a destination object that no longer matches it."""
    recorded = recorded_shard_ids(destination_url)
    with_record = [shard for shard in shards if shard.shard_id in recorded]
    with ThreadPoolExecutor(max_workers=RECORD_READ_THREADS) as pool:
        records = pool.map(lambda shard: _read_json(shard_record_url(destination_url, shard.shard_id)), with_record)
        by_id = {
            shard.shard_id: [CopiedObject(**item) for item in record]
            for shard, record in zip(with_record, records, strict=True)
        }
    destination = list_objects(destination_url)
    return [shard for shard in shards if not shard_is_copied(shard, by_id.get(shard.shard_id), destination)]


def _plan(source_url: str, destination_url: str) -> tuple[list[CopyShard], list[CopyShard]]:
    """Shard the source, report progress, and return all shards and those still to copy."""
    files = source_files(source_url)
    if not files:
        raise ValueError(f"{source_url} holds no files")
    shards = [
        CopyShard(shard_id(group), source_url, destination_url, group)
        for group in pack_shards(files, TARGET_SHARD_BYTES, MAX_SHARD_FILES)
    ]
    pending = _pending_shards(shards, destination_url)
    _report(f"source: {len(files)} files, {format_size(_shard_bytes(shards))} in {len(shards)} shards")
    _report(f"already copied and matching: {len(shards) - len(pending)} shards")
    _report(f"to copy: {len(pending)} shards, {format_size(_shard_bytes(pending))}")
    return shards, pending


@dataclass(frozen=True)
class Progress:
    """Verified totals so far, out of the whole copy."""

    done_bytes: int
    total_bytes: int
    done_files: int
    total_files: int
    done_shards: int
    total_shards: int


def progress_line(progress: Progress, rate: float) -> str:
    """One line of copy progress; *rate* is verified bytes per second during this run."""
    p = progress
    # A source of only empty files has no bytes to take a share of.
    share = f" ({100 * p.done_bytes / p.total_bytes:.1f}%)" if p.total_bytes else ""
    line = (
        f"progress: {format_size(p.done_bytes)} / {format_size(p.total_bytes)}{share}, "
        f"{p.done_files}/{p.total_files} files, {p.done_shards}/{p.total_shards} shards"
    )
    if rate <= 0:
        return line
    eta_minutes = int((p.total_bytes - p.done_bytes) / rate / 60)
    return f"{line}, {format_size(int(rate))}/s, ETA {eta_minutes // 60}h{eta_minutes % 60:02d}m"


@contextmanager
def _progress_reports(shards: Sequence[CopyShard], pending: Sequence[CopyShard], destination_url: str) -> Iterator[None]:
    """Report verified progress every ``PROGRESS_INTERVAL`` seconds while the copy runs.

    A pending shard counts once its record is rewritten during this run; a stale record from an
    earlier run does not count until it is replaced.
    """
    pending_by_id = {shard.shard_id: shard for shard in pending}
    total_bytes = _shard_bytes(shards)
    total_files = sum(len(shard.files) for shard in shards)
    earlier_bytes = total_bytes - _shard_bytes(pending)
    earlier_files = total_files - sum(len(shard.files) for shard in pending)
    records_url = shard_records_url(destination_url)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(PROGRESS_INTERVAL):
            try:
                records = list_objects(records_url)
            except Exception as error:
                # One failed listing must not end progress reports for the rest of a long copy.
                _report(f"progress report skipped: listing {records_url} failed: {error!r}")
                continue
            written = {
                path.removesuffix(SHARD_RECORD_SUFFIX)
                for path, entry in records.items()
                if entry["LastModified"] >= started_at
            }
            copied = [shard for shard_id, shard in pending_by_id.items() if shard_id in written]
            copied_bytes = _shard_bytes(copied)
            progress = Progress(
                done_bytes=earlier_bytes + copied_bytes,
                total_bytes=total_bytes,
                done_files=earlier_files + sum(len(shard.files) for shard in copied),
                total_files=total_files,
                done_shards=len(shards) - len(pending) + len(copied),
                total_shards=len(shards),
            )
            _report(progress_line(progress, copied_bytes / (time.monotonic() - started)))

    thread = threading.Thread(target=report, name="copy-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def _copy_job(source_url: str, destination_url: str, max_workers: int) -> None:
    """Iris-side entrypoint: plan, copy the pending shards, verify, and write the completion record."""
    shards, pending = _plan(source_url, destination_url)
    completion_url = f"{status_url(destination_url)}/{COMPLETION_RECORD}"
    if pending:
        filesystem, completion = filesystem_for(completion_url)
        if filesystem.exists(completion):
            filesystem.rm(completion)
        _report(f"copying {len(pending)} shards on up to {max_workers} workers")
        ctx = ZephyrContext(name="copy-prefix", resources=WORKER_RESOURCES, max_workers=max_workers)
        with _progress_reports(shards, pending, destination_url):
            ctx.execute(Dataset.from_list(pending).map(_copy_shard))

    remaining = _pending_shards(shards, destination_url)
    if remaining:
        sample = ", ".join(file.path for file in remaining[0].files[:5])
        raise RuntimeError(
            f"{len(remaining)} shards are unrecorded or differ at {destination_url}; rerun to copy them again. "
            f"First shard {remaining[0].shard_id} holds {sample}"
        )
    total_files = sum(len(shard.files) for shard in shards)
    _write_json(
        completion_url,
        {
            "source": source_url,
            "destination": destination_url,
            "total_files": total_files,
            "total_bytes": _shard_bytes(shards),
            "shards": [shard.shard_id for shard in shards],
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    _report(f"verified {total_files} files at {destination_url}")


@click.command()
@click.option("--from", "source_url", required=True, help="Source prefix, e.g. s3://marin-us-east-02a/marin/x.")
@click.option("--to", "destination_url", required=True, help="Destination prefix; files keep their relative keys.")
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
@click.option("--dry-run", is_flag=True, help="List the source and shard records from this machine and report.")
def main(source_url: str, destination_url: str, run_near: str, max_workers: int, dry_run: bool) -> None:
    """Copy a CoreWeave prefix to another CoreWeave prefix on an Iris cluster."""
    # Iris reports submission progress and streams the job's log lines through logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    source = storage_prefix(source_url)
    destination = storage_prefix(destination_url)
    if prefixes_overlap(source, destination) or prefixes_overlap(source, status_url(destination)):
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
