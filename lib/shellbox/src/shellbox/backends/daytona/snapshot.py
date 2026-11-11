# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identify, prepare and reclaim Daytona snapshots.

One snapshot corresponds to one image source and one CPU, memory, disk and GPU
request. Shellbox derives the name from both, so Harbor and the operational
tooling always address the same snapshot.
"""

import asyncio
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from daytona import (
    AsyncDaytona,
    CreateSnapshotParams,
    DaytonaConflictError,
    DaytonaError,
    DaytonaNotFoundError,
    Image,
    Resources,
)
from daytona.common.snapshot import Snapshot
from daytona_api_client_async import SnapshotState
from rigging.timing import ExponentialBackoff

from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

SNAPSHOT_PREFIX = "shellbox-"
SNAPSHOT_STALE_AFTER = timedelta(hours=1)
SNAPSHOT_PAGE_SIZE = 100
TRANSIENT_SNAPSHOT_STATUS = frozenset({502, 503, 504})
FAILED_SNAPSHOT_STATES = frozenset({SnapshotState.ERROR, SnapshotState.BUILD_FAILED, SnapshotState.INACTIVE})


@dataclass(frozen=True)
class SnapshotRequest:
    """The validated source and provider resources that identify one snapshot."""

    source: RegistryImage | DockerfileSource
    resources: Resources

    @property
    def name(self) -> str:
        return snapshot_name(self.source, self.resources)


@dataclass(frozen=True)
class SnapshotPurgeReport:
    """Observed result of one account-wide stale-snapshot purge."""

    deleted: tuple[str, ...]
    skipped: tuple[str, ...]
    unclassified: tuple[str, ...]
    remaining: tuple[str, ...]


def validate_source(
    source: RegistryImage | DockerfileSource,
) -> RegistryImage | DockerfileSource:
    """Reject Daytona build inputs it cannot snapshot."""
    if isinstance(source, RegistryImage):
        return source
    source = DockerfileSource(source.context.resolve(), source.dockerfile.resolve())
    if source.dockerfile.parent != source.context:
        raise UnsupportedMachineSpec("Daytona requires the Dockerfile at the build context root")
    if re.search(
        r"^\s*ADD(?:\s|$)",
        source.dockerfile.read_text(),
        re.IGNORECASE | re.MULTILINE,
    ):
        raise UnsupportedMachineSpec("Daytona Dockerfiles with ADD require a prebuilt registry image")
    return source


def task_snapshot_source(docker_image: str | None, environment_dir: Path) -> RegistryImage | DockerfileSource:
    """Select a Harbor task's snapshot source from its image or environment."""
    if docker_image:
        return RegistryImage(docker_image)
    dockerfile = environment_dir / "Dockerfile"
    if not dockerfile.is_file():
        raise UnsupportedMachineSpec("Daytona task requires a registry image or Dockerfile")
    return validate_source(DockerfileSource(environment_dir, dockerfile))


def snapshot_resources(*, cpus: int | None, memory_mb: int | None, storage_mb: int | None, gpus: int) -> Resources:
    """Round memory and disk up to whole GiB, as the provider requires."""
    return Resources(
        cpu=cpus,
        memory=math.ceil(memory_mb / 1024) if memory_mb is not None else None,
        disk=math.ceil(storage_mb / 1024) if storage_mb is not None else None,
        gpu=gpus or None,
    )


def snapshot_name(source: RegistryImage | DockerfileSource, resources: Resources) -> str:
    """Name the snapshot by its source identity and provider resources.

    A registry image is identified by reference, so a moved tag keeps its name.
    A Docker build context is identified by its current contents.
    """
    identity = f"{image_source_key(source)}\0" f"{json.dumps(asdict(resources), sort_keys=True)}"
    return f"{SNAPSHOT_PREFIX}{hashlib.sha256(identity.encode()).hexdigest()[:40]}"


def snapshot_request_for_spec(spec: MachineSpec) -> SnapshotRequest:
    """Derive the snapshot a machine spec would create."""
    source = spec.source
    if not isinstance(source, (RegistryImage, DockerfileSource)):
        raise UnsupportedMachineSpec("Daytona requires a registry image or Docker build context")
    return SnapshotRequest(
        source=validate_source(source),
        resources=snapshot_resources(
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
            storage_mb=spec.storage_mb,
            gpus=spec.gpus,
        ),
    )


async def _snapshot_metadata(client: AsyncDaytona, name: str) -> Snapshot:
    """Retry transient snapshot metadata reads within the caller's deadline."""
    backoff = ExponentialBackoff(initial=1, maximum=2, factor=2, jitter=0)
    attempts = 0
    while True:
        try:
            return await client.snapshot.get(name)
        except DaytonaError as error:
            attempts += 1
            if error.status_code not in TRANSIENT_SNAPSHOT_STATUS or attempts == 3:
                raise
            await asyncio.sleep(backoff.next_interval())


async def ensure_snapshot(client: AsyncDaytona, request: SnapshotRequest) -> str:
    """Create or await the active snapshot identified by ``request``."""
    name = request.name
    try:
        snapshot = await _snapshot_metadata(client, name)
    except DaytonaNotFoundError:
        image = (
            request.source.reference
            if isinstance(request.source, RegistryImage)
            else Image.from_dockerfile(request.source.dockerfile)
        )
        try:
            snapshot = await client.snapshot.create(
                CreateSnapshotParams(name=name, image=image, resources=request.resources)
            )
        except DaytonaConflictError:
            # Another rollout can create this image through a separate client.
            snapshot = await _snapshot_metadata(client, name)
    backoff = ExponentialBackoff(initial=1, maximum=10)
    while snapshot.state != SnapshotState.ACTIVE:
        if snapshot.state in FAILED_SNAPSHOT_STATES:
            raise RuntimeError(f"Daytona snapshot {name} is {snapshot.state}: {snapshot.error_reason}")
        await asyncio.sleep(backoff.next_interval())
        snapshot = await _snapshot_metadata(client, name)
    return name


async def list_snapshots(client: AsyncDaytona) -> list[Snapshot]:
    """Read every snapshot page in the authenticated account."""
    snapshots: list[Snapshot] = []
    page = 1
    while True:
        result = await client.snapshot.list(page=page, limit=SNAPSHOT_PAGE_SIZE)
        snapshots.extend(result.items)
        if not result.items or page >= max(result.total_pages, 1):
            return snapshots
        page += 1


def snapshot_last_used(snapshot: Snapshot) -> datetime | None:
    """Return the provider's UTC last-use time, or None when it is unusable."""
    used = snapshot.last_used_at
    if used is None or used.tzinfo is None:
        return None
    return used.astimezone(UTC)


def snapshot_is_stale(
    snapshot: Snapshot,
    *,
    now: datetime,
    minimum_idle: timedelta = SNAPSHOT_STALE_AFTER,
) -> bool:
    """Whether the snapshot's last use is older than ``minimum_idle``."""
    used = snapshot_last_used(snapshot)
    if used is None:
        raise ValueError(f"{snapshot.name} has no last-use evidence")
    return now - used > minimum_idle


def stale_snapshots(
    snapshots: list[Snapshot],
    *,
    now: datetime,
    minimum_idle: timedelta = SNAPSHOT_STALE_AFTER,
) -> tuple[list[Snapshot], list[Snapshot]]:
    """Split account snapshots into stale and unclassified targets.

    Provider-owned general snapshots are outside account cleanup scope.
    """
    stale: list[Snapshot] = []
    unclassified: list[Snapshot] = []
    for snapshot in snapshots:
        if snapshot.general:
            continue
        used = snapshot_last_used(snapshot)
        if used is None:
            unclassified.append(snapshot)
        elif now - used > minimum_idle:
            stale.append(snapshot)
    return stale, unclassified


async def purge_stale_snapshots(
    client: AsyncDaytona,
    *,
    now: datetime | None = None,
    minimum_idle: timedelta = SNAPSHOT_STALE_AFTER,
) -> SnapshotPurgeReport:
    """Delete every stale account snapshot after revalidating its last use.

    Revalidating each target preserves a snapshot that was used after the
    inventory was taken. Removal is asynchronous, so the report lists deleted
    names that a follow-up inventory still observes.
    """
    observed_at = now or datetime.now(UTC)
    stale, unclassified = stale_snapshots(await list_snapshots(client), now=observed_at, minimum_idle=minimum_idle)
    deleted: list[str] = []
    skipped: list[str] = []
    for candidate in stale:
        try:
            current = await _snapshot_metadata(client, candidate.name)
        except DaytonaNotFoundError:
            continue
        if not snapshot_is_stale(current, now=observed_at, minimum_idle=minimum_idle):
            skipped.append(current.name)
            continue
        await client.snapshot.delete(current)
        deleted.append(current.name)
    present = {snapshot.name for snapshot in await list_snapshots(client)}
    return SnapshotPurgeReport(
        deleted=tuple(deleted),
        skipped=tuple(skipped),
        unclassified=tuple(snapshot.name for snapshot in unclassified),
        remaining=tuple(name for name in deleted if name in present),
    )
