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
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

from daytona import (
    AsyncDaytona,
    CreateSnapshotParams,
    DaytonaConflictError,
    DaytonaNotFoundError,
    Image,
    Resources,
)
from daytona.common.snapshot import Snapshot
from daytona_api_client_async import SnapshotState
from rigging.timing import ExponentialBackoff

from shellbox.backends.daytona.provider import retry_provider_call
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

SNAPSHOT_PREFIX = "shellbox-"
SNAPSHOT_STALE_AFTER = timedelta(hours=1)
SNAPSHOT_PAGE_SIZE = 100
FAILED_SNAPSHOT_STATES = frozenset({SnapshotState.ERROR, SnapshotState.BUILD_FAILED, SnapshotState.INACTIVE})


@dataclass(frozen=True)
class SnapshotRequest:
    """The validated source and provider resources that identify one snapshot."""

    source: RegistryImage | DockerfileSource
    resources: Resources

    @property
    def name(self) -> str:
        return snapshot_name(self.source, self.resources)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON interchange consumed by ``plan`` and ``ensure``."""
        spec: dict[str, Any] = {"name": self.name, "resources": asdict(self.resources)}
        if isinstance(self.source, RegistryImage):
            spec["image"] = self.source.reference
        else:
            spec["dockerfile"] = str(self.source.dockerfile)
        return spec

    @classmethod
    def from_dict(cls, spec: Mapping[str, object]) -> "SnapshotRequest":
        """Parse the interchange emitted by :meth:`as_dict`."""
        unknown = set(spec) - {"name", "resources", "image", "dockerfile"}
        if unknown:
            raise ValueError(f"Unknown snapshot spec fields: {sorted(unknown)}")
        image = spec.get("image")
        dockerfile = spec.get("dockerfile")
        if (image is None) == (dockerfile is None):
            raise ValueError("Snapshot spec requires exactly one of image or dockerfile")
        if not isinstance(image, str | None) or not isinstance(dockerfile, str | None):
            raise ValueError("Snapshot spec image and dockerfile must be strings")
        source = (
            RegistryImage(image) if image is not None else DockerfileSource(Path(dockerfile).parent, Path(dockerfile))
        )
        values = spec.get("resources")
        if not isinstance(values, Mapping):
            raise ValueError("Snapshot spec resources must be an object")
        resources = {}
        for field_name in ("cpu", "memory", "disk", "gpu"):
            value = values.get(field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"Snapshot spec {field_name} must be an integer or null")
            resources[field_name] = value
        request = cls(source=source, resources=Resources(**resources))
        recorded = spec.get("name")
        if recorded is not None and recorded != request.name:
            raise ValueError(f"Snapshot spec name {recorded} does not match the derived name {request.name}")
        return request


@dataclass(frozen=True)
class SnapshotPurgeReport:
    """Observed result of one account-wide stale-snapshot purge."""

    deleted: tuple[str, ...]
    skipped: tuple[str, ...]
    unclassified: tuple[str, ...]
    remaining: tuple[str, ...]
    referenced: tuple[str, ...]


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
    """Derive the snapshot a machine spec will create."""
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
    return await retry_provider_call("snapshot.get", partial(client.snapshot.get, name))


async def _resolve_snapshot(client: AsyncDaytona, request: SnapshotRequest) -> Snapshot:
    name = request.name
    try:
        return await client.snapshot.get(name)
    except DaytonaNotFoundError:
        image = (
            request.source.reference
            if isinstance(request.source, RegistryImage)
            else Image.from_dockerfile(request.source.dockerfile)
        )
        try:
            return await client.snapshot.create(
                CreateSnapshotParams(name=name, image=image, resources=request.resources)
            )
        except DaytonaConflictError:
            # Another rollout can create this image through a separate client.
            return await client.snapshot.get(name)


async def ensure_snapshot(client: AsyncDaytona, request: SnapshotRequest) -> str:
    """Create or recover the active snapshot under its deterministic name."""
    name = request.name
    snapshot = await retry_provider_call("snapshot.resolve", partial(_resolve_snapshot, client, request))
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
        result = await retry_provider_call(
            "snapshot.list", partial(client.snapshot.list, page=page, limit=SNAPSHOT_PAGE_SIZE)
        )
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


class SnapshotClassification(StrEnum):
    """Account-cleanup disposition of one provider snapshot."""

    GENERAL = "general"
    STALE = "stale"
    FRESH = "fresh"
    UNCLASSIFIED = "unclassified"


def classify_snapshot(
    snapshot: Snapshot,
    *,
    now: datetime,
    minimum_idle: timedelta = SNAPSHOT_STALE_AFTER,
) -> SnapshotClassification:
    """Classify one snapshot against the account-wide idle policy.

    Provider-owned general snapshots are outside account cleanup scope, and a
    snapshot without usable last-use evidence stays unclassified rather than
    being treated as fresh or stale.
    """
    if snapshot.general:
        return SnapshotClassification.GENERAL
    used = snapshot_last_used(snapshot)
    if used is None:
        return SnapshotClassification.UNCLASSIFIED
    return SnapshotClassification.STALE if now - used > minimum_idle else SnapshotClassification.FRESH


def classify_inventory(
    snapshots: list[Snapshot],
    *,
    now: datetime,
    minimum_idle: timedelta = SNAPSHOT_STALE_AFTER,
) -> dict[str, SnapshotClassification]:
    """Classify every snapshot in one account inventory, keyed by name."""
    return {snapshot.name: classify_snapshot(snapshot, now=now, minimum_idle=minimum_idle) for snapshot in snapshots}


async def referenced_snapshots(client: AsyncDaytona) -> set[str]:
    """Snapshot names that existing account sandboxes still reference.

    Last-use evidence only records the most recent sandbox creation, so an
    old timestamp does not establish absence of references. Fails closed: an
    enumeration error raises instead of reporting an empty set, because
    deleting without reference evidence is the one unrecoverable outcome.
    """
    try:
        return {sandbox.snapshot async for sandbox in client.list() if sandbox.snapshot}
    except Exception as error:
        raise RuntimeError("Cannot enumerate sandbox references; refusing to purge") from error


async def purge_stale_snapshots(
    client: AsyncDaytona,
    *,
    now: datetime | None = None,
    minimum_idle: timedelta = SNAPSHOT_STALE_AFTER,
) -> SnapshotPurgeReport:
    """Delete every stale snapshot in the account after revalidating its last use.

    Every non-general account snapshot is a candidate, including names Shellbox
    did not create; provider-owned general snapshots are outside cleanup scope.
    Sandbox references are enumerated before any deletion, so reference evidence
    that cannot be obtained aborts the purge entirely, and a stale snapshot that
    an existing sandbox still references is preserved and reported as
    ``referenced``. Revalidating each target preserves a snapshot that was used
    after the inventory was taken, and a target whose last-use evidence
    disappears between the inventory and the revalidation is recorded as
    unclassified without blocking the remaining deletions. Removal is
    asynchronous, so the report lists deleted names that a follow-up inventory
    still observes.
    """
    observed_at = now or datetime.now(UTC)
    referenced = await referenced_snapshots(client)
    inventory_classes = classify_inventory(await list_snapshots(client), now=observed_at, minimum_idle=minimum_idle)
    stale = [
        name for name, classification in inventory_classes.items() if classification is SnapshotClassification.STALE
    ]
    unclassified = [
        name
        for name, classification in inventory_classes.items()
        if classification is SnapshotClassification.UNCLASSIFIED
    ]
    deleted: list[str] = []
    skipped: list[str] = []
    referenced_kept: list[str] = []
    for name in stale:
        if name in referenced:
            referenced_kept.append(name)
            continue
        try:
            current = await _snapshot_metadata(client, name)
        except DaytonaNotFoundError:
            continue
        current_class = classify_snapshot(current, now=observed_at, minimum_idle=minimum_idle)
        if current_class is SnapshotClassification.STALE:
            await client.snapshot.delete(current)
            deleted.append(current.name)
        elif current_class is SnapshotClassification.UNCLASSIFIED:
            unclassified.append(current.name)
        else:
            skipped.append(current.name)
    present = {snapshot.name for snapshot in await list_snapshots(client)}
    return SnapshotPurgeReport(
        deleted=tuple(deleted),
        skipped=tuple(skipped),
        unclassified=tuple(unclassified),
        remaining=tuple(name for name in deleted if name in present),
        referenced=tuple(referenced_kept),
    )
