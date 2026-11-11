"""Share bounded, run-owned Daytona snapshots across rollout workers."""

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict
from uuid import uuid4

import ray
from daytona import AsyncDaytona, CreateSnapshotParams, DaytonaConfig, DaytonaNotFoundError, Resources
from daytona_api_client_async import SnapshotState
from loguru import logger
from shellbox.backends.daytona.leases import SnapshotLease
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec

from skyrl_train.rollouts.snapshot_leases import SnapshotLeasePool


@ray.remote(num_cpus=0, max_concurrency=512, max_restarts=0)
class DaytonaSnapshotPool:
    def __init__(self, capacity: int, prefix: str, api_key: str, api_url: str, target: str):
        self.connection = DaytonaConfig(api_key=api_key, api_url=api_url, target=target)
        self.prefix = prefix
        self.sources: dict[str, tuple[str, dict]] = {}
        self.pool = SnapshotLeasePool(capacity, self.build, self.delete)

    async def acquire(self, spec: MachineSpec, token: str) -> str:
        if not isinstance(spec.source, RegistryImage):
            raise ValueError("Snapshot pool accepts registry images only")
        resources = asdict(Resources(
            cpu=spec.cpus,
            memory=math.ceil(spec.memory_mb / 1024) if spec.memory_mb is not None else None,
            disk=math.ceil(spec.storage_mb / 1024) if spec.storage_mb is not None else None,
            gpu=spec.gpus or None,
        ))
        source = (spec.source.reference, resources)
        identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        self.sources[identity] = source
        started = time.monotonic()
        name = await self.pool.acquire(identity, token)
        logger.info("snapshot_lease_acquired identity={} token={} wait_seconds={}", identity, token, time.monotonic()-started)
        return name

    async def release(self, token: str) -> None:
        await self.pool.release(token)

    async def inventory(self) -> dict:
        return await self.pool.inventory()

    async def build(self, identity: str) -> str:
        image, resources = self.sources[identity]
        name = self.prefix + identity[:32]
        started = time.monotonic()
        async with AsyncDaytona(self.connection) as client, asyncio.timeout(600):
            snapshot = await client.snapshot.create(CreateSnapshotParams(name=name, image=image, resources=Resources(**resources)))
            while snapshot.state != SnapshotState.ACTIVE:
                if snapshot.state in (SnapshotState.ERROR, SnapshotState.BUILD_FAILED, SnapshotState.INACTIVE):
                    raise RuntimeError(f"Snapshot build failed: {name}, {snapshot.state}")
                await asyncio.sleep(2)
                snapshot = await client.snapshot.get(name)
        logger.info("snapshot_created name={} elapsed_seconds={}", name, time.monotonic()-started)
        return name

    async def delete(self, name: str) -> None:
        if not name.startswith(self.prefix):
            raise ValueError("Refusing to delete a snapshot outside this run")
        started = time.monotonic()
        async with AsyncDaytona(self.connection) as client, asyncio.timeout(120):
            live = {sandbox.snapshot async for sandbox in client.list() if sandbox.snapshot}
            snapshot = await client.snapshot.get(name)
            if name in live or snapshot.id in live:
                raise RuntimeError("Snapshot still referenced by a live sandbox")
            await client.snapshot.delete(snapshot)
            while True:
                try:
                    await client.snapshot.get(name)
                except DaytonaNotFoundError:
                    break
                await asyncio.sleep(2)
        logger.info("snapshot_deleted name={} elapsed_seconds={}", name, time.monotonic()-started)


class RaySnapshotLeaser:
    def __init__(self, actor):
        self.actor = actor

    async def acquire(self, spec: MachineSpec) -> SnapshotLease:
        token = uuid4().hex
        name = await self.actor.acquire.remote(spec, token)
        return SnapshotLease(name, token)

    async def release(self, lease: SnapshotLease) -> None:
        await self.actor.release.remote(lease.token)


def shared_snapshot_leaser(*, capacity: int, api_key: str, api_url: str, target: str) -> RaySnapshotLeaser:
    job = ray.get_runtime_context().get_job_id()
    prefix = "long-loop-" + hashlib.sha256(str(job).encode()).hexdigest()[:12] + "-"
    actor = DaytonaSnapshotPool.options(name=prefix + "snapshot-pool", get_if_exists=True).remote(
        capacity, prefix, api_key, api_url, target
    )
    return RaySnapshotLeaser(actor)
