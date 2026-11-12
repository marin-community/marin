import asyncio
from types import SimpleNamespace

import pytest

from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec

from skyrl_train.rollouts.daytona_snapshot_pool import RaySnapshotLeaser
from skyrl_train.rollouts.snapshot_leases import SnapshotLeasePool


@pytest.mark.asyncio
async def test_shared_identity_waits_for_last_release_and_confirmed_deletion():
    live = set()
    deletion_started = asyncio.Event()
    deletion_allowed = asyncio.Event()

    async def build(identity):
        live.add(identity)
        assert len(live) <= 1
        return identity

    async def delete(name):
        deletion_started.set()
        await deletion_allowed.wait()
        live.remove(name)

    pool = SnapshotLeasePool(1, build, delete)
    assert await pool.acquire("a", "sample1") == "a"
    assert await pool.acquire("a", "sample2") == "a"
    await pool.release("sample1")
    assert live == {"a"}
    assert (await pool.inventory())["a"]["leases"] == ["sample2"]
    release = asyncio.create_task(pool.release("sample2"))
    await deletion_started.wait()
    next_identity = asyncio.create_task(pool.acquire("b", "sample3"))
    assert (await pool.inventory())["a"]["phase"] == "deleting"
    deletion_allowed.set()
    await release
    assert await next_identity == "b"
    assert live == {"b"}
    await pool.release("sample3")
    assert live == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["build", "delete"])
async def test_provider_failure_keeps_uncertain_capacity_reserved(operation):
    async def build(identity):
        if operation == "build":
            raise RuntimeError("provider outcome unknown")
        return identity

    async def delete(name):
        raise RuntimeError("provider outcome unknown")

    pool = SnapshotLeasePool(1, build, delete)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        await pool.acquire("a", "sample1")
        await pool.release("sample1")
    inventory = await pool.inventory()
    assert inventory["a"]["phase"] == "uncertain"
    assert len(inventory) == 1


@pytest.mark.asyncio
async def test_queued_same_image_reference_protects_snapshot_during_build():
    started = asyncio.Event()
    finish = asyncio.Event()
    joined = asyncio.Event()
    deleted = []

    async def build(identity):
        started.set()
        await finish.wait()
        return identity

    async def delete(name):
        deleted.append(name)

    pool = SnapshotLeasePool(1, build, delete)
    first = asyncio.create_task(pool.acquire("a", "first"))
    await started.wait()

    async def join():
        joined.set()
        return await pool.acquire("a", "queued")

    second = asyncio.create_task(join())
    await joined.wait()
    assert (await pool.inventory())["a"]["leases"] == ["first", "queued"]
    finish.set()
    assert await first == await second == "a"
    await pool.release("first")
    assert deleted == []
    await pool.release("queued")
    assert deleted == ["a"]


@pytest.mark.asyncio
async def test_cancelled_build_waiter_does_not_hold_snapshot_capacity():
    build_started = asyncio.Event()
    finish_build = asyncio.Event()
    waiter_started = asyncio.Event()
    deleted = []

    async def build(identity):
        build_started.set()
        await finish_build.wait()
        return identity

    async def delete(name):
        deleted.append(name)

    pool = SnapshotLeasePool(1, build, delete)
    first = asyncio.create_task(pool.acquire("a", "first"))
    await build_started.wait()

    async def wait_for_shared_build():
        waiter_started.set()
        return await pool.acquire("a", "cancelled")

    waiter = asyncio.create_task(wait_for_shared_build())
    await waiter_started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert (await pool.inventory())["a"]["leases"] == ["first"]
    finish_build.set()
    assert await first == "a"
    await pool.release("first")
    assert await pool.acquire("b", "next") == "b"
    await pool.release("next")
    assert deleted == ["a", "b"]
    assert await pool.inventory() == {}


@pytest.mark.asyncio
async def test_abandoned_build_deletes_only_after_provider_outcome_is_known():
    started = asyncio.Event()
    finish = asyncio.Event()
    deleted = []

    async def build(identity):
        started.set()
        await finish.wait()
        return identity

    async def delete(name):
        deleted.append(name)

    pool = SnapshotLeasePool(1, build, delete)
    pending = asyncio.create_task(pool.acquire("a", "lost-reply"))
    await started.wait()
    await pool.abandon("lost-reply")
    assert (await pool.inventory())["a"] == {"name": None, "phase": "building", "leases": []}
    assert deleted == []
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert deleted == ["a"]
    assert await pool.inventory() == {}
    await pool.abandon("lost-reply")
    with pytest.raises(asyncio.CancelledError):
        await pool.acquire("b", "lost-reply")
    assert await pool.inventory() == {}


@pytest.mark.asyncio
async def test_duplicate_acquisition_does_not_release_existing_holder():
    started = asyncio.Event()
    finish = asyncio.Event()

    async def build(identity):
        started.set()
        await finish.wait()
        return identity

    async def delete(name):
        pass

    pool = SnapshotLeasePool(1, build, delete)
    first = asyncio.create_task(pool.acquire("a", "holder"))
    await started.wait()
    with pytest.raises(ValueError, match="already acquired"):
        await pool.acquire("a", "holder")
    assert (await pool.inventory())["a"]["leases"] == ["holder"]
    finish.set()
    assert await first == "a"
    with pytest.raises(ValueError, match="already acquired"):
        await pool.acquire("a", "holder")
    assert (await pool.inventory())["a"]["leases"] == ["holder"]
    await pool.release("holder")
    assert await pool.inventory() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_phase", ["before_acquire", "after_acquire"])
async def test_cancelled_ray_reply_reconciles_late_acquisition(reply_phase):
    reached = asyncio.Event()
    reply_allowed = asyncio.Event()
    deleted = []
    remote_requests = []

    async def build(identity):
        return identity

    async def delete(name):
        deleted.append(name)

    pool = SnapshotLeasePool(1, build, delete)

    async def remote_acquire(spec, token):
        if reply_phase == "before_acquire":
            reached.set()
            await reply_allowed.wait()
        name = await pool.acquire("a", token)
        if reply_phase == "after_acquire":
            reached.set()
            await reply_allowed.wait()
        return name

    def submit_acquire(spec, token):
        request = asyncio.create_task(remote_acquire(spec, token))
        remote_requests.append(request)
        return request

    actor = SimpleNamespace(
        acquire=SimpleNamespace(remote=submit_acquire),
        abandon=SimpleNamespace(remote=lambda token: asyncio.create_task(pool.abandon(token))),
    )
    leaser = RaySnapshotLeaser(actor)
    acquiring = asyncio.create_task(leaser.acquire(MachineSpec(source=RegistryImage("registry/image@sha256:test"))))
    await reached.wait()
    acquiring.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquiring
    assert await pool.inventory() == {}
    reply_allowed.set()
    if reply_phase == "before_acquire":
        with pytest.raises(asyncio.CancelledError):
            await remote_requests[0]
        assert deleted == []
    else:
        assert await remote_requests[0] == "a"
        assert deleted == ["a"]
    assert await pool.acquire("b", "next") == "b"
    await pool.release("next")
    assert await pool.inventory() == {}
