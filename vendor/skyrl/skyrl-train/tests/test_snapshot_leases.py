import asyncio

import pytest

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
