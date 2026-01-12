# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PortAllocator."""

import asyncio
import socket

import pytest

from fluster.cluster.worker.manager import PortAllocator


@pytest.fixture
def allocator():
    """Create PortAllocator with small range for testing."""
    return PortAllocator(port_range=(40000, 40100))


@pytest.mark.asyncio
async def test_allocate_single_port(allocator):
    """Test allocating a single port."""
    ports = await allocator.allocate(count=1)
    assert len(ports) == 1
    assert 40000 <= ports[0] < 40100


@pytest.mark.asyncio
async def test_allocate_multiple_ports(allocator):
    """Test allocating multiple ports at once."""
    ports = await allocator.allocate(count=5)
    assert len(ports) == 5
    assert len(set(ports)) == 5  # All unique
    for port in ports:
        assert 40000 <= port < 40100


@pytest.mark.asyncio
async def test_allocated_ports_are_usable(allocator):
    """Test that allocated ports can actually be bound."""
    ports = await allocator.allocate(count=3)

    # Verify each port can be bound (it's free)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))


@pytest.mark.asyncio
async def test_no_port_reuse_before_release(allocator):
    """Test that allocated ports are not reused before release."""
    ports1 = await allocator.allocate(count=5)
    ports2 = await allocator.allocate(count=5)

    # No overlap between the two allocations
    assert len(set(ports1) & set(ports2)) == 0


@pytest.mark.asyncio
async def test_ports_reused_after_release(allocator):
    """Test that ports can be reused after release."""
    # Allocate all available ports in a small range
    allocator_small = PortAllocator(port_range=(40000, 40003))

    # Allocate 3 ports
    ports1 = await allocator_small.allocate(count=3)
    assert len(ports1) == 3

    # Release them
    await allocator_small.release(ports1)

    # Should be able to allocate again
    ports2 = await allocator_small.allocate(count=3)
    assert len(ports2) == 3

    # Ports should be reused (same set, possibly different order)
    assert set(ports1) == set(ports2)


@pytest.mark.asyncio
async def test_release_partial_ports(allocator):
    """Test releasing only some ports."""
    ports = await allocator.allocate(count=5)

    # Release first 3 ports
    await allocator.release(ports[:3])

    # Allocate 2 more - should get from the released ones
    new_ports = await allocator.allocate(count=2)

    # At least some of the new ports should be from released ones
    assert len(set(new_ports) & set(ports[:3])) > 0


@pytest.mark.asyncio
async def test_exhausted_port_range(allocator):
    """Test behavior when port range is exhausted."""
    allocator_tiny = PortAllocator(port_range=(40000, 40002))

    # Allocate all available ports (2 ports: 40000, 40001)
    ports = await allocator_tiny.allocate(count=2)
    assert len(ports) == 2

    # Trying to allocate more should raise RuntimeError
    with pytest.raises(RuntimeError, match="No free ports available"):
        await allocator_tiny.allocate(count=1)


@pytest.mark.asyncio
async def test_concurrent_allocations(allocator):
    """Test concurrent port allocations are thread-safe."""

    async def allocate_ports():
        return await allocator.allocate(count=5)

    # Run multiple concurrent allocations
    results = await asyncio.gather(
        allocate_ports(),
        allocate_ports(),
        allocate_ports(),
    )

    # Collect all allocated ports
    all_ports = []
    for ports in results:
        all_ports.extend(ports)

    # All ports should be unique (no conflicts)
    assert len(all_ports) == len(set(all_ports))


@pytest.mark.asyncio
async def test_release_nonexistent_port(allocator):
    """Test that releasing a non-allocated port doesn't cause errors."""
    # Should not raise an error
    await allocator.release([99999])


@pytest.mark.asyncio
async def test_default_port_range():
    """Test default port range is 30000-40000."""
    allocator = PortAllocator()
    ports = await allocator.allocate(count=5)

    for port in ports:
        assert 30000 <= port < 40000
