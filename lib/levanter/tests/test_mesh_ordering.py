"""
Minimal test to diagnose TPU device ordering differences between monorepo and standalone.

This test checks what device ordering JAX returns by default, without any mesh setup.
"""
import sys

import jax
import pytest


def test_default_device_ordering():
    """Test what device ordering JAX gives us by default."""
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")

    devices = jax.devices()
    device_ids = [d.id for d in devices]
    coords = [getattr(d, "coords", None) for d in devices]

    print(f"\n=== DEFAULT DEVICE ORDERING TEST ===", file=sys.stderr)
    print(f"JAX version: {jax.__version__}", file=sys.stderr)
    print(f"Number of devices: {len(devices)}", file=sys.stderr)
    print(f"Device IDs: {device_ids}", file=sys.stderr)
    print(f"Device coords: {coords}", file=sys.stderr)
    print(f"Devices: {devices}", file=sys.stderr)

    # Assert what we expect based on standalone Levanter behavior
    assert device_ids == [0, 1, 2, 3], f"Expected [0, 1, 2, 3] but got {device_ids}"

    # For v4-8, coords should form a 2x2 grid
    if coords[0] is not None:
        print(f"Physical topology detected:", file=sys.stderr)
        for i, (dev_id, coord) in enumerate(zip(device_ids, coords)):
            print(f"  Device {dev_id}: coords={coord}", file=sys.stderr)


def test_array_default_sharding():
    """Test what sharding JAX uses for arrays created without explicit mesh."""
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")

    import jax.numpy as jnp

    # Create a simple array without any mesh context
    x = jnp.ones((8, 8))

    print(f"\n=== ARRAY DEFAULT SHARDING TEST ===", file=sys.stderr)
    print(f"Array shape: {x.shape}", file=sys.stderr)
    print(f"Array sharding: {x.sharding}", file=sys.stderr)

    # Check device placement
    if hasattr(x.sharding, "device_set"):
        devices = list(x.sharding.device_set)
        device_ids = [d.id for d in devices]
        print(f"Array on devices: {device_ids}", file=sys.stderr)


def test_haliax_random_default_sharding():
    """Test what sharding haliax.random uses by default."""
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")

    import haliax as hax
    from jax import random as jrandom

    Pos = hax.Axis("Pos", 16)
    Embed = hax.Axis("Embed", 32)

    # Create array WITHOUT mesh context (like the buggy tests did)
    x = hax.random.normal(jrandom.PRNGKey(0), (Pos, Embed))

    print(f"\n=== HALIAX RANDOM DEFAULT SHARDING TEST ===", file=sys.stderr)
    print(f"Array axes: {x.axes}", file=sys.stderr)
    print(f"Array sharding: {x.array.sharding}", file=sys.stderr)

    if hasattr(x.array.sharding, "device_set"):
        devices = list(x.array.sharding.device_set)
        device_ids = sorted([d.id for d in devices])
        print(f"Array on devices: {device_ids}", file=sys.stderr)
        print(f"Sharding mesh device_ids: {getattr(x.array.sharding, 'mesh', None)}", file=sys.stderr)
