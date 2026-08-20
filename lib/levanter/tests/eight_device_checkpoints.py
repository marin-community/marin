# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-write scenarios that need more devices than the test process has.

An array needs eight devices to have more than one replica, and the XLA device count is
fixed for the life of a process, so :func:`run_on_eight_devices` spawns one. The scenarios
live here because a spawned child imports the module its target came from, and the test
module's name under xdist is not importable from a fresh interpreter.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.tensorstore_serialization import (
    TensorStoreWriteConfig,
    _shard_write_region,
    plan_array_write,
    tree_deserialize_leaves_tensorstore,
    tree_serialize_leaves_tensorstore,
)

# Set in the parent, because spawn copies the environment into the child before it starts.
EIGHT_DEVICE_ENV = {"XLA_FLAGS": "--xla_force_host_platform_device_count=8", "JAX_PLATFORMS": "cpu"}
# Small enough that every array is worth splitting, so these exercise the split path.
SPLIT_CONFIG = TensorStoreWriteConfig(min_replica_slice_bytes=1, max_chunk_bytes=4096)


def run_on_eight_devices(scenario):
    """Run ``scenario`` in a spawned interpreter, re-raising whatever it raises."""
    with mock.patch.dict(os.environ, EIGHT_DEVICE_ENV):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            return pool.submit(scenario).result()


def _mesh(shape: tuple[int, int] = (2, 4)) -> Mesh:
    assert jax.device_count() == 8, jax.device_count()
    return Mesh(np.array(jax.devices()).reshape(*shape), ("replica", "expert"))


def _writers_and_coverage(arr, plan) -> tuple[int, np.ndarray]:
    """How many devices write ``arr``, and how many times each of its elements is written."""
    times_written = np.zeros(arr.shape, dtype=int)
    writers = 0
    for shard in arr.addressable_shards:
        region = _shard_write_region(shard, plan)
        if region is None:
            continue
        writers += 1
        times_written[region.index] += 1
    return writers, times_written


def disjoint_slices_cover_the_array():
    mesh = _mesh()
    # spec -> how many of the 8 devices should write part of the array
    cases = {
        "replicated": (P(None, None), 8),
        "sharded_over_expert": (P("expert", None), 8),
        "fully_sharded": (P(("replica", "expert"), None), 8),
    }
    for name, (spec, expected_writers) in cases.items():
        arr = jax.device_put(jnp.arange(64 * 16, dtype=jnp.float32).reshape(64, 16), NamedSharding(mesh, spec))

        writers, times_written = _writers_and_coverage(arr, plan_array_write(name, arr, SPLIT_CONFIG))

        assert writers == expected_writers, (name, writers)
        assert (times_written == 1).all(), (name, "every byte written exactly once")


def replicated_arrays_survive_a_roundtrip():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        source = {
            "replicated": jax.device_put(
                jax.random.normal(jax.random.PRNGKey(0), (64, 16)), NamedSharding(mesh, P(None, None))
            ),
            "sharded": jax.device_put(
                jax.random.normal(jax.random.PRNGKey(1), (64, 16)), NamedSharding(mesh, P("expert", None))
            ),
            "scalar": jnp.array(7),
        }

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, source, write_config=SPLIT_CONFIG)
            restored = tree_deserialize_leaves_tensorstore(tmpdir, {k: jnp.zeros_like(v) for k, v in source.items()})

        for key, value in source.items():
            assert jnp.array_equal(restored[key], value), key


def a_checkpoint_loads_on_another_mesh():
    write_mesh = _mesh()
    read_mesh = _mesh(shape=(1, 8))
    with TemporaryDirectory() as tmpdir:
        with jax.set_mesh(write_mesh):
            written = jax.device_put(
                jax.random.normal(jax.random.PRNGKey(0), (64, 16)), NamedSharding(write_mesh, P(None, None))
            )
            tree_serialize_leaves_tensorstore(tmpdir, {"embed": written}, write_config=SPLIT_CONFIG)
            expected = np.asarray(written)

        with jax.set_mesh(read_mesh):
            target = {"embed": jax.device_put(jnp.zeros((64, 16)), NamedSharding(read_mesh, P("expert", None)))}

            restored = tree_deserialize_leaves_tensorstore(tmpdir, target)

            assert np.array_equal(np.asarray(restored["embed"]), expected)
            assert restored["embed"].sharding.spec == P("expert", None)


def small_arrays_are_not_split():
    mesh = _mesh()
    arr = jax.device_put(jnp.arange(64 * 16, dtype=jnp.float32).reshape(64, 16), NamedSharding(mesh, P(None, None)))

    # 4 KiB array, 8 replicas: a 1 MiB floor means splitting is not worth it.
    plan = plan_array_write("w", arr, TensorStoreWriteConfig(min_replica_slice_bytes=1024**2))
    assert plan.split_axis is None, plan
    assert plan.write_replicas == 1, plan
    assert _writers_and_coverage(arr, plan)[0] == 1

    # max_write_replicas caps the fan-out without disabling the split.
    capped = plan_array_write("w", arr, TensorStoreWriteConfig(min_replica_slice_bytes=1, max_write_replicas=2))
    assert capped.write_replicas == 2, capped
    assert _writers_and_coverage(arr, capped)[0] == 2


def a_replica_count_that_divides_nothing_still_splits():
    mesh = _mesh()
    # Neither 8 nor 7 divides an axis of a (12, 20) shard. 6 divides the first.
    arr = jax.device_put(jnp.arange(12 * 20, dtype=jnp.float32).reshape(12, 20), NamedSharding(mesh, P(None, None)))

    plan = plan_array_write("w", arr, SPLIT_CONFIG)

    assert plan.write_replicas == 6, plan
    assert plan.split_axis == 0, plan
    assert (_writers_and_coverage(arr, plan)[1] == 1).all(), "every byte written exactly once"


def un_splittable_arrays_spread_over_replicas():
    mesh = _mesh()
    # 11 is prime and above the replica count, so no split applies at any width.
    arr = jax.device_put(jnp.arange(11, dtype=jnp.float32), NamedSharding(mesh, P(None)))

    writers = set()
    for path in [f"w{i}" for i in range(20)]:
        plan = plan_array_write(path, arr, SPLIT_CONFIG)
        assert plan.write_replicas == 1, plan
        written = [s.replica_id for s in arr.addressable_shards if _shard_write_region(s, plan) is not None]
        assert len(written) == 1, (path, written)
        writers.add(written[0])

    assert len(writers) > 1, f"20 arrays all landed on replica {writers}"
