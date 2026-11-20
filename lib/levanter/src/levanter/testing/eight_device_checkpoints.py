# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint scenarios that need more devices or processes than the test runtime has.

The XLA device count is fixed for the life of a process, so the scenarios spawn fresh
interpreters for eight-device sharding and two-process distributed restores. A spawned child
imports the module its target came from, so the scenarios live on the installed package where
any interpreter can import them regardless of its working directory.
"""

import multiprocessing
import os
import socket
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.tensorstore_serialization import (
    ReplicaRestoreMode,
    TensorStoreReadConfig,
    TensorStoreWriteConfig,
    _replica_staging_sharding,
    _shard_write_region,
    plan_array_write,
    tree_deserialize_leaves_tensorstore,
    tree_serialize_leaves_tensorstore,
)

EIGHT_DEVICE_COUNT = 8
DISTRIBUTED_PROCESS_COUNT = 2
DISTRIBUTED_ARRAY_SIZE = 4096
_CHUNK_READ_METRIC = "/tensorstore/cache/chunk_cache/reads"
_LOOPBACK_ADDRESS = "127.0.0.1"
_PINNED_HOST_MEMORY_KIND = "pinned_host"
# Small enough that every array is worth splitting, so these exercise the split path.
SPLIT_CONFIG = TensorStoreWriteConfig(min_replica_slice_bytes=1, max_chunk_bytes=4096)


def _cpu_device_environment(device_count: int) -> dict[str, str]:
    return {"XLA_FLAGS": f"--xla_force_host_platform_device_count={device_count}", "JAX_PLATFORMS": "cpu"}


def run_on_eight_devices(scenario):
    """Run ``scenario`` in a spawned interpreter, re-raising whatever it raises."""
    with mock.patch.dict(os.environ, _cpu_device_environment(EIGHT_DEVICE_COUNT)):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            return pool.submit(scenario).result()


def _tensorstore_chunk_reads() -> int:
    metrics = ts.experimental_collect_matching_metrics(_CHUNK_READ_METRIC, include_zero_metrics=True)
    return sum(
        value["value"] for metric in metrics if metric["name"] == _CHUNK_READ_METRIC for value in metric["values"]
    )


@dataclass(frozen=True)
class _DistributedRestoreResult:
    arrays: dict[str, np.ndarray]
    memory_kinds: frozenset[str]
    tensorstore_chunk_reads: int


def _distributed_restore(process_id: int, coordinator: str, checkpoint_path: str) -> _DistributedRestoreResult:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(name, None)
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=DISTRIBUTED_PROCESS_COUNT,
        process_id=process_id,
        local_device_ids=[process_id],
    )
    try:
        mesh = Mesh(np.array(jax.devices()), ("replica",))
        sharding = NamedSharding(mesh, P())
        pinned = sharding.with_memory_kind(_PINNED_HOST_MEMORY_KIND)

        with jax.set_mesh(mesh):
            before = _tensorstore_chunk_reads()
            restored = tree_deserialize_leaves_tensorstore(
                checkpoint_path,
                {
                    name: jax.ShapeDtypeStruct((DISTRIBUTED_ARRAY_SIZE,), jnp.float32, sharding=pinned)
                    for name in ("a", "c")
                },
            )
            after = _tensorstore_chunk_reads()
            on_device = {name: jax.device_put(array, sharding) for name, array in restored.items()}
            return _DistributedRestoreResult(
                arrays={name: np.asarray(array) for name, array in on_device.items()},
                memory_kinds=frozenset(array.sharding.memory_kind for array in restored.values()),
                tensorstore_chunk_reads=after - before,
            )
    finally:
        jax.distributed.shutdown()


def replica_aware_restore_across_processes() -> None:
    with TemporaryDirectory() as tmpdir:
        expected = {
            "a": np.arange(DISTRIBUTED_ARRAY_SIZE, dtype=np.float32),
            "c": np.arange(DISTRIBUTED_ARRAY_SIZE, dtype=np.float32) + 1,
        }
        tree_serialize_leaves_tensorstore(tmpdir, expected)
        with socket.socket() as listener:
            listener.bind((_LOOPBACK_ADDRESS, 0))
            port = listener.getsockname()[1]
        coordinator = f"{_LOOPBACK_ADDRESS}:{port}"

        with mock.patch.dict(os.environ, _cpu_device_environment(DISTRIBUTED_PROCESS_COUNT)):
            with ProcessPoolExecutor(
                max_workers=DISTRIBUTED_PROCESS_COUNT, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [
                    pool.submit(_distributed_restore, process_id, coordinator, tmpdir)
                    for process_id in range(DISTRIBUTED_PROCESS_COUNT)
                ]
                results = [future.result() for future in futures]

    for result in results:
        for name in expected:
            np.testing.assert_array_equal(result.arrays[name], expected[name])
        assert result.memory_kinds == {_PINNED_HOST_MEMORY_KIND}
    assert sum(result.tensorstore_chunk_reads for result in results) == len(expected)


def _mesh(shape: tuple[int, int] = (2, 4)) -> Mesh:
    assert jax.device_count() == EIGHT_DEVICE_COUNT, jax.device_count()
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


def replica_aware_restore_reads_each_shard_once():
    mesh = _mesh()
    sharding = NamedSharding(mesh, P("expert", None))
    # -0.0, a NaN payload, a subnormal, and 1.0 catch arithmetic restore reductions.
    bit_patterns = np.array([0x80000000, 0x7FC00001, 0x00000001, 0x3F800000], dtype=np.uint32)
    source_bits = np.tile(bit_patterns, 64 * 16 // len(bit_patterns)).reshape(64, 16)
    source = jax.device_put(source_bits.view(np.float32), sharding)
    pinned = sharding.with_memory_kind(_PINNED_HOST_MEMORY_KIND)
    staging = _replica_staging_sharding(pinned, source.shape)
    assert staging is not None
    assert staging.memory_kind == sharding.memory_kind

    with jax.set_mesh(mesh), TemporaryDirectory() as tmpdir:
        tree_serialize_leaves_tensorstore(tmpdir, {"expert": source}, write_config=SPLIT_CONFIG)
        before = _tensorstore_chunk_reads()
        restored = tree_deserialize_leaves_tensorstore(
            tmpdir,
            {"expert": jax.ShapeDtypeStruct(source.shape, source.dtype, sharding=pinned)},
        )["expert"]
        after = _tensorstore_chunk_reads()
        control_before = _tensorstore_chunk_reads()
        tree_deserialize_leaves_tensorstore(
            tmpdir,
            {"expert": jax.ShapeDtypeStruct(source.shape, source.dtype, sharding=pinned)},
            read_config=TensorStoreReadConfig(replica_mode=ReplicaRestoreMode.EVERY_REPLICA),
        )
        control_after = _tensorstore_chunk_reads()

    assert after - before == 8
    assert control_after - control_before == 16
    assert restored.sharding.memory_kind == _PINNED_HOST_MEMORY_KIND
    on_device = jax.device_put(restored, sharding)
    np.testing.assert_array_equal(np.asarray(on_device).view(np.uint32), source_bits)


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
