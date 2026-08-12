# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import sys
import textwrap
from tempfile import TemporaryDirectory
from typing import Any

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from chex import assert_trees_all_close
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from test_utils import MLP, arrays_only, assert_trees_not_close, use_test_mesh

from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.checkpoint_manifest import CHECKPOINT_FORMAT_VERSION, manifest_path, read_manifest
from levanter.models.gpt2 import Gpt2Mlp
from levanter.tensorstore_serialization import (
    TensorStoreWriteConfig,
    _capped_chunk_shape,
    _HostStagingGate,
    _transfer_shard_to_pageable_host,
    tree_deserialize_leaves_tensorstore,
    tree_serialize_leaves_tensorstore,
)


def test_pageable_checkpoint_staging_detaches_from_donated_jax_buffer():
    source = jnp.arange(8, dtype=jnp.float32)

    staged = asyncio.run(_transfer_shard_to_pageable_host(source.addressable_shards[0]))

    source_host = np.asarray(source)
    np.testing.assert_array_equal(staged, source_host)
    assert not np.shares_memory(staged, source_host)


def test_tensorstore_checkpoint_simple():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    def make_state(key):
        model = MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
        optim = optax.adam(1e-4)
        opt_state = optim.init(arrays_only(model))

        return model, opt_state, key

    initial_model, initial_opt_state, initial_key = make_state(key0)
    rep_model, rep_state, rep_key = make_state(key1)

    assert_trees_not_close(initial_model, rep_model)

    with TemporaryDirectory() as tmpdir:
        tree_serialize_leaves_tensorstore(tmpdir, (initial_model, initial_opt_state, initial_key))
        restored_model, restored_optstate, rkey = tree_deserialize_leaves_tensorstore(
            tmpdir,
            (rep_model, rep_state, rep_key),
        )

        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(restored_model)),
            jax.tree_util.tree_leaves(arrays_only(initial_model)),
        )
        assert all(np.isclose(rkey, initial_key))


def test_tensorstore_checkpoint_eval_shape_concretizes_named_sharding_mesh():
    with use_test_mesh():
        key0 = jax.random.PRNGKey(0)
        arr = jax.random.normal(key0, (8,), dtype=jnp.float32)

        # Simulate an eval_shape-produced sharding: NamedSharding(mesh=AbstractMesh(...)).
        abs_mesh = jax.sharding.get_abstract_mesh()
        abstract_sharding = NamedSharding(abs_mesh, P("data"))
        assert isinstance(abstract_sharding.mesh, jax.sharding.AbstractMesh)
        state_shape = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=abstract_sharding)

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, {"x": arr})
            restored = tree_deserialize_leaves_tensorstore(tmpdir, {"x": state_shape})
            assert jnp.allclose(restored["x"], arr)


def test_checkpoint_steps():
    with use_test_mesh():
        key0 = jax.random.PRNGKey(0)
        key1 = jax.random.PRNGKey(1)

        optim = optax.adam(1e-4)

        def make_state(key):
            model = MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
            opt_state = optim.init(arrays_only(model))

            return model, opt_state, key

        initial_model, initial_opt_state, initial_key = make_state(key0)
        data = jax.random.uniform(key0, (2, 2))

        @eqx.filter_grad
        def loss_fn(model, data):
            m = jax.vmap(model)
            return jnp.mean(jnp.square(m(data)))

        model, state = initial_model, initial_opt_state
        for i in range(3):
            grad = loss_fn(model, data)
            updates, state = optim.update(grad, state)
            model = eqx.apply_updates(model, updates)

        assert_trees_not_close(model, initial_model)
        assert_trees_not_close(state, initial_opt_state)

        rep_model, rep_state, rep_key = make_state(key1)
        assert_trees_not_close(model, rep_model)
        assert_trees_not_close(state, rep_state)

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, (model, state, initial_key, 3))
            restored_model, restored_state, rkey, step = tree_deserialize_leaves_tensorstore(
                tmpdir,
                (rep_model, rep_state, rep_key, 0),
            )
            assert step == 3

            assert_trees_all_close(
                jax.tree_util.tree_leaves(arrays_only(restored_model)),
                jax.tree_util.tree_leaves(arrays_only(model)),
            )
            assert_trees_all_close(
                jax.tree_util.tree_leaves(arrays_only(restored_state)),
                jax.tree_util.tree_leaves(arrays_only(state)),
            )
            assert step == 3


def test_tensorstore_gpt2_mlp():
    with use_test_mesh():

        key0 = jax.random.PRNGKey(0)
        key1 = jax.random.PRNGKey(1)

        Embed = hax.Axis("embed", 64)
        Intermediate = hax.Axis("intermediate", 128)

        def make_state(key):
            model = Gpt2Mlp.init(Embed, Intermediate, jax.nn.relu, key=key)
            optim = optax.adam(1e-4)
            opt_state = optim.init(arrays_only(model))

            return arrays_only(model), arrays_only(opt_state), key

        initial_model, initial_opt_state, initial_key = make_state(key0)
        rep_model, rep_state, rep_key = make_state(key1)

        assert_trees_not_close(initial_model, rep_model)

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, (initial_model, initial_opt_state, initial_key))
            restored_model, restored_optstate, rkey = tree_deserialize_leaves_tensorstore(
                tmpdir,
                (rep_model, rep_state, rep_key),
            )

            assert_trees_all_close(
                jax.tree_util.tree_leaves(arrays_only(restored_model)),
                jax.tree_util.tree_leaves(arrays_only(initial_model)),
            )


def test_tensorstore_ok_with_nones():
    with use_test_mesh():
        A = hax.Axis("A", 10)

        class MyModule(eqx.Module):
            a: Any
            b: Any

        m = MyModule(a=None, b=hax.zeros(A))
        m2 = MyModule(a=None, b=hax.ones(A))

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, m)
            m3 = tree_deserialize_leaves_tensorstore(tmpdir, m2)
            assert m3.a is None
            assert hax.all(m3.b == hax.zeros(A))

        m3 = MyModule(a=hax.zeros(A), b=hax.ones(A))
        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, m2)
            with pytest.raises(FileNotFoundError):
                tree_deserialize_leaves_tensorstore(tmpdir, m3)


def test_tensorstore_ok_with_missing():
    with use_test_mesh():
        A = hax.Axis("A", 10)

        class MyModule(eqx.Module):
            a: Any
            b: Any

        m = MyModule(a=None, b=hax.zeros(A))
        m2 = MyModule(a=hax.full(A, 4), b=hax.ones(A))

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, m)
            m3 = tree_deserialize_leaves_tensorstore(tmpdir, m2, allow_missing=True)
            assert hax.all(m3.a == hax.full(A, 4))
            assert hax.all(m3.b == hax.zeros(A))


def test_a_staging_budget_smaller_than_one_shard_still_completes():
    """The gate must always admit the first snapshot, or a save deadlocks."""
    with use_test_mesh():
        A = hax.Axis("A", 1024)
        state = {"w": hax.full(A, 3.0)}

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(
                tmpdir, state, write_config=TensorStoreWriteConfig(max_staged_host_bytes=1)
            )
            restored = tree_deserialize_leaves_tensorstore(tmpdir, {"w": hax.zeros(A)})

        assert hax.all(restored["w"] == state["w"])


def test_staged_bytes_stay_admitted_until_their_write_is_released():
    """TensorStore holds a shard snapshot until the commit, so admission must outlive the copy."""

    async def scenario():
        gate = _HostStagingGate(100)
        await gate.acquire(60)

        queued = asyncio.create_task(gate.acquire(60))
        await asyncio.sleep(0)
        assert not queued.done(), "120 bytes must not be admitted against a 100 byte budget"

        gate.release(60)
        await asyncio.wait_for(queued, timeout=5)
        assert gate.peak_bytes == 60

    asyncio.run(scenario())


def test_a_save_larger_than_the_staging_budget_rolls_through_it():
    """Peak host memory follows the budget. The checkpoint that comes back is still complete."""
    with use_test_mesh():
        A = hax.Axis("A", 4096)
        state = {f"w{i}": hax.full(A, float(i)) for i in range(8)}

        with TemporaryDirectory() as tmpdir:
            # Every array is 16 KiB, so a 32 KiB budget admits about two at a time.
            tree_serialize_leaves_tensorstore(
                tmpdir, state, write_config=TensorStoreWriteConfig(max_staged_host_bytes=32 * 1024)
            )
            restored = tree_deserialize_leaves_tensorstore(tmpdir, {name: hax.zeros(A) for name in state})

        for name, expected in state.items():
            assert hax.all(restored[name] == expected)


@pytest.mark.parametrize(
    "local_shape, itemsize, max_bytes, expected",
    [
        ((8, 16), 4, 4096, (8, 16)),  # already under the cap
        ((64, 16), 4, 1024, (16, 16)),  # halve the largest even axis until it fits
        ((3, 5), 4, 1, (3, 5)),  # nothing divides: one chunk, cap exceeded
        ((), 4, 1024, ()),  # scalar
        ((0, 16), 4, 1024, (1, 16)),  # zarr3 rejects a zero-length chunk dimension
    ],
)
def test_capped_chunk_shape_stays_a_divisor(local_shape, itemsize, max_bytes, expected):
    chunk = _capped_chunk_shape(local_shape, itemsize, max_bytes)
    assert chunk == expected
    assert all(dim % size == 0 for dim, size in zip(local_shape, chunk)), "chunks must tile the writer's slice"


# An array needs eight devices to have more than one replica, and the XLA device count is
# process-global, so these run in a fresh interpreter (as test_snowball.py does).
_EIGHT_DEVICE_PREAMBLE = """
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["JAX_PLATFORMS"] = "cpu"
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from levanter.tensorstore_serialization import (
    TensorStoreWriteConfig,
    _shard_write_region,
    plan_array_write,
    tree_deserialize_leaves_tensorstore,
    tree_serialize_leaves_tensorstore,
)

assert jax.device_count() == 8
# Small enough that every array is worth splitting, so the test exercises the split path.
CONFIG = TensorStoreWriteConfig(min_replica_slice_bytes=1, max_chunk_bytes=4096)
MESH = Mesh(np.array(jax.devices()).reshape(2, 4), ("replica", "expert"))
"""


def _run_eight_device_script(body: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _EIGHT_DEVICE_PREAMBLE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return result.stdout


def test_every_replica_writes_a_disjoint_slice_covering_the_array():
    """The contract replica-parallel writes rest on: no idle replicas, no byte written twice."""
    _run_eight_device_script(
        """
        cases = {
            # spec -> how many of the 8 devices should write part of the array
            "replicated": (P(None, None), 8),
            "sharded_over_expert": (P("expert", None), 8),
            "fully_sharded": (P(("replica", "expert"), None), 8),
        }
        for name, (spec, expected_writers) in cases.items():
            arr = jax.device_put(
                jnp.arange(64 * 16, dtype=jnp.float32).reshape(64, 16), NamedSharding(MESH, spec)
            )
            plan = plan_array_write(name, arr, CONFIG)
            times_written = np.zeros((64, 16), dtype=int)
            writers = 0
            for shard in arr.addressable_shards:
                region = _shard_write_region(shard, plan)
                if region is None:
                    continue
                writers += 1
                times_written[region.index] += 1
            assert writers == expected_writers, (name, writers)
            assert (times_written == 1).all(), (name, "every byte written exactly once")
        print("OK")
        """
    )


def test_replicated_arrays_survive_a_replica_parallel_roundtrip():
    _run_eight_device_script(
        """
        with jax.set_mesh(MESH):
            source = {
                "replicated": jax.device_put(
                    jax.random.normal(jax.random.PRNGKey(0), (64, 16)), NamedSharding(MESH, P(None, None))
                ),
                "sharded": jax.device_put(
                    jax.random.normal(jax.random.PRNGKey(1), (64, 16)), NamedSharding(MESH, P("expert", None))
                ),
                "scalar": jnp.array(7),
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                tree_serialize_leaves_tensorstore(tmpdir, source, write_config=CONFIG)
                restored = tree_deserialize_leaves_tensorstore(
                    tmpdir, {k: jnp.zeros_like(v) for k, v in source.items()}
                )
            for key, value in source.items():
                assert jnp.array_equal(restored[key], value), key
        print("OK")
        """
    )


def test_checkpoint_written_on_one_mesh_loads_on_another():
    """Content must not depend on the mesh that wrote it, so writer layout can keep changing."""
    _run_eight_device_script(
        """
        read_mesh = Mesh(np.array(jax.devices()).reshape(1, 8), ("replica", "expert"))
        with tempfile.TemporaryDirectory() as tmpdir:
            with jax.set_mesh(MESH):
                written = jax.device_put(
                    jax.random.normal(jax.random.PRNGKey(0), (64, 16)), NamedSharding(MESH, P(None, None))
                )
                tree_serialize_leaves_tensorstore(tmpdir, {"embed": written}, write_config=CONFIG)
                expected = np.asarray(written)
            with jax.set_mesh(read_mesh):
                target = {"embed": jax.device_put(jnp.zeros((64, 16)), NamedSharding(read_mesh, P("expert", None)))}
                restored = tree_deserialize_leaves_tensorstore(tmpdir, target)
                assert np.array_equal(np.asarray(restored["embed"]), expected)
                assert restored["embed"].sharding.spec == P("expert", None)
        print("OK")
        """
    )


def test_small_arrays_are_not_split_into_tiny_objects():
    _run_eight_device_script(
        """
        arr = jax.device_put(jnp.arange(64 * 16, dtype=jnp.float32).reshape(64, 16), NamedSharding(MESH, P(None, None)))
        # 4 KiB array, 8 replicas: a 1 MiB floor means splitting is not worth it.
        plan = plan_array_write("w", arr, TensorStoreWriteConfig(min_replica_slice_bytes=1024**2))
        assert plan.split_axis is None, plan
        assert plan.write_replicas == 1, plan
        writers = sum(1 for shard in arr.addressable_shards if _shard_write_region(shard, plan) is not None)
        assert writers == 1, writers

        # max_write_replicas caps the fan-out without disabling the split.
        capped = plan_array_write("w", arr, TensorStoreWriteConfig(min_replica_slice_bytes=1, max_write_replicas=2))
        assert capped.write_replicas == 2, capped
        assert sum(1 for s in arr.addressable_shards if _shard_write_region(s, capped) is not None) == 2
        print("OK")
        """
    )


def test_a_replica_count_that_divides_nothing_still_splits():
    """A rack count that is not a power of two must not drop an array to one writer."""
    _run_eight_device_script(
        """
        # Neither 8 nor 7 divides an axis of a (12, 20) shard. 6 divides the first.
        arr = jax.device_put(
            jnp.arange(12 * 20, dtype=jnp.float32).reshape(12, 20), NamedSharding(MESH, P(None, None))
        )
        plan = plan_array_write("w", arr, CONFIG)
        assert plan.write_replicas == 6, plan
        assert plan.split_axis == 0, plan

        times_written = np.zeros((12, 20), dtype=int)
        for shard in arr.addressable_shards:
            region = _shard_write_region(shard, plan)
            if region is not None:
                times_written[region.index] += 1
        assert (times_written == 1).all(), "every byte written exactly once"
        print("OK")
        """
    )


def test_arrays_no_split_applies_to_are_spread_over_replicas():
    """Every such array otherwise lands on replica 0, one process for the whole run."""
    _run_eight_device_script(
        """
        # 11 is prime and above the replica count, so no split applies at any width.
        arr = jax.device_put(jnp.arange(11, dtype=jnp.float32), NamedSharding(MESH, P(None)))
        writers = set()
        for path in [f"w{i}" for i in range(20)]:
            plan = plan_array_write(path, arr, CONFIG)
            assert plan.write_replicas == 1, plan
            written = [s.replica_id for s in arr.addressable_shards if _shard_write_region(s, plan) is not None]
            assert len(written) == 1, (path, written)
            writers.add(written[0])
        assert len(writers) > 1, f"20 arrays all landed on replica {writers}"
        print("OK")
        """
    )


def test_host_arrays_without_a_sharding_roundtrip():
    """Numpy leaves reach the serializer through ``is_jax_array_like`` and have no shards."""
    source = {"w": np.arange(10, dtype=np.float32)}

    with TemporaryDirectory() as tmpdir:
        tree_serialize_leaves_tensorstore(tmpdir, source)
        restored = tree_deserialize_leaves_tensorstore(tmpdir, {"w": jnp.zeros(10, dtype=jnp.float32)})

    np.testing.assert_array_equal(np.asarray(restored["w"]), source["w"])


def test_arrays_with_a_zero_length_axis_roundtrip():
    """An empty array still needs a positive chunk grid, which zarr3 enforces on creation."""
    with use_test_mesh():
        source = {"empty": jnp.zeros((0, 16), dtype=jnp.float32), "w": jnp.arange(8, dtype=jnp.float32)}

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, source)
            restored = tree_deserialize_leaves_tensorstore(
                tmpdir, {key: jnp.zeros_like(value) for key, value in source.items()}
            )

        assert restored["empty"].shape == (0, 16)
        np.testing.assert_array_equal(np.asarray(restored["w"]), np.asarray(source["w"]))


def test_reading_a_newer_format_version_fails_loudly(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION + 1,
                "array_driver": "zarr3",
                "kvstore_driver": "ocdbt",
                "arrays": [],
            }
        )
    )

    with pytest.raises(ValueError, match="Upgrade levanter"):
        read_manifest(str(tmp_path))


def test_saving_writes_a_manifest_listing_every_array(tmp_path):
    with use_test_mesh():
        A = hax.Axis("A", 8)

        save_checkpoint({"model": {"w": hax.zeros(A)}, "step": jnp.array(3)}, step=3, checkpoint_path=str(tmp_path))

        manifest = read_manifest(str(tmp_path))
        assert manifest is not None
        assert {array.path: array.shape for array in manifest.arrays} == {"model/w": (8,), "step": ()}


def test_checkpoints_without_a_manifest_still_load(tmp_path):
    """Checkpoints predating the manifest have none, so the reader has to fall back."""
    with use_test_mesh():
        A = hax.Axis("A", 8)
        state = {"model": {"w": hax.full(A, 5.0)}}

        save_checkpoint(state, step=0, checkpoint_path=str(tmp_path))
        (tmp_path / manifest_path("")).unlink()
        assert read_manifest(str(tmp_path)) is None

        restored = load_checkpoint({"model": {"w": hax.zeros(A)}}, str(tmp_path))

        assert jax.numpy.array_equal(restored["model"]["w"].array, state["model"]["w"].array)
