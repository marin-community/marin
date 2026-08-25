# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorstore as ts
from chex import assert_trees_all_close
from jax.experimental.array_serialization import serialization as array_ser
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.testing.helpers import MLP, arrays_only, assert_trees_not_close, use_test_mesh

from levanter import tensorstore_serialization
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.checkpoint_manifest import CHECKPOINT_FORMAT_VERSION, manifest_path, read_manifest
from levanter.models.gpt2 import Gpt2Mlp
from levanter.tensorstore_serialization import (
    TensorStoreReadConfig,
    TensorStoreWriteConfig,
    _capped_chunk_shape,
    _HostByteBudget,
    _trim_host_memory_after_commits,
    _transfer_shard_to_pageable_host,
    build_kvstore_spec,
    tree_deserialize_leaves_tensorstore,
    tree_serialize_leaves_tensorstore,
)
from levanter.testing import eight_device_checkpoints
from levanter.testing.eight_device_checkpoints import run_on_eight_devices

_ASYNC_TEST_TIMEOUT = 5


def test_build_kvstore_spec_normalizes_file_uri(tmp_path):
    spec = build_kvstore_spec(f"file://{tmp_path}/cache")

    assert spec == {"driver": "file", "path": str(tmp_path / "cache")}


def test_pageable_checkpoint_staging_detaches_from_donated_jax_buffer():
    source = jnp.arange(8, dtype=jnp.float32)

    staged = asyncio.run(_transfer_shard_to_pageable_host(source.addressable_shards[0]))

    source_host = np.asarray(source)
    np.testing.assert_array_equal(staged, source_host)
    assert not np.shares_memory(staged, source_host)


def test_jemalloc_checkpoint_skips_glibc_trim_callback(monkeypatch):
    callbacks = []
    future = SimpleNamespace(add_done_callback=callbacks.append)

    monkeypatch.setenv("LD_PRELOAD", "libjemalloc.so.2")
    _trim_host_memory_after_commits([future])
    assert callbacks == []

    monkeypatch.delenv("LD_PRELOAD")
    _trim_host_memory_after_commits([future])
    assert len(callbacks) == 1


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


def test_tensorstore_checkpoint_restores_pinned_host_memory_kind():
    # Regression for #8441. A run with offload_opt_state/FP32_PINNED_HOST holds its master and
    # optimizer leaves on `pinned_host`, so the restore template (produced by eval_shape, hence an
    # abstract mesh) carries that memory kind. Passing it straight to tensorstore materializes device
    # buffers and fails the memory-kind check on assembly, so the checkpoint could not be read back.
    with use_test_mesh() as mesh:
        arr = jax.random.normal(jax.random.PRNGKey(0), (8,), dtype=jnp.float32)
        abs_mesh = jax.sharding.get_abstract_mesh()
        for spec in (P(), P("data")):
            pinned = NamedSharding(abs_mesh, spec).with_memory_kind("pinned_host")
            template = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=pinned)
            with TemporaryDirectory() as tmpdir:
                tree_serialize_leaves_tensorstore(tmpdir, {"x": arr})
                restored = tree_deserialize_leaves_tensorstore(tmpdir, {"x": template})["x"]

            assert restored.sharding.memory_kind == "pinned_host"
            # allclose refuses to compare across memory spaces, so pull the host copy to device first.
            on_device = jax.device_put(restored, NamedSharding(mesh, P()))
            assert jnp.allclose(on_device, arr)


def test_tensorstore_checkpoint_restores_mixed_memory_kinds_in_tree_order():
    with use_test_mesh() as mesh:
        first = jnp.arange(8, dtype=jnp.float32)
        second = first + 10
        pinned = NamedSharding(jax.sharding.get_abstract_mesh(), P("data")).with_memory_kind("pinned_host")
        template = {
            "device": jax.ShapeDtypeStruct(first.shape, first.dtype, sharding=NamedSharding(mesh, P("data"))),
            "host": jax.ShapeDtypeStruct(second.shape, second.dtype, sharding=pinned),
        }

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, {"device": first, "host": second})
            restored = tree_deserialize_leaves_tensorstore(tmpdir, template)

        assert restored["device"].sharding.memory_kind == "device"
        assert restored["host"].sharding.memory_kind == "pinned_host"
        np.testing.assert_array_equal(restored["device"], first)
        np.testing.assert_array_equal(jax.device_put(restored["host"], NamedSharding(mesh, P())), second)


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
        gate = _HostByteBudget(100)
        await gate.acquire(60)

        queued = asyncio.create_task(gate.acquire(60))
        await asyncio.sleep(0)
        assert not queued.done(), "120 bytes must not be admitted against a 100 byte budget"

        gate.release(60)
        await asyncio.wait_for(queued, timeout=5)
        assert gate.peak_bytes == 60

    asyncio.run(scenario())


def test_serialization_stages_all_leaves_before_tensorstore_open_completes(monkeypatch, tmp_path):
    source = {
        "first": np.arange(8, dtype=np.float32),
        "second": np.arange(8, dtype=np.float32) + 100,
    }
    expected = {name: value.copy() for name, value in source.items()}
    open_promises_and_futures = [ts.Promise.new() for _ in source]
    commit_promise, commit_future = ts.Promise.new()
    writes = []
    write_started = threading.Event()

    class FakeStore:
        def __getitem__(self, index):
            return self

        def write(self, data, *, can_reference_source_data_indefinitely):
            writes.append(np.array(data, copy=True))
            write_started.set()
            return SimpleNamespace(commit=commit_future)

    open_futures = iter(future for _, future in open_promises_and_futures)
    monkeypatch.setattr(tensorstore_serialization.ts, "open", lambda *args, **kwargs: next(open_futures))

    manager = array_ser.GlobalAsyncCheckpointManager()
    returned = threading.Event()
    errors = []

    def serialize():
        try:
            tree_serialize_leaves_tensorstore(str(tmp_path), {"w": source}, manager=manager)
        except BaseException as error:
            errors.append(error)
        else:
            returned.set()

    serializer = threading.Thread(target=serialize)
    serializer.start()
    try:
        assert returned.wait(timeout=_ASYNC_TEST_TIMEOUT), "serialization waited for TensorStore to open after staging"
        for value in source.values():
            value.fill(-1)
    finally:
        for open_promise, _ in open_promises_and_futures:
            open_promise.set_result(FakeStore())
        assert write_started.wait(timeout=_ASYNC_TEST_TIMEOUT)
        commit_promise.set_result(None)
        serializer.join(timeout=_ASYNC_TEST_TIMEOUT)
        manager.wait_until_finished()

    assert not serializer.is_alive()
    assert errors == []
    writes_by_first_value = {int(value[0]): value for value in writes}
    for value in expected.values():
        np.testing.assert_array_equal(writes_by_first_value[int(value[0])], value)


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
    "arrays, budget",
    [
        # One shard larger than the whole budget: it must be admitted anyway, or the read deadlocks.
        (1, 1),
        # Eight 16 KiB arrays through a budget admitting about two: the read rolls through it.
        (8, 32 * 1024),
    ],
)
def test_a_restore_completes_whatever_the_read_budget_admits(arrays, budget):
    """Concurrent reads stay bounded by the budget and still return every array intact."""
    with use_test_mesh():
        A = hax.Axis("A", 4096)
        state = {f"w{i}": hax.full(A, float(i)) for i in range(arrays)}

        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(tmpdir, state)
            restored = tree_deserialize_leaves_tensorstore(
                tmpdir,
                {name: hax.zeros(A) for name in state},
                read_config=TensorStoreReadConfig(max_in_flight_bytes=budget),
            )

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


def test_every_replica_writes_a_disjoint_slice_covering_the_array():
    """The contract replica-parallel writes rest on: no idle replicas, no byte written twice."""
    run_on_eight_devices(eight_device_checkpoints.disjoint_slices_cover_the_array)


def test_replicated_arrays_survive_a_replica_parallel_roundtrip():
    run_on_eight_devices(eight_device_checkpoints.replicated_arrays_survive_a_roundtrip)


def test_replica_aware_restore_reads_each_shard_once():
    run_on_eight_devices(eight_device_checkpoints.replica_aware_restore_reads_each_shard_once)


def test_replica_aware_restore_broadcasts_across_processes():
    eight_device_checkpoints.replica_aware_restore_across_processes()


def test_checkpoint_written_on_one_mesh_loads_on_another():
    """Content must not depend on the mesh that wrote it, so writer layout can keep changing."""
    run_on_eight_devices(eight_device_checkpoints.a_checkpoint_loads_on_another_mesh)


def test_small_arrays_are_not_split_into_tiny_objects():
    run_on_eight_devices(eight_device_checkpoints.small_arrays_are_not_split)


def test_a_replica_count_that_divides_nothing_still_splits():
    """A rack count that is not a power of two must not drop an array to one writer."""
    run_on_eight_devices(eight_device_checkpoints.a_replica_count_that_divides_nothing_still_splits)


def test_arrays_no_split_applies_to_are_spread_over_replicas():
    """Every such array otherwise lands on replica 0, one process for the whole run."""
    run_on_eight_devices(eight_device_checkpoints.un_splittable_arrays_spread_over_replicas)


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
