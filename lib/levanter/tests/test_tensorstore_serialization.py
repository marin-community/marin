# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

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
from jax.experimental.array_serialization import tensorstore_impl as ts_impl
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from test_utils import MLP, arrays_only, assert_trees_not_close, use_test_mesh

from levanter.models.gpt2 import Gpt2Mlp
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore


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


def test_serialize_uses_fresh_context_per_save():
    """Each save must run through its own tensorstore Context, not the shared `_TS_CONTEXT`.

    JAX's process-lifetime `_TS_CONTEXT` strongly owns its cache pool; reusing it across saves
    of distinct OCDBT databases pins each save's pinned-host source buffers forever, leaking host
    RAM ~linearly per checkpoint. The fix injects a fresh Context per save, so this guards that
    (a) the global singleton is never the context handed to `async_serialize`, (b) successive
    saves get distinct contexts, and (c) the wrapper is restored after each save.
    """
    original_async_serialize = ts_impl.async_serialize
    seen_contexts: list[Any] = []

    def recording_async_serialize(*args, **kwargs):
        seen_contexts.append(kwargs.get("context"))
        return original_async_serialize(*args, **kwargs)

    model = MLP(in_size=2, out_size=1, width_size=2, depth=1, key=jax.random.PRNGKey(0))

    ts_impl.async_serialize = recording_async_serialize
    try:
        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(f"{tmpdir}/step-0", model)
        with TemporaryDirectory() as tmpdir:
            tree_serialize_leaves_tensorstore(f"{tmpdir}/step-1", model)
        # The per-save wrapper must restore the prior `async_serialize` (our recorder) on exit,
        # never leaving a leaked wrapper installed.
        restored_to = ts_impl.async_serialize
    finally:
        ts_impl.async_serialize = original_async_serialize

    assert restored_to is recording_async_serialize

    assert len(seen_contexts) >= 2
    assert all(ctx is not None for ctx in seen_contexts)
    assert all(ctx is not ts_impl._TS_CONTEXT for ctx in seen_contexts)
    # Distinct saves must not share a Context, otherwise the prior save's pool is never released.
    assert len(set(id(ctx) for ctx in seen_contexts)) >= 2


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
