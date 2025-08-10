from tempfile import TemporaryDirectory
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from chex import assert_trees_all_close

import haliax as hax

from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore
from test_utils import MLP, arrays_only, assert_trees_not_close


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


def test_checkpoint_steps():
    mesh = jax.sharding.Mesh(jax.devices(), ("device",))
    with mesh:
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
    mesh = jax.sharding.Mesh(jax.devices(), ("device",))
    with mesh:
        from levanter.models.gpt2 import Gpt2Mlp

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
    mesh = jax.sharding.Mesh(jax.devices(), ("device",))
    with mesh:
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
    mesh = jax.sharding.Mesh(jax.devices(), ("device",))
    with mesh:
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
