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
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from test_utils import MLP, arrays_only, assert_trees_not_close, use_test_mesh

from levanter.models.gpt2 import Gpt2Mlp
from levanter.tensorstore_serialization import (
    build_kvstore_spec,
    tree_deserialize_leaves_tensorstore,
    tree_serialize_leaves_tensorstore,
)


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


@pytest.fixture
def _clean_s3_env(monkeypatch):
    """Start each S3 spec test from a known environment."""
    for name in (
        "AWS_ENDPOINT_URL",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "LEVANTER_S3_VIRTUAL_HOSTED",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_build_kvstore_spec_s3_path_style_default(_clean_s3_env):
    # Without the virtual-hosted flag, a custom endpoint is passed through unchanged
    # (path-style addressing), preserving the bucket name. This is the default that keeps
    # path-style providers like Cloudflare R2 working.
    _clean_s3_env.setenv("AWS_ENDPOINT_URL", "https://cwobject.com")

    spec = build_kvstore_spec("s3://my-bucket/checkpoints/step-100")
    assert spec["driver"] == "s3"
    assert spec["bucket"] == "my-bucket"
    assert spec["path"] == "checkpoints/step-100"
    assert spec["endpoint"] == "https://cwobject.com"
    assert spec["aws_region"] == "us-east-1"
    assert "aws_credentials" not in spec


def test_build_kvstore_spec_s3_virtual_hosted(_clean_s3_env):
    # With the flag set, the bucket is folded into the endpoint host as a subdomain and the
    # bucket field is emptied, producing the virtual-hosted spec CoreWeave cwobject requires.
    _clean_s3_env.setenv("AWS_ENDPOINT_URL", "https://cwobject.com")
    _clean_s3_env.setenv("LEVANTER_S3_VIRTUAL_HOSTED", "1")
    _clean_s3_env.setenv("AWS_ACCESS_KEY_ID", "key")
    _clean_s3_env.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    spec = build_kvstore_spec("s3://marin-us-east-02a/checkpoints/step-100")
    assert spec == {
        "driver": "s3",
        "bucket": "",
        "path": "checkpoints/step-100",
        "endpoint": "https://marin-us-east-02a.cwobject.com",
        "aws_region": "us-east-1",
        "aws_credentials": {"type": "environment"},
    }


def test_build_kvstore_spec_virtual_hosted_noop_without_endpoint(_clean_s3_env):
    # The flag only takes effect when a custom endpoint is configured; without one it is a
    # no-op and ordinary AWS S3 (default path/virtual handling) is left untouched.
    _clean_s3_env.setenv("LEVANTER_S3_VIRTUAL_HOSTED", "true")

    spec = build_kvstore_spec("s3://my-bucket/checkpoints/step-100")
    assert spec["bucket"] == "my-bucket"
    assert "endpoint" not in spec
