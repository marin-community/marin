import equinox as eqx
import jax
import pytest
from chex import assert_trees_all_close
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import haliax
import haliax as hax
import haliax.nn as hnn

from levanter.grad_accum import microbatched


class Mlp(eqx.Module):
    """
    Simple 1 hidden layer MLP implementation
    """

    w_in: hax.NamedArray
    w_out: hax.NamedArray
    In: hax.Axis = eqx.field(static=True)
    Out: hax.Axis = eqx.field(static=True)
    Mid: hax.Axis = eqx.field(static=True)

    @staticmethod
    def init(In: hax.Axis, Out: hax.Axis, Mid: hax.Axis, *, key):
        w_in = hax.random.normal(key, hax.concat_axis_specs(In, Mid)) * 0.02
        w_out = hax.random.normal(key, hax.concat_axis_specs(Mid, Out)) * 0.02
        return Mlp(w_in, w_out, In, Out, Mid)

    def __call__(self, x):
        x = hax.dot(self.w_in, x, axis=self.In)
        x = hnn.relu(x)
        x = hax.dot(self.w_out, x, axis=self.Mid)
        return x


@pytest.mark.parametrize("parallelism", [1, 2, 4])
@pytest.mark.parametrize("accum_steps", [1, 3])
def test_accumulate_gradients_sharded(parallelism, accum_steps):
    In = hax.Axis("In", 32)
    Out = hax.Axis("Out", 32)
    Mid = hax.Axis("Mid", 32)
    Batch = hax.Axis("Batch", len(jax.devices()) * parallelism * accum_steps)
    mlp = Mlp.init(In, Out, Mid, key=jax.random.PRNGKey(0))

    def loss_fn(mlp, x):
        return mlp(x).mean().scalar()

    x = hax.random.normal(jax.random.PRNGKey(0), (Batch, In))

    mesh_shard = Mesh(jax.devices(), ("data",))
    x = jax.device_put(x, NamedSharding(mesh_shard, PartitionSpec("data", None)))

    axis_mapping = {"Batch": "data"}

    mesh = Mesh(jax.devices(), ("data",))

    @hax.partitioning.named_jit(axis_resources=axis_mapping)
    def jit_grad_accum(mlp, x):
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=False)
        grad_fn = microbatched(grad_fn, Batch, parallelism, axis_mapping, axis_mapping)
        acc_v, acc_g = grad_fn(
            mlp,
            x,
        )
        return acc_v, acc_g

    with mesh:
        mlp = haliax.shard(mlp, axis_mapping)
        x = haliax.shard(x, axis_mapping)
        grad_fn = eqx.filter_value_and_grad(loss_fn)
        acc_v, acc_g = jit_grad_accum(mlp, x)
        v, g = grad_fn(mlp, x)

        assert_trees_all_close(acc_v, v, atol=1e-3, rtol=1e-3)

        for l1, l2 in zip(jax.tree_util.tree_leaves(acc_g), jax.tree_util.tree_leaves(g)):
            assert_trees_all_close(l1, l2, atol=1e-3, rtol=1e-3)
