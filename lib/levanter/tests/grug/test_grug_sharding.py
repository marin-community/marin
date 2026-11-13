# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard

from levanter.grug.grug_moe import MoEExpertMlp, QBRoutedMoE, moe_mlp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import _axis_names, keep_reduced, replicate_keeping_reduced
from levanter.testing.devices import use_test_devices

_AXES = ("replica_dcn", "data", "expert")
_BATCH_REDUCED = frozenset({"replica_dcn", "data"})


def _mesh(shape, devices):
    return jax.make_mesh(shape, _AXES, axis_types=(AxisType.Explicit,) * 3, devices=devices)


def _weight_gradient_reductions(fn, weights) -> list[str]:
    """The all-reduces and reduce-scatters in ``fn``'s compiled program with a result the size of a weight's shard."""
    text = jax.jit(fn).lower(*weights).compile().as_text()
    shard_sizes = {math.prod(w.sharding.shard_shape(w.shape)) for w in weights}
    results = re.findall(r"= (.+?) (?:all-reduce|reduce-scatter)(?:-start)?\(", text)
    return [
        result
        for result in results
        if any(
            math.prod(map(int, filter(None, dims.split(",")))) in shard_sizes
            for dims in re.findall(r"\[([\d,]*)\]", result)
        )
    ]


@pytest.mark.multidevice
def test_keep_reduced_leaves_out_axes_the_spec_shards_over():
    # A weight reduced over replica_dcn and data enters a shard_map that shards it over data and all-gathers it there,
    # as the chunked sonic_cute path does. Its gradient stays unreduced over both axes and sums to the plain gradient.
    with use_test_devices(8) as devices:
        mesh = _mesh((2, 2, 2), devices)

        def loss(w, x):
            spec = keep_reduced(P("expert", "data", None), w)
            assert spec.reduced == (jax.typeof(w).sharding.spec.reduced - {"data"}), spec

            def local(w, x):
                return jnp.sum((x @ jax.lax.all_gather(w, "data", axis=1, tiled=True)[0]) ** 2)[None]

            sharded = shard_map(local, mesh=mesh, in_specs=(spec, P(_AXES, None)), out_specs=P(_AXES), check_vma=False)
            return jnp.sum(sharded(reshard(w, spec), x))

        def reduced_grad(w, x):
            grad = jax.grad(loss)(reshard(w, P("expert", None, None, reduced=_BATCH_REDUCED)), x)
            assert jax.typeof(grad).sharding.spec.unreduced == _BATCH_REDUCED, jax.typeof(grad)
            return reshard(grad, P("expert", None, None))

        with jax.set_mesh(mesh):
            w = jax.device_put(np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 10, NamedSharding(mesh, P("expert")))
            x = jax.device_put(np.arange(32, dtype=np.float32).reshape(8, 4) / 7, NamedSharding(mesh, P(_AXES)))
            np.testing.assert_allclose(jax.jit(reduced_grad)(w, x), jax.jit(jax.grad(loss))(w, x), rtol=1e-6)


@pytest.mark.multidevice
def test_unreduced_weights_keep_their_specs_and_programs():
    # The grug shard_maps receive exactly the specs they had before, so their programs do not change.
    with use_test_devices(8) as devices:
        mesh = _mesh((2, 2, 2), devices)
        with jax.set_mesh(mesh):
            w = jax.device_put(np.ones((4, 6), np.float32), NamedSharding(mesh, P("expert", None)))
            for spec in (P(None, None), P("expert", None), P(("replica_dcn", "data"), "expert")):
                assert keep_reduced(spec, w) == spec
            replicated = jax.jit(lambda w: reshard(w, P(None, None))).lower(w).as_text()
            assert jax.jit(lambda w: replicate_keeping_reduced(w)).lower(w).as_text() == replicated


def _lm_head_case(mesh):
    hidden = jax.device_put(np.random.RandomState(0).randn(8, 16).astype(np.float32), NamedSharding(mesh, P(_AXES)))
    labels = jax.device_put(np.arange(8, dtype=np.int32) % 32, NamedSharding(mesh, P(_AXES)))
    lm_head = jax.device_put(np.random.RandomState(1).randn(16, 32).astype(np.float32) / 4, NamedSharding(mesh, P()))

    def loss(lm_head):
        return fused_linear_softmax_cross_entropy_loss(hidden, lm_head, labels, reduction="mean")

    return loss, (lm_head,)


def _moe_case(mesh):
    rng = np.random.RandomState(0)
    tokens, hidden, intermediate, experts = 16, 8, 4, 4
    x_sharding = NamedSharding(mesh, P(_AXES, None))
    x = jax.device_put(rng.randn(tokens, hidden).astype(np.float32), x_sharding)
    selected = jax.device_put(rng.randint(0, experts, (tokens, 2)).astype(np.int32), x_sharding)
    combine = jax.device_put(np.full((tokens, 2), 0.5, np.float32), x_sharding)
    # Without expert parallelism the expert bank is replicated, so it is reduced over every token axis.
    w_sharding = NamedSharding(mesh, P("expert", None, None) if mesh.shape["expert"] > 1 else P())
    w_up_gate = jax.device_put(rng.randn(experts, hidden, 2 * intermediate).astype(np.float32) / 4, w_sharding)
    w_down = jax.device_put(rng.randn(experts, intermediate, hidden).astype(np.float32) / 4, w_sharding)

    def loss(w_up_gate, w_down):
        return jnp.sum(moe_mlp(x, selected, combine, w_up_gate, w_down, mesh=mesh) ** 2)

    return loss, (w_up_gate, w_down)


def _router_case(mesh):
    rng = np.random.RandomState(0)
    tokens, hidden, experts = 16, 8, 4
    x = jax.device_put(rng.randn(tokens, hidden).astype(np.float32), NamedSharding(mesh, P(_AXES, None)))
    token_valid = jax.device_put(np.ones(tokens, bool), NamedSharding(mesh, P(_AXES)))
    router = jax.device_put(rng.randn(hidden, experts).astype(np.float32), NamedSharding(mesh, P()))
    router_bias = jax.device_put(np.zeros(experts, np.float32), NamedSharding(mesh, P()))
    expert_mlp = MoEExpertMlp.init(
        num_experts=experts, hidden_dim=hidden, intermediate_dim=4, initializer_std=0.25, key=jax.random.key(0)
    )

    def loss(router):
        routed, _ = QBRoutedMoE(num_experts_per_token=2, batch_axes=_AXES)(
            x, token_valid, router=router, router_bias=router_bias, expert_mlp=expert_mlp, mesh=mesh
        )
        return jnp.sum(routed**2)

    return loss, (router,)


@pytest.mark.multidevice
@pytest.mark.parametrize(
    ("case", "mesh_shape"),
    [
        pytest.param(_lm_head_case, (2, 2, 2), id="lm_head"),
        pytest.param(_moe_case, (2, 2, 2), id="moe_expert_parallel"),
        pytest.param(_moe_case, (2, 4, 1), id="moe_local"),
        pytest.param(_router_case, (2, 2, 2), id="router"),
    ],
)
def test_reduced_weights_get_unreduced_gradients(case, mesh_shape):
    # Weights the caller marks reduced over the token axes come back with gradients unreduced over them: the backward
    # runs no weight-sized reduction, and reducing the gradient afterwards gives the plain gradient.
    with use_test_devices(8) as devices:
        mesh = _mesh(mesh_shape, devices)
        with jax.set_mesh(mesh):
            loss, weights = case(mesh)
            argnums = tuple(range(len(weights)))
            specs = [jax.typeof(w).sharding.spec for w in weights]
            # Tokens are sharded over every mesh axis; each weight is reduced over those it is not sharded over.
            reduced = [frozenset(_AXES) - {name for entry in spec for name in _axis_names(entry)} for spec in specs]

            def unreduced_grads(*weights):
                marked = [reshard(w, spec.update(reduced=r)) for w, spec, r in zip(weights, specs, reduced)]
                return jax.grad(loss, argnums=argnums)(*marked)

            def reduced_afterwards(*weights):
                return [reshard(g, spec) for g, spec in zip(unreduced_grads(*weights), specs)]

            for grad, r in zip(jax.eval_shape(unreduced_grads, *weights), reduced, strict=True):
                assert jax.typeof(grad).sharding.spec.unreduced == r, jax.typeof(grad)
            assert _weight_gradient_reductions(unreduced_grads, weights) == []
            assert _weight_gradient_reductions(jax.grad(loss, argnums=argnums), weights) != []

            plain = jax.jit(jax.grad(loss, argnums=argnums))(*weights)
            for actual, expected in zip(jax.jit(reduced_afterwards)(*weights), plain, strict=True):
                np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)
