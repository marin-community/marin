# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`moe_mlp(implementation="marin_ep")` end-to-end vs the oracle, on GPUs.

The levanter dispatch must wire the fused backend with per-owner pooling
(G = local experts). Importing `grug_moe` pulls in the sonic backends and
their `jax_triton` dependency, which is broken on some dev-pod envs (cu13
plugin vs jaxlib version skew) — hence the module-level importorskip so
the transport conformance tests in sibling modules still collect there.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.ep_marin import _static_capacity

from experiments.marin_ep.oracle import moe_oracle, pooled_keep_mask

grug_moe = pytest.importorskip("levanter.grug.grug_moe", reason="needs a functional jax_triton install")


def test_moe_mlp_marin_ep_implementation_matches_oracle_on_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("needs real GPUs")
    devices = len(jax.devices())
    if devices < 2:
        pytest.skip("needs >= 2 GPUs")
    tokens, topk, hidden, intermediate = 128, 3, 256, 256
    num_experts = devices * 4
    local_experts = 4
    capacity_factor = 1.1

    rng = np.random.default_rng(seed=11)
    probs = rng.dirichlet(np.full(num_experts, 0.4))
    experts = rng.choice(num_experts, size=(devices * tokens, topk), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)
    weights = (rng.random((devices * tokens, topk)) + 0.05).astype(np.float32)
    w13 = (0.3 * rng.standard_normal((num_experts, hidden, 2 * intermediate))).astype(np.float32)
    w2 = (0.3 * rng.standard_normal((num_experts, intermediate, hidden))).astype(np.float32)

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, devices, 1),
        ("data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    batch_spec = P(("data", "expert"))

    def put(a, spec):
        return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))

    with jax.set_mesh(mesh):
        y, dropped = jax.jit(
            partial(
                grug_moe.moe_mlp,
                implementation="marin_ep",
                capacity_factor=capacity_factor,
                report_capacity_overflow=True,
            )
        )(
            put(x, batch_spec),
            put(experts, batch_spec),
            put(weights, batch_spec),
            put(w13, P("expert", None, None)),
            put(w2, P("expert", None, None)),
        )

    capacity = _static_capacity(devices * tokens * topk, num_experts, capacity_factor)
    keep, dropped_oracle = pooled_keep_mask(
        experts.reshape(devices, tokens, topk), num_experts=num_experts, capacity=capacity, group_size=local_experts
    )
    y_oracle = moe_oracle(
        jnp.asarray(x),
        jnp.asarray(experts),
        jnp.asarray(weights),
        jnp.asarray(w13),
        jnp.asarray(w2),
        jnp.asarray(keep.reshape(devices * tokens, topk)),
        activation_fn=jax.nn.silu,
    )
    assert int(dropped) == dropped_oracle
    assert dropped_oracle > 0
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_oracle), rtol=2e-2, atol=0.2)


@pytest.mark.parametrize("fused_implementation", ["marin_ep_mgpu_fused", "marin_ep_mgpu_fused2"])
def test_moe_mlp_mgpu_fused_matches_mgpu_brd_on_gpu(fused_implementation):
    """The fused flavors must reproduce the multi-launch brd flavor.

    Both run identical GEMM kernels on identical pool layouts; only the
    dispatch fusion differs, so values and gradients must match bitwise.
    Dims straddle the fused kernel's tile constants: hidden 512 = 2 transport
    lanes, 2I = 640 exercises the N padding to the 256-wide collective tile.
    """
    if jax.default_backend() != "gpu":
        pytest.skip("needs real GPUs")
    if float(jax.devices()[0].compute_capability) < 10:
        pytest.skip("needs Blackwell (tcgen05 ragged dot)")
    devices = len(jax.devices())
    if devices < 2:
        pytest.skip("needs >= 2 GPUs")
    tokens, topk, hidden, intermediate = 1024, 3, 512, 320
    num_experts = devices * 4
    capacity_factor = 1.1

    rng = np.random.default_rng(seed=13)
    probs = rng.dirichlet(np.full(num_experts, 0.4))
    experts = rng.choice(num_experts, size=(devices * tokens, topk), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)
    weights = (rng.random((devices * tokens, topk)) + 0.05).astype(np.float32)
    w13 = (0.3 * rng.standard_normal((num_experts, hidden, 2 * intermediate))).astype(np.float32)
    w2 = (0.3 * rng.standard_normal((num_experts, intermediate, hidden))).astype(np.float32)

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, devices, 1),
        ("data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    batch_spec = P(("data", "expert"))

    def put(a, spec):
        return jax.device_put(jnp.asarray(a, dtype=jnp.bfloat16), NamedSharding(mesh, spec))

    experts_sharded = jax.device_put(jnp.asarray(experts), NamedSharding(mesh, batch_spec))

    def run(implementation):
        def loss(xb, w13b, w2b):
            y, dropped = grug_moe.moe_mlp(
                xb,
                experts_sharded,
                put(weights, batch_spec),
                w13b,
                w2b,
                implementation=implementation,
                capacity_factor=capacity_factor,
                report_capacity_overflow=True,
            )
            return jnp.sum(jnp.square(y.astype(jnp.float32))), (y, dropped)

        with jax.set_mesh(mesh):
            (val, (y, dropped)), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
                put(x, batch_spec),
                put(w13, P("expert", None, None)),
                put(w2, P("expert", None, None)),
            )
        del val
        return jax.block_until_ready((y, dropped, grads))

    y_b, dropped_b, grads_b = run("marin_ep_mgpu_brd")
    y_f, dropped_f, grads_f = run(fused_implementation)

    assert int(dropped_f) == int(dropped_b)
    np.testing.assert_array_equal(np.asarray(y_f, np.float32), np.asarray(y_b, np.float32))
    for gf, gb, name in zip(grads_f, grads_b, ("dx", "dw13", "dw2"), strict=True):
        np.testing.assert_array_equal(np.asarray(gf, np.float32), np.asarray(gb, np.float32), err_msg=name)
