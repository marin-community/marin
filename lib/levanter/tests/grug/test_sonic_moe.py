# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import jax
import jax.numpy as jnp

from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sonic_moe import (
    SonicGatherRaggedDotBlockSizes,
    SonicGatherSumBlockSizes,
    SonicMetadataBlockSizes,
    sonic_gather_ragged_dot_pallas_triton,
    sonic_gather_ragged_dot_reference,
    sonic_gather_sum_pallas_mgpu,
    sonic_gather_sum_pallas_triton,
    sonic_gather_sum_pallas_triton_faithful,
    sonic_gather_sum_reference,
    sonic_topk_metadata_pallas_triton,
    sonic_topk_metadata_reference,
)


def _make_moe_inputs() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(0)
    k_x, k_logits, k_w13, k_w2 = jax.random.split(key, 4)
    selected_experts = jnp.array(
        [
            [1, 0],
            [2, 1],
            [0, 2],
            [1, 2],
            [3, 0],
        ],
        dtype=jnp.int32,
    )
    x = jax.random.normal(k_x, (selected_experts.shape[0], 8), dtype=jnp.float32)
    combine_logits = jax.random.normal(k_logits, selected_experts.shape, dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1)
    w_up_gate = jax.random.normal(k_w13, (4, 8, 12), dtype=jnp.float32)
    w_down = jax.random.normal(k_w2, (4, 6, 8), dtype=jnp.float32)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def test_sonic_gather_sum_pallas_interpret_matches_reference_and_grad():
    dispatch_output = jnp.arange(10 * 9, dtype=jnp.float32).reshape(10, 9) / 10
    dispatch_positions = jnp.array(
        [
            [2, 4],
            [1, 7],
            [0, 8],
            [3, 6],
            [5, 9],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.array(
        [
            [0.7, 0.3],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.5, 0.5],
        ],
        dtype=jnp.float32,
    )
    block_sizes = SonicGatherSumBlockSizes(token_block_size=4, hidden_block_size=8)

    y_ref = sonic_gather_sum_reference(dispatch_output, dispatch_positions, combine_weights)
    y_pallas = sonic_gather_sum_pallas_mgpu(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=block_sizes,
        interpret=True,
    )
    np.testing.assert_allclose(np.asarray(y_pallas), np.asarray(y_ref), rtol=1e-6, atol=1e-6)

    y_triton = sonic_gather_sum_pallas_triton(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=block_sizes,
        interpret=True,
    )
    np.testing.assert_allclose(np.asarray(y_triton), np.asarray(y_ref), rtol=1e-6, atol=1e-6)

    faithful_block_sizes = SonicGatherSumBlockSizes(token_block_size=1, hidden_block_size=5, k_block_size=2)
    y_faithful = sonic_gather_sum_pallas_triton_faithful(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=faithful_block_sizes,
        interpret=True,
    )
    np.testing.assert_allclose(np.asarray(y_faithful), np.asarray(y_ref), rtol=1e-6, atol=1e-6)
    exact_hidden_block_sizes = SonicGatherSumBlockSizes(token_block_size=1, hidden_block_size=3, k_block_size=2)
    y_exact_hidden = sonic_gather_sum_pallas_triton_faithful(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=exact_hidden_block_sizes,
        interpret=True,
    )
    np.testing.assert_allclose(np.asarray(y_exact_hidden), np.asarray(y_ref), rtol=1e-6, atol=1e-6)

    def loss(fn, dispatch_output, combine_weights):
        y = fn(dispatch_output, dispatch_positions, combine_weights)
        return jnp.sum(y * y)

    ref_grads = jax.grad(lambda out, weights: loss(sonic_gather_sum_reference, out, weights), argnums=(0, 1))(
        dispatch_output,
        combine_weights,
    )
    pallas_grads = jax.grad(
        lambda out, weights: loss(
            lambda y, pos, w: sonic_gather_sum_pallas_mgpu(
                y,
                pos,
                w,
                block_sizes=block_sizes,
                interpret=True,
            ),
            out,
            weights,
        ),
        argnums=(0, 1),
    )(dispatch_output, combine_weights)
    triton_grads = jax.grad(
        lambda out, weights: loss(
            lambda y, pos, w: sonic_gather_sum_pallas_triton(
                y,
                pos,
                w,
                block_sizes=block_sizes,
                interpret=True,
            ),
            out,
            weights,
        ),
        argnums=(0, 1),
    )(dispatch_output, combine_weights)
    faithful_grads = jax.grad(
        lambda out, weights: loss(
            lambda y, pos, w: sonic_gather_sum_pallas_triton_faithful(
                y,
                pos,
                w,
                block_sizes=faithful_block_sizes,
                interpret=True,
            ),
            out,
            weights,
        ),
        argnums=(0, 1),
    )(dispatch_output, combine_weights)

    np.testing.assert_allclose(np.asarray(pallas_grads[0]), np.asarray(ref_grads[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pallas_grads[1]), np.asarray(ref_grads[1]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(triton_grads[0]), np.asarray(ref_grads[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(triton_grads[1]), np.asarray(ref_grads[1]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(faithful_grads[0]), np.asarray(ref_grads[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(faithful_grads[1]), np.asarray(ref_grads[1]), rtol=1e-6, atol=1e-6)


def test_sonic_gather_sum_faithful_pallas_interpret_matches_topk_four_reference():
    dispatch_output = jnp.arange(12 * 7, dtype=jnp.float32).reshape(12, 7) / 10
    dispatch_positions = jnp.array(
        [
            [2, 4, 1, 7],
            [0, 8, 3, 6],
            [5, 9, 10, 11],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=jnp.float32,
    )
    block_sizes = SonicGatherSumBlockSizes(token_block_size=1, hidden_block_size=5, k_block_size=3)

    y_ref = sonic_gather_sum_reference(dispatch_output, dispatch_positions, combine_weights)
    y_faithful = sonic_gather_sum_pallas_triton_faithful(
        dispatch_output,
        dispatch_positions,
        combine_weights,
        block_sizes=block_sizes,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(y_faithful), np.asarray(y_ref), rtol=1e-6, atol=1e-6)


def test_moe_mlp_sonic_xla_matches_scatter_values_and_gradients(monkeypatch):
    monkeypatch.setenv("RAGGED_DOT_IMPL", "xla")
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_moe_inputs()

    y_scatter = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        mesh=None,
        local_implementation="scatter",
    )
    y_sonic = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        mesh=None,
        local_implementation="sonic_xla",
    )
    np.testing.assert_allclose(np.asarray(y_sonic), np.asarray(y_scatter), rtol=1e-5, atol=1e-5)

    def loss(local_implementation, x, combine_weights, w_up_gate, w_down):
        y = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=None,
            local_implementation=local_implementation,
        )
        return jnp.mean(y * y)

    scatter_grads = jax.grad(loss, argnums=(1, 2, 3, 4))("scatter", x, combine_weights, w_up_gate, w_down)
    sonic_grads = jax.grad(loss, argnums=(1, 2, 3, 4))("sonic_xla", x, combine_weights, w_up_gate, w_down)

    for sonic_grad, scatter_grad in zip(sonic_grads, scatter_grads, strict=True):
        np.testing.assert_allclose(np.asarray(sonic_grad), np.asarray(scatter_grad), rtol=1e-5, atol=1e-5)


def test_sonic_gather_ragged_dot_pallas_interpret_matches_reference_and_grad():
    key = jax.random.key(1)
    k_x, k_w = jax.random.split(key)
    x = jax.random.normal(k_x, (6, 8), dtype=jnp.float32)
    rhs = jax.random.normal(k_w, (3, 8, 16), dtype=jnp.float32)
    token_ids_sort = jnp.array([1, 4, 0, 3, 5, 2, 0, 4], dtype=jnp.int32)
    group_sizes = jnp.array([2, 3, 3], dtype=jnp.int32)
    block_sizes = SonicGatherRaggedDotBlockSizes(row_block_size=4, contraction_block_size=4)

    y_ref = sonic_gather_ragged_dot_reference(x, token_ids_sort, rhs, group_sizes)
    y_pallas = sonic_gather_ragged_dot_pallas_triton(
        x,
        token_ids_sort,
        rhs,
        group_sizes,
        block_sizes=block_sizes,
        interpret=True,
    )
    np.testing.assert_allclose(np.asarray(y_pallas), np.asarray(y_ref), rtol=1e-5, atol=1e-5)

    def loss(fn, x, rhs):
        y = fn(x, token_ids_sort, rhs, group_sizes)
        return jnp.mean(y * y)

    ref_grads = jax.grad(lambda x, rhs: loss(sonic_gather_ragged_dot_reference, x, rhs), argnums=(0, 1))(x, rhs)
    pallas_grads = jax.grad(
        lambda x, rhs: loss(
            lambda x, token_ids_sort, rhs, group_sizes: sonic_gather_ragged_dot_pallas_triton(
                x,
                token_ids_sort,
                rhs,
                group_sizes,
                block_sizes=block_sizes,
                interpret=True,
            ),
            x,
            rhs,
        ),
        argnums=(0, 1),
    )(x, rhs)

    np.testing.assert_allclose(np.asarray(pallas_grads[0]), np.asarray(ref_grads[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(pallas_grads[1]), np.asarray(ref_grads[1]), rtol=1e-5, atol=1e-5)


def test_sonic_topk_metadata_pallas_interpret_matches_reference():
    selected_experts = jnp.array(
        [
            [1, 0],
            [2, 1],
            [0, 2],
            [1, 2],
            [3, 0],
            [2, 3],
            [1, 0],
            [3, 2],
        ],
        dtype=jnp.int32,
    )
    block_sizes = SonicMetadataBlockSizes(assignments_per_tile=4)

    ref = sonic_topk_metadata_reference(selected_experts, num_experts=4)
    pallas = sonic_topk_metadata_pallas_triton(
        selected_experts,
        num_experts=4,
        block_sizes=block_sizes,
        interpret=True,
    )

    for pallas_value, ref_value in zip(pallas, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(pallas_value), np.asarray(ref_value))


def test_moe_mlp_sonic_xla_gather_w13_matches_scatter_values_and_gradients(monkeypatch):
    monkeypatch.setenv("RAGGED_DOT_IMPL", "xla")
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_moe_inputs()

    y_scatter = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        mesh=None,
        local_implementation="scatter",
    )
    y_sonic = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        mesh=None,
        local_implementation="sonic_xla_gather_w13",
    )
    np.testing.assert_allclose(np.asarray(y_sonic), np.asarray(y_scatter), rtol=1e-5, atol=1e-5)

    def loss(local_implementation, x, combine_weights, w_up_gate, w_down):
        y = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=None,
            local_implementation=local_implementation,
        )
        return jnp.mean(y * y)

    scatter_grads = jax.grad(loss, argnums=(1, 2, 3, 4))("scatter", x, combine_weights, w_up_gate, w_down)
    sonic_grads = jax.grad(loss, argnums=(1, 2, 3, 4))(
        "sonic_xla_gather_w13",
        x,
        combine_weights,
        w_up_gate,
        w_down,
    )

    for sonic_grad, scatter_grad in zip(sonic_grads, scatter_grads, strict=True):
        np.testing.assert_allclose(np.asarray(sonic_grad), np.asarray(scatter_grad), rtol=1e-5, atol=1e-5)
