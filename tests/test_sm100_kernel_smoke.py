# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.nn import ragged_dot
from levanter.grug.attention import AttentionMask, gpu_fa4_thd_attention, reference_attention

pytestmark = pytest.mark.sm100


def _require_sm100() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("requires an NVIDIA SM100 GPU")
    compute_capability = jax.devices("gpu")[0].compute_capability
    if callable(compute_capability):
        compute_capability = compute_capability()
    if str(compute_capability) != "10.0":
        pytest.skip(f"requires compute capability 10.0, got {compute_capability}")


def _weighted_sum(output: jax.Array, cotangent: jax.Array) -> jax.Array:
    return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))


@pytest.mark.parametrize("sliding_window", [None, 5], ids=["full-causal", "sliding-window"])
def test_sm100_fa4_thd_matches_reference_value_and_gradients(sliding_window):
    _require_sm100()
    q_key, k_key, v_key, cotangent_key = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(q_key, (1, 64, 4, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 64, 2, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 64, 2, 128), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    segment_ids = jnp.array([[3] * 31 + [8] * 33], dtype=jnp.int32)
    mask = AttentionMask.causal(sliding_window=sliding_window).with_segment_ids(segment_ids, max_segments=2)

    def fa4(q_arg, k_arg, v_arg):
        return gpu_fa4_thd_attention(q_arg, k_arg, v_arg, mask)

    def reference(q_arg, k_arg, v_arg):
        return reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)

    actual = jax.jit(fa4)(q, k, v)
    expected = jax.jit(reference)(q, k, v)
    np.testing.assert_allclose(actual, expected, atol=7e-2, rtol=7e-2)

    actual_grads = jax.jit(jax.grad(lambda *args: _weighted_sum(fa4(*args), cotangent), argnums=(0, 1, 2)))(q, k, v)
    expected_grads = jax.jit(jax.grad(lambda *args: _weighted_sum(reference(*args), cotangent), argnums=(0, 1, 2)))(
        q, k, v
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)


@pytest.mark.parametrize(
    ("n", "group_sizes"),
    [
        (160, (256, 256)),
        (128, (257, 255)),
        (160, (257, 255)),
    ],
    ids=["aligned-groups-n160", "uneven-groups-n128", "uneven-groups-n160"],
)
def test_sm100_ragged_dot_matches_xla_value_and_gradients(n, group_sizes):
    _require_sm100()
    m, k = 512, 64
    lhs = ((jnp.arange(m * k).reshape(m, k) % 5) - 2).astype(jnp.bfloat16) / 2
    rhs = ((jnp.arange(2 * k * n).reshape(2, k, n) % 7) - 3).astype(jnp.bfloat16) / 2
    cotangent = ((jnp.arange(m * n).reshape(m, n) % 3) - 1).astype(jnp.bfloat16)
    group_sizes = jnp.array(group_sizes, dtype=jnp.int32)
    dimension_numbers = jax.lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((1,), (1,)), ((), ())),
        lhs_ragged_dimensions=(0,),
        rhs_group_dimensions=(0,),
    )

    def triton(lhs_arg, rhs_arg):
        return ragged_dot(lhs_arg, rhs_arg, group_sizes, implementation="triton")

    def xla(lhs_arg, rhs_arg):
        return jax.lax.ragged_dot_general(
            lhs=lhs_arg,
            rhs=rhs_arg,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=dimension_numbers,
        )

    actual = jax.jit(triton)(lhs, rhs)
    expected = jax.jit(xla)(lhs, rhs)
    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-2)

    actual_grads = jax.jit(jax.grad(lambda *args: _weighted_sum(triton(*args), cotangent), argnums=(0, 1)))(lhs, rhs)
    expected_grads = jax.jit(jax.grad(lambda *args: _weighted_sum(xla(*args), cotangent), argnums=(0, 1)))(lhs, rhs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-2, rtol=1e-2)
