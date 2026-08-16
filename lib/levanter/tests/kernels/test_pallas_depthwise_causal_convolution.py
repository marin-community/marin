# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# copied from https://github.com/open-lm-engine/accelerated-model-architectures
# **************************************************

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from levanter.kernels.pallas.depthwise_causal_convolution import depthwise_causal_convolution


_TOLERANCE = {"atol": 2e-4, "rtol": 0}
_PALLAS_TOLERANCE = {jnp.float32: {"atol": 2e-4, "rtol": 0}, jnp.bfloat16: {"atol": 2e-2, "rtol": 0}}
_PALLAS_GRAD_TOLERANCE = {jnp.float32: {"atol": 1e-2, "rtol": 0}, jnp.bfloat16: {"atol": 5e-2, "rtol": 0}}


def _reference_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    input_state: np.ndarray | None,
    output_state: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    B, S, H = x.shape
    K = weight.shape[-1]

    xt = np.transpose(x, (0, 2, 1))
    state = np.zeros((B, H, K - 1), dtype=np.float32) if input_state is None else input_state.astype(np.float32)
    full = np.concatenate([state, xt.astype(np.float32)], axis=-1)

    y = np.zeros((B, H, S), dtype=np.float32)
    for j in range(S):
        window = full[:, :, j : j + K]
        y[:, :, j] = (window * weight[None, :, :].astype(np.float32)).sum(-1)

    if bias is not None:
        y = y + bias[None, :, None].astype(np.float32)

    y = np.transpose(y, (0, 2, 1)).astype(x.dtype)
    final_state = full[:, :, 1 - K :].astype(x.dtype) if output_state else None

    return y, final_state


def _generate_args() -> list:
    return list(
        product(
            [2, 4],  # kernel_size; K == 1 is intentionally unsupported
            [1, 2, 6, 9],  # sequence length: shorter than, equal to, or longer than kernel_size
            [False, True],  # has_input_state
            [False, True],  # add_bias
            [False, True],  # output_state
        )
    )


@pytest.mark.parametrize("kernel_size,S,has_input_state,add_bias,output_state", _generate_args())
def test_depthwise_causal_convolution_xla_matches_reference(
    kernel_size: int, S: int, has_input_state: bool, add_bias: bool, output_state: bool
) -> None:
    B, H = 2, 5
    std = 0.1

    key_x, key_w, key_b, key_h0 = jax.random.split(jax.random.PRNGKey(0), 4)

    x = jax.random.normal(key_x, (B, S, H), dtype=jnp.float32) * std
    weight = jax.random.normal(key_w, (H, kernel_size), dtype=jnp.float32) * std
    bias = jax.random.normal(key_b, (H,), dtype=jnp.float32) * std if add_bias else None
    input_state = (
        jax.random.normal(key_h0, (B, H, kernel_size - 1), dtype=jnp.float32) * std if has_input_state else None
    )

    y, final_state = depthwise_causal_convolution(
        x, weight, bias, input_state, output_state=output_state, implementation="xla"
    )

    y_ref, final_state_ref = _reference_numpy(
        np.asarray(x),
        np.asarray(weight),
        np.asarray(bias) if bias is not None else None,
        np.asarray(input_state) if input_state is not None else None,
        output_state,
    )

    assert_allclose(np.asarray(y), y_ref, **_TOLERANCE)

    if output_state:
        assert final_state is not None
        assert final_state.shape == (B, H, kernel_size - 1)
        assert_allclose(np.asarray(final_state), final_state_ref, **_TOLERANCE)
    else:
        assert final_state is None


@pytest.mark.parametrize("kernel_size", [2, 4])
def test_depthwise_causal_convolution_xla_grad_runs(kernel_size: int) -> None:
    B, S, H = 2, 6, 5
    std = 0.1

    key_x, key_w, key_b, key_h0 = jax.random.split(jax.random.PRNGKey(1), 4)

    x = jax.random.normal(key_x, (B, S, H), dtype=jnp.float32) * std
    weight = jax.random.normal(key_w, (H, kernel_size), dtype=jnp.float32) * std
    bias = jax.random.normal(key_b, (H,), dtype=jnp.float32) * std
    input_state = jax.random.normal(key_h0, (B, H, kernel_size - 1), dtype=jnp.float32) * std

    def f(x, weight, bias, input_state):
        y, _ = depthwise_causal_convolution(x, weight, bias, input_state, output_state=False, implementation="xla")

        return y.sum()

    dx, dweight, dbias, dinput_state = jax.grad(f, argnums=(0, 1, 2, 3))(x, weight, bias, input_state)

    for name, grad, expected_shape in [
        ("dx", dx, x.shape),
        ("dweight", dweight, weight.shape),
        ("dbias", dbias, bias.shape),
        ("dinput_state", dinput_state, input_state.shape),
    ]:
        assert grad.shape == expected_shape, name
        assert bool(jnp.all(jnp.isfinite(grad))), name


def _generate_pallas_args() -> list:
    return list(
        product(
            [2, 4],  # kernel_size: the pallas_tpu implementation assumes kernel_size > 1
            # sequence length: shorter than, equal to, or not a multiple of the internal block size.
            # 1 and 2 also cover S < kernel_size - 1, where ht keeps part of input_state rather than
            # being filled entirely from input.
            [1, 2, 3, 16, 37, 130],
            [False, True],  # has_input_state
            [False, True],  # add_bias
            [False, True],  # output_state
            [jnp.float32, jnp.bfloat16],
            [None, "silu", "swish"],
        )
    )


@pytest.mark.parametrize(
    "kernel_size,S,has_input_state,add_bias,output_state,dtype,activation_function", _generate_pallas_args()
)
def test_depthwise_causal_convolution_pallas_tpu_matches_xla(
    kernel_size: int,
    S: int,
    has_input_state: bool,
    add_bias: bool,
    output_state: bool,
    dtype: jnp.dtype,
    activation_function: str | None,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("pallas_tpu implementation is only supported on TPU")

    B, H = 2, 5
    std = 0.1
    tolerance = _PALLAS_TOLERANCE[dtype]
    grad_tolerance = _PALLAS_GRAD_TOLERANCE[dtype]

    key_x, key_w, key_b, key_h0, key_dy = jax.random.split(jax.random.PRNGKey(3), 5)

    x = (jax.random.normal(key_x, (B, S, H), dtype=jnp.float32) * std).astype(dtype)
    weight = (jax.random.normal(key_w, (H, kernel_size), dtype=jnp.float32) * std).astype(dtype)
    bias = (jax.random.normal(key_b, (H,), dtype=jnp.float32) * std).astype(dtype) if add_bias else None
    input_state = (
        (jax.random.normal(key_h0, (B, H, kernel_size - 1), dtype=jnp.float32) * std).astype(dtype)
        if has_input_state
        else None
    )

    def _run(implementation: str, x: jax.Array, weight: jax.Array, bias, input_state):
        return depthwise_causal_convolution(
            x,
            weight,
            bias,
            input_state,
            output_state=output_state,
            activation_function=activation_function,
            implementation=implementation,
        )

    def _run_vjp(implementation: str, x: jax.Array, weight: jax.Array, bias, input_state):
        return jax.vjp(
            lambda x, weight, bias, input_state: _run(implementation, x, weight, bias, input_state),
            x,
            weight,
            bias,
            input_state,
        )

    (y_kernel, ht_kernel), vjp_kernel = _run_vjp("pallas_tpu", x, weight, bias, input_state)
    (y_expected, ht_expected), vjp_expected = _run_vjp("xla", x, weight, bias, input_state)

    assert_allclose(np.asarray(y_kernel, dtype=np.float32), np.asarray(y_expected, dtype=np.float32), **tolerance)

    if output_state:
        assert ht_kernel is not None
        assert ht_expected is not None
        assert ht_kernel.shape == (B, H, kernel_size - 1)
        assert_allclose(
            np.asarray(ht_kernel, dtype=np.float32), np.asarray(ht_expected, dtype=np.float32), **tolerance
        )
    else:
        assert ht_kernel is None
        assert ht_expected is None

    dy = (jax.random.normal(key_dy, y_kernel.shape, dtype=jnp.float32) * std).astype(dtype)
    dht = (jax.random.normal(key_dy, ht_kernel.shape, dtype=jnp.float32) * std).astype(dtype) if output_state else None

    dx_kernel, dweight_kernel, dbias_kernel, dinput_state_kernel = vjp_kernel((dy, dht))
    dx_expected, dweight_expected, dbias_expected, dinput_state_expected = vjp_expected((dy, dht))

    assert_allclose(
        np.asarray(dx_kernel, dtype=np.float32), np.asarray(dx_expected, dtype=np.float32), **grad_tolerance
    )
    assert_allclose(
        np.asarray(dweight_kernel, dtype=np.float32), np.asarray(dweight_expected, dtype=np.float32), **grad_tolerance
    )

    if add_bias:
        assert_allclose(
            np.asarray(dbias_kernel, dtype=np.float32), np.asarray(dbias_expected, dtype=np.float32), **grad_tolerance
        )
    else:
        assert dbias_kernel is None
        assert dbias_expected is None

    if has_input_state:
        assert_allclose(
            np.asarray(dinput_state_kernel, dtype=np.float32),
            np.asarray(dinput_state_expected, dtype=np.float32),
            **grad_tolerance,
        )
    else:
        assert dinput_state_kernel is None
        assert dinput_state_expected is None
