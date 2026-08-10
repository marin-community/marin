# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.mok import MokBf16Config, MokRuntimeHandle
import levanter.kernels.mok.ffi as mok_ffi
import levanter.kernels.mok.runtime as mok_runtime
from levanter.kernels.mok.ffi import (
    _backward_ffi_inputs,
    _pack_weights,
    _row_major_layout,
    _schedule_capacity,
    _unpack_weight_grads,
)


def _dense_swiglu(x, gate, up, down):
    return (jax.nn.silu(x @ gate) * (x @ up)) @ down


def test_two_shared_experts_packed_value_matches_independent_sum():
    key = jax.random.key(0)
    keys = jax.random.split(key, 8)
    x = jax.random.normal(keys[0], (7, 8))
    shared = (
        jax.random.normal(keys[1], (8, 4)),
        jax.random.normal(keys[2], (8, 4)),
        jax.random.normal(keys[3], (4, 8)),
        jax.random.normal(keys[4], (8, 4)),
        jax.random.normal(keys[5], (8, 4)),
        jax.random.normal(keys[6], (4, 8)),
    )
    routed = (jnp.zeros((2, 8, 4)), jnp.zeros((2, 8, 4)), jnp.zeros((2, 4, 8)))

    packed_gate, packed_up, packed_down, *_ = _pack_weights(*shared, *routed)
    actual = (jax.nn.silu(x @ packed_gate.T) * (x @ packed_up.T)) @ packed_down.T
    expected = _dense_swiglu(x, *shared[:3]) + _dense_swiglu(x, *shared[3:])

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_native_weight_gradients_unpack_to_canonical_leaves():
    canonical = (
        jnp.zeros((8, 4)),
        jnp.zeros((8, 4)),
        jnp.zeros((4, 8)),
        jnp.zeros((8, 4)),
        jnp.zeros((8, 4)),
        jnp.zeros((4, 8)),
        jnp.zeros((2, 8, 4)),
        jnp.zeros((2, 8, 4)),
        jnp.zeros((2, 4, 8)),
    )
    packed, pullback = jax.vjp(lambda *weights: _pack_weights(*weights), *canonical)
    packed_cotangents = tuple(jnp.arange(value.size, dtype=jnp.float32).reshape(value.shape) for value in packed)

    expected = pullback(packed_cotangents)
    actual = _unpack_weight_grads(*packed_cotangents)

    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_backward_ffi_wire_signature_omits_selected_experts():
    grad_y = jnp.zeros((4, 8), jnp.bfloat16)
    x = jnp.zeros((4, 8), jnp.bfloat16)
    selected_experts = jnp.zeros((4, 2), jnp.int32)
    router_weights = jnp.zeros((4, 2), jnp.float32)
    packed_weights = (
        jnp.zeros((8, 8), jnp.bfloat16),
        jnp.zeros((8, 8), jnp.bfloat16),
        jnp.zeros((8, 8), jnp.bfloat16),
        jnp.zeros((2, 4, 8), jnp.bfloat16),
        jnp.zeros((2, 4, 8), jnp.bfloat16),
        jnp.zeros((2, 8, 4), jnp.bfloat16),
    )
    forward_context = tuple(jnp.zeros((1,), jnp.bfloat16) for _ in range(11))

    inputs = _backward_ffi_inputs(
        grad_y,
        (x, selected_experts, router_weights, *packed_weights),
        forward_context,
    )

    assert len(inputs) == 20
    assert inputs[1] is x
    assert inputs[2] is router_weights
    assert all(value is not selected_experts for value in inputs)


@pytest.mark.parametrize(
    ("ep_size", "multiplier", "expected"),
    ((4, 0.1, 2048), (4, 0.5, 2048), (64, 0.5, 32768)),
)
def test_schedule_capacity_matches_dropless_workspace_contract(ep_size, multiplier, expected):
    assert _schedule_capacity(tokens=512, topk=2, ep_size=ep_size, multiplier=multiplier) == expected


def test_ffi_layout_uses_jax_major_to_minor_row_major_order():
    assert _row_major_layout((2, 3, 5)) == (0, 1, 2)


def test_mok_package_import_does_not_import_torch_or_native_extension():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import levanter.kernels.mok; "
            "assert 'torch' not in sys.modules; assert 'mok' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_config_rejects_unaligned_minibatches_before_runtime_setup():
    with pytest.raises(ValueError, match="minibatch_size"):
        MokBf16Config(minibatch_size=1000)


def test_runtime_handle_closes_registered_workspace_once(monkeypatch):
    closed_workspaces = []
    monkeypatch.setattr(mok_runtime, "close_mok_runtime", closed_workspaces.append)
    handle = MokRuntimeHandle(workspace_id=7)

    handle.close()
    handle.close()

    assert closed_workspaces == [7]


def test_custom_vjp_traces_through_checkpoint_without_ffi_effect(monkeypatch):
    class ScratchQueries:
        @staticmethod
        def levanter_mok_bf16_forward_scratch_bytes_v1(*args):
            return 0

        @staticmethod
        def levanter_mok_bf16_backward_scratch_bytes_v1(*args):
            return 0

    monkeypatch.setattr(mok_ffi, "_native_extension", ScratchQueries)
    monkeypatch.setattr(mok_ffi, "register_ffi_targets", lambda: None)
    tokens, hidden_size, intermediate_size, num_experts, topk = 4, 8, 4, 4, 2
    args = (
        jnp.ones((tokens, hidden_size), jnp.bfloat16),
        jnp.zeros((tokens, topk), jnp.int32),
        jnp.ones((tokens, topk), jnp.float32),
        jnp.ones((hidden_size, intermediate_size), jnp.bfloat16),
        jnp.ones((hidden_size, intermediate_size), jnp.bfloat16),
        jnp.ones((intermediate_size, hidden_size), jnp.bfloat16),
        jnp.ones((hidden_size, intermediate_size), jnp.bfloat16),
        jnp.ones((hidden_size, intermediate_size), jnp.bfloat16),
        jnp.ones((intermediate_size, hidden_size), jnp.bfloat16),
        jnp.ones((num_experts, hidden_size, intermediate_size), jnp.bfloat16),
        jnp.ones((num_experts, hidden_size, intermediate_size), jnp.bfloat16),
        jnp.ones((num_experts, intermediate_size, hidden_size), jnp.bfloat16),
    )
    config = MokBf16Config(minibatch_size=256, macrobatch_size=256)

    def loss(*operands):
        output = mok_ffi._mok_bf16_local(*operands, 4, config)
        return jnp.sum(output.astype(jnp.float32))

    rematerialized_grad = jax.grad(jax.checkpoint(loss), argnums=(0, 2, 3))
    jaxpr = jax.make_jaxpr(rematerialized_grad)(*args)

    assert not jaxpr.effects
