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
from levanter.kernels.mok.reference import mok_bf16_reference, rmsnorm_reference
import levanter.kernels.mok.runtime as mok_runtime
from levanter.kernels.mok.ffi import (
    _backward_ffi_inputs,
    _pack_weights,
    _row_major_layout,
    _schedule_capacity,
    _validate_shapes,
    disabled_latent_weights,
    _unpack_weight_grads,
)


def _dense_swiglu(x, gate, up, down):
    return (jax.nn.silu(x @ gate) * (x @ up)) @ down


def _zero_latent(hidden_size):
    return (
        jnp.zeros((hidden_size, 0), jnp.bfloat16),
        jnp.zeros((0,), jnp.float32),
        jnp.zeros((0, hidden_size), jnp.bfloat16),
    )


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

    packed_gate, packed_up, packed_down, *_ = _pack_weights(*shared, *routed, *_zero_latent(8))
    actual = (jax.nn.silu(x @ packed_gate.T) * (x @ packed_up.T)) @ packed_down.T
    expected = _dense_swiglu(x, *shared[:3]) + _dense_swiglu(x, *shared[3:])

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_latent_projections_are_transposed_in_opposite_directions():
    # Marin builds w_latent_down as (hidden, latent) and w_latent_up as (latent, hidden); the
    # kernel wants the reverse of each. At hidden != latent a missed transpose is a shape error,
    # but at hidden == latent it would be silent numerical garbage, so pin the orientation.
    hidden_size, latent_size = 8, 4
    latent_down = jnp.arange(hidden_size * latent_size, dtype=jnp.bfloat16).reshape(hidden_size, latent_size)
    latent_norm = jnp.arange(latent_size, dtype=jnp.float32)
    latent_up = jnp.arange(latent_size * hidden_size, dtype=jnp.bfloat16).reshape(latent_size, hidden_size)
    shared = (jnp.zeros((hidden_size, 2), jnp.bfloat16),) * 2 + (jnp.zeros((2, hidden_size), jnp.bfloat16),)
    routed = (jnp.zeros((2, latent_size, 2), jnp.bfloat16),) * 2 + (jnp.zeros((2, 2, latent_size), jnp.bfloat16),)

    packed = _pack_weights(*shared, *shared, *routed, latent_down, latent_norm, latent_up)
    packed_down, packed_norm, packed_up = packed[6:]

    assert packed_down.shape == (latent_size, hidden_size)
    assert packed_up.shape == (hidden_size, latent_size)
    np.testing.assert_array_equal(np.asarray(packed_down, np.float32), np.asarray(latent_down, np.float32).T)
    np.testing.assert_array_equal(np.asarray(packed_up, np.float32), np.asarray(latent_up, np.float32).T)
    # The norm gain is float32 on both sides and must not be transposed or cast.
    np.testing.assert_array_equal(np.asarray(packed_norm), np.asarray(latent_norm))
    assert packed_norm.dtype == jnp.float32


def test_disabled_latent_weights_zero_the_latent_axis_in_each_leafs_own_position():
    latent_down, latent_norm, latent_up = disabled_latent_weights(jnp.zeros((4, 512), jnp.bfloat16))
    assert latent_down.shape == (512, 0)
    assert latent_norm.shape == (0,)
    assert latent_up.shape == (0, 512)
    assert latent_norm.dtype == jnp.float32


def test_validate_shapes_checks_the_shared_and_routed_widths_independently():
    hidden_size, latent_size, intermediate = 512, 256, 256
    x = jnp.zeros((256, hidden_size), jnp.bfloat16)
    selected = jnp.zeros((256, 2), jnp.int32)
    router = jnp.zeros((256, 2), jnp.float32)
    shared = (
        jnp.zeros((hidden_size, intermediate), jnp.bfloat16),
        jnp.zeros((hidden_size, intermediate), jnp.bfloat16),
        jnp.zeros((intermediate, hidden_size), jnp.bfloat16),
    ) * 2
    routed = (
        jnp.zeros((2, latent_size, intermediate), jnp.bfloat16),
        jnp.zeros((2, latent_size, intermediate), jnp.bfloat16),
        jnp.zeros((2, intermediate, latent_size), jnp.bfloat16),
    )
    latent = (
        jnp.zeros((hidden_size, latent_size), jnp.bfloat16),
        jnp.zeros((latent_size,), jnp.float32),
        jnp.zeros((latent_size, hidden_size), jnp.bfloat16),
    )

    _validate_shapes(x, selected, router, shared, routed, latent, latent_size=latent_size)

    # Routed weights sized at the shared width are the single-width bug this ABI exists to stop.
    wide_routed = (
        jnp.zeros((2, hidden_size, intermediate), jnp.bfloat16),
        jnp.zeros((2, hidden_size, intermediate), jnp.bfloat16),
        jnp.zeros((2, intermediate, hidden_size), jnp.bfloat16),
    )
    with pytest.raises(ValueError, match="routed weights"):
        _validate_shapes(x, selected, router, shared, wide_routed, latent, latent_size=latent_size)

    # latent_up transposed the wrong way is (latent, hidden) -> (hidden, latent): legal-looking.
    flipped = (latent[0], latent[1], jnp.zeros((hidden_size, latent_size), jnp.bfloat16))
    with pytest.raises(ValueError, match="latent weights"):
        _validate_shapes(x, selected, router, shared, routed, flipped, latent_size=latent_size)


def test_mok_bf16_rejects_a_partially_supplied_shared_slot():
    x = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    present = jnp.zeros((256, 256), dtype=jnp.bfloat16)
    routed = (
        jnp.zeros((2, 256, 256), dtype=jnp.bfloat16),
        jnp.zeros((2, 256, 256), dtype=jnp.bfloat16),
        jnp.zeros((2, 256, 256), dtype=jnp.bfloat16),
    )

    with pytest.raises(ValueError, match="not optional"):
        mok_ffi.mok_bf16(
            x,
            jnp.zeros((512, 2), dtype=jnp.int32),
            jnp.zeros((512, 2), dtype=jnp.float32),
            present,
            None,
            None,
            None,
            None,
            None,
            *routed,
            config=MokBf16Config(),
        )


def test_mok_bf16_requires_latent_weights_whenever_latent_size_is_set():
    x = jnp.zeros((512, 256), dtype=jnp.bfloat16)
    shared = (jnp.zeros((256, 256), dtype=jnp.bfloat16),) * 6
    routed = (jnp.zeros((2, 256, 256), dtype=jnp.bfloat16),) * 3

    with pytest.raises(ValueError, match="latent_size=256"):
        mok_ffi.mok_bf16(
            x,
            jnp.zeros((512, 2), dtype=jnp.int32),
            jnp.zeros((512, 2), dtype=jnp.float32),
            *shared,
            *routed,
            config=MokBf16Config(latent_size=256),
        )


def test_native_weight_gradients_unpack_to_canonical_leaves():
    canonical = (
        jnp.zeros((8, 4)),
        jnp.zeros((8, 4)),
        jnp.zeros((4, 8)),
        jnp.zeros((8, 4)),
        jnp.zeros((8, 4)),
        jnp.zeros((4, 8)),
        jnp.zeros((2, 6, 4)),
        jnp.zeros((2, 6, 4)),
        jnp.zeros((2, 4, 6)),
        jnp.zeros((8, 6)),
        jnp.zeros((6,)),
        jnp.zeros((6, 8)),
    )
    packed, pullback = jax.vjp(lambda *weights: _pack_weights(*weights), *canonical)
    packed_cotangents = tuple(jnp.arange(value.size, dtype=jnp.float32).reshape(value.shape) for value in packed)

    expected = pullback(packed_cotangents)
    actual = _unpack_weight_grads(*packed_cotangents)

    assert len(actual) == 12
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
        jnp.zeros((2, 4, 6), jnp.bfloat16),
        jnp.zeros((2, 4, 6), jnp.bfloat16),
        jnp.zeros((2, 6, 4), jnp.bfloat16),
        jnp.zeros((6, 8), jnp.bfloat16),
        jnp.zeros((6,), jnp.float32),
        jnp.zeros((8, 6), jnp.bfloat16),
    )
    forward_context = tuple(jnp.zeros((1,), jnp.bfloat16) for _ in range(14))

    inputs = _backward_ffi_inputs(
        grad_y,
        (x, selected_experts, router_weights, *packed_weights),
        forward_context,
    )

    # 12 operands then the 14 forward residuals: A0..A25.
    assert len(inputs) == 26
    assert inputs[1] is x
    assert inputs[2] is router_weights
    assert inputs[9:12] == packed_weights[6:]
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


def test_config_rejects_a_latent_width_the_workspace_cannot_align():
    with pytest.raises(ValueError, match="latent_size"):
        MokBf16Config(latent_size=1000)


def test_runtime_handle_closes_registered_workspace_once(monkeypatch):
    closed_workspaces = []
    monkeypatch.setattr(mok_runtime, "close_mok_runtime", closed_workspaces.append)
    handle = MokRuntimeHandle(workspace_id=7)

    handle.close()
    handle.close()

    assert closed_workspaces == [7]


class _ScratchQueries:
    """A stand-in native extension that records how the ``_v2`` queries were called."""

    calls: list[tuple[str, tuple[int, ...]]] = []

    @classmethod
    def levanter_mok_bf16_forward_scratch_bytes_v2(cls, *args):
        cls.calls.append(("forward", args))
        return 0

    @classmethod
    def levanter_mok_bf16_backward_scratch_bytes_v2(cls, *args):
        cls.calls.append(("backward", args))
        return 0


def _vjp_operands(tokens, hidden_size, latent_size, intermediate_size, num_experts, topk):
    routed_width = latent_size or hidden_size
    return (
        jnp.ones((tokens, hidden_size), jnp.bfloat16),
        jnp.zeros((tokens, topk), jnp.int32),
        jnp.ones((tokens, topk), jnp.float32),
        *(jnp.ones((hidden_size, intermediate_size), jnp.bfloat16),) * 2,
        jnp.ones((intermediate_size, hidden_size), jnp.bfloat16),
        *(jnp.ones((hidden_size, intermediate_size), jnp.bfloat16),) * 2,
        jnp.ones((intermediate_size, hidden_size), jnp.bfloat16),
        jnp.ones((num_experts, routed_width, intermediate_size), jnp.bfloat16),
        jnp.ones((num_experts, routed_width, intermediate_size), jnp.bfloat16),
        jnp.ones((num_experts, intermediate_size, routed_width), jnp.bfloat16),
        jnp.ones((hidden_size, latent_size), jnp.bfloat16),
        jnp.ones((latent_size,), jnp.float32),
        jnp.ones((latent_size, hidden_size), jnp.bfloat16),
    )


@pytest.mark.parametrize("latent_size", (0, 4))
def test_custom_vjp_traces_through_checkpoint_without_ffi_effect(monkeypatch, latent_size):
    _ScratchQueries.calls = []
    monkeypatch.setattr(mok_ffi, "_native_extension", lambda: _ScratchQueries)
    monkeypatch.setattr(mok_ffi, "register_ffi_targets", lambda: None)
    tokens, hidden_size, intermediate_size, num_experts, topk = 4, 8, 4, 4, 2
    args = _vjp_operands(tokens, hidden_size, latent_size, intermediate_size, num_experts, topk)
    config = MokBf16Config(minibatch_size=256, macrobatch_size=256)
    # latent_size is not 256-aligned here; bypass the dataclass check to keep the trace cheap.
    object.__setattr__(config, "latent_size", latent_size)

    def loss(*operands):
        output = mok_ffi._mok_bf16_local(*operands, 4, config)
        return jnp.sum(output.astype(jnp.float32))

    rematerialized_grad = jax.grad(jax.checkpoint(loss), argnums=(0, 2, 3, 12, 13, 14))
    jaxpr = jax.make_jaxpr(rematerialized_grad)(*args)

    assert not jaxpr.effects
    # latent_size rides in the third positional slot of both _v2 queries.
    assert _ScratchQueries.calls, "the scratch queries were never consulted"
    for _which, call_args in _ScratchQueries.calls:
        assert call_args[0] == tokens
        assert call_args[1] == hidden_size
        assert call_args[2] == latent_size
        assert call_args[3] == topk


def test_reference_matches_a_hand_rolled_latent_moe_composition():
    keys = jax.random.split(jax.random.key(0), 12)
    tokens, hidden_size, latent_size, intermediate_size, num_experts, topk = 5, 8, 4, 3, 4, 2
    x = jax.random.normal(keys[0], (tokens, hidden_size))
    selected = jnp.stack(
        [jnp.arange(topk, dtype=jnp.int32) for _ in range(tokens)],
        axis=0,
    )
    router = jax.random.uniform(keys[1], (tokens, topk), dtype=jnp.float32)
    shared = tuple(
        jax.random.normal(keys[2 + i], shape)
        for i, shape in enumerate(
            [
                (hidden_size, intermediate_size),
                (hidden_size, intermediate_size),
                (intermediate_size, hidden_size),
                (hidden_size, intermediate_size),
                (hidden_size, intermediate_size),
                (intermediate_size, hidden_size),
            ]
        )
    )
    routed = (
        jax.random.normal(keys[8], (num_experts, latent_size, intermediate_size)),
        jax.random.normal(keys[9], (num_experts, latent_size, intermediate_size)),
        jax.random.normal(keys[10], (num_experts, intermediate_size, latent_size)),
    )
    latent_down = jax.random.normal(keys[11], (hidden_size, latent_size))
    latent_norm = jnp.ones((latent_size,), jnp.float32)
    latent_up = jax.random.normal(keys[0], (latent_size, hidden_size))

    actual = mok_bf16_reference(
        x, selected, router, *shared, *routed, latent_down, latent_norm, latent_up, latent_norm_eps=1e-5
    )

    projected = rmsnorm_reference(x @ latent_down, latent_norm, 1e-5)
    combined = jnp.zeros((tokens, latent_size), jnp.float32)
    for slot in range(topk):
        for expert in range(num_experts):
            mask = (selected[:, slot] == expert)[:, None]
            contribution = _dense_swiglu(projected, routed[0][expert], routed[1][expert], routed[2][expert])
            combined = combined + jnp.where(mask, contribution * router[:, slot : slot + 1], 0.0)
    expected = _dense_swiglu(x, *shared[:3]) + _dense_swiglu(x, *shared[3:]) + combined @ latent_up

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4)


def test_reference_without_latent_weights_runs_the_experts_at_the_hidden_width():
    tokens, hidden_size, intermediate_size, num_experts, topk = 4, 8, 3, 2, 1
    x = jnp.ones((tokens, hidden_size))
    selected = jnp.zeros((tokens, topk), jnp.int32)
    router = jnp.ones((tokens, topk), jnp.float32)
    shared = (jnp.zeros((hidden_size, intermediate_size)),) * 2 + (jnp.zeros((intermediate_size, hidden_size)),)
    routed = (
        jnp.ones((num_experts, hidden_size, intermediate_size)),
        jnp.ones((num_experts, hidden_size, intermediate_size)),
        jnp.ones((num_experts, intermediate_size, hidden_size)),
    )

    actual = mok_bf16_reference(x, selected, router, *shared, *shared, *routed)
    expected = _dense_swiglu(x, routed[0][0], routed[1][0], routed[2][0])

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
