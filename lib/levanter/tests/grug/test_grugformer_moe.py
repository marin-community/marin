# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import importlib.util
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax._src import config as jax_config
from jax.extend import core as jax_core
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh
from haliax.nn.ragged_dot import ragged_dot

import levanter.grug.grug_moe as grug_moe
from levanter.grug._moe.common import (
    _deinterleave_gate_up,
    _interleave_gate_up,
    _interleave_halves,
    _prepare_moe_dispatch,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    _scaled_capacity,
    _swiglu_gate_up_backward,
    CapacityDrops,
)
from levanter.grug._moe.ep_deepep import _pack_deepep_local_assignments
from levanter.grug._moe.ep_fixed_all_to_all import _moe_mlp_ep_fixed_a2a_local
from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import (
    _moe_mlp_ep_fixed_pooled_wave_a2a_local,
    _interleaved_receiver_ranks,
    _receiver_ranks,
)
from levanter.grug._moe import ep_ragged_all_to_all
from levanter.grug._moe.ep_ragged_all_to_all import (
    _accepted_assignments,
    _gather_dispatch_rows,
    _RaggedDotExpertMlp,
    _RoutingWeightGradient,
    _transport_buffer,
    _TransportBufferSite,
    _unpermute_from_global_expert,
)
from levanter.grug._moe.sonic import sonic_gather_sum, sonic_scatter_rows
from levanter.grug._moe.topk import top_k_indices
from levanter.grug.grug_moe import (
    MoEExpertMlp,
    MoEExpertMlpPspecs,
    MoeImplementation,
    _clip_receiver_group_sizes,
    _expert_granular_a2a_params,
    moe_mlp,
)
from levanter.utils.activation import ActivationFunctionEnum


_BF16_MOE_RELATIVE_TOLERANCE = 0.02
_FP32_MOE_RELATIVE_TOLERANCE = 1e-4


def _make_dense_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_ep_mesh_or_none() -> Mesh | None:
    """An expert-parallel mesh, or None when the runtime has too few devices to build one.

    Callers skip on None. Under CI's single CPU device that silences every test below that runs a
    backend end to end, so those tests assert nothing on a green run. The repository has no marker
    or fixture for declaring a device requirement; #8704 tracks adding one.
    """
    devices = jax.devices()
    if len(devices) < 2 or len(devices) % 2 != 0:
        return None
    mesh_devices = np.array(devices).reshape(len(devices) // 2, 2, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _make_abstract_moe_mesh(*, data: int, expert: int, model: int) -> AbstractMesh:
    return AbstractMesh(
        axis_sizes=(data, expert, model),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _make_single_expert_mesh() -> Mesh:
    return Mesh(
        np.asarray([jax.devices()[0]]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )


def _count_jaxpr_primitives(value, primitive_name: str, where=lambda eqn: True) -> int:
    """Count the ``primitive_name`` equations that satisfy ``where``, including in sub-programs."""
    jaxpr = getattr(value, "jaxpr", value)
    if isinstance(jaxpr, jax_core.Jaxpr):
        return sum(eqn.primitive.name == primitive_name and where(eqn) for eqn in jaxpr.eqns) + sum(
            _count_jaxpr_primitives(param, primitive_name, where)
            for eqn in jaxpr.eqns
            for param in eqn.params.values()
        )
    if isinstance(value, dict):
        return sum(_count_jaxpr_primitives(item, primitive_name, where) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_jaxpr_primitives(item, primitive_name, where) for item in value)
    return 0


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _make_inputs(
    *,
    key: jax.Array,
    tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    k_x, k_sel, k_logits, k_w13, k_w2 = jax.random.split(key, 5)
    x = jax.random.normal(k_x, (tokens, hidden_dim), dtype=jnp.float32)
    selected_experts = jax.random.randint(k_sel, (tokens, topk), 0, num_experts, dtype=jnp.int32)
    combine_logits = jax.random.normal(k_logits, (tokens, topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1)
    w_up_gate = jax.random.normal(k_w13, (num_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.float32)
    w_down = jax.random.normal(k_w2, (num_experts, intermediate_dim, hidden_dim), dtype=jnp.float32)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def _dense_moe_output(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> jax.Array:
    selected_w_up_gate = w_up_gate[selected_experts]
    hidden = jnp.einsum("th,tkhi->tki", x, selected_w_up_gate)
    intermediate_dim = w_down.shape[1]
    gate, up = jnp.split(hidden, [intermediate_dim], axis=-1)
    expert_output = jnp.einsum(
        "tki,tkih->tkh",
        jax.nn.silu(gate) * up,
        w_down[selected_experts],
    )
    return jnp.einsum("tkh,tk->th", expert_output, combine_weights)


def _make_unique_topk_experts(*, tokens: int, topk: int, num_experts: int) -> jax.Array:
    if topk > num_experts:
        raise ValueError(f"topk must be <= num_experts, got topk={topk}, num_experts={num_experts}")
    token_ids = jnp.arange(tokens, dtype=jnp.int32)[:, None]
    expert_offsets = jnp.arange(topk, dtype=jnp.int32)[None, :]
    return (token_ids + expert_offsets) % num_experts


def _gather_sum_reference(
    dispatch_output: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
) -> jax.Array:
    out = jnp.zeros((dispatch_positions.shape[0], dispatch_output.shape[1]), dtype=dispatch_output.dtype)
    weights = combine_weights.astype(dispatch_output.dtype)
    for topk_index in range(dispatch_positions.shape[1]):
        out = out + dispatch_output[dispatch_positions[:, topk_index]] * weights[:, topk_index, None]
    return out


def _skip_without_sonic_gpu_runtime() -> None:
    optional_modules = ("jax_triton", "triton")
    if not all(importlib.util.find_spec(module) is not None for module in optional_modules):
        pytest.skip("raw Sonic optional dependencies are not installed")
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("raw Sonic triton_call tests require a GPU")


def test_interleaved_receiver_ranks_allocate_capacity_round_robin_over_sources():
    """The interleaved receiver ranks must (1) keep the tokens a round-robin-over-sources fill would
    keep, (2) leave the per-expert drop count at `min(count, capacity)`, and (3) keep, within any one
    source, a prefix of pool positions per expert -- so a later token never displaces an earlier one in
    the same sequence (each expert shard holds whole sequences)."""
    expert_shards, pool_capacity, local_experts, receiver_capacity = 4, 5, 2, 3
    send_size = expert_shards * pool_capacity
    rng = np.random.default_rng(0)
    received = rng.integers(-1, local_experts, size=send_size)  # -1 marks an empty slot
    received_experts = jnp.asarray(received, dtype=jnp.int32)

    ranks = _interleaved_receiver_ranks(
        received_experts,
        local_experts=local_experts,
        expert_shards=expert_shards,
        pool_capacity=pool_capacity,
    )
    keep = np.asarray((received_experts >= 0) & (ranks < receiver_capacity))

    # Reference: visit each source's slot `pos` before any source's slot `pos + 1` (round-robin over
    # sources), keeping the first `receiver_capacity` per expert.
    counts = np.zeros(local_experts, dtype=int)
    reference = np.zeros(send_size, dtype=bool)
    for pos in range(pool_capacity):
        for shard in range(expert_shards):
            i = shard * pool_capacity + pos
            e = int(received[i])
            if e >= 0 and counts[e] < receiver_capacity:
                reference[i] = True
                counts[e] += 1
    np.testing.assert_array_equal(keep, reference)

    # Per-expert kept count is exactly min(total, capacity) -- the transpose does not change drop counts.
    for e in range(local_experts):
        total = int((received == e).sum())
        assert int(((received == e) & keep).sum()) == min(total, receiver_capacity)

    # Within a source, kept slots for an expert are the lowest pool positions: causality is preserved.
    for shard in range(expert_shards):
        block = slice(shard * pool_capacity, (shard + 1) * pool_capacity)
        for e in range(local_experts):
            positions = np.where(received[block] == e)[0]
            kept_positions = positions[keep[block][positions]]
            assert list(kept_positions) == list(positions[: len(kept_positions)])


def test_interleaved_receiver_ranks_spread_overflow_evenly_across_sources():
    """When every slot carries one oversubscribed expert, round-robin allocation keeps within one token
    of `capacity / expert_shards` from each source, instead of a source-major prefix that starves the
    last sources (the bias this replaces)."""
    expert_shards, pool_capacity, local_experts, receiver_capacity = 4, 3, 1, 6
    received_experts = jnp.zeros(expert_shards * pool_capacity, dtype=jnp.int32)  # all carry expert 0

    ranks = _interleaved_receiver_ranks(
        received_experts,
        local_experts=local_experts,
        expert_shards=expert_shards,
        pool_capacity=pool_capacity,
    )
    keep = np.asarray(ranks < receiver_capacity).reshape(expert_shards, pool_capacity)
    kept_per_source = keep.sum(axis=1)

    assert int(keep.sum()) == receiver_capacity
    assert kept_per_source.max() - kept_per_source.min() <= 1  # even, not a source prefix

    # The source-major baseline instead keeps a prefix of whole sources (the starvation this fixes).
    plain = np.asarray(_receiver_ranks(received_experts, local_experts=local_experts) < receiver_capacity).reshape(
        expert_shards, pool_capacity
    )
    assert plain.sum(axis=1).max() - plain.sum(axis=1).min() == pool_capacity


def test_moe_mlp_runs_without_ep_axis():
    mesh = _make_dense_mesh()
    tokens = max(8, len(jax.devices()) * 8)
    hidden_dim = 32
    intermediate_dim = 64
    num_experts = 4
    topk = 2

    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
            key=jax.random.key(0),
            tokens=tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            topk=topk,
        )

        out = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            mesh=None,
        )
        assert out.shape == (tokens, hidden_dim)
        assert jnp.isfinite(out).all()
        assert getattr(out.sharding, "spec", None) == P("data")

        jit_fn = jax.jit(
            lambda x, sel, cw, up_gate, down: moe_mlp(
                x, sel, cw, up_gate, down, activation=ActivationFunctionEnum.silu, mesh=None
            )
        )
        out_jit = jit_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
        np.testing.assert_allclose(np.asarray(out), np.asarray(out_jit), rtol=1e-5, atol=1e-5)


def test_moe_mlp_default_matches_explicit_ring_without_ep_axis():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(8),
        tokens=16,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=8,
        topk=2,
    )

    y_default = moe_mlp(x, selected_experts, combine_weights, w_up_gate, w_down, mesh=None)
    y_ring = moe_mlp(x, selected_experts, combine_weights, w_up_gate, w_down, implementation="ring", mesh=None)
    np.testing.assert_allclose(np.asarray(y_default), np.asarray(y_ring), rtol=1e-5, atol=1e-5)


def test_moe_mlp_padding_matches_compact_value_and_gradients():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(52),
        tokens=6,
        hidden_dim=8,
        intermediate_dim=12,
        num_experts=4,
        topk=2,
    )
    token_valid = jnp.array([True, False, True, True, False, True])
    valid_indices = jnp.array([0, 2, 3, 5], dtype=jnp.int32)
    cotangent = jax.random.normal(jax.random.key(53), x.shape)

    def padded_output(x, w_up_gate, w_down):
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            token_valid=token_valid,
            implementation="scatter",
            report_capacity_overflow=True,
        )

    compact_x = x[valid_indices]
    compact_selected_experts = selected_experts[valid_indices]
    compact_combine_weights = combine_weights[valid_indices]

    def compact_output(x, w_up_gate, w_down):
        return moe_mlp(
            x,
            compact_selected_experts,
            compact_combine_weights,
            w_up_gate,
            w_down,
            implementation="scatter",
        )

    actual, overflow = padded_output(x, w_up_gate, w_down)
    expected_compact = compact_output(compact_x, w_up_gate, w_down)
    expected = jnp.zeros_like(actual).at[valid_indices].set(expected_compact)
    actual_gradients = jax.grad(
        lambda x, w_up_gate, w_down: jnp.sum(padded_output(x, w_up_gate, w_down)[0] * cotangent),
        argnums=(0, 1, 2),
    )(x, w_up_gate, w_down)
    expected_gradients = jax.grad(
        lambda x, w_up_gate, w_down: jnp.sum(compact_output(x, w_up_gate, w_down) * cotangent[valid_indices]),
        argnums=(0, 1, 2),
    )(compact_x, w_up_gate, w_down)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_gradients[0][valid_indices], expected_gradients[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual_gradients[0][~token_valid], jnp.zeros((2, x.shape[1])))
    for actual_gradient, expected_gradient in zip(actual_gradients[1:], expected_gradients[1:], strict=True):
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)
    assert int(overflow.dropped) == 0
    assert int(overflow.padding_skipped) == 4


def test_moe_mlp_all_padding_has_no_expert_output_or_gradients():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(54),
        tokens=4,
        hidden_dim=8,
        intermediate_dim=12,
        num_experts=4,
        topk=2,
    )
    token_valid = jnp.zeros((x.shape[0],), dtype=jnp.bool_)

    def loss(x, w_up_gate, w_down):
        out, _ = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            token_valid=token_valid,
            implementation="scatter",
            report_capacity_overflow=True,
        )
        return jnp.sum(out), out

    gradients, out = jax.grad(loss, argnums=(0, 1, 2), has_aux=True)(x, w_up_gate, w_down)
    _, overflow = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        token_valid=token_valid,
        implementation="scatter",
        report_capacity_overflow=True,
    )

    np.testing.assert_array_equal(out, jnp.zeros_like(out))
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))
    assert int(overflow.dropped) == 0
    assert int(overflow.padding_skipped) == x.shape[0] * selected_experts.shape[1]


@pytest.mark.parametrize(
    "count, factor, divisor, expected",
    [
        (130_967_264, 1.15, 64, 2_353_319),
        (33_554_256, 1.15, 64, 602_929),
        (16_777_217, 1.0, 1, 16_777_217),
        (7680, 4.05, 8, 3888),
        (0, 1.15, 64, 0),
    ],
)
def test_scaled_capacity_preserves_large_assignment_counts(count, factor, divisor, expected):
    def capacity(assignments):
        return _scaled_capacity(
            assignments,
            capacity_factor=factor,
            divisor=divisor,
            minimum=0,
            maximum=max(expected + 1, 1),
        )

    assert int(jax.jit(capacity)(jnp.int32(count))) == expected


def test_scaled_capacity_preserves_buffer_and_empty_demand_bounds():
    capacity = jax.jit(
        lambda count: _scaled_capacity(count, capacity_factor=1.15, divisor=64, minimum=6, maximum=2_000_000)
    )
    assert int(capacity(jnp.int32(0))) == 6
    assert int(capacity(jnp.int32(130_967_264))) == 2_000_000


def test_deepep_local_assignment_packing_uses_local_expert_ids():
    recv_x = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=jnp.float32,
    )
    recv_topk_idx = jnp.array(
        [
            [0, 1],
            [1, -1],
            [0, 0],
        ],
        dtype=jnp.int32,
    )
    recv_topk_weights = jnp.array(
        [
            [0.1, 0.2],
            [0.3, 0.0],
            [0.4, 0.5],
        ],
        dtype=jnp.float32,
    )

    local_assignments = _pack_deepep_local_assignments(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        local_experts=2,
        num_recv_tokens=jnp.array(2, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(local_assignments.local_group_sizes), np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(local_assignments.recv_token_indices[:3]),
        np.array([0, 0, 1], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(local_assignments.x_dispatch[:3]),
        np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(local_assignments.assignment_weights[:3]),
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(local_assignments.x_dispatch[3:]), 0, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(local_assignments.assignment_weights[3:]), 0, rtol=0, atol=0)


@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("drop_every", [0, 3], ids=["all-accepted", "some-dropped"])
def test_dispatch_gradient_sums_each_tokens_accepted_assignments(topk, dtype, drop_every):
    tokens, hidden = 7, 3
    indices = np.random.default_rng(1).permutation(tokens * topk).astype(np.int32)
    accepted = np.ones((tokens, topk), dtype=bool)
    if drop_every:
        accepted.reshape(-1)[::drop_every] = False
    positions = np.argsort(indices)
    x = jnp.arange(tokens * hidden, dtype=dtype).reshape(tokens, hidden)
    cotangent = np.arange(tokens * topk * hidden, dtype=np.float32).reshape(tokens * topk, hidden) / 256
    # The transport never writes a dropped slot's cotangent row, so it may hold anything.
    cotangent[positions[~accepted.reshape(-1)]] = np.nan
    cotangent = jnp.asarray(cotangent, dtype=dtype)

    @jax.jit
    def evaluate(x, cotangent):
        output, backward = jax.vjp(
            lambda value: _gather_dispatch_rows(value, jnp.asarray(indices), jnp.asarray(accepted, jnp.float32), topk),
            x,
        )
        return output, backward(cotangent)[0]

    actual_output, actual_gradient = evaluate(x, cotangent)
    expected_gradient = np.zeros((tokens, hidden), dtype=np.float32)
    kept = accepted.reshape(-1)[indices]  # per sorted row
    np.add.at(expected_gradient, indices[kept] // topk, np.asarray(cotangent, dtype=np.float32)[kept])
    # Only the accepted slots are specified; the GPU path never writes the others.
    np.testing.assert_array_equal(np.asarray(actual_output)[kept], np.asarray(x)[indices // topk][kept])
    np.testing.assert_array_equal(np.asarray(actual_gradient), np.asarray(expected_gradient, dtype=dtype))


def test_accepted_assignments_keep_each_groups_accepted_prefix():
    num_experts, tokens, topk = 5, 9, 2
    rng = np.random.default_rng(3)
    selected = rng.integers(0, num_experts, size=tokens * topk).astype(np.int32)
    selected[[4, 11]] = num_experts  # invalid assignments
    group_sizes = np.bincount(selected, minlength=num_experts + 1)[:num_experts].astype(np.int32)
    accepted_sizes = np.minimum(group_sizes, [0, 1, 2, 9, 3]).astype(np.int32)
    sorted_indices = np.argsort(selected, kind="stable").astype(np.int32)

    actual = jax.jit(_accepted_assignments)(
        jnp.asarray(selected), jnp.asarray(sorted_indices), jnp.asarray(group_sizes), jnp.asarray(accepted_sizes)
    )

    expected = np.zeros(tokens * topk, dtype=bool)
    for expert in range(num_experts):
        members = sorted_indices[selected[sorted_indices] == expert]
        expected[members[: accepted_sizes[expert]]] = True
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_combine_skips_dropped_rows_and_differentiates_accepted_zero_weights():
    tokens, topk, hidden = 6, 3, 4
    rng = np.random.default_rng(4)
    sorted_indices = rng.permutation(tokens * topk).astype(np.int32)
    weights = rng.random((tokens, topk)).astype(np.float32)
    accepted = np.ones((tokens, topk), dtype=bool)
    accepted[[0, 2, 5], [1, 0, 2]] = False
    # Accepted assignments with weight zero still have the weight gradient <dout, y>.
    weights[[1, 4], [2, 0]] = 0
    rows = rng.standard_normal((tokens * topk, hidden)).astype(np.float32)
    cotangent = rng.standard_normal((tokens, hidden)).astype(np.float32)
    # Assignment a's row sits at sorted position positions[a].
    positions = np.argsort(sorted_indices).reshape(tokens, topk)
    expected = np.einsum("tkh,tk->th", rows[positions], np.where(accepted, weights, 0))
    expected_weight_gradient = np.where(accepted, np.einsum("th,tkh->tk", cotangent, rows[positions]), 0)
    # A dropped slot's row is never written, so it may hold anything.
    rows[positions[~accepted]] = np.nan

    def combine(weights):
        # As in the ragged MoE, the `where` zeroes the dropped weights and discards their gradients.
        return _unpermute_from_global_expert(
            jnp.asarray(rows),
            jnp.asarray(sorted_indices),
            jnp.where(accepted, weights, 0),
            jnp.asarray(accepted),
            tokens_per_shard=tokens,
            topk=topk,
        )

    actual, pullback = jax.vjp(jax.jit(combine), jnp.asarray(weights))
    (actual_weight_gradient,) = pullback(jnp.asarray(cotangent))

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_weight_gradient), expected_weight_gradient, rtol=1e-5, atol=1e-6)


def test_portable_expert_mlp_backward_ignores_rows_past_the_active_count():
    capacity, hidden, inter, experts = 10, 4, 6, 2
    active_group_sizes = jnp.asarray([3, 4], dtype=jnp.int32)
    physical_group_sizes = jnp.asarray([3, 7], dtype=jnp.int32)
    k_x, k_w13, k_w2, k_ct = jax.random.split(jax.random.key(5), 4)
    x = jax.random.normal(k_x, (capacity, hidden))
    w13 = jax.random.normal(k_w13, (experts, hidden, 2 * inter))
    w2 = jax.random.normal(k_w2, (experts, inter, hidden))
    cotangent = jax.random.normal(k_ct, (capacity, hidden))
    # Rows 7.. are past the active count: their input rows and output cotangent rows are unspecified.
    x_unspecified = x.at[7:].set(jnp.nan)
    cotangent_unspecified = cotangent.at[7:].set(jnp.nan)

    expert_mlp = _RaggedDotExpertMlp(jax.nn.silu, routing_weight_gradient=_RoutingWeightGradient.EXACT)

    def run(x, cotangent):
        out, residuals = expert_mlp.forward(x, w13, w2, physical_group_sizes, active_group_sizes)
        return out, expert_mlp.backward(residuals, cotangent)

    clean_out, (clean_dx, clean_dw13, clean_dw2, clean_output_dot) = run(x.at[7:].set(0), cotangent.at[7:].set(0))
    out, (dx, dw13, dw2, output_dot) = run(x_unspecified, cotangent_unspecified)

    np.testing.assert_array_equal(np.asarray(out[:7]), np.asarray(clean_out[:7]))
    np.testing.assert_array_equal(np.asarray(dx[:7]), np.asarray(clean_dx[:7]))
    np.testing.assert_array_equal(np.asarray(dw13), np.asarray(clean_dw13))
    np.testing.assert_array_equal(np.asarray(dw2), np.asarray(clean_dw2))
    np.testing.assert_array_equal(np.asarray(output_dot[:7]), np.asarray(clean_output_dot[:7]))
    # The row dot is the gradient of a per-row output scale, <y, dy>.
    np.testing.assert_allclose(
        np.asarray(clean_output_dot[:7]),
        np.sum(np.asarray(clean_out[:7]) * np.asarray(cotangent[:7]), axis=-1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_prepare_moe_dispatch_indices_match_materialized_dispatch():
    x, selected_experts, combine_weights, _w_up_gate, _w_down = _make_inputs(
        key=jax.random.key(28),
        tokens=20,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=5,
        topk=2,
    )

    token_valid = jnp.ones((x.shape[0],), dtype=jnp.bool_)
    x_sort, w_sort, token_ids_sort, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        token_valid,
        num_experts=5,
    )
    token_ids_from_indices, dispatch_positions, index_group_sizes, sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            token_valid,
            num_experts=5,
        )
    )

    np.testing.assert_array_equal(np.asarray(token_ids_from_indices), np.asarray(token_ids_sort))
    np.testing.assert_array_equal(np.asarray(index_group_sizes), np.asarray(group_sizes))
    np.testing.assert_allclose(np.asarray(x[token_ids_from_indices]), np.asarray(x_sort), rtol=0, atol=0)

    dispatch_weights = combine_weights.reshape(-1)
    np.testing.assert_allclose(
        np.asarray(dispatch_weights[sorted_assignment_ids].astype(x.dtype)),
        np.asarray(w_sort),
        rtol=0,
        atol=0,
    )

    expected_sorted_positions = np.arange(selected_experts.size, dtype=np.int32)
    flat_dispatch_positions = np.asarray(dispatch_positions).reshape(-1)
    np.testing.assert_array_equal(
        flat_dispatch_positions[np.asarray(sorted_assignment_ids)], expected_sorted_positions
    )


def _arange_w13(dtype, *, experts: int = 2, hidden: int = 3, moe_dim: int = 4) -> jax.Array:
    values = jnp.arange(experts * hidden * 2 * moe_dim, dtype=jnp.float32)
    return values.reshape(experts, hidden, 2 * moe_dim).astype(dtype)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_interleave_places_gate_and_up_in_alternating_columns(dtype):
    moe_dim = 4
    w13 = _arange_w13(dtype, moe_dim=moe_dim)

    interleaved = _interleave_gate_up(w13, moe_dim)

    assert interleaved.shape == w13.shape
    assert interleaved.dtype == w13.dtype
    np.testing.assert_array_equal(np.asarray(interleaved[..., 0::2]), np.asarray(w13[..., :moe_dim]))
    np.testing.assert_array_equal(np.asarray(interleaved[..., 1::2]), np.asarray(w13[..., moe_dim:]))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_deinterleave_matches_the_interleave_transpose(dtype):
    moe_dim = 4
    w13 = _arange_w13(dtype, moe_dim=moe_dim)
    cotangent = (jnp.arange(w13.size, dtype=jnp.float32).reshape(w13.shape) / 7).astype(dtype)

    _, transpose = jax.vjp(lambda w: _interleave_gate_up(w, moe_dim), w13)
    (expected,) = transpose(cotangent)

    np.testing.assert_array_equal(np.asarray(_deinterleave_gate_up(cotangent)), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(_deinterleave_gate_up(_interleave_gate_up(w13, moe_dim))), np.asarray(w13)
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_the_interleave_transpose_de_interleaves_the_cotangent(dtype):
    # `bitcast_convert_type` has no AD rule, so the pack carries a hand-written VJP. Its
    # correctness is what keeps `dw13` pointing at the right half of the fused weight.
    gate = _arange_w13(dtype, moe_dim=4)[..., :4]
    up = -gate
    # The cotangent carries the interleaved layout, one value per output element.
    cotangent = _arange_w13(dtype, moe_dim=4)

    _, vjp = jax.vjp(_interleave_halves, gate, up)
    gate_ct, up_ct = vjp(cotangent)

    np.testing.assert_array_equal(np.asarray(gate_ct), np.asarray(cotangent[..., 0::2]))
    np.testing.assert_array_equal(np.asarray(up_ct), np.asarray(cotangent[..., 1::2]))


@pytest.mark.parametrize(
    ("dtype", "max_abs_error", "mean_abs_error"),
    [(jnp.bfloat16, 8e-3, 5e-4), (jnp.float32, 2e-7, 2e-8)],
)
def test_swiglu_backward_matches_autodiff_of_the_forward(dtype, max_abs_error, mean_abs_error):
    tokens, moe_dim = 5, 4
    gu = jnp.linspace(-2.0, 2.0, tokens * 2 * moe_dim, dtype=jnp.float32).reshape(tokens, 2 * moe_dim).astype(dtype)
    dh = jnp.linspace(1.0, -1.0, tokens * moe_dim, dtype=jnp.float32).reshape(tokens, moe_dim).astype(dtype)

    def swiglu(x):
        gate, up = x[:, 0::2], x[:, 1::2]
        return jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)

    expected = jax.vjp(swiglu, gu)[1](dh.astype(jnp.float32))[0]

    actual = _swiglu_gate_up_backward(gu, dh)

    assert actual.dtype == gu.dtype
    error = np.abs(np.asarray(actual, dtype=np.float32) - np.asarray(expected, dtype=np.float32))
    assert np.max(error) <= max_abs_error
    assert np.mean(error) <= mean_abs_error


def test_moe_expert_mlp_init_matches_across_backends():
    k_mlp = jax.random.key(26)
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4

    scatter_mlp = MoEExpertMlp.init(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        initializer_std=0.02,
        key=k_mlp,
        implementation="scatter",
    )
    sonic_mlp = MoEExpertMlp.init(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        initializer_std=0.02,
        key=k_mlp,
        implementation="sonic",
    )

    np.testing.assert_allclose(
        np.asarray(sonic_mlp.w_gate),
        np.asarray(scatter_mlp.w_gate),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sonic_mlp.w_up),
        np.asarray(scatter_mlp.w_up),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(sonic_mlp.w_down), np.asarray(scatter_mlp.w_down), rtol=1e-5, atol=1e-5)


def test_moe_mlp_sonic_backend_reports_missing_optional_dependencies():
    optional_modules = ("jax_triton", "triton")
    if all(importlib.util.find_spec(module) is not None for module in optional_modules):
        pytest.skip("raw Sonic optional dependencies are installed in this environment")

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(20),
        tokens=8,
        hidden_dim=8,
        intermediate_dim=12,
        num_experts=4,
        topk=2,
    )

    with pytest.raises(ImportError, match="implementation='sonic' requires jax-triton and triton"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=None,
            implementation="sonic",
        )


def test_sonic_gather_sum_matches_jax_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    tokens = 32
    topk = 2
    hidden_dim = 64
    num_experts = 8
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    combine_weights = jax.nn.softmax(
        jax.random.normal(jax.random.key(29), (tokens, topk), dtype=jnp.float32),
        axis=-1,
    )
    dispatch_output = jax.random.normal(jax.random.key(30), (tokens * topk, hidden_dim), dtype=jnp.float32)
    _token_ids, dispatch_positions, _group_sizes, _assignment_ids = _prepare_moe_dispatch_indices_with_assignment_ids(
        selected_experts,
        jnp.ones((tokens,), dtype=jnp.bool_),
        num_experts=num_experts,
    )

    @jax.jit
    def gather_sum(dispatch_output, dispatch_positions, combine_weights):
        return (
            sonic_gather_sum(dispatch_output, dispatch_positions, combine_weights),
            _gather_sum_reference(dispatch_output, dispatch_positions, combine_weights),
        )

    sonic_out, reference_out = gather_sum(dispatch_output, dispatch_positions, combine_weights)
    sonic_out.block_until_ready()
    reference_out.block_until_ready()
    np.testing.assert_allclose(np.asarray(sonic_out), np.asarray(reference_out), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("weighted", [False, True], ids=["copy", "weighted"])
def test_sonic_scatter_rows_matches_gather_on_kept_rows_on_gpu(weighted):
    _skip_without_sonic_gpu_runtime()
    tokens, topk, hidden = 64, 8, 320
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.standard_normal((tokens, hidden), dtype=np.float32), jnp.bfloat16)
    sorted_indices = rng.permutation(tokens * topk).astype(np.int32)
    positions = np.argsort(sorted_indices).astype(np.int32).reshape(tokens, topk)
    keep = rng.random((tokens, topk)) < 0.8
    weights = np.where(rng.random((tokens, topk)) < 0.1, 0.0, rng.random((tokens, topk))).astype(np.float32)

    actual = jax.jit(
        lambda x, positions, keep, weights: sonic_scatter_rows(
            x, positions, keep, rows=tokens * topk, weights=weights if weighted else None
        )
    )(x, jnp.asarray(positions), jnp.asarray(keep), jnp.asarray(weights))

    sorted_rows = np.asarray(x, np.float32)[sorted_indices // topk]
    if weighted:
        sorted_rows = sorted_rows * weights.reshape(-1)[sorted_indices][:, None]
    expected = np.asarray(jnp.asarray(sorted_rows, jnp.bfloat16))
    kept = keep.reshape(-1)[sorted_indices]
    np.testing.assert_array_equal(np.asarray(actual)[kept].view(np.uint16), expected[kept].view(np.uint16))


# 0xFFFFFFFF is the negative NaN whose total-order key is the smallest int32.
_TOP_K_SPECIAL_VALUES = np.array(
    [0x00000000, 0x80000000, 0x7F800000, 0xFF800000]  # signed zeros and infinities
    + [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF],  # NaNs of either sign
    np.uint32,
).view(np.float32)


def _top_k_adversarial_rows(rows: int, width: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, width)).astype(np.float32)
    values[0::5] = rng.integers(0, 3, size=values[0::5].shape)  # heavy ties
    values[1::5] = np.where(rng.random(values[1::5].shape) < 0.5, -0.0, 0.0)  # signed zeros
    values[2::5] = np.where(
        rng.random(values[2::5].shape) < 0.3, rng.choice(_TOP_K_SPECIAL_VALUES, values[2::5].shape), values[2::5]
    )
    values[3::5] = rng.choice(_TOP_K_SPECIAL_VALUES, values[3::5].shape)
    values[4] = _TOP_K_SPECIAL_VALUES[-1]  # every key ties with the smallest int32
    return jnp.asarray(values)


# Widths cover one tile, exact or padded, and three tiles, exact (384) or padded (37). k = 37 runs the
# steps in a loop rather than unrolled.
_TOP_K_CASES = [
    (61, 1, 1),
    (61, 7, 7),
    (97, 37, 9),
    (61, 37, 37),
    (61, 100, 9),
    (61, 128, 9),
    (61, 256, 9),
    (97, 384, 9),
    (64, 384, 8),
    (61, 1000, 9),
    (61, 1023, 9),
]


def test_top_k_indices_run_inside_a_checking_shard_map_on_gpu():
    # The hero calls the kernel per token shard, inside a shard_map that checks varying axes.
    _skip_without_sonic_gpu_runtime()
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("data",), axis_types=(AxisType.Explicit,))
    values = _top_k_adversarial_rows(16 * len(devices), 384, seed=11)

    with jax.set_mesh(mesh):
        actual = jax.jit(
            jax.shard_map(
                lambda local: top_k_indices(local, 9),
                mesh=mesh,
                in_specs=P("data", None),
                out_specs=P("data", None),
            )
        )(jax.sharding.reshard(values, P("data", None)))

    expected = jax.jit(lambda v: jax.lax.top_k(v, 9)[1])(values)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(("rows", "width", "k"), _TOP_K_CASES)
def test_top_k_indices_match_lax_top_k_on_gpu(rows, width, k):
    _skip_without_sonic_gpu_runtime()
    values = _top_k_adversarial_rows(rows, width, seed=width)

    actual = jax.jit(lambda v: top_k_indices(v, k))(values)

    expected = jax.jit(lambda v: jax.lax.top_k(v, k)[1])(values)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_moe_mlp_sonic_matches_jax_gather_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    tokens = 512
    hidden_dim = 128
    intermediate_dim = 256
    num_experts = 8
    topk = 2
    k_x, k_logits, k_w13, k_w2 = jax.random.split(jax.random.key(31), 4)
    dtype = jnp.bfloat16
    x = jax.random.normal(k_x, (tokens, hidden_dim), dtype=dtype)
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    combine_weights = jax.nn.softmax(
        jax.random.normal(k_logits, (tokens, topk), dtype=jnp.float32),
        axis=-1,
    )
    w_up_gate = jax.random.normal(k_w13, (num_experts, hidden_dim, 2 * intermediate_dim), dtype=dtype)
    w_down = jax.random.normal(k_w2, (num_experts, intermediate_dim, hidden_dim), dtype=dtype)

    @jax.jit
    def run_moe_with_reference(x, selected_experts, combine_weights, w_up_gate, w_down):
        token_ids, dispatch_positions, group_sizes, _assignment_ids = (
            _prepare_moe_dispatch_indices_with_assignment_ids(
                selected_experts,
                jnp.ones((tokens,), dtype=jnp.bool_),
                num_experts=num_experts,
            )
        )
        x_dispatch = x[token_ids]
        w13_dispatch = ragged_dot(x_dispatch, w_up_gate, group_sizes)
        gate_dispatch, up_dispatch = grug_moe.split_moe_w13_output(
            w13_dispatch,
            intermediate_dim=intermediate_dim,
            interleaved=False,
        )
        dispatch_out = ragged_dot(jax.nn.silu(gate_dispatch) * up_dispatch, w_down, group_sizes)
        reference_out = _gather_sum_reference(dispatch_out, dispatch_positions, combine_weights)
        sonic_out = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="sonic",
            mesh=None,
        )
        return sonic_out, reference_out

    sonic_out, reference_out = run_moe_with_reference(x, selected_experts, combine_weights, w_up_gate, w_down)
    sonic_out.block_until_ready()
    reference_out.block_until_ready()
    max_abs = jnp.max(jnp.abs(sonic_out.astype(jnp.float32) - reference_out.astype(jnp.float32)))
    assert float(max_abs) <= 64.0


def test_moe_expert_mlp_init_uses_logical_weight_pspecs():
    mesh = _make_dense_mesh()
    pspecs = MoEExpertMlpPspecs(expert=None, hidden="data", intermediate="model")

    with jax.set_mesh(mesh):
        mlp = MoEExpertMlp.init(
            num_experts=4,
            hidden_dim=16,
            intermediate_dim=24,
            initializer_std=0.02,
            key=jax.random.key(27),
            implementation="sonic",
            pspecs=pspecs,
        )

    assert mlp.w_gate.sharding.spec == P(None, "data", "model")
    assert mlp.w_up.sharding.spec == P(None, "data", "model")
    assert mlp.w_down.sharding.spec == P(None, "model", "data")


@pytest.mark.parametrize(
    "implementation",
    ["ring", "ring_gather_combine", "ragged_all_to_all", "fixed_all_to_all", "fixed_pooled_wave_all_to_all"],
)
def test_moe_ep_path_lowers_on_abstract_mesh(implementation: MoeImplementation):
    mesh = _make_abstract_moe_mesh(data=2, expert=2, model=1)

    tokens = 16
    hidden_dim = 32
    intermediate_dim = 64
    num_experts = 4
    topk = 2

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        x = jax.ShapeDtypeStruct(
            shape=(tokens, hidden_dim),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )
        selected_experts = jax.ShapeDtypeStruct(
            shape=(tokens, topk),
            dtype=jnp.int32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )
        combine_weights = jax.ShapeDtypeStruct(
            shape=(tokens, topk),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )
        token_valid = jax.ShapeDtypeStruct(
            shape=(tokens,),
            dtype=jnp.bool_,
            sharding=NamedSharding(mesh, P(("data", "expert"))),
        )
        w_up_gate = jax.ShapeDtypeStruct(
            shape=(num_experts, hidden_dim, 2 * intermediate_dim),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P("expert", None, None)),
        )
        w_down = jax.ShapeDtypeStruct(
            shape=(num_experts, intermediate_dim, hidden_dim),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P("expert", None, None)),
        )

        def f(x, sel, cw, valid, up_gate, down):
            return moe_mlp(
                x,
                sel,
                cw,
                up_gate,
                down,
                token_valid=valid,
                activation=ActivationFunctionEnum.silu,
                implementation=implementation,
                mesh=mesh,
                pooled_transport_capacity_factor=(1.05 if implementation == "fixed_pooled_wave_all_to_all" else None),
                num_expert_waves=1,
            )

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = (
            jax.jit(f)
            .trace(x, selected_experts, combine_weights, token_valid, w_up_gate, w_down)
            .lower(lowering_platforms=(platform,))
        )
        assert lowered is not None


def test_fixed_all_to_all_drops_assignments_over_capacity():
    mesh = Mesh(
        np.asarray([jax.devices()[0]]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )
    tokens = 4
    hidden_dim = 4
    intermediate_dim = 6
    num_experts = 2
    topk = 2
    x, _, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(41),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = jnp.tile(jnp.arange(topk, dtype=jnp.int32), (tokens, 1))

    def fixed_a2a(x, selected_experts, combine_weights, w_up_gate, w_down):
        return _moe_mlp_ep_fixed_a2a_local(
            x,
            selected_experts,
            combine_weights,
            jnp.ones((tokens,), dtype=jnp.bool_),
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=0.5,
            token_sharding_axes=("expert",),
        )

    sharded_fixed_a2a = jax.shard_map(
        fixed_a2a,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P()),
        out_specs=(P(), CapacityDrops(sender_dropped=P(), receiver_dropped=P())),
        check_vma=False,
    )
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        actual, overflow = jax.jit(sharded_fixed_a2a)(x, selected_experts, combine_weights, w_up_gate, w_down)

    keep = jnp.asarray([[True, True], [True, True], [False, False], [False, False]])

    def dense_output(x, w_up_gate, w_down):
        return _dense_moe_output(x, selected_experts, combine_weights * keep, w_up_gate, w_down)

    cotangent = jax.random.normal(jax.random.key(42), x.shape)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        actual_gradients = jax.jit(
            jax.grad(
                lambda x, w_up_gate, w_down: jnp.sum(
                    sharded_fixed_a2a(x, selected_experts, combine_weights, w_up_gate, w_down)[0] * cotangent
                ),
                argnums=(0, 1, 2),
            )
        )(x, w_up_gate, w_down)

        expected = jax.jit(dense_output)(x, w_up_gate, w_down)
        expected_gradients = jax.jit(
            jax.grad(
                lambda x, w_up_gate, w_down: jnp.sum(dense_output(x, w_up_gate, w_down) * cotangent),
                argnums=(0, 1, 2),
            )
        )(x, w_up_gate, w_down)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_gradient),
            np.asarray(expected_gradient),
            rtol=1e-5,
            atol=1e-5,
        )
    assert int(overflow.sender_dropped) == 4
    assert int(overflow.receiver_dropped) == 0


def test_fixed_all_to_all_padding_does_not_change_capacity_acceptance():
    mesh = _make_single_expert_mesh()
    x, _, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(55),
        tokens=6,
        hidden_dim=4,
        intermediate_dim=6,
        num_experts=2,
        topk=2,
    )
    selected_experts = jnp.tile(jnp.arange(2, dtype=jnp.int32), (x.shape[0], 1))
    token_valid = jnp.array([True, False, True, True, False, True])
    valid_indices = jnp.array([0, 2, 3, 5], dtype=jnp.int32)
    cotangent = jax.random.normal(jax.random.key(56), x.shape)

    def fixed_a2a(x, selected_experts, combine_weights, token_valid, w_up_gate, w_down):
        return _moe_mlp_ep_fixed_a2a_local(
            x,
            selected_experts,
            combine_weights,
            token_valid,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=2,
            capacity_factor=0.5,
            token_sharding_axes=("expert",),
        )

    sharded_fixed_a2a = jax.shard_map(
        fixed_a2a,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P()),
        out_specs=(P(), CapacityDrops(sender_dropped=P(), receiver_dropped=P())),
        check_vma=False,
    )

    def padded_output(x, w_up_gate, w_down):
        return sharded_fixed_a2a(
            x,
            selected_experts,
            combine_weights,
            token_valid,
            w_up_gate,
            w_down,
        )

    compact_x = x[valid_indices]
    compact_selected_experts = selected_experts[valid_indices]
    compact_combine_weights = combine_weights[valid_indices]
    compact_valid = jnp.ones((valid_indices.shape[0],), dtype=jnp.bool_)

    def compact_output(x, w_up_gate, w_down):
        return sharded_fixed_a2a(
            x,
            compact_selected_experts,
            compact_combine_weights,
            compact_valid,
            w_up_gate,
            w_down,
        )

    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        actual, padded_overflow = jax.jit(padded_output)(x, w_up_gate, w_down)
        expected_compact, compact_overflow = jax.jit(compact_output)(compact_x, w_up_gate, w_down)
        actual_gradients = jax.jit(
            jax.grad(
                lambda x, w_up_gate, w_down: jnp.sum(padded_output(x, w_up_gate, w_down)[0] * cotangent),
                argnums=(0, 1, 2),
            )
        )(x, w_up_gate, w_down)
        expected_gradients = jax.jit(
            jax.grad(
                lambda x, w_up_gate, w_down: jnp.sum(
                    compact_output(x, w_up_gate, w_down)[0] * cotangent[valid_indices]
                ),
                argnums=(0, 1, 2),
            )
        )(compact_x, w_up_gate, w_down)

    np.testing.assert_allclose(actual[valid_indices], expected_compact, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual[~token_valid], jnp.zeros((2, x.shape[1])))
    np.testing.assert_allclose(actual_gradients[0][valid_indices], expected_gradients[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual_gradients[0][~token_valid], jnp.zeros((2, x.shape[1])))
    for actual_gradient, expected_gradient in zip(actual_gradients[1:], expected_gradients[1:], strict=True):
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)
    assert padded_overflow.dropped == compact_overflow.dropped == 4


@pytest.mark.timeout(180)
def test_fixed_pooled_wave_all_to_all_matches_dense_value_and_gradients():
    mesh = _make_single_expert_mesh()
    tokens = 6
    hidden_dim = 4
    intermediate_dim = 3
    num_experts = 6
    topk = 2
    num_expert_waves = 3
    x, _, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(49),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = jnp.asarray(
        [[0, 1], [2, 3], [4, 5], [0, 2], [1, 3], [4, 5]],
        dtype=jnp.int32,
    )
    cotangent = jax.random.normal(jax.random.key(50), x.shape)

    def pooled_output(x, combine_weights, w_up_gate, w_down):
        return _moe_mlp_ep_fixed_pooled_wave_a2a_local(
            x,
            selected_experts,
            combine_weights,
            jnp.ones((tokens,), dtype=jnp.bool_),
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=4.0,
            token_sharding_axes=("expert",),
            transport_capacity_factor=4.0,
            num_expert_waves=num_expert_waves,
        )[0]

    sharded_pooled_output = jax.shard_map(
        pooled_output,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    rematerialized_pooled_output = jax.checkpoint(sharded_pooled_output)

    def dense_output(x, combine_weights, w_up_gate, w_down):
        return _dense_moe_output(x, selected_experts, combine_weights, w_up_gate, w_down)

    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        actual = jax.jit(sharded_pooled_output)(x, combine_weights, w_up_gate, w_down)
        actual_gradient_fn = jax.grad(
            lambda x, combine_weights, w_up_gate, w_down: jnp.sum(
                sharded_pooled_output(x, combine_weights, w_up_gate, w_down) * cotangent
            ),
            argnums=(0, 1, 2, 3),
        )
        actual_gradients = jax.jit(actual_gradient_fn)(x, combine_weights, w_up_gate, w_down)
        rematerialized_gradient_fn = jax.grad(
            lambda x, combine_weights, w_up_gate, w_down: jnp.sum(
                rematerialized_pooled_output(x, combine_weights, w_up_gate, w_down) * cotangent
            ),
            argnums=(0, 1, 2, 3),
        )
        gradient_jaxpr = jax.make_jaxpr(rematerialized_gradient_fn)(x, combine_weights, w_up_gate, w_down)
        expected = jax.jit(dense_output)(x, combine_weights, w_up_gate, w_down)
        expected_gradients = jax.jit(
            jax.grad(
                lambda x, combine_weights, w_up_gate, w_down: jnp.sum(
                    dense_output(x, combine_weights, w_up_gate, w_down) * cotangent
                ),
                argnums=(0, 1, 2, 3),
            )
        )(x, combine_weights, w_up_gate, w_down)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_gradient),
            np.asarray(expected_gradient),
            rtol=1e-5,
            atol=1e-5,
        )
    assert _count_jaxpr_primitives(gradient_jaxpr, "all_to_all") == 6 * num_expert_waves


def test_fixed_pooled_wave_all_to_all_reports_sender_and_receiver_drops():
    mesh = _make_single_expert_mesh()
    tokens = 6
    hidden_dim = 4
    intermediate_dim = 3
    num_experts = 6
    topk = 2
    x, _, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(51),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = jnp.tile(jnp.arange(topk, dtype=jnp.int32), (tokens, 1))

    def pooled_output(x, combine_weights, w_up_gate, w_down):
        return _moe_mlp_ep_fixed_pooled_wave_a2a_local(
            x,
            selected_experts,
            combine_weights,
            jnp.ones((tokens,), dtype=jnp.bool_),
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=1.33,
            token_sharding_axes=("expert",),
            transport_capacity_factor=0.75,
            num_expert_waves=3,
        )

    sharded_pooled_output = jax.shard_map(
        pooled_output,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=(P(), CapacityDrops(sender_dropped=P(), receiver_dropped=P())),
        check_vma=False,
    )
    with jax.set_mesh(mesh):
        actual, overflow = jax.jit(sharded_pooled_output)(x, combine_weights, w_up_gate, w_down)

    keep = jnp.arange(tokens)[:, None] < 3
    expected = _dense_moe_output(x, selected_experts, combine_weights * keep, w_up_gate, w_down)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(overflow.sender_dropped) == 3
    assert int(overflow.receiver_dropped) == 3


@pytest.mark.parametrize(
    "implementation", ["ring", "ring_gather_combine", "fixed_all_to_all", "fixed_pooled_wave_all_to_all"]
)
@pytest.mark.parametrize(
    "token_valid",
    [[True, True, True, True], [True, False, True, True]],
    ids=["all_valid", "padded"],
)
def test_portable_ep_backends_match_dense_cross_shard_value_and_gradients(
    implementation: MoeImplementation,
    token_valid: list[bool],
):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    script = """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.grug.grug_moe import moe_mlp

        assert jax.device_count() == 4
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(2, 2, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        x = jax.random.normal(jax.random.key(0), (4, 4))
        selected_experts = jnp.asarray(
            [[2, 3], [4, 5], [6, 7], [0, 1]],
            dtype=jnp.int32,
        )
        combine_weights = jax.nn.softmax(jax.random.normal(jax.random.key(1), (4, 2)), axis=-1)
        token_valid = jnp.asarray(__TOKEN_VALID__)
        w_up_gate = jax.random.normal(jax.random.key(2), (8, 4, 6))
        w_down = jax.random.normal(jax.random.key(3), (8, 3, 4))
        cotangent = jax.random.normal(jax.random.key(4), (4, 4))

        def dense_output(x, w_up_gate, w_down):
            selected_w13 = w_up_gate[selected_experts]
            hidden = jnp.einsum("th,tkhi->tki", x, selected_w13)
            gate, up = jnp.split(hidden, [3], axis=-1)
            expert_output = jnp.einsum(
                "tki,tkih->tkh",
                jax.nn.silu(gate) * up,
                w_down[selected_experts],
            )
            return jnp.einsum("tkh,tk->th", expert_output, combine_weights * token_valid[:, None])

        expected = jax.jit(dense_output)(x, w_up_gate, w_down)
        expected_gradients = jax.jit(
            jax.grad(
                lambda x, w_up_gate, w_down: jnp.sum(dense_output(x, w_up_gate, w_down) * cotangent),
                argnums=(0, 1, 2),
            )
        )(x, w_up_gate, w_down)

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        token_sharding = NamedSharding(mesh, P(("data", "expert")))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.device_put(x, batch_sharding)
        selected_experts = jax.device_put(selected_experts, batch_sharding)
        combine_weights = jax.device_put(combine_weights, batch_sharding)
        token_valid = jax.device_put(token_valid, token_sharding)
        w_up_gate = jax.device_put(w_up_gate, expert_sharding)
        w_down = jax.device_put(w_down, expert_sharding)
        cotangent = jax.device_put(cotangent, batch_sharding)

        implementation = "__IMPLEMENTATION__"
        extra = {}
        if implementation == "fixed_pooled_wave_all_to_all":
            extra["pooled_transport_capacity_factor"] = 4.0

        def backend_output(x, w_up_gate, w_down):
            return moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                token_valid=token_valid,
                activation=jax.nn.silu,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=4.0,
                report_capacity_overflow=True,
                **extra,
            )

        with jax.set_mesh(mesh):
            actual, overflow = jax.jit(backend_output)(x, w_up_gate, w_down)
            actual_gradients = jax.jit(
                jax.grad(
                    lambda x, w_up_gate, w_down: jnp.sum(backend_output(x, w_up_gate, w_down)[0] * cotangent),
                    argnums=(0, 1, 2),
                )
            )(x, w_up_gate, w_down)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual_gradient),
                np.asarray(expected_gradient),
                rtol=1e-5,
                atol=1e-5,
            )
        assert int(overflow.dropped) == 0
        assert int(overflow.padding_skipped) == int(jnp.sum(~token_valid)) * 2
    """
    script = script.replace("__IMPLEMENTATION__", implementation).replace("__TOKEN_VALID__", repr(token_valid))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "token_valid",
    [[True] * 16, [True, False, True, True] * 4],
    ids=["all_valid", "padded"],
)
def test_ring_gather_combine_matches_scatter_combine_with_drops(token_valid: list[bool]):
    """`ring_gather_combine` gives `ring`'s values, drops and gradients without token-buffer scatter-adds."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    script = """
        import functools

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.grug.grug_moe import moe_mlp

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(2, 2, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        tokens, hidden, inter, experts, topk = 16, 10, 6, 8, 3
        keys = jax.random.split(jax.random.key(0), 6)
        x = jax.random.normal(keys[0], (tokens, hidden))
        # Skewed routing so some shards run out of capacity.
        logits = jax.random.normal(keys[1], (tokens, experts)) + jnp.linspace(2.0, 0.0, experts)
        selected_experts = jax.lax.top_k(logits, topk)[1].astype(jnp.int32)
        combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, topk)), axis=-1)
        token_valid = jnp.asarray(__TOKEN_VALID__)
        w_up_gate = jax.random.normal(keys[3], (experts, hidden, 2 * inter))
        w_down = jax.random.normal(keys[4], (experts, inter, hidden))
        cotangent = jax.random.normal(keys[5], (tokens, hidden))

        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert = NamedSharding(mesh, P("expert", None, None))
        x, selected_experts, combine_weights, cotangent = (
            jax.device_put(a, batch) for a in (x, selected_experts, combine_weights, cotangent)
        )
        token_valid = jax.device_put(token_valid, NamedSharding(mesh, P(("data", "expert"))))
        w_up_gate, w_down = jax.device_put(w_up_gate, expert), jax.device_put(w_down, expert)

        def loss(implementation, x, combine_weights, w_up_gate, w_down):
            out, counts = moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                token_valid=token_valid,
                activation=jax.nn.silu,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=0.5,
                report_capacity_overflow=True,
            )
            return jnp.sum(out * cotangent), counts.dropped

        def token_row_scatter_adds(jaxpr):
            # Scatter-adds that write [rows, hidden] token buffers, searched through nested jaxprs.
            count = 0
            for eqn in jaxpr.eqns:
                shape = eqn.outvars[0].aval.shape if eqn.outvars else ()
                count += eqn.primitive.name == "scatter-add" and len(shape) == 2 and shape[1] == hidden
                for param in eqn.params.values():
                    for sub in param if isinstance(param, (tuple, list)) else (param,):
                        sub = getattr(sub, "jaxpr", sub)
                        if hasattr(sub, "eqns"):
                            count += token_row_scatter_adds(sub)
            return count

        results = {}
        for implementation in ("ring", "ring_gather_combine"):
            with jax.set_mesh(mesh):
                grad_fn = jax.value_and_grad(
                    functools.partial(loss, implementation), argnums=(0, 1, 2, 3), has_aux=True
                )
                args = (x, combine_weights, w_up_gate, w_down)
                scatter_adds = token_row_scatter_adds(jax.make_jaxpr(grad_fn)(*args).jaxpr)
                results[implementation] = (grad_fn(*args), scatter_adds)

        ((value_s, dropped_s), grads_s), scatter_adds_s = results["ring"]
        ((value_g, dropped_g), grads_g), scatter_adds_g = results["ring_gather_combine"]
        assert scatter_adds_s > 0, scatter_adds_s
        assert scatter_adds_g == 0, scatter_adds_g
        assert int(dropped_s) > 0, int(dropped_s)
        assert int(dropped_g) == int(dropped_s)
        np.testing.assert_allclose(np.asarray(value_g), np.asarray(value_s), rtol=1e-5, atol=1e-5)
        for g, s in zip(grads_g, grads_s, strict=True):
            np.testing.assert_allclose(np.asarray(g), np.asarray(s), rtol=1e-5, atol=1e-5)
    """
    script = script.replace("__TOKEN_VALID__", repr(token_valid))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _simulate_ragged_a2a(operands, outputs, params):
    """Reference semantics of ``ragged_all_to_all``: slice i of sender s goes to shard i // spd.

    Checks the receiver's ``recv_sizes`` against what each sender actually writes. The real
    collective sizes incoming transfers from that vector, so building it from the wrong direction
    -- the easiest mistake in this arithmetic, since the two are transposes of one another -- moves
    the right bytes here but mis-sizes the receive on a real multi-shard run.
    """
    num_shards = len(operands)
    for sender in range(num_shards):
        in_off, send, out_off, _ = (np.asarray(a) for a in params[sender])
        slices_per_device = len(in_off) // num_shards
        for i in range(len(in_off)):
            dst = i // slices_per_device
            n = send[i]
            recv = np.asarray(params[dst].recv_sizes)[sender * slices_per_device + i % slices_per_device]
            assert recv == n, f"recv_sizes {recv} != send_sizes {n} for update {i} from {sender} to {dst}"
            outputs[dst][out_off[i] : out_off[i] + n] = operands[sender][in_off[i] : in_off[i] + n]


def test_expert_granular_a2a_params_roundtrip_with_drops():
    """Dispatch packs receivers expert-major with sender order inside each expert, and the
    return direction restores each accepted row to its unclipped sorted position, leaving
    dropped rows at the output operand's values -- all under forced capacity clipping. The
    expert MLP leaves unused receiver capacity unspecified, so it starts as NaN here and must
    never reach the return."""
    shards, local_experts, tokens, topk, hidden = 4, 3, 10, 2, 5
    num_experts = shards * local_experts
    assignments = tokens * topk
    capacity = int(0.7 * assignments)  # force drops

    rng = np.random.default_rng(0)
    selected = rng.integers(0, num_experts, size=(shards, tokens, topk))
    payload = rng.normal(size=(shards, assignments, hidden)).astype(np.float32)
    sorted_payload = np.stack([payload[s][np.argsort(selected[s].reshape(-1), kind="stable")] for s in range(shards)])
    group_sizes = np.stack(
        [np.bincount(selected[s].reshape(-1), minlength=num_experts) for s in range(shards)]
    ).astype(np.int32)
    starts = np.cumsum(group_sizes, axis=1) - group_sizes

    clipped = np.asarray(
        _clip_receiver_group_sizes(
            jnp.asarray(group_sizes), local_expert_size=local_experts, receiver_capacity=capacity
        )
    )
    assert clipped.sum() < group_sizes.sum()  # drops actually happen

    params = [
        _expert_granular_a2a_params(
            jnp.asarray(group_sizes),
            jnp.asarray(clipped),
            jnp.asarray(s),
            local_expert_size=local_experts,
        )
        for s in range(shards)
    ]

    received = [np.full((capacity, hidden), np.nan, np.float32) for _ in range(shards)]
    _simulate_ragged_a2a(sorted_payload, received, [p[0] for p in params])
    for receiver in range(shards):
        rows = [
            sorted_payload[s][starts[s, g] : starts[s, g] + clipped[s, g]]
            for e in range(local_experts)
            for g in [receiver * local_experts + e]
            for s in range(shards)
        ]
        expected = np.concatenate(rows, axis=0)
        np.testing.assert_array_equal(received[receiver][: len(expected)], expected)
        assert np.isnan(received[receiver][len(expected) :]).all()

    returned = [np.zeros((assignments, hidden), np.float32) for _ in range(shards)]
    _simulate_ragged_a2a(received, returned, [p[1] for p in params])
    for s in range(shards):
        expected = np.zeros_like(sorted_payload[s])
        for g in range(num_experts):
            expected[starts[s, g] : starts[s, g] + clipped[s, g]] = sorted_payload[s][
                starts[s, g] : starts[s, g] + clipped[s, g]
            ]
        np.testing.assert_array_equal(returned[s], expected)


def test_expert_granular_a2a_params_are_each_others_transpose():
    """The return parameters equal what JAX's ragged_all_to_all transpose rule derives from the
    dispatch parameters with its two offset all-to-alls, and vice versa, so the backend's backward
    can use them directly. Checked under forced capacity clipping."""
    shards, local_experts, tokens, topk = 4, 3, 10, 2
    num_experts = shards * local_experts
    rng = np.random.default_rng(1)
    selected = rng.integers(0, num_experts, size=(shards, tokens * topk))
    group_sizes = np.stack([np.bincount(selected[s], minlength=num_experts) for s in range(shards)]).astype(np.int32)
    clipped = _clip_receiver_group_sizes(
        jnp.asarray(group_sizes), local_expert_size=local_experts, receiver_capacity=int(0.7 * tokens * topk)
    )
    assert int(jnp.sum(clipped)) < group_sizes.sum()  # drops actually happen
    params = [
        _expert_granular_a2a_params(jnp.asarray(group_sizes), clipped, jnp.asarray(s), local_expert_size=local_experts)
        for s in range(shards)
    ]

    def exchanged(field, direction, shard):
        # A tiled all_to_all over the expert axis: chunk ``shard`` of every peer's vector.
        return np.concatenate(
            [
                np.asarray(getattr(params[peer][direction], field)).reshape(shards, local_experts)[shard]
                for peer in range(shards)
            ]
        )

    for direction, mirror in ((0, 1), (1, 0)):
        for s in range(shards):
            forward, transpose = params[s][direction], params[s][mirror]
            np.testing.assert_array_equal(exchanged("output_offsets", direction, s), transpose.input_offsets)
            np.testing.assert_array_equal(exchanged("input_offsets", direction, s), transpose.output_offsets)
            np.testing.assert_array_equal(forward.recv_sizes, transpose.send_sizes)
            np.testing.assert_array_equal(forward.send_sizes, transpose.recv_sizes)


def test_expert_granular_a2a_params_chunked_masking_composes():
    """Masking the clip to one expert chunk at a time (full sender starts, chained returns)
    reproduces the whole layer: each chunk's receiver packs only its experts from offset zero,
    and the chained returns cover exactly the per-chunk accepted prefixes, never a chunk's
    unused capacity."""
    shards, local_experts, tokens, topk, hidden = 4, 3, 10, 2, 5
    num_experts = shards * local_experts
    assignments = tokens * topk
    capacity = int(0.7 * assignments)
    chunks = 3
    chunk_capacity = -(-capacity // chunks)
    chunk_of_expert = (np.arange(num_experts) % local_experts) // (local_experts // chunks)

    rng = np.random.default_rng(0)
    selected = rng.integers(0, num_experts, size=(shards, tokens, topk))
    payload = rng.normal(size=(shards, assignments, hidden)).astype(np.float32)
    sorted_payload = np.stack([payload[s][np.argsort(selected[s].reshape(-1), kind="stable")] for s in range(shards)])
    group_sizes = np.stack(
        [np.bincount(selected[s].reshape(-1), minlength=num_experts) for s in range(shards)]
    ).astype(np.int32)
    starts = np.cumsum(group_sizes, axis=1) - group_sizes

    returned = [np.zeros((assignments, hidden), np.float32) for _ in range(shards)]
    accepted = np.zeros((shards, num_experts), np.int32)
    for chunk in range(chunks):
        masked = np.where(chunk_of_expert[None, :] == chunk, group_sizes, 0)
        clipped = np.asarray(
            _clip_receiver_group_sizes(
                jnp.asarray(masked), local_expert_size=local_experts, receiver_capacity=chunk_capacity
            )
        )
        accepted += clipped
        params = [
            _expert_granular_a2a_params(
                jnp.asarray(group_sizes),
                jnp.asarray(clipped),
                jnp.asarray(s),
                local_expert_size=local_experts,
            )
            for s in range(shards)
        ]
        received = [np.full((chunk_capacity, hidden), np.nan, np.float32) for _ in range(shards)]
        _simulate_ragged_a2a(sorted_payload, received, [p[0] for p in params])
        _simulate_ragged_a2a(received, returned, [p[1] for p in params])

    for s in range(shards):
        expected = np.zeros_like(sorted_payload[s])
        for g in range(num_experts):
            expected[starts[s, g] : starts[s, g] + accepted[s, g]] = sorted_payload[s][
                starts[s, g] : starts[s, g] + accepted[s, g]
            ]
        np.testing.assert_array_equal(returned[s], expected)


def _force_routing_weight_gradient(
    monkeypatch: pytest.MonkeyPatch, routing_weight_gradient: _RoutingWeightGradient
) -> None:
    """Make the ragged backend run the expert MLP it selects with ``routing_weight_gradient``.

    The backend takes the routing-weight gradient from the expert MLP it selects, and off SM100 that
    is always the portable one, whose own gradient is EXACT.
    """
    select_expert_mlp = ep_ragged_all_to_all._select_expert_mlp
    monkeypatch.setattr(
        ep_ragged_all_to_all,
        "_select_expert_mlp",
        lambda activation_fn, dtype: dataclasses.replace(
            select_expert_mlp(activation_fn, dtype), routing_weight_gradient=routing_weight_gradient
        ),
    )


@pytest.mark.parametrize(
    ("implementation", "routing_weight_gradient"),
    [
        ("ring", _RoutingWeightGradient.EXACT),
        ("ragged_all_to_all", _RoutingWeightGradient.EXACT),
        ("ragged_all_to_all", _RoutingWeightGradient.EXPERT_SIDE),
    ],
    ids=["ring", "ragged", "ragged_expert_side"],
)
@pytest.mark.parametrize("padded", [False, True], ids=["all_valid", "padded"])
def test_moe_mlp_ep_backends_match_dense_value_and_gradients_when_available(
    implementation: MoeImplementation,
    routing_weight_gradient: _RoutingWeightGradient,
    padded: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")

    platform = jax.devices()[0].platform
    if platform == "cpu":
        pytest.skip("ragged_all_to_all is not implemented on XLA:CPU")
    if platform == "tpu":
        monkeypatch.setenv("RAGGED_DOT_IMPL", "megablox")
    # The ring backend differentiates the combine weights exactly and ignores this.
    _force_routing_weight_gradient(monkeypatch, routing_weight_gradient)

    tokens = len(jax.devices()) * 8
    gpu_runtime = platform == "gpu"
    hidden_dim = 16 if gpu_runtime else 128
    # Keep the TPU GMM rectangular so its VJP must swap the K and N dimensions.
    intermediate_dim = 24 if gpu_runtime else 16
    num_experts = 4
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(23),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    dtype = jnp.bfloat16 if platform in {"gpu", "tpu"} else jnp.float32
    relative_tolerance = _BF16_MOE_RELATIVE_TOLERANCE if dtype == jnp.bfloat16 else _FP32_MOE_RELATIVE_TOLERANCE
    x = x.astype(dtype)
    # Two accepted assignments whose weighted output cotangent w * dout rounds to zero: token 0's
    # first has weight zero, and token 2's first has a normal weight while the product underflows,
    # because token 2's output cotangent is tiny. Their weight gradient is still <dout, y>.
    edge_assignments = (np.array([0, 2]), np.array([0, 0]))
    combine_weights = combine_weights.astype(dtype).at[0, 0].set(0).at[2, 0].set(2.0**-10)
    token_valid = (jnp.arange(tokens) % 4 != 1) if padded else jnp.ones((tokens,), dtype=jnp.bool_)
    w_up_gate = w_up_gate.astype(dtype)
    w_down = w_down.astype(dtype)
    cotangent = jax.random.normal(jax.random.key(24), x.shape, dtype=dtype).at[2].multiply(2.0**-120)

    x_reference = x.astype(jnp.float32)
    combine_weights_reference = combine_weights.astype(jnp.float32) * token_valid[:, None]
    w_up_gate_reference = w_up_gate.astype(jnp.float32)
    w_down_reference = w_down.astype(jnp.float32)
    cotangent_reference = cotangent.astype(jnp.float32)
    expected = _dense_moe_output(
        x_reference,
        selected_experts,
        combine_weights_reference,
        w_up_gate_reference,
        w_down_reference,
    )
    expected_gradients = jax.jit(
        jax.grad(
            lambda x, w_up_gate, w_down, combine_weights: jnp.sum(
                _dense_moe_output(x, selected_experts, combine_weights, w_up_gate, w_down) * cotangent_reference
            ),
            argnums=(0, 1, 2, 3),
        )
    )(x_reference, w_up_gate_reference, w_down_reference, combine_weights_reference)
    # Padding tokens take no part in routing, so their routing weights get no gradient.
    expected_gradients = (*expected_gradients[:3], expected_gradients[3] * token_valid[:, None])

    def assignment_output(token, slot):
        # y for one assignment: the dense output's derivative with respect to its weight.
        one_hot = jnp.zeros_like(combine_weights_reference).at[token, slot].set(1)
        return _dense_moe_output(x_reference, selected_experts, one_hot, w_up_gate_reference, w_down_reference)[token]

    # The edge gradients <dout, y> are far below the gradient's maximum, so check each against the
    # rounding of its own products.
    edge_scales = [
        float(jnp.sum(jnp.abs(cotangent_reference[token] * assignment_output(token, slot))))
        for token, slot in zip(*edge_assignments, strict=True)
    ]

    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    token_sharding = NamedSharding(mesh, P(("data", "expert")))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    x = jax.sharding.reshard(x, batch_sharding)
    selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
    combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
    token_valid = jax.sharding.reshard(token_valid, token_sharding)
    w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
    w_down = jax.sharding.reshard(w_down, expert_sharding)
    cotangent = jax.sharding.reshard(cotangent, batch_sharding)

    def backend_output(x, w_up_gate, w_down, combine_weights):
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            token_valid=token_valid,
            implementation=implementation,
            mesh=mesh,
            report_capacity_overflow=True,
            capacity_factor=2.0,
        )

    with jax.set_mesh(mesh):
        actual, overflow = jax.jit(backend_output)(x, w_up_gate, w_down, combine_weights)
        actual_gradients = jax.jit(
            jax.grad(
                lambda x, w_up_gate, w_down, combine_weights: jnp.sum(
                    backend_output(x, w_up_gate, w_down, combine_weights)[0] * cotangent
                ),
                argnums=(0, 1, 2, 3),
            )
        )(x, w_up_gate, w_down, combine_weights)

    def relative_max_error(actual, expected):
        actual = np.asarray(actual, dtype=np.float32)
        expected = np.asarray(expected, dtype=np.float32)
        return np.max(np.abs(actual - expected)) / np.max(np.abs(expected))

    assert relative_max_error(actual, expected) < relative_tolerance
    for actual_gradient in actual_gradients:
        assert np.isfinite(np.asarray(actual_gradient)).all()
    for actual_gradient, expected_gradient in zip(actual_gradients[:3], expected_gradients[:3], strict=True):
        assert relative_max_error(actual_gradient, expected_gradient) < relative_tolerance
    actual_weight_gradient = np.asarray(actual_gradients[3], dtype=np.float32)
    expected_weight_gradient = np.asarray(expected_gradients[3])
    if routing_weight_gradient == _RoutingWeightGradient.EXPERT_SIDE:
        # The edge assignments are outside EXPERT_SIDE's contract; compare every other weight.
        inside = np.ones(expected_weight_gradient.shape, dtype=bool)
        inside[edge_assignments] = False
        assert (
            relative_max_error(actual_weight_gradient[inside], expected_weight_gradient[inside]) < relative_tolerance
        )
    else:
        assert relative_max_error(actual_weight_gradient, expected_weight_gradient) < relative_tolerance
        edge_errors = np.abs(actual_weight_gradient[edge_assignments] - expected_weight_gradient[edge_assignments])
        np.testing.assert_array_less(edge_errors, relative_tolerance * np.asarray(edge_scales))
    assert int(overflow.dropped) == 0
    assert int(overflow.padding_skipped) == int(jnp.sum(~token_valid)) * topk


@pytest.mark.parametrize(
    ("routing_weight_gradient", "row_dot_transports"),
    [(_RoutingWeightGradient.EXACT, 0), (_RoutingWeightGradient.EXPERT_SIDE, 2)],
)
def test_ragged_backward_takes_the_routing_weight_gradient_of_its_expert_mlp(
    routing_weight_gradient: _RoutingWeightGradient, row_dot_transports: int, monkeypatch: pytest.MonkeyPatch
):
    # EXPERT_SIDE sends each expert row's <h, dh> back in a [rows, 1] float32 transport, one per expert
    # chunk (two here); EXACT differentiates the combine over the kept expert outputs instead.
    _force_routing_weight_gradient(monkeypatch, routing_weight_gradient)
    mesh = _make_abstract_moe_mesh(data=2, expert=2, model=1)
    tokens, hidden_dim, intermediate_dim, num_experts, topk = 16, 32, 64, 4, 2

    def spec(shape, dtype, *partition):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, P(*partition)))

    def loss(x, combine_weights, w_up_gate, w_down, selected_experts):
        out = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation="ragged_all_to_all",
            mesh=mesh,
        )
        return jnp.sum(out)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        # bfloat16, the dtype the GPU expert kernels take; the row dots stay float32.
        jaxpr = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2, 3)))(
            spec((tokens, hidden_dim), jnp.bfloat16, ("data", "expert"), None),
            spec((tokens, topk), jnp.bfloat16, ("data", "expert"), None),
            spec((num_experts, hidden_dim, 2 * intermediate_dim), jnp.bfloat16, "expert", None, None),
            spec((num_experts, intermediate_dim, hidden_dim), jnp.bfloat16, "expert", None, None),
            spec((tokens, topk), jnp.int32, ("data", "expert"), None),
        )

    def is_row_dot(eqn):
        return eqn.invars[0].aval.shape[-1:] == (1,) and eqn.invars[0].aval.dtype == jnp.float32

    assert _count_jaxpr_primitives(jaxpr, "ragged_all_to_all", is_row_dot) == row_dot_transports


def _filled_transport_buffer(fill: float):
    """A `_transport_buffer` whose unspecified contents are ``fill`` everywhere."""

    def transport_buffer(rows, hidden_dim, dtype, tie, site):
        del tie, site
        return jnp.full((rows, hidden_dim), fill, dtype)

    return transport_buffer


# The unwritten-rows test's layout: with two GPUs on the expert axis, each holds 6 experts and runs
# them in two chunks of 3, as the hero's GPUs do.
_UNWRITTEN_ROWS_EXPERTS = 12
_UNWRITTEN_ROWS_RANDOM_ROUTINGS = 16


def _unwritten_rows_routings(
    *, tokens: int, num_experts: int, topk: int, shards: int
) -> list[tuple[str, jax.Array, jax.Array]]:
    """Routings that leave transport rows unwritten in different ways: (name, selected, valid).

    Experts ``[0, 6)`` live on the first expert-axis rank and ``[6, 12)`` on the second; each rank's
    chunks hold three experts. Token shards are contiguous blocks of ``tokens // shards``.
    """
    local = num_experts // 2
    chunk = local // 2
    every_fourth_padded = jnp.arange(tokens) % 4 != 1

    def random_routing(seed: int) -> jax.Array:
        return jax.random.randint(jax.random.key(seed), (tokens, topk), 0, num_experts, dtype=jnp.int32)

    routings = [
        (f"random-{seed}", random_routing(seed), every_fourth_padded)
        for seed in range(_UNWRITTEN_ROWS_RANDOM_ROUTINGS)
    ]
    base = random_routing(100)
    per_shard = tokens // shards
    first_shard = jnp.arange(tokens) < per_shard
    # Shard i's tokens pick experts 3i and 3i + 1 (mod 12), so consecutive shards use different chunks.
    pair_start = (jnp.arange(tokens) // per_shard * chunk) % num_experts
    spread_pairs = jnp.stack([pair_start, pair_start + 1], axis=1).astype(jnp.int32)
    routings += [
        # An expert that no token selects has an empty group on every receiver.
        ("expert-unused", jnp.where(base == 0, 1, base), every_fourth_padded),
        # The second rank's first chunk receives no rows at all.
        (
            "chunk-unused",
            jnp.where((base >= local) & (base < local + chunk), base - local, base),
            every_fourth_padded,
        ),
        # The second rank receives no rows in either chunk, and the first rank drops many.
        ("rank-unused", base % local, every_fourth_padded),
        # The first shard sends nothing.
        ("sender-padded", base, every_fourth_padded & ~first_shard),
        # Every token picks the same two experts: nearly every assignment drops.
        ("two-experts", jnp.broadcast_to(jnp.array([3, 4], jnp.int32), (tokens, topk)), every_fourth_padded),
        # One valid token per shard, its two experts in one chunk that no other valid token uses:
        # nothing drops, and most of each receiver's capacity stays unwritten.
        ("mostly-padded", spread_pairs, jnp.arange(tokens) % per_shard == 0),
    ]
    return routings


@pytest.mark.parametrize("routing_weight_gradient", list(_RoutingWeightGradient), ids=lambda g: str(g))
def test_ragged_moe_reads_no_unwritten_transport_rows_on_gpu(
    routing_weight_gradient: _RoutingWeightGradient, monkeypatch: pytest.MonkeyPatch
):
    # The transport buffers start with unspecified contents, and every consumer must read only the
    # rows a collective wrote. Filling them with NaN instead of zero must change no output or
    # gradient, in the forward, the backward, or a recompute, so a reader of an unwritten row
    # anywhere in the layer fails here. The routings cover drops, padding, empty expert groups and
    # chunks, a receiver and a sender with no rows, and multi-expert chunks, all in one executable.
    mesh = _make_ep_mesh_or_none()
    if mesh is None or jax.devices()[0].platform != "gpu":
        pytest.skip("requires an even number of >=2 GPUs")
    _force_routing_weight_gradient(monkeypatch, routing_weight_gradient)

    tokens = len(jax.devices()) * 8
    hidden_dim, intermediate_dim, topk = 16, 24, 2
    num_experts = _UNWRITTEN_ROWS_EXPERTS
    x, _selected, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(41),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    cotangent = jax.random.normal(jax.random.key(43), (tokens, hidden_dim), dtype=jnp.bfloat16)

    batch = NamedSharding(mesh, P(("data", "expert"), None))
    token_axis = NamedSharding(mesh, P(("data", "expert")))
    experts = NamedSharding(mesh, P("expert", None, None))
    x, combine_weights, cotangent = (
        jax.sharding.reshard(a, batch)
        for a in (x.astype(jnp.bfloat16), combine_weights.astype(jnp.bfloat16), cotangent)
    )
    w_up_gate = jax.sharding.reshard(w_up_gate.astype(jnp.bfloat16), experts)
    w_down = jax.sharding.reshard(w_down.astype(jnp.bfloat16), experts)

    def layer(x, w_up_gate, w_down, combine_weights, selected_experts, token_valid):
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            token_valid=token_valid,
            implementation="ragged_all_to_all",
            mesh=mesh,
            report_capacity_overflow=True,
            capacity_factor=0.5,
        )

    def loss(x, w_up_gate, w_down, combine_weights, routing):
        # A fresh function for each fill: jax.checkpoint caches its trace by function, and a cached
        # trace would keep the previous fill in the recompute and the backward.
        out, _ = jax.checkpoint(lambda *operands: layer(*operands))(x, w_up_gate, w_down, combine_weights, *routing)
        return jnp.sum(out * cotangent)

    def run(x, w_up_gate, w_down, combine_weights, routing):
        out, counts = layer(x, w_up_gate, w_down, combine_weights, *routing)
        return out, counts.dropped, jax.grad(loss, argnums=range(4))(x, w_up_gate, w_down, combine_weights, routing)

    def zero_and_nan(*args):
        # One executable for every routing: each further program with ragged transports asks NCCL
        # for another symmetric-memory window, which a preallocated test process may not have room for.
        results = []
        for fill in (0.0, jnp.nan):
            monkeypatch.setattr(ep_ragged_all_to_all, "_transport_buffer", _filled_transport_buffer(fill))
            results.append(run(*args))
        return results

    routings = _unwritten_rows_routings(tokens=tokens, num_experts=num_experts, topk=topk, shards=len(jax.devices()))
    dropped = {}
    with jax.set_mesh(mesh):
        step = jax.jit(zero_and_nan)
        for name, selected_experts, token_valid in routings:
            routing = (jax.sharding.reshard(selected_experts, batch), jax.sharding.reshard(token_valid, token_axis))
            zero_filled, nan_filled = step(x, w_up_gate, w_down, combine_weights, routing)
            dropped[name] = int(zero_filled[1])
            for zero, nan in zip(jax.tree.leaves(zero_filled), jax.tree.leaves(nan_filled), strict=True):
                nan = np.asarray(nan, dtype=np.float32)
                assert np.isfinite(nan).all(), f"{name}: a NaN-filled unwritten row reached an output"
                np.testing.assert_array_equal(nan, np.asarray(zero, dtype=np.float32), err_msg=name)

    # The routings must do what they claim, or the cases they name go untested.
    assert dropped["two-experts"] > 0 and dropped["rank-unused"] > 0, dropped
    assert dropped["mostly-padded"] == 0, dropped


def test_moe_mlp_runs_with_ep_axis_when_available():
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")

    tokens = len(jax.devices()) * 8
    hidden_dim = 32
    intermediate_dim = 64
    num_experts = 4
    topk = 2

    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
            key=jax.random.key(1),
            tokens=tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            topk=topk,
        )

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        out = jax.jit(functools.partial(moe_mlp, activation=ActivationFunctionEnum.silu, mesh=None))(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
        assert out.shape == (tokens, hidden_dim)
        assert jnp.isfinite(out).all()

        out_ragged = jax.jit(
            functools.partial(
                moe_mlp,
                activation=ActivationFunctionEnum.silu,
                implementation="ragged_all_to_all",
                mesh=None,
            )
        )(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
        assert out_ragged.shape == (tokens, hidden_dim)
        assert jnp.isfinite(out_ragged).all()


def test_functional_moe_mlp_accepts_enum_and_callable_activation():
    tokens = 16
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 8
    topk = 2

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(2),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )

    y_enum = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=ActivationFunctionEnum.silu,
        mesh=None,
    )
    y_callable = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=lambda t: jax.nn.silu(t),
        mesh=None,
    )
    np.testing.assert_allclose(np.asarray(y_callable), np.asarray(y_enum), rtol=1e-5, atol=1e-5)


def test_moe_mlp_reports_positive_drop_count_in_ring_ep_when_over_capacity():
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")

    tokens = len(jax.devices()) * 8
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4
    topk = 2

    key = jax.random.key(5)
    x = jax.random.normal(key, (tokens, hidden_dim), dtype=jnp.float32)
    selected_experts = jnp.zeros((tokens, topk), dtype=jnp.int32)
    combine_weights = jnp.full((tokens, topk), 0.5, dtype=jnp.float32)
    w_up_gate = jax.random.normal(
        jax.random.key(6), (num_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.float32
    )
    w_down = jax.random.normal(jax.random.key(7), (num_experts, intermediate_dim, hidden_dim), dtype=jnp.float32)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        out, dispatch_counts = jax.jit(
            functools.partial(moe_mlp, implementation="ring", mesh=None, report_capacity_overflow=True)
        )(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )

    assert out.shape == (tokens, hidden_dim)
    assert dispatch_counts.dropped.shape == ()
    assert int(dispatch_counts.dropped) > 0


def test_moe_mlp_reports_positive_drop_count_in_ragged_a2a_when_over_capacity():
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")

    tokens = len(jax.devices()) * 8
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4
    topk = 2

    key = jax.random.key(15)
    x = jax.random.normal(key, (tokens, hidden_dim), dtype=jnp.float32)
    selected_experts = jnp.zeros((tokens, topk), dtype=jnp.int32)
    combine_weights = jnp.full((tokens, topk), 0.5, dtype=jnp.float32)
    w_up_gate = jax.random.normal(
        jax.random.key(16), (num_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.float32
    )
    w_down = jax.random.normal(jax.random.key(17), (num_experts, intermediate_dim, hidden_dim), dtype=jnp.float32)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        out, dispatch_counts = jax.jit(
            functools.partial(moe_mlp, implementation="ragged_all_to_all", mesh=None, report_capacity_overflow=True)
        )(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )

    assert out.shape == (tokens, hidden_dim)
    assert dispatch_counts.dropped.shape == ()
    assert int(dispatch_counts.dropped) > 0


@pytest.mark.parametrize("traced_capacity", [False, True])
@pytest.mark.parametrize(
    "capacity, expected",
    [
        (0, [[0, 0, 0, 0], [0, 0, 0, 0]]),
        (3, [[3, 0, 0, 0], [0, 0, 3, 0]]),
        (4, [[3, 0, 0, 0], [1, 0, 4, 0]]),
        (5, [[3, 0, 0, 0], [2, 0, 4, 1]]),
        (6, [[3, 1, 0, 0], [2, 0, 4, 1]]),
        (20, [[3, 1, 0, 0], [2, 0, 4, 1]]),
    ],
)
def test_ragged_a2a_receiver_clipping_respects_capacity(capacity, expected, traced_capacity):
    group_sizes = jnp.array(
        [
            [3, 1, 0, 0],
            [2, 0, 4, 1],
        ],
        dtype=jnp.int32,
    )

    def clip(counts, limit):
        return grug_moe._clip_receiver_group_sizes(counts, local_expert_size=2, receiver_capacity=limit)

    if traced_capacity:
        clipped = jax.jit(clip)(group_sizes, jnp.asarray(capacity, dtype=jnp.int32))
    else:
        clipped = jax.jit(clip, static_argnums=1)(group_sizes, capacity)
    np.testing.assert_array_equal(clipped, np.asarray(expected, dtype=np.int32))


# The transport buffer's traced minimum, as XLA names the opcode in optimized HLO.
MINIMUM_OPCODE = "kMinimum"


def _optimized_hlo_opcode_count(fill_fn, opcode_name: str) -> int:
    tie = jnp.asarray([1, 7, 0, 3], dtype=jnp.int32)
    executable = jax.jit(fill_fn).lower(tie).compile().runtime_executable()
    return sum(
        instruction.opcode.name == opcode_name
        for module in executable.hlo_modules()
        for computation in module.computations()
        for instruction in computation.instructions()
    )


def test_transport_buffer_is_not_a_foldable_constant():
    assert (
        _optimized_hlo_opcode_count(
            lambda tie: _transport_buffer(4, 3, jnp.float32, tie, site=_TransportBufferSite.DISPATCH_OUTPUT),
            MINIMUM_OPCODE,
        )
        == 1
    )
    assert (
        _optimized_hlo_opcode_count(
            lambda tie: jnp.broadcast_to((jnp.minimum(tie[0], 5) * 0).astype(jnp.float32), (4, 3)), MINIMUM_OPCODE
        )
        == 0
    ), "the folding probe no longer folds, so this test can no longer detect a foldable fill"


def test_transport_buffer_sites_prevent_cse():
    def distinct_sites(tie):
        return (
            _transport_buffer(4, 3, jnp.float32, tie, site=_TransportBufferSite.DISPATCH_OUTPUT),
            _transport_buffer(4, 3, jnp.float32, tie, site=_TransportBufferSite.DISPATCH_COTANGENT),
        )

    def repeated_site(tie):
        fill = _transport_buffer(4, 3, jnp.float32, tie, site=_TransportBufferSite.DISPATCH_OUTPUT)
        return fill, fill

    assert _optimized_hlo_opcode_count(distinct_sites, MINIMUM_OPCODE) == 2
    assert (
        _optimized_hlo_opcode_count(repeated_site, MINIMUM_OPCODE) == 1
    ), "the CSE probe no longer merges repeated sites, so this test can no longer detect a site collision"
