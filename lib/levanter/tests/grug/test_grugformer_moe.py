# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax._src import config as jax_config
from jax.extend import core as jax_core
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh
from haliax.nn.ragged_dot import ragged_dot

import levanter.grug.grug_moe as grug_moe
from levanter.grug._moe.cudnn_wgrad_cute import (
    _GROUP_ALIGNMENT,
    aligned_group_capacity,
    full_partition_offsets,
    pad_grouped_rows,
)
from levanter.grug._moe.common import (
    _interleave_gate_up,
    _interleave_halves,
    _prepare_moe_dispatch,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    _swiglu_gate_up_backward,
    CapacityOverflow,
)
from levanter.grug._moe.ep_common import _align_up_rows
from levanter.grug._moe.ep_deepep import _pack_deepep_local_assignments
import levanter.grug._moe.ep_ragged_all_to_all as ep_ragged_all_to_all
from levanter.grug._moe.ep_ragged_all_to_all import (
    RAGGED_CUDNN_RECEIVER_ALIGNMENT,
    _ragged_dot_expert_mlp,
    _resolve_receiver_alignment,
)
from levanter.grug._moe.ep_fixed_all_to_all import _moe_mlp_ep_fixed_a2a_local
from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import (
    _moe_mlp_ep_fixed_pooled_wave_a2a_local,
    _interleaved_receiver_ranks,
    _receiver_ranks,
)
from levanter.grug._moe.sonic import sonic_gather_sum
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


def _count_jaxpr_primitives(value, primitive_name: str) -> int:
    jaxpr = getattr(value, "jaxpr", value)
    if isinstance(jaxpr, jax_core.Jaxpr):
        return sum(eqn.primitive.name == primitive_name for eqn in jaxpr.eqns) + sum(
            _count_jaxpr_primitives(param, primitive_name) for eqn in jaxpr.eqns for param in eqn.params.values()
        )
    if isinstance(value, dict):
        return sum(_count_jaxpr_primitives(item, primitive_name) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_jaxpr_primitives(item, primitive_name) for item in value)
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


@pytest.mark.parametrize(
    "capacity,sizes",
    [(1024, [300, 400, 200]), (256, [33, 32, 38, 25]), (100, [100]), (700, [0, 700, 0])],
)
def test_wgrad_padding_satisfies_both_halves_of_the_kernel_contract(capacity, sizes):
    """Every extent divides the alignment AND the final offset is the buffer's row count.

    Checking the padder against the kernel's contract, rather than against the transport's idea
    of the same layout, is the property the old self-consistency tests were missing: they passed
    at alignment 8 while the kernel required 256.
    """
    values = jnp.arange(capacity * 2, dtype=jnp.bfloat16).reshape(capacity, 2)
    padded, offsets = pad_grouped_rows(values, jnp.array(sizes, dtype=jnp.int32))

    extents = np.diff(np.concatenate([[0], np.asarray(offsets)]))
    assert all(int(extent) % _GROUP_ALIGNMENT == 0 for extent in extents)
    assert int(offsets[-1]) == padded.shape[0]
    assert all(int(extent) >= size for extent, size in zip(extents, sizes, strict=True))


def test_the_wgrad_group_alignment_matches_the_installed_cudnn_kernel():
    """The padding constant must equal the kernel's own requirement.

    The kernel does not validate its group sizes and returns silently wrong gradients when they
    are misaligned, so this is the only thing standing between a refactor and corrupted expert
    weight gradients (issue #8339: the constant read 8 against a 256-row contract). Asserting the
    two against each other, rather than against a literal, is what makes it non-regressable.
    """
    assert _GROUP_ALIGNMENT == 256, "cuDNN Frontend's grouped-Wgrad contract requires 256-row groups"

    try:
        kernel_module = importlib.import_module("cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad")
    except ImportError:
        pytest.skip("cuDNN Frontend is only installed with the GPU extra")
    assert _GROUP_ALIGNMENT == kernel_module.MoEGroupedGemmWgradBF16Kernel.FIX_PAD_SIZE


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


def test_prepare_moe_dispatch_indices_match_materialized_dispatch():
    x, selected_experts, combine_weights, _w_up_gate, _w_down = _make_inputs(
        key=jax.random.key(28),
        tokens=20,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=5,
        topk=2,
    )

    x_sort, w_sort, token_ids_sort, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        num_experts=5,
    )
    token_ids_from_indices, dispatch_positions, index_group_sizes, sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
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
def test_a_mismatched_moe_dim_is_rejected_rather_than_broadcast(dtype):
    # The packed path broadcasts the two halves against each other, so a wrong `moe_dim` would
    # return a wrong-width array instead of failing the way the stack path does.
    with pytest.raises(ValueError, match="w13 output last dimension"):
        _interleave_gate_up(_arange_w13(dtype, moe_dim=4), 3)


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


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_swiglu_backward_matches_autodiff_of_the_forward(dtype):
    tokens, moe_dim = 5, 4
    gu = jnp.linspace(-2.0, 2.0, tokens * 2 * moe_dim, dtype=jnp.float32).reshape(tokens, 2 * moe_dim).astype(dtype)
    dh = jnp.linspace(1.0, -1.0, tokens * moe_dim, dtype=jnp.float32).reshape(tokens, moe_dim).astype(dtype)

    def swiglu(x):
        gate, up = x[:, 0::2], x[:, 1::2]
        return jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)

    expected = jax.vjp(swiglu, gu)[1](dh.astype(jnp.float32))[0]

    actual = _swiglu_gate_up_backward(gu, dh)

    assert actual.dtype == gu.dtype
    # The reference runs in float32, so the gap is the storage dtype's rounding: one bfloat16
    # unit in the last place at these magnitudes is about 1e-3.
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=1e-2, atol=1e-3
    )


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
    ["ring", "ragged_all_to_all", "fixed_all_to_all", "fixed_pooled_wave_all_to_all"],
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

        def f(x, sel, cw, up_gate, down):
            return moe_mlp(
                x,
                sel,
                cw,
                up_gate,
                down,
                activation=ActivationFunctionEnum.silu,
                implementation=implementation,
                mesh=mesh,
                pooled_transport_capacity_factor=(1.05 if implementation == "fixed_pooled_wave_all_to_all" else None),
                num_expert_waves=1,
            )

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = (
            jax.jit(f)
            .trace(x, selected_experts, combine_weights, w_up_gate, w_down)
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
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=0.5,
        )

    sharded_fixed_a2a = jax.shard_map(
        fixed_a2a,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P()),
        out_specs=(P(), CapacityOverflow(sender=P(), receiver=P())),
        check_vma=False,
    )
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        actual, overflow = sharded_fixed_a2a(x, selected_experts, combine_weights, w_up_gate, w_down)

    keep = jnp.asarray([[True, True], [True, True], [False, False], [False, False]])

    def dense_output(x, w_up_gate, w_down):
        return _dense_moe_output(x, selected_experts, combine_weights * keep, w_up_gate, w_down)

    cotangent = jax.random.normal(jax.random.key(42), x.shape)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        actual_gradients = jax.grad(
            lambda x, w_up_gate, w_down: jnp.sum(
                sharded_fixed_a2a(x, selected_experts, combine_weights, w_up_gate, w_down)[0] * cotangent
            ),
            argnums=(0, 1, 2),
        )(x, w_up_gate, w_down)

        expected = dense_output(x, w_up_gate, w_down)
        expected_gradients = jax.grad(
            lambda x, w_up_gate, w_down: jnp.sum(dense_output(x, w_up_gate, w_down) * cotangent),
            argnums=(0, 1, 2),
        )(x, w_up_gate, w_down)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_gradient),
            np.asarray(expected_gradient),
            rtol=1e-5,
            atol=1e-5,
        )
    assert int(overflow.sender) == 4
    assert int(overflow.receiver) == 0


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
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=4.0,
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
        actual = sharded_pooled_output(x, combine_weights, w_up_gate, w_down)
        actual_gradient_fn = jax.grad(
            lambda x, combine_weights, w_up_gate, w_down: jnp.sum(
                sharded_pooled_output(x, combine_weights, w_up_gate, w_down) * cotangent
            ),
            argnums=(0, 1, 2, 3),
        )
        actual_gradients = actual_gradient_fn(x, combine_weights, w_up_gate, w_down)
        rematerialized_gradient_fn = jax.grad(
            lambda x, combine_weights, w_up_gate, w_down: jnp.sum(
                rematerialized_pooled_output(x, combine_weights, w_up_gate, w_down) * cotangent
            ),
            argnums=(0, 1, 2, 3),
        )
        gradient_jaxpr = jax.make_jaxpr(rematerialized_gradient_fn)(x, combine_weights, w_up_gate, w_down)
        expected = dense_output(x, combine_weights, w_up_gate, w_down)
        expected_gradients = jax.grad(
            lambda x, combine_weights, w_up_gate, w_down: jnp.sum(
                dense_output(x, combine_weights, w_up_gate, w_down) * cotangent
            ),
            argnums=(0, 1, 2, 3),
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
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=1.33,
            transport_capacity_factor=0.75,
            num_expert_waves=3,
        )

    sharded_pooled_output = jax.shard_map(
        pooled_output,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=(P(), CapacityOverflow(sender=P(), receiver=P())),
        check_vma=False,
    )
    with jax.set_mesh(mesh):
        actual, overflow = sharded_pooled_output(x, combine_weights, w_up_gate, w_down)

    keep = jnp.arange(tokens)[:, None] < 3
    expected = _dense_moe_output(x, selected_experts, combine_weights * keep, w_up_gate, w_down)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(overflow.sender) == 3
    assert int(overflow.receiver) == 3


@pytest.mark.parametrize("implementation", ["ring", "fixed_all_to_all", "fixed_pooled_wave_all_to_all"])
def test_portable_ep_backends_match_dense_cross_shard_value_and_gradients(implementation: MoeImplementation):
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
            return jnp.einsum("tkh,tk->th", expert_output, combine_weights)

        expected = dense_output(x, w_up_gate, w_down)
        expected_gradients = jax.grad(
            lambda x, w_up_gate, w_down: jnp.sum(dense_output(x, w_up_gate, w_down) * cotangent),
            argnums=(0, 1, 2),
        )(x, w_up_gate, w_down)

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.device_put(x, batch_sharding)
        selected_experts = jax.device_put(selected_experts, batch_sharding)
        combine_weights = jax.device_put(combine_weights, batch_sharding)
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
                activation=jax.nn.silu,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=4.0,
                **extra,
            )

        with jax.set_mesh(mesh):
            actual = backend_output(x, w_up_gate, w_down)
            actual_gradients = jax.grad(
                lambda x, w_up_gate, w_down: jnp.sum(backend_output(x, w_up_gate, w_down) * cotangent),
                argnums=(0, 1, 2),
            )(x, w_up_gate, w_down)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual_gradient),
                np.asarray(expected_gradient),
                rtol=1e-5,
                atol=1e-5,
            )
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script.replace("__IMPLEMENTATION__", implementation))],
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


@pytest.mark.parametrize("receiver_alignment", [1, 8])
def test_expert_granular_a2a_params_roundtrip_with_drops(receiver_alignment: int):
    """Dispatch packs receivers expert-major with sender order inside each expert, and the
    return direction restores each accepted row to its unclipped sorted position, leaving
    dropped rows at the output operand's values -- all under forced capacity clipping.

    Aligning the receiver's group starts must not change any of that: it decides where a group
    sits in the receiver buffer, not which rows are accepted or where they come home to.
    """
    shards, local_experts, tokens, topk, hidden = 4, 3, 10, 2, 5
    num_experts = shards * local_experts
    assignments = tokens * topk
    capacity = int(0.7 * assignments)  # force drops
    receiver_rows = aligned_group_capacity(capacity, local_experts, receiver_alignment)

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
            receiver_alignment=receiver_alignment,
        )
        for s in range(shards)
    ]

    received = [np.zeros((receiver_rows, hidden), np.float32) for _ in range(shards)]
    _simulate_ragged_a2a(sorted_payload, received, [p[0] for p in params])
    for receiver in range(shards):
        cursor = 0
        for e in range(local_experts):
            g = receiver * local_experts + e
            expected = np.concatenate(
                [sorted_payload[s][starts[s, g] : starts[s, g] + clipped[s, g]] for s in range(shards)], axis=0
            )
            assert cursor % receiver_alignment == 0
            np.testing.assert_array_equal(received[receiver][cursor : cursor + len(expected)], expected)
            extent = -(-len(expected) // receiver_alignment) * receiver_alignment
            # Slack inside the group, which the grouped kernels contract as zeros.
            np.testing.assert_array_equal(received[receiver][cursor + len(expected) : cursor + extent], 0)
            cursor += extent
        np.testing.assert_array_equal(received[receiver][cursor:], 0)

    returned = [np.zeros((assignments, hidden), np.float32) for _ in range(shards)]
    _simulate_ragged_a2a(received, returned, [p[1] for p in params])
    for s in range(shards):
        expected = np.zeros_like(sorted_payload[s])
        for g in range(num_experts):
            expected[starts[s, g] : starts[s, g] + clipped[s, g]] = sorted_payload[s][
                starts[s, g] : starts[s, g] + clipped[s, g]
            ]
        np.testing.assert_array_equal(returned[s], expected)


@pytest.mark.parametrize("receiver_alignment", [1, 8])
def test_expert_granular_a2a_params_chunked_masking_composes(receiver_alignment: int):
    """Masking the clip to one expert chunk at a time (full sender starts, chained returns)
    reproduces the whole layer: each chunk's receiver packs only its experts from offset zero,
    and the chained returns cover exactly the per-chunk accepted prefixes.

    The aligned layout composes the same way: a chunk's slack lives in that chunk's receiver
    buffer, and the accepted set it returns is unchanged."""
    shards, local_experts, tokens, topk, hidden = 4, 3, 10, 2, 5
    num_experts = shards * local_experts
    assignments = tokens * topk
    capacity = int(0.7 * assignments)
    chunks = 3
    chunk_experts = local_experts // chunks
    chunk_capacity = -(-capacity // chunks)
    receiver_rows = aligned_group_capacity(chunk_capacity, chunk_experts, receiver_alignment)
    chunk_of_expert = (np.arange(num_experts) % local_experts) // chunk_experts

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
                receiver_alignment=receiver_alignment,
            )
            for s in range(shards)
        ]
        received = [np.zeros((receiver_rows, hidden), np.float32) for _ in range(shards)]
        _simulate_ragged_a2a(sorted_payload, received, [p[0] for p in params])
        _simulate_ragged_a2a(received, returned, [p[1] for p in params])

    for s in range(shards):
        expected = np.zeros_like(sorted_payload[s])
        for g in range(num_experts):
            expected[starts[s, g] : starts[s, g] + accepted[s, g]] = sorted_payload[s][
                starts[s, g] : starts[s, g] + accepted[s, g]
            ]
        np.testing.assert_array_equal(returned[s], expected)


def test_aligned_receiver_layout_is_what_the_wgrad_pad_would_have_built():
    """The claim the pre-aligned cuDNN entry point rests on.

    `pad_grouped_rows` turns an unaligned receiver buffer into the kernel's aligned layout by
    copying it. An aligned receiver layout is that same layout, built by the collective instead.
    Both the operand and the offsets the kernel sees must therefore agree row for row -- if they
    did not, skipping the pad would change what the kernel computes.
    """
    shards, local_experts, tokens, topk, hidden = 4, 3, 10, 2, 5
    num_experts = shards * local_experts
    assignments = tokens * topk
    capacity = int(0.7 * assignments)
    alignment = _GROUP_ALIGNMENT

    rng = np.random.default_rng(0)
    selected = rng.integers(0, num_experts, size=(shards, tokens, topk))
    payload = rng.normal(size=(shards, assignments, hidden)).astype(np.float32)
    sorted_payload = np.stack([payload[s][np.argsort(selected[s].reshape(-1), kind="stable")] for s in range(shards)])
    group_sizes = np.stack(
        [np.bincount(selected[s].reshape(-1), minlength=num_experts) for s in range(shards)]
    ).astype(np.int32)
    clipped = np.asarray(
        _clip_receiver_group_sizes(
            jnp.asarray(group_sizes), local_expert_size=local_experts, receiver_capacity=capacity
        )
    )
    assert clipped.sum() < group_sizes.sum()  # drops actually happen

    buffers = {}
    for receiver_alignment in (1, alignment):
        params = [
            _expert_granular_a2a_params(
                jnp.asarray(group_sizes),
                jnp.asarray(clipped),
                jnp.asarray(s),
                local_expert_size=local_experts,
                receiver_alignment=receiver_alignment,
            )
            for s in range(shards)
        ]
        rows = aligned_group_capacity(capacity, local_experts, receiver_alignment)
        received = [np.zeros((rows, hidden), np.float32) for _ in range(shards)]
        _simulate_ragged_a2a(sorted_payload, received, [p[0] for p in params])
        buffers[receiver_alignment] = received

    receiver_rows = aligned_group_capacity(capacity, local_experts, alignment)
    arrivals = clipped.reshape(shards, shards, local_experts).sum(axis=0)  # [receiver, local expert]
    for receiver in range(shards):
        active = arrivals[receiver].astype(np.int32)
        padded, padded_offsets = pad_grouped_rows(jnp.asarray(buffers[1][receiver]), jnp.asarray(active))
        aligned = buffers[alignment][receiver]

        # The backend hands the pre-aligned entry point the rounded extents; the kernel wrapper
        # then charges the buffer's leftover rows to the last group, which is the full partition
        # the kernel requires. `pad_grouped_rows` builds the same partition of a same-sized buffer.
        extents = np.asarray(_align_up_rows(jnp.asarray(active), alignment))
        physical = extents.copy()
        physical[-1] += receiver_rows - extents.sum()

        assert padded.shape[0] == receiver_rows
        np.testing.assert_array_equal(np.asarray(padded), aligned)
        np.testing.assert_array_equal(np.asarray(padded_offsets), np.cumsum(physical))
        np.testing.assert_array_equal(
            np.asarray(full_partition_offsets(receiver_rows, jnp.asarray(extents[:-1]))), np.cumsum(physical)
        )
        assert int(padded_offsets[-1]) == padded.shape[0]


def test_expert_mlp_over_an_aligned_receiver_buffer_matches_the_packed_layout():
    """The other half of the claim: the expert MLP reads the aligned layout as if it were packed.

    Each grouped GEMM here is row-wise, so a group's alignment slack -- zero rows the collective
    never writes -- contributes nothing and produces zeros, and every arriving row gets the same
    value it would have got packed. This is the portable `ragged_dot` path, which is what runs
    wherever the cuDNN kernels do not.
    """
    alignment = _GROUP_ALIGNMENT
    experts, hidden, intermediate = 3, 8, 16
    active = np.array([5, 12, 3], np.int32)
    extents = -(-active // alignment) * alignment
    capacity = int(active.sum()) + 4  # trailing rows past the last group, as the backend has

    rng = np.random.default_rng(0)
    w13 = jnp.asarray(rng.normal(size=(experts, hidden, 2 * intermediate)), jnp.float32)
    w2 = jnp.asarray(rng.normal(size=(experts, intermediate, hidden)), jnp.float32)
    rows = rng.normal(size=(int(active.sum()), hidden)).astype(np.float32)

    packed = np.zeros((capacity, hidden), np.float32)
    packed[: active.sum()] = rows
    aligned = np.zeros((aligned_group_capacity(capacity, experts, alignment), hidden), np.float32)
    starts = np.cumsum(extents) - extents
    read = 0
    for e in range(experts):
        aligned[starts[e] : starts[e] + active[e]] = rows[read : read + active[e]]
        read += active[e]

    def run(buffer, sizes):
        physical = jnp.asarray(sizes).at[-1].add(buffer.shape[0] - int(sizes.sum()))
        return _ragged_dot_expert_mlp(jnp.asarray(buffer), w13, w2, physical, jnp.asarray(sizes), jax.nn.silu)

    packed_out = np.asarray(run(packed, active))
    aligned_out = np.asarray(run(aligned, extents))

    read = 0
    for e in range(experts):
        np.testing.assert_array_equal(
            aligned_out[starts[e] : starts[e] + active[e]], packed_out[read : read + active[e]]
        )
        np.testing.assert_array_equal(aligned_out[starts[e] + active[e] : starts[e] + extents[e]], 0)
        read += active[e]


@pytest.mark.parametrize(
    "capacity,groups,arrivals",
    [
        # The hero geometry: one chunk of three experts over the production receiver capacity.
        (301466, 3, [301466, 0, 0]),
        (301466, 3, [100000, 100000, 101466]),
        (301466, 3, [99999, 100001, 65536]),
        (301466, 3, [0, 0, 0]),
        # Edges: a single group, an empty group, an exactly aligned capacity.
        (301466, 1, [301466]),
        (301466, 1, [0]),
        (1024, 4, [0, 512, 0, 512]),
        (1024, 4, [1, 1, 1, 1]),
        (768, 3, [256, 256, 256]),
    ],
)
def test_prealigned_offsets_are_a_full_aligned_partition_of_the_receiver_buffer(capacity, groups, arrivals):
    """Both halves of the kernel's contract, on the layout the transport hands it.

    The pre-aligned path skips `pad_grouped_rows`, so nothing downstream repairs the offsets: the
    transport's buffer sizing and `full_partition_offsets` have to produce every extent a multiple
    of `_GROUP_ALIGNMENT` *and* a final offset equal to the operand's row count. Rounding the
    extents up alone gives the first and not the second.
    """
    alignment = _GROUP_ALIGNMENT
    receiver_rows = aligned_group_capacity(capacity, groups, alignment)
    assert receiver_rows % alignment == 0
    assert receiver_rows >= capacity

    extents = np.asarray(_align_up_rows(jnp.asarray(arrivals, jnp.int32), alignment))
    assert extents.sum() <= receiver_rows, "the static buffer must cover every rounding of the routing"

    offsets = np.asarray(full_partition_offsets(receiver_rows, jnp.asarray(extents[:-1])))
    sizes = np.diff(np.concatenate([[0], offsets]))

    assert int(offsets[-1]) == receiver_rows
    assert all(int(size) % alignment == 0 for size in sizes)
    assert all(int(size) >= arrival for size, arrival in zip(sizes, arrivals, strict=True))

    # The QuACK GEMMs run over the extents, not over this partition: their row count must stay
    # within one alignment per expert of the arrivals, or the aligned arm buys the copy back.
    assert extents.sum() - sum(arrivals) < alignment * groups


def test_receiver_alignment_must_satisfy_the_wgrad_kernels_group_alignment():
    """The pre-aligned path skips the kernel's own pad, so the layout has to already satisfy it."""
    assert RAGGED_CUDNN_RECEIVER_ALIGNMENT == _GROUP_ALIGNMENT == 256
    assert _resolve_receiver_alignment(None) == 1
    assert _resolve_receiver_alignment(_GROUP_ALIGNMENT) == _GROUP_ALIGNMENT
    assert _resolve_receiver_alignment(2 * _GROUP_ALIGNMENT) == 2 * _GROUP_ALIGNMENT
    # The alignments this option was written against before the kernel's real requirement was
    # known. An arm configured at one of them must not quietly run a misaligned layout.
    for stale in (1, 8, 64, 128, _GROUP_ALIGNMENT + 1, _GROUP_ALIGNMENT + 128):
        with pytest.raises(ValueError, match="multiple"):
            _resolve_receiver_alignment(stale)
    with pytest.raises(ValueError, match="positive"):
        _resolve_receiver_alignment(0)
    with pytest.raises(ValueError, match="positive"):
        _resolve_receiver_alignment(-256)


def test_receiver_alignment_routes_the_expert_mlp_to_the_prealigned_wgrad(monkeypatch):
    """An aligned receiver layout is the only thing that lets the cute path skip the pad."""
    monkeypatch.setattr(ep_ragged_all_to_all, "_quack_grouped_gemm_available", lambda: True)
    select = ep_ragged_all_to_all._select_expert_mlp
    assert select(jax.nn.silu) is ep_ragged_all_to_all._cute_expert_mlp
    assert select(jax.nn.silu, receiver_alignment=1) is ep_ragged_all_to_all._cute_expert_mlp
    assert select(jax.nn.silu, receiver_alignment=_GROUP_ALIGNMENT) is ep_ragged_all_to_all._cute_expert_mlp_prealigned
    # The portable path reads group sizes off whatever buffer it is handed, aligned or not.
    assert select(jax.nn.gelu, receiver_alignment=_GROUP_ALIGNMENT) is _ragged_dot_expert_mlp
    # The harness records which entry point ran off this name, so the two must not share one.
    assert select(jax.nn.silu).__name__ != select(jax.nn.silu, receiver_alignment=_GROUP_ALIGNMENT).__name__


def test_the_prealigned_cute_path_runs_the_gemms_over_the_arrivals_not_the_whole_buffer(monkeypatch):
    """The aligned arm must not buy its copy saving back in grouped-GEMM row work.

    The weight-gradient kernel needs a partition of the whole receiver buffer, but the QuACK
    GEMMs do not: driving them off the physical sizes instead of the arrival extents would put
    every unused capacity row through every grouped GEMM. At the hero that is ~302k rows against
    ~262k, a 15% inflation of a 4 s/step leg, against 365 ms of copy saved. So the ``cu`` this
    path builds must end at the arrivals' aligned extents, exactly as the packed path's does.
    """
    recorded = {}

    def fake_expert_mlp_cudnn(x_dispatch, w13_il, moe_w2, group_sizes, cu):
        recorded["group_sizes"] = np.asarray(group_sizes)
        recorded["cu"] = np.asarray(cu)
        return x_dispatch

    monkeypatch.setitem(
        sys.modules,
        "levanter.grug._moe.sonic_cute",
        SimpleNamespace(
            _expert_mlp_cudnn=fake_expert_mlp_cudnn,
            _expert_mlp_cudnn_prealigned=fake_expert_mlp_cudnn,
        ),
    )

    experts, hidden, intermediate = 3, 8, 16
    arrivals = np.array([300, 400, 200], np.int32)
    receiver_rows = aligned_group_capacity(1024, experts, _GROUP_ALIGNMENT)
    extents = _align_up_rows(jnp.asarray(arrivals), _GROUP_ALIGNMENT)
    physical = extents.at[-1].add(receiver_rows - int(np.asarray(extents).sum()))

    x = jnp.zeros((receiver_rows, hidden), jnp.bfloat16)
    w13 = jnp.zeros((experts, hidden, 2 * intermediate), jnp.bfloat16)
    w2 = jnp.zeros((experts, intermediate, hidden), jnp.bfloat16)
    ep_ragged_all_to_all._cute_expert_mlp_prealigned(x, w13, w2, physical, extents, jax.nn.silu)

    np.testing.assert_array_equal(recorded["group_sizes"], np.asarray(extents))
    assert int(recorded["cu"][-1]) == int(np.asarray(extents).sum())
    # ...which is strictly less than the buffer, and within one alignment per expert of arrivals.
    assert int(recorded["cu"][-1]) < receiver_rows == int(np.asarray(physical).sum())
    assert int(recorded["cu"][-1]) - int(arrivals.sum()) < _GROUP_ALIGNMENT * experts


@pytest.mark.parametrize("implementation", ["ring", "ragged_all_to_all"])
def test_moe_mlp_ep_backends_match_dense_value_and_gradients_when_available(implementation: MoeImplementation):
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")
    if jax.devices()[0].platform == "cpu":
        pytest.skip("ragged_all_to_all is not implemented on XLA:CPU")

    tokens = len(jax.devices()) * 8
    gpu_runtime = jax.devices()[0].platform == "gpu"
    hidden_dim = 16 if gpu_runtime else 128
    intermediate_dim = 24 if gpu_runtime else 128
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
    dtype = jnp.bfloat16 if gpu_runtime else jnp.float32
    relative_tolerance = _BF16_MOE_RELATIVE_TOLERANCE if dtype == jnp.bfloat16 else _FP32_MOE_RELATIVE_TOLERANCE
    x = x.astype(dtype)
    combine_weights = combine_weights.astype(dtype)
    w_up_gate = w_up_gate.astype(dtype)
    w_down = w_down.astype(dtype)
    cotangent = jax.random.normal(jax.random.key(24), x.shape, dtype=dtype)

    x_reference = x.astype(jnp.float32)
    combine_weights_reference = combine_weights.astype(jnp.float32)
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
    expected_gradients = jax.grad(
        lambda x, w_up_gate, w_down: jnp.sum(
            _dense_moe_output(x, selected_experts, combine_weights_reference, w_up_gate, w_down) * cotangent_reference
        ),
        argnums=(0, 1, 2),
    )(x_reference, w_up_gate_reference, w_down_reference)

    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    x = jax.sharding.reshard(x, batch_sharding)
    selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
    combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
    w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
    w_down = jax.sharding.reshard(w_down, expert_sharding)
    cotangent = jax.sharding.reshard(cotangent, batch_sharding)

    def backend_output(x, w_up_gate, w_down):
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation=implementation,
            mesh=mesh,
            report_capacity_overflow=True,
            capacity_factor=2.0,
        )

    with jax.set_mesh(mesh):
        actual, overflow = backend_output(x, w_up_gate, w_down)
        actual_gradients = jax.grad(
            lambda x, w_up_gate, w_down: jnp.sum(backend_output(x, w_up_gate, w_down)[0] * cotangent),
            argnums=(0, 1, 2),
        )(x, w_up_gate, w_down)

    def relative_max_error(actual, expected):
        actual = np.asarray(actual, dtype=np.float32)
        expected = np.asarray(expected, dtype=np.float32)
        return np.max(np.abs(actual - expected)) / np.max(np.abs(expected))

    assert relative_max_error(actual, expected) < relative_tolerance
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        assert np.isfinite(np.asarray(actual_gradient)).all()
        assert relative_max_error(actual_gradient, expected_gradient) < relative_tolerance
    assert int(overflow.total) == 0


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

        out_ragged = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="ragged_all_to_all",
            mesh=None,
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

        out, dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation="ring",
            mesh=None,
            report_capacity_overflow=True,
        )

    assert out.shape == (tokens, hidden_dim)
    assert dropped.total.shape == ()
    assert int(dropped.total) > 0


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

        out, dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation="ragged_all_to_all",
            mesh=None,
            report_capacity_overflow=True,
        )

    assert out.shape == (tokens, hidden_dim)
    assert dropped.total.shape == ()
    assert int(dropped.total) > 0


def test_ragged_a2a_receiver_clipping_respects_capacity():
    group_sizes = jnp.array(
        [
            [3, 1, 0, 0],
            [2, 0, 4, 1],
        ],
        dtype=jnp.int32,
    )

    clipped = grug_moe._clip_receiver_group_sizes(
        group_sizes,
        local_expert_size=2,
        receiver_capacity=3,
    )

    np.testing.assert_array_equal(
        np.asarray(clipped),
        np.asarray(
            [
                [3, 0, 0, 0],
                [0, 0, 3, 0],
            ],
            dtype=np.int32,
        ),
    )
    assert int(jnp.sum(clipped)) < int(jnp.sum(group_sizes))
