# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

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
    _prepare_moe_dispatch,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    CapacityOverflow,
)
from levanter.grug._moe.ep_deepep import _pack_deepep_local_assignments
from levanter.grug._moe.ep_fixed_all_to_all import _moe_mlp_ep_fixed_a2a_local
from levanter.grug._moe.ep_fixed_pooled_wave_all_to_all import (
    _moe_mlp_ep_fixed_pooled_wave_a2a_local,
    _interleaved_receiver_ranks,
    _receiver_ranks,
)
from levanter.grug._moe.ep_ragged_all_to_all import _loop_local_zeros, _LoopLocalZeroSite
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


def test_expert_granular_a2a_params_roundtrip_with_drops():
    """Dispatch packs receivers expert-major with sender order inside each expert, and the
    return direction restores each accepted row to its unclipped sorted position, leaving
    dropped rows at the output operand's values -- all under forced capacity clipping."""
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

    received = [np.zeros((capacity, hidden), np.float32) for _ in range(shards)]
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
        np.testing.assert_array_equal(received[receiver][len(expected) :], 0)

    returned = [np.zeros((assignments, hidden), np.float32) for _ in range(shards)]
    _simulate_ragged_a2a(received, returned, [p[1] for p in params])
    for s in range(shards):
        expected = np.zeros_like(sorted_payload[s])
        for g in range(num_experts):
            expected[starts[s, g] : starts[s, g] + clipped[s, g]] = sorted_payload[s][
                starts[s, g] : starts[s, g] + clipped[s, g]
            ]
        np.testing.assert_array_equal(returned[s], expected)


def test_expert_granular_a2a_params_chunked_masking_composes():
    """Masking the clip to one expert chunk at a time (full sender starts, chained returns)
    reproduces the whole layer: each chunk's receiver packs only its experts from offset zero,
    and the chained returns cover exactly the per-chunk accepted prefixes."""
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
        received = [np.zeros((chunk_capacity, hidden), np.float32) for _ in range(shards)]
        _simulate_ragged_a2a(sorted_payload, received, [p[0] for p in params])
        _simulate_ragged_a2a(received, returned, [p[1] for p in params])

    for s in range(shards):
        expected = np.zeros_like(sorted_payload[s])
        for g in range(num_experts):
            expected[starts[s, g] : starts[s, g] + accepted[s, g]] = sorted_payload[s][
                starts[s, g] : starts[s, g] + accepted[s, g]
            ]
        np.testing.assert_array_equal(returned[s], expected)


@pytest.mark.parametrize("implementation", ["ring", "ragged_all_to_all"])
def test_moe_mlp_ep_backends_match_dense_value_and_gradients_when_available(
    implementation: MoeImplementation,
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


@pytest.mark.parametrize("site", list(_LoopLocalZeroSite))
@pytest.mark.parametrize(
    "tie",
    [
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([1, 7, 0, 3], dtype=np.int32),
        np.array([2**20, 5, 5, 5], dtype=np.int32),
    ],
    ids=["all-empty-groups", "mixed", "large-first-group"],
)
def test_loop_local_zeros_fills_exact_zeros(site: _LoopLocalZeroSite, tie: np.ndarray):
    filled = _loop_local_zeros(4, 3, jnp.float32, jnp.asarray(tie), site=site)

    assert filled.shape == (4, 3)
    np.testing.assert_array_equal(np.asarray(filled), np.zeros((4, 3), dtype=np.float32))


def _optimized_hlo_opcode_count(fill_fn, opcode_name: str) -> int:
    tie = jnp.asarray([1, 7, 0, 3], dtype=jnp.int32)
    executable = jax.jit(fill_fn).lower(tie).compile().runtime_executable()
    return sum(
        instruction.opcode.name == opcode_name
        for module in executable.hlo_modules()
        for computation in module.computations()
        for instruction in computation.instructions()
    )


def test_loop_local_zeros_is_not_a_foldable_constant():
    assert (
        _optimized_hlo_opcode_count(
            lambda tie: _loop_local_zeros(4, 3, jnp.float32, tie, site=_LoopLocalZeroSite.DISPATCH_OUTPUT),
            "kMinimum",
        )
        == 1
    )
    assert (
        _optimized_hlo_opcode_count(
            lambda tie: jnp.broadcast_to((jnp.minimum(tie[0], 5) * 0).astype(jnp.float32), (4, 3)), "kMinimum"
        )
        == 0
    ), "the folding probe no longer folds, so this test can no longer detect a foldable fill"


def test_loop_local_zeros_sites_prevent_cse():
    def distinct_sites(tie):
        return (
            _loop_local_zeros(4, 3, jnp.float32, tie, site=_LoopLocalZeroSite.DISPATCH_OUTPUT),
            _loop_local_zeros(4, 3, jnp.float32, tie, site=_LoopLocalZeroSite.OPERAND_COTANGENT),
        )

    def repeated_site(tie):
        fill = _loop_local_zeros(4, 3, jnp.float32, tie, site=_LoopLocalZeroSite.DISPATCH_OUTPUT)
        return fill, fill

    assert _optimized_hlo_opcode_count(distinct_sites, "kMinimum") == 2
    assert (
        _optimized_hlo_opcode_count(repeated_site, "kMinimum") == 1
    ), "the CSE probe no longer merges repeated sites, so this test can no longer detect a site collision"
