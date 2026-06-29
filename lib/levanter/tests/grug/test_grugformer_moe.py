# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh
from haliax.nn.ragged_dot import ragged_dot

import levanter.grug.grug_moe as grug_moe
from levanter.grug._moe.common import (
    _DEFAULT_EP_CAPACITY_FACTOR,
    _prepare_moe_dispatch,
    _prepare_moe_dispatch_indices_with_assignment_ids,
)
from levanter.grug._moe.ep_deepep import _pack_deepep_local_assignments
from levanter.grug._moe.pallas_mgpu import (
    GroupInfo,
    MoeMgpuConfig,
    _MoeMgpuUpMetadata,
    _effective_padded_capacity_factor,
    _moe_mgpu_dispatch_w13_activation,
    _receiver_capacity,
    down_unpermute_mgpu,
    infer_moe_mgpu_config,
    moe_mlp_pallas_mgpu,
    moe_mlp_pallas_mgpu_reference,
    moe_mlp_pallas_mgpu_staged,
    permute_mgpu,
    permute_mgpu_reference,
    permute_up_mgpu,
    permute_up_mgpu_reference,
    prepare_mgpu_moe_metadata,
    ragged_w2_mgpu,
    ragged_w2_reference,
    unpermute_mgpu,
    unpermute_mgpu_reference,
)
from levanter.grug._moe.sonic import sonic_gather_sum
from levanter.grug.grug_moe import (
    MoEExpertMlp,
    MoEExpertMlpPspecs,
    MoeImplementation,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _shard_a2a_params,
    moe_mlp,
)
from levanter.utils.activation import ActivationFunctionEnum


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
    devices = jax.devices()
    if len(devices) < 2 or len(devices) % 2 != 0:
        return None
    mesh_devices = np.array(devices).reshape(len(devices) // 2, 2, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _make_expert_only_mesh_or_none() -> Mesh | None:
    devices = jax.devices()
    if len(devices) < 2:
        return None
    mesh_devices = np.array(devices).reshape(1, len(devices), 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _make_two_expert_mesh_or_none() -> Mesh | None:
    devices = jax.devices()
    if len(devices) < 2:
        return None
    mesh_devices = np.array(devices[:2]).reshape(1, 2, 1)
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


def test_prepare_moe_dispatch_indices_preserve_assignment_order_within_expert():
    selected_experts = jnp.array(
        [
            [2, 1],
            [2, 2],
            [1, 2],
        ],
        dtype=jnp.int32,
    )
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    combine_weights = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)

    x_sort, w_sort, token_ids_sort, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        num_experts=3,
    )
    token_ids_from_indices, dispatch_positions, index_group_sizes, sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            num_experts=3,
        )
    )

    np.testing.assert_array_equal(np.asarray(sorted_assignment_ids), np.array([1, 4, 0, 2, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(token_ids_from_indices), np.array([0, 2, 0, 1, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(token_ids_sort), np.asarray(token_ids_from_indices))
    np.testing.assert_array_equal(np.asarray(index_group_sizes), np.asarray(group_sizes))
    np.testing.assert_allclose(np.asarray(x[token_ids_from_indices]), np.asarray(x_sort), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(combine_weights.reshape(-1)[np.asarray(sorted_assignment_ids)]),
        np.asarray(w_sort),
        rtol=0,
        atol=0,
    )
    expected_dispatch_positions = np.array([[2, 0], [3, 4], [1, 5]], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(dispatch_positions), expected_dispatch_positions)


def test_moe_expert_mlp_init_uses_concat_w13_for_sonic_backend():
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
        np.asarray(sonic_mlp.w_gate_up),
        np.asarray(scatter_mlp.w_gate_up),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(sonic_mlp.w_down), np.asarray(scatter_mlp.w_down), rtol=1e-5, atol=1e-5)


def test_moe_expert_mlp_ordered_implementation_falls_back_at_call_boundary():
    k_mlp = jax.random.key(65)
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4
    x, selected_experts, combine_weights, _w_up_gate, _w_down = _make_inputs(
        key=jax.random.key(66),
        tokens=8,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=2,
    )
    scatter_mlp = MoEExpertMlp.init(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        initializer_std=0.02,
        key=k_mlp,
        implementation="scatter",
    )
    fallback_mlp = MoEExpertMlp.init(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        initializer_std=0.02,
        key=k_mlp,
        implementation=["pallas_mgpu", "scatter"],
    )

    expected = scatter_mlp(x, selected_experts, combine_weights, mesh=None)
    with pytest.warns(RuntimeWarning, match="implementation 'pallas_mgpu' failed; trying next fallback"):
        actual = fallback_mlp(x, selected_experts, combine_weights, mesh=None)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_moe_expert_mlp_ordered_implementation_preserves_capacity_overflow_report():
    k_mlp = jax.random.key(68)
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4
    x, selected_experts, combine_weights, _w_up_gate, _w_down = _make_inputs(
        key=jax.random.key(69),
        tokens=8,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=2,
    )
    scatter_mlp = MoEExpertMlp.init(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        initializer_std=0.02,
        key=k_mlp,
        implementation="scatter",
    )
    fallback_mlp = MoEExpertMlp.init(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        initializer_std=0.02,
        key=k_mlp,
        implementation=("pallas_mgpu", "scatter"),
    )

    expected, expected_dropped = scatter_mlp(
        x,
        selected_experts,
        combine_weights,
        mesh=None,
        report_capacity_overflow=True,
    )
    with pytest.warns(RuntimeWarning, match="implementation 'pallas_mgpu' failed; trying next fallback"):
        actual, actual_dropped = fallback_mlp(
            x,
            selected_experts,
            combine_weights,
            mesh=None,
            report_capacity_overflow=True,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(actual_dropped) == int(expected_dropped)


def test_moe_expert_mlp_init_rejects_empty_implementation_sequence():
    with pytest.raises(ValueError, match="implementation sequence must contain at least one implementation"):
        MoEExpertMlp.init(
            num_experts=4,
            hidden_dim=16,
            intermediate_dim=24,
            initializer_std=0.02,
            key=jax.random.key(67),
            implementation=(),
        )


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

    assert mlp.w_gate_up.sharding.spec == P(None, "data", "model")
    assert mlp.w_down.sharding.spec == P(None, "model", "data")


def test_prepare_mgpu_moe_metadata_sorts_assignment_ids_stably():
    selected_experts = jnp.array(
        [
            [3, 1],
            [1, 2],
        ],
        dtype=jnp.int32,
    )

    metadata = prepare_mgpu_moe_metadata(selected_experts, local_experts=2, ep_size=2)

    np.testing.assert_array_equal(np.asarray(metadata.assignment_ids_sorted), np.array([1, 2, 3, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(metadata.token_ids_sorted), np.array([0, 1, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(metadata.expert_ids_sorted), np.array([1, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(metadata.dst_ranks_sorted), np.array([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(metadata.local_experts_sorted), np.array([1, 1, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(metadata.local_pos_sorted), np.array([0, 1, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(metadata.send_counts),
        np.array(
            [
                [0, 2],
                [1, 1],
            ],
            dtype=np.int32,
        ),
    )
    assert metadata.global_counts.shape == (1, 2, 2)


def test_pallas_mgpu_group_info_does_not_select_middle_empty_group():
    group_lengths = jnp.array([5, 0, 3], dtype=jnp.int32)

    empty_gap_info = GroupInfo.create(group_lengths, 4, jnp.array(2, dtype=jnp.int32))
    later_group_info = GroupInfo.create(group_lengths, 4, jnp.array(3, dtype=jnp.int32))

    assert int(empty_gap_info.actual_size) == 0
    assert int(empty_gap_info.group_id) != 1
    assert int(later_group_info.group_id) == 2
    assert int(later_group_info.actual_start) == 5
    assert int(later_group_info.actual_end) == 8
    assert int(later_group_info.start_within_block) == 1
    assert int(later_group_info.actual_size) == 3


def test_permute_mgpu_reference_builds_expert_major_receive_layout():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")

    x = jnp.stack(
        [
            jnp.arange(6, dtype=jnp.float32),
            100 + jnp.arange(6, dtype=jnp.float32),
        ],
        axis=1,
    )
    selected_experts = jnp.array(
        [
            [0, 2],
            [1, 3],
            [2, 3],
            [0, 2],
            [1, 1],
            [3, 0],
        ],
        dtype=jnp.int32,
    )

    def shard_local(x_local: jax.Array, selected_experts_local: jax.Array):
        layout = permute_mgpu_reference(
            x_local,
            selected_experts_local,
            local_experts=2,
            config=MoeMgpuConfig(capacity_factor=1.0),
        )
        return (
            layout.recv_x,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            layout.rows_per_expert,
            layout.dropped,
        )

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        recv_x, recv_src_rank, recv_src_assignment, rows_per_expert, dropped = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert", None)),
            out_specs=(P("expert", None), P("expert"), P("expert"), P("expert"), P()),
            check_vma=False,
        )(x, selected_experts)

    expected_recv_x = np.array(
        [
            [[0, 100], [3, 103], [5, 105], [1, 101], [4, 104], [4, 104]],
            [[0, 100], [2, 102], [3, 103], [1, 101], [2, 102], [5, 105]],
        ],
        dtype=np.float32,
    )
    expected_src_rank = np.array(
        [
            [0, 1, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=np.int32,
    )
    expected_src_assignment = np.array(
        [
            [0, 0, 5, 2, 2, 3],
            [1, 4, 1, 3, 5, 4],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(np.asarray(recv_x).reshape(2, 6, 2), expected_recv_x)
    np.testing.assert_array_equal(np.asarray(recv_src_rank).reshape(2, 6), expected_src_rank)
    np.testing.assert_array_equal(np.asarray(recv_src_assignment).reshape(2, 6), expected_src_assignment)
    np.testing.assert_array_equal(
        np.asarray(rows_per_expert).reshape(2, 2), np.array([[3, 3], [3, 3]], dtype=np.int32)
    )
    assert int(dropped) == 0


def test_permute_mgpu_matches_reference_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("permute_mgpu requires H100/Hopper GPUs")

    x = jnp.stack(
        [
            jnp.arange(6, dtype=jnp.bfloat16),
            (100 + jnp.arange(6, dtype=jnp.float32)).astype(jnp.bfloat16),
        ],
        axis=1,
    )
    selected_experts = jnp.array(
        [
            [0, 2],
            [1, 3],
            [2, 3],
            [0, 2],
            [1, 1],
            [3, 0],
        ],
        dtype=jnp.int32,
    )

    def shard_local(x_local: jax.Array, selected_experts_local: jax.Array):
        config = MoeMgpuConfig(capacity_factor=1.0)
        actual = permute_mgpu(
            x_local,
            selected_experts_local,
            local_experts=2,
            config=config,
        )
        expected = permute_mgpu_reference(
            x_local,
            selected_experts_local,
            local_experts=2,
            config=config,
        )
        return (
            actual.recv_x,
            expected.recv_x,
            actual.recv_src_rank,
            expected.recv_src_rank,
            actual.recv_src_assignment,
            expected.recv_src_assignment,
            actual.rows_per_expert,
            expected.rows_per_expert,
            actual.dropped,
            expected.dropped,
        )

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        (
            recv_x,
            expected_recv_x,
            recv_src_rank,
            expected_src_rank,
            recv_src_assignment,
            expected_src_assignment,
            rows_per_expert,
            expected_rows_per_expert,
            dropped,
            expected_dropped,
        ) = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert", None)),
            out_specs=(
                P("expert", None),
                P("expert", None),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P(),
                P(),
            ),
            check_vma=False,
        )(
            x, selected_experts
        )

    np.testing.assert_array_equal(np.asarray(recv_x), np.asarray(expected_recv_x))
    np.testing.assert_array_equal(np.asarray(recv_src_rank), np.asarray(expected_src_rank))
    np.testing.assert_array_equal(np.asarray(recv_src_assignment), np.asarray(expected_src_assignment))
    np.testing.assert_array_equal(np.asarray(rows_per_expert), np.asarray(expected_rows_per_expert))
    assert int(dropped) == int(expected_dropped)


def test_pallas_mgpu_permute_dispatch_copy_tile_config_matches_reference_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("permute_mgpu requires H100/Hopper GPUs")

    tokens = 12
    hidden_dim = 256
    x = (jnp.arange(tokens * hidden_dim, dtype=jnp.float32).reshape(tokens, hidden_dim) / 1024.0).astype(jnp.bfloat16)
    selected_experts = jnp.array(
        [
            [0, 2],
            [1, 3],
            [2, 3],
            [0, 2],
            [1, 1],
            [3, 0],
            [0, 3],
            [2, 1],
            [3, 2],
            [1, 0],
            [2, 0],
            [3, 1],
        ],
        dtype=jnp.int32,
    )

    def shard_local(x_local: jax.Array, selected_experts_local: jax.Array):
        config = MoeMgpuConfig(capacity_factor=1.0, dispatch_chunk_copy_tile=256)
        actual = permute_mgpu(
            x_local,
            selected_experts_local,
            local_experts=2,
            config=config,
        )
        expected = permute_mgpu_reference(
            x_local,
            selected_experts_local,
            local_experts=2,
            config=config,
        )
        return (
            actual.recv_x,
            expected.recv_x,
            actual.recv_src_rank,
            expected.recv_src_rank,
            actual.recv_src_assignment,
            expected.recv_src_assignment,
            actual.rows_per_expert,
            expected.rows_per_expert,
            actual.dropped,
            expected.dropped,
        )

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        (
            recv_x,
            expected_recv_x,
            recv_src_rank,
            expected_src_rank,
            recv_src_assignment,
            expected_src_assignment,
            rows_per_expert,
            expected_rows_per_expert,
            dropped,
            expected_dropped,
        ) = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert", None)),
            out_specs=(
                P("expert", None),
                P("expert", None),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P(),
                P(),
            ),
            check_vma=False,
        )(
            x, selected_experts
        )

    np.testing.assert_array_equal(np.asarray(recv_x), np.asarray(expected_recv_x))
    np.testing.assert_array_equal(np.asarray(recv_src_rank), np.asarray(expected_src_rank))
    np.testing.assert_array_equal(np.asarray(recv_src_assignment), np.asarray(expected_src_assignment))
    np.testing.assert_array_equal(np.asarray(rows_per_expert), np.asarray(expected_rows_per_expert))
    assert int(dropped) == int(expected_dropped)


def test_permute_up_mgpu_reference_builds_hidden_layout():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")

    x = jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8) / 32.0
    selected_experts = jnp.stack(
        [
            jnp.arange(16, dtype=jnp.int32) % 4,
            (jnp.arange(16, dtype=jnp.int32) + 1) % 4,
        ],
        axis=1,
    )
    moe_w13 = jax.random.normal(jax.random.key(36), (4, 8, 24), dtype=jnp.float32) * 0.1

    def shard_local(x_local: jax.Array, selected_experts_local: jax.Array, moe_w13_local: jax.Array):
        layout = permute_up_mgpu_reference(
            x_local,
            selected_experts_local,
            moe_w13_local,
            local_experts=2,
            activation_fn=jax.nn.silu,
            config=MoeMgpuConfig(capacity_factor=1.0),
        )
        return layout.hidden, layout.recv_src_rank, layout.recv_src_assignment, layout.rows_per_expert, layout.dropped

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        moe_w13 = jax.sharding.reshard(moe_w13, NamedSharding(mesh, P("expert", None, None)))
        hidden, recv_src_rank, recv_src_assignment, rows_per_expert, dropped = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert", None), P("expert", None, None)),
            out_specs=(P("expert", None), P("expert"), P("expert"), P("expert"), P()),
            check_vma=False,
        )(x, selected_experts, moe_w13)

    assert hidden.shape == (2, 16, 12)
    assert jnp.isfinite(hidden).all()
    assert recv_src_rank.shape == (2, 16)
    assert recv_src_assignment.shape == (2, 16)
    np.testing.assert_array_equal(np.asarray(rows_per_expert).reshape(2, 2).sum(axis=1), np.array([16, 16]))
    assert int(dropped) == 0


def test_permute_up_mgpu_matches_reference_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("permute_up_mgpu requires H100/Hopper GPUs")

    tokens = 64
    hidden_dim = 128
    intermediate_dim = 128
    local_experts = 2
    num_experts = 4
    topk = 2
    x = jax.random.normal(jax.random.key(37), (tokens, hidden_dim), dtype=jnp.bfloat16) * 0.1
    token_ids = jnp.arange(tokens, dtype=jnp.int32)[:, None]
    route_slots = jnp.arange(topk, dtype=jnp.int32)[None, :]
    selected_experts = (token_ids + route_slots) % num_experts
    moe_w13 = jax.random.normal(
        jax.random.key(38),
        (num_experts, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.bfloat16,
    ) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    def shard_local(x_local: jax.Array, selected_experts_local: jax.Array, moe_w13_local: jax.Array):
        config = MoeMgpuConfig(capacity_factor=1.0)
        actual = permute_up_mgpu(
            x_local,
            selected_experts_local,
            moe_w13_local,
            local_experts=local_experts,
            activation_fn=jax.nn.silu,
            config=config,
        )
        expected = permute_up_mgpu_reference(
            x_local,
            selected_experts_local,
            moe_w13_local,
            local_experts=local_experts,
            activation_fn=jax.nn.silu,
            config=config,
        )
        return (
            actual.hidden,
            expected.hidden,
            actual.recv_src_rank,
            expected.recv_src_rank,
            actual.recv_src_assignment,
            expected.recv_src_assignment,
            actual.rows_per_expert,
            expected.rows_per_expert,
            actual.dropped,
            expected.dropped,
        )

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        moe_w13 = jax.sharding.reshard(moe_w13, NamedSharding(mesh, P("expert", None, None)))
        (
            hidden,
            expected_hidden,
            recv_src_rank,
            expected_src_rank,
            recv_src_assignment,
            expected_src_assignment,
            rows_per_expert,
            expected_rows_per_expert,
            dropped,
            expected_dropped,
        ) = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert", None), P("expert", None, None)),
            out_specs=(
                P("expert", None),
                P("expert", None),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P(),
                P(),
            ),
            check_vma=False,
        )(
            x, selected_experts, moe_w13
        )

    np.testing.assert_allclose(np.asarray(hidden), np.asarray(expected_hidden), rtol=1e-2, atol=0.1)
    np.testing.assert_array_equal(np.asarray(recv_src_rank), np.asarray(expected_src_rank))
    np.testing.assert_array_equal(np.asarray(recv_src_assignment), np.asarray(expected_src_assignment))
    np.testing.assert_array_equal(np.asarray(rows_per_expert), np.asarray(expected_rows_per_expert))
    assert int(dropped) == int(expected_dropped)


def test_permute_up_mgpu_chunked_matches_staged_on_balanced_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("permute_up_mgpu requires H100/Hopper GPUs")

    tokens = 256
    hidden_dim = 128
    intermediate_dim = 128
    local_experts = 2
    num_experts = 4
    topk = 2
    x = jax.random.normal(jax.random.key(51), (tokens, hidden_dim), dtype=jnp.bfloat16) * 0.1
    selected_experts = jnp.tile(
        jnp.arange(num_experts, dtype=jnp.int32),
        tokens * topk // num_experts,
    ).reshape(tokens, topk)
    moe_w13 = jax.random.normal(
        jax.random.key(52),
        (num_experts, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.bfloat16,
    ) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    def shard_local(x_local: jax.Array, selected_experts_local: jax.Array, moe_w13_local: jax.Array):
        base_config = MoeMgpuConfig(
            block_m=64,
            block_n=128,
            block_k=64,
            capacity_factor=1.0,
            dispatch_expert_group_size=2,
            dispatch_chunked_permute_up=False,
        )
        chunked_config = MoeMgpuConfig(
            block_m=64,
            block_n=128,
            block_k=64,
            capacity_factor=1.0,
            dispatch_expert_group_size=2,
            dispatch_chunked_permute_up=True,
        )
        expected = permute_up_mgpu(
            x_local,
            selected_experts_local,
            moe_w13_local,
            local_experts=local_experts,
            activation_fn=jax.nn.silu,
            config=base_config,
        )
        actual = permute_up_mgpu(
            x_local,
            selected_experts_local,
            moe_w13_local,
            local_experts=local_experts,
            activation_fn=jax.nn.silu,
            config=chunked_config,
        )
        return (
            actual.hidden,
            expected.hidden,
            actual.recv_src_rank,
            expected.recv_src_rank,
            actual.recv_src_assignment,
            expected.recv_src_assignment,
            actual.rows_per_expert,
            expected.rows_per_expert,
            actual.dropped,
            expected.dropped,
        )

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        moe_w13 = jax.sharding.reshard(moe_w13, NamedSharding(mesh, P("expert", None, None)))
        (
            hidden,
            expected_hidden,
            recv_src_rank,
            expected_src_rank,
            recv_src_assignment,
            expected_src_assignment,
            rows_per_expert,
            expected_rows_per_expert,
            dropped,
            expected_dropped,
        ) = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert", None), P("expert", None, None)),
            out_specs=(
                P("expert", None),
                P("expert", None),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P("expert"),
                P(),
                P(),
            ),
            check_vma=False,
        )(
            x, selected_experts, moe_w13
        )

    np.testing.assert_allclose(np.asarray(hidden), np.asarray(expected_hidden), rtol=1e-2, atol=0.1)
    np.testing.assert_array_equal(np.asarray(recv_src_rank), np.asarray(expected_src_rank))
    np.testing.assert_array_equal(np.asarray(recv_src_assignment), np.asarray(expected_src_assignment))
    np.testing.assert_array_equal(np.asarray(rows_per_expert), np.asarray(expected_rows_per_expert))
    assert int(dropped) == int(expected_dropped)


def test_ragged_w2_reference_matches_manual_grouped_matmul_with_padding():
    hidden = jnp.arange(10 * 4, dtype=jnp.float32).reshape(10, 4) / 10.0
    moe_w2 = jnp.arange(3 * 4 * 6, dtype=jnp.float32).reshape(3, 4, 6) / 25.0
    rows_per_expert = jnp.array([3, 0, 5], dtype=jnp.int32)

    actual = ragged_w2_reference(hidden, moe_w2, rows_per_expert)

    expected = []
    start = 0
    group_sizes = [3, 0, 7]
    for expert, group_size in enumerate(group_sizes):
        end = start + group_size
        if group_size:
            expected.append(hidden[start:end] @ moe_w2[expert])
        start = end
    expected = jnp.concatenate(expected, axis=0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_ragged_w2_mgpu_matches_reference_on_hopper_when_available():
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("ragged_w2_mgpu requires H100/Hopper GPUs")

    capacity = 128
    intermediate_dim = 64
    hidden_dim = 128
    local_experts = 3
    config = MoeMgpuConfig(block_m=64, block_n=128, block_k=64, capacity_factor=1.0)
    hidden = jax.random.normal(jax.random.key(39), (capacity, intermediate_dim), dtype=jnp.bfloat16) * 0.1
    moe_w2 = jax.random.normal(
        jax.random.key(40),
        (local_experts, intermediate_dim, hidden_dim),
        dtype=jnp.bfloat16,
    ) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    rows_per_expert = jnp.array([0, 65, 47], dtype=jnp.int32)

    @jax.jit
    def run(hidden: jax.Array, moe_w2: jax.Array, rows_per_expert: jax.Array):
        return (
            ragged_w2_mgpu(hidden, moe_w2, rows_per_expert, config=config),
            ragged_w2_reference(hidden, moe_w2, rows_per_expert),
        )

    actual, expected = run(hidden, moe_w2, rows_per_expert)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=0.1)


def test_unpermute_mgpu_reference_combines_return_slots_in_route_order():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")

    y_dispatch = jnp.array(
        [
            [[1, 10], [2, 20], [3, 30], [0, 0]],
            [[4, 40], [5, 50], [6, 60], [0, 0]],
        ],
        dtype=jnp.float32,
    ).reshape(8, 2)
    recv_src_rank = jnp.array([[0, 0, 1, -1], [0, 1, 1, -1]], dtype=jnp.int32).reshape(8)
    recv_src_assignment = jnp.array([[0, 3, 0, -1], [1, 2, 3, -1]], dtype=jnp.int32).reshape(8)
    combine_weights = jnp.array(
        [
            [[1.0, 0.25], [0.5, 2.0]],
            [[0.25, 1.0], [2.0, 0.5]],
        ],
        dtype=jnp.float32,
    ).reshape(4, 2)

    def shard_local(
        y_dispatch_local: jax.Array,
        recv_src_rank_local: jax.Array,
        recv_src_assignment_local: jax.Array,
        combine_weights_local: jax.Array,
    ):
        return unpermute_mgpu_reference(
            y_dispatch_local,
            recv_src_rank_local,
            recv_src_assignment_local,
            combine_weights_local,
        )

    with jax.set_mesh(mesh):
        y_dispatch = jax.sharding.reshard(y_dispatch, NamedSharding(mesh, P("expert", None)))
        recv_src_rank = jax.sharding.reshard(recv_src_rank, NamedSharding(mesh, P("expert")))
        recv_src_assignment = jax.sharding.reshard(recv_src_assignment, NamedSharding(mesh, P("expert")))
        combine_weights = jax.sharding.reshard(combine_weights, NamedSharding(mesh, P("expert", None)))
        out = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert"), P("expert"), P("expert", None)),
            out_specs=P("expert", None),
            check_vma=False,
        )(y_dispatch, recv_src_rank, recv_src_assignment, combine_weights)

    expected = np.array(
        [
            [[2.0, 20.0], [4.0, 40.0]],
            [[0.75, 7.5], [13.0, 130.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(out).reshape(2, 2, 2), expected, rtol=1e-5, atol=1e-5)


def test_unpermute_mgpu_reference_ignores_invalid_rows_and_zero_weight_routes():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")

    y_dispatch = jnp.array(
        [
            [[10, 1], [20, 2], [999, 999]],
            [[30, 3], [40, 4], [50, 5]],
        ],
        dtype=jnp.float32,
    ).reshape(6, 2)
    recv_src_rank = jnp.array([[0, 1, -1], [0, 0, 1]], dtype=jnp.int32).reshape(6)
    recv_src_assignment = jnp.array([[0, 2, -1], [1, 3, 1]], dtype=jnp.int32).reshape(6)
    combine_weights = jnp.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[0.0, 3.0], [4.0, 0.0]],
        ],
        dtype=jnp.float32,
    ).reshape(4, 2)

    def shard_local(
        y_dispatch_local: jax.Array,
        recv_src_rank_local: jax.Array,
        recv_src_assignment_local: jax.Array,
        combine_weights_local: jax.Array,
    ):
        return unpermute_mgpu_reference(
            y_dispatch_local,
            recv_src_rank_local,
            recv_src_assignment_local,
            combine_weights_local,
        )

    with jax.set_mesh(mesh):
        y_dispatch = jax.sharding.reshard(y_dispatch, NamedSharding(mesh, P("expert", None)))
        recv_src_rank = jax.sharding.reshard(recv_src_rank, NamedSharding(mesh, P("expert")))
        recv_src_assignment = jax.sharding.reshard(recv_src_assignment, NamedSharding(mesh, P("expert")))
        combine_weights = jax.sharding.reshard(combine_weights, NamedSharding(mesh, P("expert", None)))
        out = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert"), P("expert"), P("expert", None)),
            out_specs=P("expert", None),
            check_vma=False,
        )(y_dispatch, recv_src_rank, recv_src_assignment, combine_weights)

    expected = np.array(
        [
            [[10.0, 1.0], [80.0, 8.0]],
            [[150.0, 15.0], [80.0, 8.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(out).reshape(2, 2, 2), expected, rtol=1e-5, atol=1e-5)


def test_unpermute_mgpu_matches_reference_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("unpermute_mgpu requires H100/Hopper GPUs")

    tokens = 8
    topk = 2
    hidden_dim = 8
    local_experts = 2
    x = jax.random.normal(jax.random.key(41), (tokens, hidden_dim), dtype=jnp.bfloat16)
    token_ids = jnp.arange(tokens, dtype=jnp.int32)[:, None]
    route_slots = jnp.arange(topk, dtype=jnp.int32)[None, :]
    selected_experts = (token_ids + route_slots) % (2 * local_experts)
    combine_weights = jax.nn.softmax(
        jax.random.normal(jax.random.key(42), (tokens, topk), dtype=jnp.float32),
        axis=-1,
    ).astype(jnp.bfloat16)

    def shard_local(x_local: jax.Array, selected_experts_local: jax.Array, combine_weights_local: jax.Array):
        layout = permute_mgpu_reference(
            x_local,
            selected_experts_local,
            local_experts=local_experts,
            config=MoeMgpuConfig(capacity_factor=1.0),
        )
        rank = jax.lax.axis_index("expert")
        row_ids = jnp.arange(layout.recv_x.shape[0] * hidden_dim, dtype=jnp.float32).reshape(
            layout.recv_x.shape[0],
            hidden_dim,
        )
        y_dispatch = (row_ids + 100 * rank).astype(jnp.bfloat16)
        actual = unpermute_mgpu(
            y_dispatch,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            combine_weights_local,
        )
        expected = unpermute_mgpu_reference(
            y_dispatch,
            layout.recv_src_rank,
            layout.recv_src_assignment,
            combine_weights_local,
        )
        return actual, expected

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        combine_weights = jax.sharding.reshard(combine_weights, NamedSharding(mesh, P("expert", None)))
        actual, expected = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(P("expert", None), P("expert", None), P("expert", None)),
            out_specs=(P("expert", None), P("expert", None)),
            check_vma=False,
        )(x, selected_experts, combine_weights)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=0.1)


def test_down_unpermute_mgpu_matches_reference_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("down_unpermute_mgpu requires H100/Hopper GPUs")

    tokens = 64
    hidden_dim = 128
    intermediate_dim = 128
    local_experts = 2
    num_experts = 4
    topk = 2
    config = MoeMgpuConfig(capacity_factor=1.0)
    x, selected_experts, combine_weights, moe_w13, moe_w2 = _make_inputs(
        key=jax.random.key(55),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)
    moe_w13 = moe_w13.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    moe_w2 = moe_w2.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    def shard_local(
        x_local: jax.Array,
        selected_experts_local: jax.Array,
        combine_weights_local: jax.Array,
        moe_w13_local: jax.Array,
        moe_w2_local: jax.Array,
    ):
        up_layout = permute_up_mgpu_reference(
            x_local,
            selected_experts_local,
            moe_w13_local,
            local_experts=local_experts,
            activation_fn=jax.nn.silu,
            config=config,
        )
        actual = down_unpermute_mgpu(
            up_layout.hidden,
            up_layout.recv_src_rank,
            up_layout.recv_src_assignment,
            up_layout.rows_per_expert,
            moe_w2_local,
            combine_weights_local,
            selected_experts_local,
            config=config,
        )
        y_dispatch = ragged_w2_reference(up_layout.hidden, moe_w2_local, up_layout.rows_per_expert)
        expected = unpermute_mgpu_reference(
            y_dispatch,
            up_layout.recv_src_rank,
            up_layout.recv_src_assignment,
            combine_weights_local,
        )
        return actual, expected

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        combine_weights = jax.sharding.reshard(combine_weights, NamedSharding(mesh, P("expert", None)))
        moe_w13 = jax.sharding.reshard(moe_w13, NamedSharding(mesh, P("expert", None, None)))
        moe_w2 = jax.sharding.reshard(moe_w2, NamedSharding(mesh, P("expert", None, None)))
        actual, expected = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=(P("expert", None), P("expert", None)),
            check_vma=False,
        )(x, selected_experts, combine_weights, moe_w13, moe_w2)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=0.1)


def test_moe_mlp_pallas_mgpu_staged_matches_reference_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("moe_mlp_pallas_mgpu_staged requires H100/Hopper GPUs")

    tokens = 64
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = 4
    topk = 2
    config = MoeMgpuConfig(capacity_factor=1.0)
    x, selected_experts, combine_weights, moe_w13, moe_w2 = _make_inputs(
        key=jax.random.key(43),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)
    moe_w13 = moe_w13.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    moe_w2 = moe_w2.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    def shard_local(
        x_local: jax.Array,
        selected_experts_local: jax.Array,
        combine_weights_local: jax.Array,
        moe_w13_local: jax.Array,
        moe_w2_local: jax.Array,
    ):
        actual, actual_dropped = moe_mlp_pallas_mgpu_staged(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=jax.nn.silu,
            config=config,
        )
        expected, expected_dropped = moe_mlp_pallas_mgpu_reference(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=jax.nn.silu,
            config=config,
            expert_axis="expert",
            num_experts=num_experts,
        )
        return actual, expected, actual_dropped, expected_dropped

    with jax.set_mesh(mesh):
        x = jax.sharding.reshard(x, NamedSharding(mesh, P("expert", None)))
        selected_experts = jax.sharding.reshard(selected_experts, NamedSharding(mesh, P("expert", None)))
        combine_weights = jax.sharding.reshard(combine_weights, NamedSharding(mesh, P("expert", None)))
        moe_w13 = jax.sharding.reshard(moe_w13, NamedSharding(mesh, P("expert", None, None)))
        moe_w2 = jax.sharding.reshard(moe_w2, NamedSharding(mesh, P("expert", None, None)))
        actual, expected, actual_dropped, expected_dropped = jax.shard_map(
            shard_local,
            mesh=mesh,
            in_specs=(
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=(P("expert", None), P("expert", None), P(), P()),
            check_vma=False,
        )(x, selected_experts, combine_weights, moe_w13, moe_w2)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-2, atol=0.2)
    assert int(actual_dropped) == int(expected_dropped)


def test_moe_mlp_pallas_mgpu_reference_matches_scatter_without_ep_axis():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(31),
        tokens=12,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )

    expected, expected_dropped = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=ActivationFunctionEnum.silu,
        implementation="scatter",
        mesh=None,
        report_capacity_overflow=True,
    )
    actual, actual_dropped = moe_mlp_pallas_mgpu_reference(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation_fn=jax.nn.silu,
        config=MoeMgpuConfig(),
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(actual_dropped) == int(expected_dropped)


def test_moe_mlp_pallas_mgpu_reference_reports_capacity_drops_without_ep_axis():
    tokens = 8
    topk = 2
    hidden_dim = 8
    intermediate_dim = 12
    num_experts = 4
    x = jax.random.normal(jax.random.key(32), (tokens, hidden_dim), dtype=jnp.float32)
    selected_experts = jnp.zeros((tokens, topk), dtype=jnp.int32)
    combine_weights = jnp.full((tokens, topk), 0.5, dtype=jnp.float32)
    w_up_gate = jax.random.normal(
        jax.random.key(33), (num_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.float32
    )
    w_down = jax.random.normal(jax.random.key(34), (num_experts, intermediate_dim, hidden_dim), dtype=jnp.float32)

    out, dropped = moe_mlp_pallas_mgpu_reference(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation_fn=jax.nn.silu,
        config=MoeMgpuConfig(capacity_factor=0.25),
    )

    assert out.shape == (tokens, hidden_dim)
    assert int(dropped) > 0


def test_moe_mlp_pallas_mgpu_reference_rejects_static_shape_mismatch_without_ep_axis():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(74),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )

    with pytest.raises(ValueError, match="combine_weights must have the same"):
        moe_mlp_pallas_mgpu_reference(
            x,
            selected_experts,
            combine_weights[:, :1],
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            config=MoeMgpuConfig(),
        )

    with pytest.raises(ValueError, match="output dimension must be even"):
        moe_mlp_pallas_mgpu_reference(
            x,
            selected_experts,
            combine_weights,
            jnp.zeros((4, 16, 49), dtype=jnp.float32),
            jnp.zeros((4, 24, 16), dtype=jnp.float32),
            activation_fn=jax.nn.silu,
            config=MoeMgpuConfig(),
        )

    with pytest.raises(ValueError, match="combine_weights must have the same"):
        moe_mlp_pallas_mgpu_reference(
            x,
            selected_experts,
            combine_weights[:, :1],
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            config=MoeMgpuConfig(),
            expert_axis="expert",
            num_experts=4,
        )


@pytest.mark.parametrize(
    ("x_shape", "selected_shape", "w13_shape", "w2_shape", "message"),
    [
        ((0, 16), (0, 2), (4, 16, 48), (4, 24, 16), "positive token dimension"),
        ((8, 16), (8, 0), (4, 16, 48), (4, 24, 16), "positive top-k route dimension"),
        ((8, 16), (8, 2), (0, 16, 48), (0, 24, 16), "positive local expert dimension"),
        ((8, 0), (8, 2), (4, 0, 48), (4, 24, 0), "positive hidden dimension"),
        ((8, 16), (8, 2), (4, 16, 0), (4, 0, 16), "positive intermediate dimension"),
    ],
)
def test_moe_mlp_pallas_mgpu_reference_rejects_empty_static_shapes_without_backend(
    x_shape,
    selected_shape,
    w13_shape,
    w2_shape,
    message,
):
    x = jnp.zeros(x_shape, dtype=jnp.float32)
    selected_experts = jnp.zeros(selected_shape, dtype=jnp.int32)
    combine_weights = jnp.zeros(selected_shape, dtype=jnp.float32)
    w_up_gate = jnp.zeros(w13_shape, dtype=jnp.float32)
    w_down = jnp.zeros(w2_shape, dtype=jnp.float32)

    with pytest.raises(ValueError, match=message):
        moe_mlp_pallas_mgpu_reference(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            config=MoeMgpuConfig(),
        )


def test_moe_mlp_pallas_mgpu_requires_expert_axis():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(35),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )

    with pytest.raises(ValueError, match="requires an expert mesh axis"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=None,
        )


def test_moe_mlp_ordered_implementation_falls_back_after_pallas_mgpu_validation_failure():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(62),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )

    expected = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=ActivationFunctionEnum.silu,
        implementation="scatter",
        mesh=None,
    )
    with pytest.warns(RuntimeWarning, match="implementation 'pallas_mgpu' failed; trying next fallback"):
        actual = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation=["pallas_mgpu", "scatter"],
            mesh=None,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_moe_mlp_ordered_implementation_preserves_capacity_overflow_report_after_fallback():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(70),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )

    expected, expected_dropped = moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=ActivationFunctionEnum.silu,
        implementation="scatter",
        mesh=None,
        report_capacity_overflow=True,
    )
    with pytest.warns(RuntimeWarning, match="implementation 'pallas_mgpu' failed; trying next fallback"):
        actual, actual_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation=("pallas_mgpu", "scatter"),
            mesh=None,
            report_capacity_overflow=True,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(actual_dropped) == int(expected_dropped)


def test_moe_mlp_ordered_implementation_raises_when_all_fallbacks_fail():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(71),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.warns(RuntimeWarning, match="implementation 'scatter' failed; trying next fallback"):
        with pytest.raises(RuntimeError, match="No requested MoE implementation succeeded") as exc_info:
            moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation=ActivationFunctionEnum.silu,
                implementation=("scatter", "sonic"),
                mesh=mesh,
            )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_moe_mlp_ordered_implementation_rejects_empty_sequence():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(63),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )

    with pytest.raises(ValueError, match="implementation sequence must contain at least one implementation"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation=(),
            mesh=None,
        )


def test_moe_mlp_pallas_mgpu_rejects_data_axis_parallelism_before_backend():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(36),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=2, expert=2, model=1)

    with pytest.raises(ValueError, match="requires data mesh axis size 1"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


def test_moe_mlp_pallas_mgpu_rejects_ep_size_above_single_node_limit_before_backend():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(57),
        tokens=16,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=16,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=16, model=1)

    with pytest.raises(ValueError, match="expert axis size <= 8"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


def test_moe_mlp_pallas_mgpu_rejects_num_experts_not_divisible_by_ep_before_backend():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(61),
        tokens=8,
        hidden_dim=128,
        intermediate_dim=128,
        num_experts=5,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match="num_experts=5 must be divisible by expert axis size=2"):
        moe_mlp(
            x.astype(jnp.bfloat16),
            selected_experts,
            combine_weights.astype(jnp.bfloat16),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


def test_moe_mlp_pallas_mgpu_rejects_non_bf16_activations_before_backend():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(58),
        tokens=8,
        hidden_dim=128,
        intermediate_dim=128,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match="requires bfloat16 activations and weights"):
        moe_mlp(
            x.astype(jnp.float32),
            selected_experts,
            combine_weights.astype(jnp.bfloat16),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


def test_moe_mlp_pallas_mgpu_rejects_invalid_public_route_dtypes_before_backend():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(59),
        tokens=8,
        hidden_dim=128,
        intermediate_dim=128,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match="selected_experts dtype int32"):
        moe_mlp(
            x.astype(jnp.bfloat16),
            selected_experts.astype(jnp.float32),
            combine_weights.astype(jnp.bfloat16),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )

    with pytest.raises(ValueError, match="combine_weights dtype bfloat16 or float32"):
        moe_mlp(
            x.astype(jnp.bfloat16),
            selected_experts,
            combine_weights.astype(jnp.int32),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


@pytest.mark.parametrize(
    ("key_seed", "hidden_dim", "intermediate_dim", "expected_error"),
    [
        (72, 96, 128, "dispatch_chunk_copy_tile=128"),
        (73, 128, 96, "block_n=128"),
    ],
)
def test_moe_mlp_pallas_mgpu_rejects_public_tile_mismatches_before_backend(
    key_seed: int,
    hidden_dim: int,
    intermediate_dim: int,
    expected_error: str,
):
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(key_seed),
        tokens=8,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match=expected_error):
        moe_mlp(
            x.astype(jnp.bfloat16),
            selected_experts,
            combine_weights.astype(jnp.bfloat16),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


def test_moe_mlp_pallas_mgpu_rejects_missing_local_gpu_before_backend():
    if any(device.platform == "gpu" for device in jax.local_devices()):
        pytest.skip("missing-GPU validation requires a CPU-only local runtime")

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(74),
        tokens=8,
        hidden_dim=128,
        intermediate_dim=128,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match="requires a GPU backend"):
        moe_mlp(
            x.astype(jnp.bfloat16),
            selected_experts,
            combine_weights.astype(jnp.bfloat16),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


def test_moe_mlp_pallas_mgpu_rejects_mixed_local_gpu_topology_before_backend(monkeypatch):
    class FakeDevice:
        platform = "gpu"

        def __init__(self, device_kind: str):
            self.device_kind = device_kind

    monkeypatch.setattr(
        jax,
        "local_devices",
        lambda: [FakeDevice("NVIDIA H100 80GB HBM3"), FakeDevice("NVIDIA A100-SXM4-80GB")],
    )

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(75),
        tokens=8,
        hidden_dim=128,
        intermediate_dim=128,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match="all participating local GPU devices to be Hopper/H100"):
        moe_mlp(
            x.astype(jnp.bfloat16),
            selected_experts,
            combine_weights.astype(jnp.bfloat16),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


@pytest.mark.parametrize(
    ("x_shape", "selected_shape", "w_up_gate_shape", "w_down_shape", "message"),
    [
        ((0, 128), (0, 2), (4, 128, 256), (4, 128, 128), "positive token dimension"),
        ((8, 128), (8, 0), (4, 128, 256), (4, 128, 128), "positive top-k route dimension"),
        ((8, 128), (8, 2), (0, 128, 256), (0, 128, 128), "positive expert dimension"),
        ((8, 0), (8, 2), (4, 0, 256), (4, 128, 0), "positive hidden dimension"),
        ((8, 128), (8, 2), (4, 128, 0), (4, 0, 128), "positive intermediate dimension"),
    ],
)
def test_moe_mlp_pallas_mgpu_rejects_empty_public_static_dimensions_before_backend(
    x_shape: tuple[int, int],
    selected_shape: tuple[int, int],
    w_up_gate_shape: tuple[int, int, int],
    w_down_shape: tuple[int, int, int],
    message: str,
):
    x = jnp.zeros(x_shape, dtype=jnp.bfloat16)
    selected_experts = jnp.zeros(selected_shape, dtype=jnp.int32)
    combine_weights = jnp.zeros(selected_shape, dtype=jnp.bfloat16)
    w_up_gate = jnp.zeros(w_up_gate_shape, dtype=jnp.bfloat16)
    w_down = jnp.zeros(w_down_shape, dtype=jnp.bfloat16)
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match=message):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=mesh,
        )


def test_moe_mlp_ep_rejects_non_positive_capacity_factor_before_backend():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(60),
        tokens=8,
        hidden_dim=32,
        intermediate_dim=64,
        num_experts=4,
        topk=2,
    )
    mesh = _make_abstract_moe_mesh(data=1, expert=2, model=1)

    with pytest.raises(ValueError, match="capacity_factor must be positive"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="ragged_all_to_all",
            mesh=mesh,
            capacity_factor=0.0,
        )


@pytest.mark.parametrize(
    ("malformed_argument", "message"),
    [
        ("combine_weight_shape", "combine_weights.*same \\[T, K\\] shape"),
        ("combine_weight_dtype", "combine_weights.*bfloat16 or float32"),
        ("selected_expert_dtype", "selected_experts dtype int32"),
        ("activation_dtype", "requires bfloat16 activations and weights"),
        ("w2_shape", "moe_w2 must have shape"),
    ],
)
def test_moe_mlp_pallas_mgpu_rejects_invalid_direct_entrypoint_inputs_before_backend(
    malformed_argument: str,
    message: str,
):
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(49),
        tokens=8,
        hidden_dim=128,
        intermediate_dim=128,
        num_experts=4,
        topk=2,
    )

    x = x.astype(jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)
    w_up_gate = w_up_gate.astype(jnp.bfloat16)
    w_down = w_down.astype(jnp.bfloat16)
    if malformed_argument == "combine_weight_shape":
        combine_weights = combine_weights[:, :1]
    elif malformed_argument == "combine_weight_dtype":
        combine_weights = jnp.ones(selected_experts.shape, dtype=jnp.int32)
    elif malformed_argument == "selected_expert_dtype":
        selected_experts = selected_experts.astype(jnp.float32)
    elif malformed_argument == "activation_dtype":
        x = x.astype(jnp.float32)
    elif malformed_argument == "w2_shape":
        w_down = w_down[:, :-1, :]
    else:
        raise AssertionError(f"unhandled malformed argument {malformed_argument}")

    with pytest.raises(ValueError, match=message):
        moe_mlp_pallas_mgpu(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
        )


@pytest.mark.parametrize(
    ("x", "selected_experts", "combine_weights", "w_up_gate", "w_down", "message"),
    [
        (
            jnp.zeros((0, 128), dtype=jnp.bfloat16),
            jnp.zeros((0, 2), dtype=jnp.int32),
            jnp.zeros((0, 2), dtype=jnp.bfloat16),
            jnp.zeros((4, 128, 256), dtype=jnp.bfloat16),
            jnp.zeros((4, 128, 128), dtype=jnp.bfloat16),
            "positive token dimension",
        ),
        (
            jnp.zeros((8, 128), dtype=jnp.bfloat16),
            jnp.zeros((8, 0), dtype=jnp.int32),
            jnp.zeros((8, 0), dtype=jnp.bfloat16),
            jnp.zeros((4, 128, 256), dtype=jnp.bfloat16),
            jnp.zeros((4, 128, 128), dtype=jnp.bfloat16),
            "positive top-k route dimension",
        ),
        (
            jnp.zeros((8, 0), dtype=jnp.bfloat16),
            jnp.zeros((8, 2), dtype=jnp.int32),
            jnp.zeros((8, 2), dtype=jnp.bfloat16),
            jnp.zeros((4, 0, 256), dtype=jnp.bfloat16),
            jnp.zeros((4, 128, 0), dtype=jnp.bfloat16),
            "positive hidden dimension",
        ),
        (
            jnp.zeros((8, 128), dtype=jnp.bfloat16),
            jnp.zeros((8, 2), dtype=jnp.int32),
            jnp.zeros((8, 2), dtype=jnp.bfloat16),
            jnp.zeros((4, 128, 0), dtype=jnp.bfloat16),
            jnp.zeros((4, 0, 128), dtype=jnp.bfloat16),
            "positive intermediate dimension",
        ),
    ],
)
def test_moe_mlp_pallas_mgpu_rejects_empty_static_route_shapes_before_backend(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        moe_mlp_pallas_mgpu(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
        )


def test_moe_mlp_pallas_mgpu_rejects_empty_local_expert_axis_before_backend():
    x = jnp.zeros((8, 128), dtype=jnp.bfloat16)
    selected_experts = jnp.zeros((8, 2), dtype=jnp.int32)
    combine_weights = jnp.zeros((8, 2), dtype=jnp.bfloat16)
    w_up_gate = jnp.zeros((0, 128, 256), dtype=jnp.bfloat16)
    w_down = jnp.zeros((0, 128, 128), dtype=jnp.bfloat16)

    with pytest.raises(ValueError, match="positive local expert dimension"):
        moe_mlp_pallas_mgpu(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
        )


@pytest.mark.parametrize(
    ("hidden_dim", "intermediate_dim", "config", "message"),
    [
        (128, 128, MoeMgpuConfig(dispatch_chunk_copy_tile=96), "dispatch_chunk_copy_tile=96"),
        (
            128,
            128,
            MoeMgpuConfig(dispatch_copy_schedule="expert_group_peer", dispatch_expert_group_size=3),
            "E_local to be divisible by dispatch_expert_group_size",
        ),
        (130, 128, MoeMgpuConfig(dispatch_chunk_copy_tile=1), "block_k=64, got D=130"),
        (128, 130, MoeMgpuConfig(), "block_n=128, got I=130"),
    ],
)
def test_moe_mlp_pallas_mgpu_rejects_tile_config_mismatch_before_backend(
    hidden_dim: int,
    intermediate_dim: int,
    config: MoeMgpuConfig,
    message: str,
):
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(52),
        tokens=8,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=4,
        topk=2,
    )

    with pytest.raises(ValueError, match=message):
        moe_mlp_pallas_mgpu(
            x.astype(jnp.bfloat16),
            selected_experts,
            combine_weights.astype(jnp.bfloat16),
            w_up_gate.astype(jnp.bfloat16),
            w_down.astype(jnp.bfloat16),
            activation_fn=jax.nn.silu,
            config=config,
        )


def test_pallas_mgpu_receiver_capacity_pads_default_like_small_shapes():
    assert MoeMgpuConfig().capacity_factor == _DEFAULT_EP_CAPACITY_FACTOR
    assert _receiver_capacity(tokens_per_shard=4, topk=2, local_experts=2, capacity_factor=1.25) == 16
    assert _receiver_capacity(tokens_per_shard=8, topk=4, local_experts=4, capacity_factor=1.25) == 40
    assert _receiver_capacity(tokens_per_shard=3, topk=1, local_experts=4, capacity_factor=1.0) == 4
    assert (
        _effective_padded_capacity_factor(
            tokens_per_shard=4,
            topk=2,
            local_experts=2,
            capacity_factor=1.25,
        )
        == 2.0
    )


def test_pallas_mgpu_config_rejects_nondeterministic_mode():
    with pytest.raises(ValueError, match="deterministic=True"):
        MoeMgpuConfig(deterministic=False)


def test_pallas_mgpu_tuned_config_infers_h100_bf16_bucket_defaults():
    config = infer_moe_mgpu_config(
        hidden_dim=2560,
        intermediate_dim=1280,
        ep_size=8,
        dtype=jnp.bfloat16,
        capacity_factor=1.25,
    )

    assert config.max_concurrent_steps == 4
    assert config.grid_block_n == 2
    assert config.dispatch_expert_group_size == 8
    assert config.dispatch_fuse_metadata
    assert config.capacity_factor == 1.25


def test_pallas_mgpu_tuned_config_preserves_capacity_factor_for_matched_bucket():
    config = infer_moe_mgpu_config(
        hidden_dim=2560,
        intermediate_dim=1280,
        ep_size=8,
        dtype=jnp.bfloat16,
        capacity_factor=0.75,
    )

    assert config == MoeMgpuConfig(capacity_factor=0.75)


def test_pallas_mgpu_tuned_config_falls_back_and_preserves_capacity_factor():
    config = infer_moe_mgpu_config(
        hidden_dim=2560,
        intermediate_dim=1280,
        ep_size=8,
        dtype=jnp.float32,
        capacity_factor=0.75,
    )

    assert config == MoeMgpuConfig(capacity_factor=0.75)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_m": 0}, "block_m must be positive"),
        ({"block_n": 0}, "block_n must be positive"),
        ({"block_k": 0}, "block_k must be positive"),
        ({"max_concurrent_steps": 0}, "max_concurrent_steps must be positive"),
        ({"grid_block_n": 0}, "grid_block_n must be positive"),
        ({"capacity_factor": 0.0}, "capacity_factor must be positive"),
        ({"num_sms": 0}, "num_sms must be positive"),
        ({"dispatch_expert_group_size": 0}, "dispatch_expert_group_size must be positive"),
        ({"dispatch_chunk_copy_tile": 0}, "dispatch_chunk_copy_tile must be positive"),
        ({"dispatch_chunk_copy_rows": 0}, "dispatch_chunk_copy_rows must be positive"),
        ({"combine_bwd_block_n": 0}, "combine_bwd_block_n must be positive"),
        ({"dx_unpermute_block_n": 0}, "dx_unpermute_block_n must be positive"),
    ],
)
def test_pallas_mgpu_config_rejects_non_positive_numeric_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MoeMgpuConfig(**kwargs)


def test_pallas_mgpu_config_rejects_unknown_dispatch_copy_schedule():
    with pytest.raises(ValueError, match="unknown dispatch_copy_schedule"):
        MoeMgpuConfig(dispatch_copy_schedule="per_expert")


def test_pallas_mgpu_config_rejects_split_wg_without_chunked_permute_up():
    with pytest.raises(ValueError, match="dispatch_split_wg_permute_up requires dispatch_chunked_permute_up"):
        MoeMgpuConfig(dispatch_split_wg_permute_up=True)


def test_pallas_mgpu_config_rejects_overlap_without_split_wg_permute_up():
    with pytest.raises(ValueError, match="dispatch_split_wg_overlap_permute_up requires dispatch_split_wg_permute_up"):
        MoeMgpuConfig(dispatch_chunked_permute_up=True, dispatch_split_wg_overlap_permute_up=True)


def test_moe_mgpu_dispatch_w13_activation_pads_non_tile_aligned_rows_on_hopper_when_available():
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    tokens = 10
    hidden_dim = 128
    intermediate_dim = 128
    local_experts = 2
    group_sizes = jnp.array([5, 5], dtype=jnp.int32)
    tokens_sorted = jax.random.normal(jax.random.key(47), (tokens, hidden_dim), dtype=jnp.bfloat16) * 0.1
    moe_w13 = (
        jax.random.normal(
            jax.random.key(48),
            (local_experts, hidden_dim, 2 * intermediate_dim),
            dtype=jnp.bfloat16,
        )
        * 0.1
    )
    metadata = _MoeMgpuUpMetadata(global_expert_counts=group_sizes[None, :])

    with pytest.warns(RuntimeWarning, match="padded WGMMA M dimension"):
        actual = _moe_mgpu_dispatch_w13_activation(
            tokens_sorted,
            moe_w13,
            activation_fn=jax.nn.silu,
            metadata=metadata,
            config=MoeMgpuConfig(),
        )

    gate = ragged_dot(tokens_sorted, moe_w13[..., :intermediate_dim], group_sizes)
    up = ragged_dot(tokens_sorted, moe_w13[..., intermediate_dim:], group_sizes)
    expected = jax.nn.silu(gate) * up
    assert actual.shape == (tokens, intermediate_dim)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=5e-2, atol=1e-2)


def test_moe_mlp_pallas_mgpu_matches_ragged_a2a_on_hopper_when_available():
    mesh = _make_expert_only_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    ep_size = mesh.shape["expert"]
    tokens = ep_size * 4
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = ep_size * 2
    topk = 2

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(36),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)
    w_up_gate = w_up_gate.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    w_down = w_down.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        expected, expected_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="ragged_all_to_all",
            mesh=None,
            capacity_factor=1.0,
            report_capacity_overflow=True,
        )
        actual, actual_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=None,
            capacity_factor=1.0,
            report_capacity_overflow=True,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=0.1)
    assert int(actual_dropped) == int(expected_dropped)


def test_moe_mlp_pallas_mgpu_accepts_fp32_combine_weights_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    ep_size = mesh.shape["expert"]
    tokens = ep_size * 4
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = ep_size * 2
    topk = 2

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(65),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.float32)
    w_up_gate = w_up_gate.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    w_down = w_down.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P("expert", None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        expected, expected_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="ragged_all_to_all",
            mesh=None,
            capacity_factor=1.0,
            report_capacity_overflow=True,
        )
        actual, actual_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=None,
            capacity_factor=1.0,
            report_capacity_overflow=True,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=0.1)
    assert int(actual_dropped) == int(expected_dropped)


def test_moe_mlp_pallas_mgpu_is_repeatable_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    ep_size = mesh.shape["expert"]
    tokens = ep_size * 4
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = ep_size * 2
    topk = 2

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(64),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)
    w_up_gate = w_up_gate.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    w_down = w_down.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P("expert", None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        first_out, first_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=None,
            capacity_factor=1.0,
            report_capacity_overflow=True,
        )
        second_out, second_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=None,
            capacity_factor=1.0,
            report_capacity_overflow=True,
        )

    np.testing.assert_array_equal(np.asarray(second_out), np.asarray(first_out))
    assert int(second_dropped) == int(first_dropped)


def test_moe_mlp_pallas_mgpu_matches_ragged_a2a_ep8_topk4_on_hopper_when_available():
    mesh = _make_expert_only_mesh_or_none()
    if mesh is None or mesh.shape["expert"] < 8:
        pytest.skip("requires at least 8 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    ep_size = 8
    tokens = ep_size * 8
    hidden_dim = 128
    intermediate_dim = 128
    local_experts = 4
    num_experts = ep_size * local_experts
    topk = 4

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(45),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)
    w_up_gate = w_up_gate.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    w_down = w_down.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P("expert", None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        expected, expected_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="ragged_all_to_all",
            mesh=None,
            capacity_factor=1.25,
            report_capacity_overflow=True,
        )
        actual, actual_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=None,
            capacity_factor=1.25,
            report_capacity_overflow=True,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=0.1)
    assert int(actual_dropped) == int(expected_dropped)


def test_moe_mlp_pallas_mgpu_reports_capacity_drops_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    ep_size = mesh.shape["expert"]
    tokens = ep_size * 8
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = ep_size * 2
    topk = 2
    capacity_factor = 0.5

    x = jax.random.normal(jax.random.key(61), (tokens, hidden_dim), dtype=jnp.bfloat16)
    selected_experts = jnp.zeros((tokens, topk), dtype=jnp.int32)
    combine_weights = jnp.full((tokens, topk), 0.5, dtype=jnp.bfloat16)
    w_up_gate = jax.random.normal(
        jax.random.key(62),
        (num_experts, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.bfloat16,
    ) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    w_down = jax.random.normal(
        jax.random.key(63),
        (num_experts, intermediate_dim, hidden_dim),
        dtype=jnp.bfloat16,
    ) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P("expert", None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        expected, expected_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="ragged_all_to_all",
            mesh=None,
            capacity_factor=capacity_factor,
            report_capacity_overflow=True,
        )
        actual, actual_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="pallas_mgpu",
            mesh=None,
            capacity_factor=capacity_factor,
            report_capacity_overflow=True,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-2, atol=0.1)
    assert int(actual_dropped) == int(expected_dropped)
    assert int(actual_dropped) > 0


def test_moe_mlp_pallas_mgpu_warns_and_pads_non_tile_aligned_capacity_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(44),
        tokens=8,
        hidden_dim=128,
        intermediate_dim=128,
        num_experts=4,
        topk=2,
    )
    x = x.astype(jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)
    w_up_gate = w_up_gate.astype(jnp.bfloat16)
    w_down = w_down.astype(jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P("expert", None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        with pytest.warns(RuntimeWarning, match="padded receiver capacity"):
            out = moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation=ActivationFunctionEnum.silu,
                implementation="pallas_mgpu",
                mesh=None,
                capacity_factor=1.25,
            )

    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize("combine_weight_dtype", [jnp.bfloat16, jnp.float32], ids=["bf16_combine", "fp32_combine"])
def test_moe_mlp_pallas_mgpu_grad_matches_ragged_a2a_on_hopper_when_available(combine_weight_dtype):
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    ep_size = mesh.shape["expert"]
    tokens = ep_size * 4
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = ep_size * 2
    topk = 2

    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(49),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    target = jax.random.normal(jax.random.key(50), (tokens, hidden_dim), dtype=jnp.bfloat16)
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(combine_weight_dtype)
    w_up_gate = w_up_gate.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    w_down = w_down.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)

    def loss_fn(x_arg, selected_arg, weights_arg, w13_arg, w2_arg, target_arg, *, implementation):
        out = moe_mlp(
            x_arg,
            selected_arg,
            weights_arg,
            w13_arg,
            w2_arg,
            activation=ActivationFunctionEnum.silu,
            implementation=implementation,
            mesh=None,
            capacity_factor=1.0,
        )
        return jnp.sum(out.astype(jnp.float32) * target_arg.astype(jnp.float32))

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        expected_target = jax.sharding.reshard(target, batch_sharding)
        actual_target = jax.sharding.reshard(target, NamedSharding(mesh, P("expert", None)))
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        expected_value, expected_grads = jax.value_and_grad(loss_fn, argnums=(0, 2, 3, 4))(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            expected_target,
            implementation="ragged_all_to_all",
        )
        actual_value, actual_grads = jax.value_and_grad(loss_fn, argnums=(0, 2, 3, 4))(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            actual_target,
            implementation="pallas_mgpu",
        )

    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_value), rtol=1e-2, atol=0.1)
    grad_names = ("x", "combine_weights", "w_up_gate", "w_down")
    for grad_name, actual_grad, expected_grad in zip(grad_names, actual_grads, expected_grads, strict=True):
        actual_np = np.asarray(actual_grad)
        expected_np = np.asarray(expected_grad)
        diff = np.abs(actual_np.astype(np.float32) - expected_np.astype(np.float32))
        max_index = np.unravel_index(np.argmax(diff), diff.shape)
        np.testing.assert_allclose(
            actual_np,
            expected_np,
            rtol=1e-2,
            atol=0.1,
            err_msg=(
                f"{grad_name} gradient mismatch: max_index={max_index}, "
                f"actual={actual_np[max_index]}, expected={expected_np[max_index]}, max_diff={diff[max_index]}"
            ),
        )


def test_moe_expert_mlp_pallas_mgpu_training_step_on_hopper_when_available():
    mesh = _make_two_expert_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least 2 devices")
    first_device = jax.devices()[0]
    if first_device.platform != "gpu" or "h100" not in getattr(first_device, "device_kind", "").lower():
        pytest.skip("pallas_mgpu requires H100/Hopper GPUs")

    ep_size = mesh.shape["expert"]
    tokens = ep_size * 4
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = ep_size * 2
    topk = 2

    x, selected_experts, combine_weights, _w_up_gate, _w_down = _make_inputs(
        key=jax.random.key(51),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    target = jax.random.normal(jax.random.key(52), (tokens, hidden_dim), dtype=jnp.bfloat16)
    x = x.astype(jnp.bfloat16) * jnp.asarray(0.1, dtype=jnp.bfloat16)
    combine_weights = combine_weights.astype(jnp.bfloat16)

    def cast_module_to_bf16(module: MoEExpertMlp) -> MoEExpertMlp:
        return eqx.tree_at(
            lambda m: (m.w_gate_up, m.w_down),
            module,
            (module.w_gate_up.astype(jnp.bfloat16), module.w_down.astype(jnp.bfloat16)),
        )

    def loss_fn(module_arg, x_arg, selected_arg, weights_arg, target_arg):
        out = module_arg(x_arg, selected_arg, weights_arg, mesh=None)
        return jnp.sum(out.astype(jnp.float32) * target_arg.astype(jnp.float32))

    with jax.set_mesh(mesh):
        module = MoEExpertMlp.init(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            initializer_std=0.02,
            key=jax.random.key(53),
            implementation="pallas_mgpu",
            capacity_factor=1.0,
        )
        module = cast_module_to_bf16(module)

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        target = jax.sharding.reshard(target, NamedSharding(mesh, P("expert", None)))

        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(module, x, selected_experts, combine_weights, target)
        array_grads = [grad for grad in jax.tree.leaves(grads) if isinstance(grad, jax.Array)]
        update_norm = sum(float(jnp.sum(jnp.abs(grad).astype(jnp.float32))) for grad in array_grads)
        updates = jax.tree.map(
            lambda grad: None if grad is None else -jnp.asarray(1e-2, dtype=grad.dtype) * grad,
            grads,
            is_leaf=lambda value: value is None,
        )
        updated_module = eqx.apply_updates(module, updates)
        updated_loss_value = loss_fn(updated_module, x, selected_experts, combine_weights, target)

    assert array_grads
    assert jnp.isfinite(loss_value)
    assert jnp.isfinite(updated_loss_value)
    assert update_norm > 0


@pytest.mark.parametrize("implementation", ["ring", "ragged_all_to_all"])
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
            )

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = (
            jax.jit(f)
            .trace(x, selected_experts, combine_weights, w_up_gate, w_down)
            .lower(lowering_platforms=(platform,))
        )
        assert lowered.out_info == jax.ShapeDtypeStruct((tokens, hidden_dim), jnp.float32)


def test_moe_ep_ordered_implementation_lowers_after_pallas_mgpu_fallback_on_abstract_mesh():
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
                implementation=("pallas_mgpu", "ragged_all_to_all"),
                mesh=mesh,
            )

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        with pytest.warns(RuntimeWarning, match="implementation 'pallas_mgpu' failed; trying next fallback"):
            lowered = (
                jax.jit(f)
                .trace(x, selected_experts, combine_weights, w_up_gate, w_down)
                .lower(lowering_platforms=(platform,))
            )
        assert lowered.out_info == jax.ShapeDtypeStruct((tokens, hidden_dim), jnp.float32)


def test_shard_a2a_params_uses_sender_side_output_offsets():
    shard_counts = jnp.array(
        [
            [1, 7, 2],
            [3, 5, 4],
            [6, 8, 9],
        ],
        dtype=jnp.int32,
    )

    input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(
        shard_counts, jnp.array(1, dtype=jnp.int32)
    )

    np.testing.assert_array_equal(np.asarray(send_sizes), np.array([3, 5, 4], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(input_offsets), np.array([0, 3, 8], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(recv_sizes), np.array([7, 5, 8], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(output_offsets), np.array([1, 7, 2], dtype=np.int32))


def test_moe_mlp_ragged_matches_ring_with_ep_axis_when_available():
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")
    if jax.devices()[0].platform == "cpu":
        pytest.skip("ragged_all_to_all is not implemented on XLA:CPU")

    tokens = len(jax.devices()) * 8
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4
    topk = 2

    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
            key=jax.random.key(23),
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

        ring_out, ring_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation="ring",
            mesh=None,
            report_capacity_overflow=True,
            capacity_factor=1.0,
        )
        ragged_out, ragged_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation="ragged_all_to_all",
            mesh=None,
            report_capacity_overflow=True,
            capacity_factor=1.0,
        )

    np.testing.assert_allclose(np.asarray(ragged_out), np.asarray(ring_out), rtol=1e-5, atol=1e-5)
    assert int(ragged_dropped) == int(ring_dropped)


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


def test_compact_and_expand_from_keep_mask_roundtrip():
    inputs = jnp.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ],
        dtype=jnp.float32,
    )
    keep_mask = jnp.array([True, False, True, True, False])

    compacted = _compact_by_keep_mask(inputs, keep_mask)
    expanded = _expand_from_keep_mask(compacted, keep_mask)

    np.testing.assert_allclose(
        np.asarray(compacted),
        np.asarray(
            [
                [1.0, 10.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(expanded),
        np.asarray(
            [
                [1.0, 10.0],
                [0.0, 0.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [0.0, 0.0],
            ],
        ),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(expanded)[np.asarray(keep_mask)],
        np.asarray(inputs)[np.asarray(keep_mask)],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(expanded)[~np.asarray(keep_mask)],
        np.zeros((2, 2), dtype=np.float32),
        rtol=0,
        atol=0,
    )


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
    assert dropped.shape == ()
    assert int(dropped) > 0


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
    assert dropped.shape == ()
    assert int(dropped) > 0


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
