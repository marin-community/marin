# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh
from haliax.nn.ragged_dot import ragged_dot

import levanter.grug.grug_moe as grug_moe
from levanter.grug._moe.common import _prepare_moe_dispatch, _prepare_moe_dispatch_indices_with_assignment_ids
from levanter.grug._moe.ep_deepep import (
    _collapse_local_assignments_gather_jax,
    _collapse_local_assignments_jax,
    _tokens_per_rdma_rank,
)
from levanter.grug._moe.ep_common import _pack_same_route_payloads, _split_same_route_payloads
from levanter.grug._moe.sonic import sonic_gather_sum
from levanter.grug.grug_moe import (
    GroupedMoEExpertMlp,
    MoEExpertMlp,
    MoEExpertMlpPspecs,
    MoeImplementation,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _shard_a2a_params,
    grouped_moe_mlp,
    moe_mlp,
)
from levanter.kernels.deepep.availability import TRANSPORT_REQUIRED_FILES, deepep_preflight_status
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


def test_grouped_moe_mlp_matches_per_layer_moe_mlp_without_ep_axis():
    group_size = 3
    tokens = 12
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4
    topk = 2
    keys = jax.random.split(jax.random.key(81), group_size)
    grouped_inputs = [
        _make_inputs(
            key=key,
            tokens=tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            topk=topk,
        )
        for key in keys
    ]
    x = jnp.stack([inputs[0] for inputs in grouped_inputs], axis=0)
    selected_experts = jnp.stack([inputs[1] for inputs in grouped_inputs], axis=0)
    combine_weights = jnp.stack([inputs[2] for inputs in grouped_inputs], axis=0)
    w_up_gate = jnp.stack([inputs[3] for inputs in grouped_inputs], axis=0)
    w_down = jnp.stack([inputs[4] for inputs in grouped_inputs], axis=0)

    grouped_out = grouped_moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=ActivationFunctionEnum.silu,
        implementation="scatter",
        mesh=None,
    )
    loop_out = jnp.stack(
        [
            moe_mlp(
                x[i],
                selected_experts[i],
                combine_weights[i],
                w_up_gate[i],
                w_down[i],
                activation=ActivationFunctionEnum.silu,
                implementation="scatter",
                mesh=None,
            )
            for i in range(group_size)
        ],
        axis=0,
    )

    np.testing.assert_allclose(np.asarray(grouped_out), np.asarray(loop_out), rtol=1e-5, atol=1e-5)


def test_grouped_moe_expert_mlp_layer_view_matches_grouped_call():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(82),
        tokens=10,
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )
    grouped_mlp = GroupedMoEExpertMlp(
        w_gate_up=jnp.stack([w_up_gate, w_up_gate * 1.5], axis=0),
        w_down=jnp.stack([w_down, w_down * 0.5], axis=0),
        implementation="scatter",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.25,
        remat_mode="none",
        valid_group_size=2,
    )
    x_group = jnp.stack([x, x * 0.25], axis=0)
    selected_group = jnp.stack([selected_experts, selected_experts], axis=0)
    weights_group = jnp.stack([combine_weights, combine_weights], axis=0)

    grouped_out = grouped_mlp(x_group, selected_group, weights_group, mesh=None)
    layer_out = grouped_mlp.layer(1)(x_group[1], selected_group[1], weights_group[1], mesh=None)

    np.testing.assert_allclose(np.asarray(grouped_out[1]), np.asarray(layer_out), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "implementation",
    ["ring", "ragged_all_to_all", "padded_all_to_all", "grouped_assigned_token"],
)
def test_grouped_moe_mlp_ep_path_lowers_on_abstract_mesh(implementation: MoeImplementation):
    mesh = _make_abstract_moe_mesh(data=2, expert=2, model=1)

    group_size = 2
    tokens = 16
    hidden_dim = 32
    intermediate_dim = 64
    num_experts = 4
    topk = 2

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        x = jax.ShapeDtypeStruct(
            shape=(group_size, tokens, hidden_dim),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(None, ("data", "expert"), None)),
        )
        selected_experts = jax.ShapeDtypeStruct(
            shape=(group_size, tokens, topk),
            dtype=jnp.int32,
            sharding=NamedSharding(mesh, P(None, ("data", "expert"), None)),
        )
        combine_weights = jax.ShapeDtypeStruct(
            shape=(group_size, tokens, topk),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(None, ("data", "expert"), None)),
        )
        w_up_gate = jax.ShapeDtypeStruct(
            shape=(group_size, num_experts, hidden_dim, 2 * intermediate_dim),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(None, "expert", None, None)),
        )
        w_down = jax.ShapeDtypeStruct(
            shape=(group_size, num_experts, intermediate_dim, hidden_dim),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(None, "expert", None, None)),
        )

        def f(x, sel, cw, up_gate, down):
            return grouped_moe_mlp(
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
        assert lowered is not None


def test_grouped_assigned_token_single_layer_alias_lowers_on_abstract_mesh():
    mesh = _make_abstract_moe_mesh(data=2, expert=2, model=1)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        x = jax.ShapeDtypeStruct(
            shape=(16, 32),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )
        selected_experts = jax.ShapeDtypeStruct(
            shape=(16, 2),
            dtype=jnp.int32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )
        combine_weights = jax.ShapeDtypeStruct(
            shape=(16, 2),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P(("data", "expert"), None)),
        )
        w_up_gate = jax.ShapeDtypeStruct(
            shape=(4, 32, 128),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P("expert", None, None)),
        )
        w_down = jax.ShapeDtypeStruct(
            shape=(4, 64, 32),
            dtype=jnp.float32,
            sharding=NamedSharding(mesh, P("expert", None, None)),
        )

        out = jax.eval_shape(
            lambda x, sel, cw, up_gate, down: moe_mlp(
                x,
                sel,
                cw,
                up_gate,
                down,
                implementation="grouped_assigned_token",
                mesh=mesh,
            ),
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
        assert out.shape == (16, 32)


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


@pytest.mark.parametrize("remat_mode", ["save_moe", "offload_moe"])
def test_moe_expert_mlp_init_preserves_remat_mode(remat_mode: str):
    mlp = MoEExpertMlp.init(
        num_experts=4,
        hidden_dim=16,
        intermediate_dim=24,
        initializer_std=0.02,
        key=jax.random.key(28),
        implementation="deepep",
        remat_mode=remat_mode,
    )

    assert mlp.remat_mode == remat_mode


def test_moe_expert_mlp_init_accepts_deepep_internode():
    mlp = MoEExpertMlp.init(
        num_experts=16,
        hidden_dim=16,
        intermediate_dim=24,
        initializer_std=0.02,
        key=jax.random.key(2801),
        implementation="deepep_internode",
        remat_mode="save_moe",
    )

    assert mlp.implementation == "deepep_internode"
    assert mlp.remat_mode == "save_moe"


def test_deepep_internode_tokens_per_rdma_rank_groups_by_local_ranks():
    tokens_per_rank = jnp.arange(16, dtype=jnp.int32)

    np.testing.assert_array_equal(
        np.asarray(_tokens_per_rdma_rank(tokens_per_rank, num_local_ranks=8)),
        np.array([28, 92], dtype=np.int32),
    )


def test_deepep_internode_tokens_per_rdma_rank_rejects_single_node():
    tokens_per_rank = jnp.arange(8, dtype=jnp.int32)

    with pytest.raises(ValueError, match="more than one RDMA node rank"):
        _tokens_per_rdma_rank(tokens_per_rank, num_local_ranks=8)


def test_deepep_internode_gather_collapse_matches_scatter_collapse():
    recv_capacity = 4
    hidden_dim = 3
    out_dispatch = (jnp.arange(5 * hidden_dim, dtype=jnp.float32).reshape(5, hidden_dim) / 7).astype(jnp.bfloat16)
    assignment_weights = jnp.array([1.0, 0.5, 0.25, 0.75, 1.25], dtype=jnp.bfloat16)
    recv_token_indices = jnp.array([0, 0, 1, 2, 2], dtype=jnp.int32)
    assignment_destinations = jnp.array([0, 1, 2, -1, 3, 4, -1, -1], dtype=jnp.int32)
    local_group_sizes = jnp.array([2, 3], dtype=jnp.int32)
    num_recv_tokens = jnp.array([3], dtype=jnp.int32)

    scatter = _collapse_local_assignments_jax(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
        recv_capacity=recv_capacity,
    )
    gather = _collapse_local_assignments_gather_jax(
        out_dispatch,
        assignment_weights,
        assignment_destinations,
        local_group_sizes,
        num_recv_tokens,
        recv_capacity=recv_capacity,
    )

    np.testing.assert_allclose(np.asarray(gather), np.asarray(scatter), rtol=0, atol=0)


def test_deepep_internode_gather_collapse_gradients_match_scatter_collapse():
    recv_capacity = 4
    hidden_dim = 3
    out_dispatch = (jnp.arange(5 * hidden_dim, dtype=jnp.float32).reshape(5, hidden_dim) / 7).astype(jnp.float32)
    assignment_weights = jnp.array([1.0, 0.5, 0.25, 0.75, 1.25], dtype=jnp.float32)
    recv_token_indices = jnp.array([0, 0, 1, 2, 2], dtype=jnp.int32)
    assignment_destinations = jnp.array([0, 1, 2, -1, 3, 4, -1, -1], dtype=jnp.int32)
    local_group_sizes = jnp.array([2, 3], dtype=jnp.int32)
    num_recv_tokens = jnp.array([3], dtype=jnp.int32)

    def scatter_loss(dispatch_rows, weights):
        return jnp.sum(
            _collapse_local_assignments_jax(
                dispatch_rows,
                weights,
                recv_token_indices,
                local_group_sizes,
                recv_capacity=recv_capacity,
            ).astype(jnp.float32)
        )

    def gather_loss(dispatch_rows, weights):
        return jnp.sum(
            _collapse_local_assignments_gather_jax(
                dispatch_rows,
                weights,
                assignment_destinations,
                local_group_sizes,
                num_recv_tokens,
                recv_capacity=recv_capacity,
            ).astype(jnp.float32)
        )

    scatter_dispatch_grad, scatter_weight_grad = jax.grad(scatter_loss, argnums=(0, 1))(
        out_dispatch,
        assignment_weights,
    )
    gather_dispatch_grad, gather_weight_grad = jax.grad(gather_loss, argnums=(0, 1))(
        out_dispatch,
        assignment_weights,
    )

    np.testing.assert_allclose(np.asarray(gather_dispatch_grad), np.asarray(scatter_dispatch_grad), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(gather_weight_grad), np.asarray(scatter_weight_grad), rtol=0, atol=0)


def test_moe_expert_mlp_init_rejects_unknown_remat_mode():
    with pytest.raises(ValueError, match="remat_mode"):
        MoEExpertMlp.init(
            num_experts=4,
            hidden_dim=16,
            intermediate_dim=24,
            initializer_std=0.02,
            key=jax.random.key(29),
            implementation="deepep",
            remat_mode="save_everything",
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


@pytest.mark.parametrize(
    "implementation", ["ring", "assigned_token", "ragged_all_to_all", "padded_all_to_all", "deepep"]
)
def test_moe_ep_path_lowers_on_abstract_mesh(implementation: MoeImplementation):
    if implementation == "deepep":
        status = deepep_preflight_status(required_files=TRANSPORT_REQUIRED_FILES)
        if not status.ok:
            pytest.skip("DeepEP source/runtime is not available for FFI lowering")

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
        assert lowered is not None


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


@pytest.mark.parametrize("implementation", ["ragged_all_to_all", "padded_all_to_all"])
def test_moe_mlp_a2a_backend_matches_ring_with_ep_axis_when_available(implementation: MoeImplementation):
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
        a2a_out, a2a_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation=implementation,
            mesh=None,
            report_capacity_overflow=True,
            capacity_factor=1.0,
        )

    np.testing.assert_allclose(np.asarray(a2a_out), np.asarray(ring_out), rtol=1e-5, atol=1e-5)
    assert int(a2a_dropped) == int(ring_dropped)


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


def test_pack_same_route_payloads_roundtrip():
    activations = jnp.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.array([[0.25], [0.5], [0.75]], dtype=jnp.float32)

    packed, widths = _pack_same_route_payloads((activations, weights))
    restored_activations, restored_weights = _split_same_route_payloads(packed, widths)

    assert widths == (3, 1)
    np.testing.assert_allclose(
        np.asarray(packed),
        np.asarray(
            [
                [1.0, 10.0, 100.0, 0.25],
                [2.0, 20.0, 200.0, 0.5],
                [3.0, 30.0, 300.0, 0.75],
            ],
            dtype=np.float32,
        ),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(np.asarray(restored_activations), np.asarray(activations), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(restored_weights), np.asarray(weights), rtol=0, atol=0)


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
