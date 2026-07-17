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
from levanter.grug._moe.ep_deepep import _pack_deepep_local_assignments
import levanter.grug._moe.source_push_combine as source_push_combine
import levanter.grug._moe.source_push_forward as source_push_forward
from levanter.grug._moe.source_push_forward import make_source_push_forward_source_plan_raw_inputs
import levanter.grug._moe.source_push_inbox as source_push_inbox
from levanter.grug._moe.source_push_inbox import PushInboxConfig
import levanter.grug._moe.source_push_public as source_push_public
import levanter.grug._moe.source_push_w2_return as source_push_w2_return
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


def _skip_without_h100x8() -> None:
    devices = jax.devices()
    if len(devices) < 8:
        pytest.skip("source-push MGPU smoke requires at least 8 visible devices")
    if not all(device.platform == "gpu" for device in devices[:8]):
        pytest.skip("source-push MGPU smoke requires GPUs")
    if not all("H100" in getattr(device, "device_kind", "").upper() for device in devices[:8]):
        pytest.skip("source-push MGPU smoke is restricted to H100")


def _shard_public_ep_arrays(
    mesh: Mesh,
    x_source: jax.Array,
    selected_source: jax.Array,
    combine_source: jax.Array,
    w_gate_up_source: jax.Array,
    w_down_source: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    ep_size, tokens_per_rank, hidden_dim = x_source.shape
    experts_per_rank = w_gate_up_source.shape[1]
    intermediate_dim = w_down_source.shape[2]
    topk = selected_source.shape[2]

    x = jnp.asarray(x_source.reshape(ep_size * tokens_per_rank, hidden_dim), dtype=jnp.bfloat16)
    selected_experts = jnp.asarray(selected_source.reshape(ep_size * tokens_per_rank, topk), dtype=jnp.int32)
    combine_weights = jnp.asarray(combine_source.reshape(ep_size * tokens_per_rank, topk), dtype=jnp.bfloat16)
    w_gate_up = jnp.asarray(
        w_gate_up_source.reshape(ep_size * experts_per_rank, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.bfloat16,
    )
    w_down = jnp.asarray(
        w_down_source.reshape(ep_size * experts_per_rank, intermediate_dim, hidden_dim),
        dtype=jnp.bfloat16,
    )

    batch_sharding = NamedSharding(mesh, P("expert", None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    return (
        jax.device_put(x, batch_sharding),
        jax.device_put(selected_experts, batch_sharding),
        jax.device_put(combine_weights, batch_sharding),
        jax.device_put(w_gate_up, expert_sharding),
        jax.device_put(w_down, expert_sharding),
    )


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


def test_moe_mlp_source_push_backend_requires_concrete_expert_mesh():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(101),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=16,
        num_experts=2,
        topk=1,
    )

    with pytest.raises(ValueError, match="requires a concrete expert-parallel source-push MGPU mesh"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation="pallas_mgpu_source_push",
            mesh=None,
        )


def test_moe_mlp_blackwell_source_push_backend_requires_concrete_expert_mesh():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(102),
        tokens=8,
        hidden_dim=16,
        intermediate_dim=16,
        num_experts=2,
        topk=1,
    )

    with pytest.raises(ValueError, match="requires a concrete expert-parallel source-push MGPU mesh"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            implementation="pallas_mgpu_source_push_blackwell",
            mesh=None,
        )


def test_source_push_public_blackwell_config_uses_tuned_transport_defaults():
    ep_size = 8
    tokens_per_rank = 512
    topk = 4
    experts_per_rank = 32
    token_ids = jnp.arange(tokens_per_rank, dtype=jnp.int32)[None, :, None]
    source_ids = jnp.arange(ep_size, dtype=jnp.int32)[:, None, None]
    route_slots = jnp.arange(topk, dtype=jnp.int32)[None, None, :]
    selected = (token_ids * topk + route_slots + source_ids) % (ep_size * experts_per_rank)
    weights = jnp.ones((ep_size, tokens_per_rank, topk), dtype=jnp.float32)

    hopper_config = source_push_public._source_push_config_from_public_inputs(
        selected,
        weights,
        ep_size=ep_size,
        tokens_per_rank=tokens_per_rank,
        topk=topk,
        hidden_dim=3072,
        intermediate_dim=3072,
        experts_per_rank=experts_per_rank,
        capacity_factor=1.25,
        implementation=source_push_public.SOURCE_PUSH_PUBLIC_IMPLEMENTATION,
    )
    blackwell_config = source_push_public._source_push_config_from_public_inputs(
        selected,
        weights,
        ep_size=ep_size,
        tokens_per_rank=tokens_per_rank,
        topk=topk,
        hidden_dim=3072,
        intermediate_dim=3072,
        experts_per_rank=experts_per_rank,
        capacity_factor=1.25,
        implementation=source_push_public.SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL,
    )

    assert blackwell_config.entries_per_rank > 24
    assert hopper_config.inbox_slots == 12
    assert hopper_config.send_worker_programs_per_peer == 2
    assert blackwell_config.inbox_slots == 24
    assert blackwell_config.send_worker_programs_per_peer == 4
    assert blackwell_config.worker_programs_per_peer == 32
    assert (
        source_push_public._source_push_execution_mode(source_push_public.SOURCE_PUSH_PUBLIC_IMPLEMENTATION)
        == source_push_forward.FORWARD_EXECUTION_STAGED_HOST_SYNC
    )
    assert (
        source_push_public._source_push_execution_mode(source_push_public.SOURCE_PUSH_PUBLIC_IMPLEMENTATION_BLACKWELL)
        == source_push_forward.FORWARD_EXECUTION_STAGED_DEVICE_SYNC
    )


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


def test_source_push_forward_matches_public_ep_backends_on_h100():
    _skip_without_h100x8()

    config = PushInboxConfig(
        ep_size=8,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=128,
        intermediate_dim=128,
        block_m=64,
        block_n=128,
        block_k=64,
        n_group=1,
        n_groups_per_job=1,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=8,
        send_pipeline_depth=1,
        routing="balanced",
        tokens_per_rank=64,
        topk=2,
        capacity_factor=1.25,
    )
    raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
    mesh = Mesh(
        np.asarray(jax.devices()[: config.ep_size]),
        ("expert",),
        axis_types=(AxisType.Explicit,),
    )
    x = jnp.asarray(
        raw_inputs.x.reshape(config.ep_size * config.tokens_per_rank, config.hidden_dim),
        dtype=jnp.bfloat16,
    )
    selected_experts = jnp.asarray(
        raw_inputs.selected_experts.reshape(config.ep_size * config.tokens_per_rank, config.topk),
        dtype=jnp.int32,
    )
    combine_weights = jnp.asarray(
        raw_inputs.combine_weights.reshape(config.ep_size * config.tokens_per_rank, config.topk),
        dtype=jnp.bfloat16,
    )
    w_gate_up = jnp.asarray(
        raw_inputs.w_gate_up.reshape(
            config.ep_size * config.experts_per_rank,
            config.hidden_dim,
            2 * config.intermediate_dim,
        ),
        dtype=jnp.bfloat16,
    )
    w_down = jnp.asarray(
        raw_inputs.w_down.reshape(
            config.ep_size * config.experts_per_rank,
            config.intermediate_dim,
            config.hidden_dim,
        ),
        dtype=jnp.bfloat16,
    )

    batch_sharding = NamedSharding(mesh, P("expert", None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    x = jax.device_put(x, batch_sharding)
    selected_experts = jax.device_put(selected_experts, batch_sharding)
    combine_weights = jax.device_put(combine_weights, batch_sharding)
    w_gate_up = jax.device_put(w_gate_up, expert_sharding)
    w_down = jax.device_put(w_down, expert_sharding)

    with jax.set_mesh(mesh):
        source_push_out, source_push_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_gate_up,
            w_down,
            implementation="pallas_mgpu_source_push",
            mesh=mesh,
            capacity_factor=config.capacity_factor,
            report_capacity_overflow=True,
        )
        source_push_out_repeat, source_push_dropped_repeat = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_gate_up,
            w_down,
            implementation="pallas_mgpu_source_push",
            mesh=mesh,
            capacity_factor=config.capacity_factor,
            report_capacity_overflow=True,
        )
        baselines = {
            implementation: moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_gate_up,
                w_down,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=config.capacity_factor,
                report_capacity_overflow=True,
            )
            for implementation in ("ragged_all_to_all", "ring")
        }

    source_push_raw = np.asarray(jax.device_get(source_push_out))
    source_push_repeat_raw = np.asarray(jax.device_get(source_push_out_repeat))
    np.testing.assert_array_equal(source_push_repeat_raw, source_push_raw)
    assert int(jax.device_get(source_push_dropped_repeat)) == int(jax.device_get(source_push_dropped))

    source_push_host = np.asarray(source_push_raw, dtype=np.float32)
    assert source_push_host.shape == (config.ep_size * config.tokens_per_rank, config.hidden_dim)
    assert int(jax.device_get(source_push_dropped)) == 0
    for baseline_out, baseline_dropped in baselines.values():
        baseline_host = np.asarray(jax.device_get(baseline_out), dtype=np.float32)
        diff = np.abs(source_push_host - baseline_host)
        assert int(jax.device_get(baseline_dropped)) == 0
        assert float(np.max(diff)) <= 0.03125
        assert float(np.mean(diff)) <= 0.002


@pytest.mark.parametrize(
    ("ep_size", "topk"),
    [(2, 2), (2, 4), (8, 2)],
    ids=["ep2_topk2", "ep2_topk4", "ep8_topk2"],
)
def test_source_push_stage_kernels_match_references_on_h100(ep_size, topk):
    _skip_without_h100x8()

    config = PushInboxConfig(
        ep_size=ep_size,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=128,
        intermediate_dim=128,
        block_m=64,
        block_n=128,
        block_k=64,
        n_group=1,
        n_groups_per_job=1,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=8,
        send_pipeline_depth=1,
        routing="balanced",
        tokens_per_rank=64,
        topk=topk,
        capacity_factor=1.25,
    )

    w13_rows = source_push_inbox.run_source_push_inbox_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=True,
        debug_exceptions=True,
    )
    w13_row = w13_rows[0]
    assert w13_row["error_type"] is None
    assert w13_row["metadata_mismatches"] == 0
    assert w13_row["input_mode"] == "source_push_plan"
    assert w13_row["row_start_mode"] == "source_padded_row_start"
    assert w13_row["hidden_max_abs_diff"] <= 0.03125
    assert w13_row["hidden_mean_abs_diff"] <= 0.002
    assert w13_row["hidden_unwritten_max_abs"] == 0.0

    w2_rows = source_push_w2_return.run_source_push_w2_return_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=True,
        debug_exceptions=True,
        hidden_input_mode=source_push_w2_return.W2_HIDDEN_INPUT_W13_REFERENCE,
        return_mode=source_push_w2_return.W2_RETURN_MODE_DIRECT_REMOTE,
    )
    w2_row = w2_rows[0]
    assert w2_row["error_type"] is None
    assert w2_row["return_mode"] == source_push_w2_return.W2_RETURN_MODE_DIRECT_REMOTE
    assert w2_row["direct_to_source"]
    assert w2_row["w2_input_mode"] == "source_push_plan"
    assert w2_row["w2_hidden_input_mode"] == source_push_w2_return.W2_HIDDEN_INPUT_W13_REFERENCE
    assert w2_row["source_queue_max_abs_diff"] <= 0.03125
    assert w2_row["mean_abs_diff"] <= 0.002

    combine_rows = source_push_combine.run_source_push_combine_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=True,
        debug_exceptions=True,
    )
    combine_row = combine_rows[0]
    assert combine_row["error_type"] is None
    assert combine_row["combine_mode"] == source_push_combine.SOURCE_COMBINE_MODE_DIRECT_GATHER_SUM
    assert combine_row["dropped_routes"] == 0
    assert combine_row["max_abs_diff"] <= 0.03125
    assert combine_row["mean_abs_diff"] <= 0.002


def test_source_push_exact_expert_major_w13_matches_reference_on_h100():
    _skip_without_h100x8()

    config = PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=128,
        intermediate_dim=128,
        block_m=64,
        block_n=128,
        block_k=64,
        n_group=1,
        n_groups_per_job=1,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=8,
        send_pipeline_depth=1,
        routing="balanced",
        tokens_per_rank=128,
        topk=2,
        capacity_factor=1.25,
    )

    rows = source_push_inbox._run_one(
        config,
        source_push_inbox.SourcePushInboxRunSettings(
            warmup=0,
            steps=1,
            repeat_runs=1,
            check=True,
            debug_exceptions=True,
        ),
        input_builder=source_push_inbox._make_exact_source_push_plan_inputs,
    )

    row = rows[0]
    assert row["error_type"] is None
    assert row["metadata_mismatches"] == 0
    assert row["row_start_mode"] == source_push_inbox.ROW_START_MODE_EXACT_EXPERT_MAJOR
    assert row["row_layout"] == source_push_inbox.ROW_LAYOUT_EXACT_EXPERT_MAJOR
    assert row["plan_layout_padding_rows_total"] == 0
    assert row["hidden_max_abs_diff"] <= 0.03125
    assert row["hidden_mean_abs_diff"] <= 0.002
    assert row["hidden_unwritten_max_abs"] == 0.0


def test_source_push_exact_expert_major_w2_return_matches_reference_on_h100():
    _skip_without_h100x8()

    config = PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=128,
        intermediate_dim=128,
        block_m=64,
        block_n=128,
        block_k=64,
        n_group=1,
        n_groups_per_job=1,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=8,
        send_pipeline_depth=1,
        routing="balanced",
        tokens_per_rank=128,
        topk=2,
        capacity_factor=1.25,
    )

    rows = source_push_w2_return._run_w2_return_one(
        config,
        source_push_inbox.SourcePushInboxRunSettings(
            warmup=0,
            steps=1,
            repeat_runs=1,
            check=True,
            debug_exceptions=True,
        ),
        input_builder=lambda run_config: source_push_w2_return.make_w2_return_exact_source_plan_inputs(
            run_config,
            hidden_input_mode=source_push_w2_return.W2_HIDDEN_INPUT_W13_REFERENCE,
        ),
        return_mode=source_push_w2_return.W2_RETURN_MODE_DIRECT_REMOTE,
    )

    row = rows[0]
    assert row["error_type"] is None
    assert row["return_mode"] == source_push_w2_return.W2_RETURN_MODE_DIRECT_REMOTE
    assert row["direct_to_source"]
    assert row["w2_input_mode"] == "exact_source_push_plan"
    assert row["w2_hidden_input_mode"] == source_push_w2_return.W2_HIDDEN_INPUT_W13_REFERENCE
    assert row["row_start_mode"] == source_push_inbox.ROW_START_MODE_EXACT_EXPERT_MAJOR
    assert row["row_layout"] == source_push_inbox.ROW_LAYOUT_EXACT_EXPERT_MAJOR
    assert row["source_queue_max_abs_diff"] <= 0.03125
    assert row["mean_abs_diff"] <= 0.002


def test_source_push_exact_expert_major_forward_matches_reference_on_h100():
    _skip_without_h100x8()

    config = PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=1,
        hidden_dim=128,
        intermediate_dim=128,
        block_m=64,
        block_n=128,
        block_k=64,
        n_group=1,
        n_groups_per_job=1,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=8,
        send_pipeline_depth=1,
        routing="balanced",
        tokens_per_rank=128,
        topk=2,
        capacity_factor=1.25,
    )

    rows = source_push_forward.run_source_push_forward_exact_source_plan(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=True,
        debug_exceptions=True,
        execution_mode=source_push_forward.FORWARD_EXECUTION_STAGED_HOST_SYNC,
    )

    row = next(
        row for row in rows if row["row_type"] == "repeat" and row["stage"] == source_push_forward.FORWARD_STAGE_TOTAL
    )
    assert row["error_type"] is None
    assert row["execution_mode"] == source_push_forward.FORWARD_EXECUTION_STAGED_HOST_SYNC
    assert row["input_mode"] == "exact_source_push_plan"
    assert row["row_start_mode"] == source_push_inbox.ROW_START_MODE_EXACT_EXPERT_MAJOR
    assert row["row_layout"] == source_push_inbox.ROW_LAYOUT_EXACT_EXPERT_MAJOR
    assert row["plan_layout_padding_rows_total"] == 0
    assert row["dropped_routes"] == 0
    assert row["max_abs_diff"] <= 0.03125
    assert row["mean_abs_diff"] <= 0.002


def test_source_push_forward_handles_tail_blocks_empty_experts_topk4_on_h100():
    _skip_without_h100x8()

    ep_size = 8
    tokens_per_rank = 65
    hidden_dim = 128
    intermediate_dim = 128
    experts_per_rank = 2
    topk = 4
    capacity_factor = 2.0

    token_ids = np.arange(tokens_per_rank, dtype=np.int32)
    route_offsets = np.arange(topk, dtype=np.int32)
    selected_by_source = []
    for src in range(ep_size):
        dst_ranks = (token_ids[:, None] + route_offsets[None, :] + src) % ep_size
        selected_by_source.append(dst_ranks * experts_per_rank)
    selected_source = jnp.asarray(np.stack(selected_by_source, axis=0), dtype=jnp.int32)
    selected_host = np.asarray(selected_source)
    np.testing.assert_array_equal(selected_host % experts_per_rank, np.zeros_like(selected_host))
    counts_by_src_dst = np.zeros((ep_size, ep_size), dtype=np.int32)
    for src in range(ep_size):
        counts_by_src_dst[src] = np.bincount(selected_host[src].reshape(-1) // experts_per_rank, minlength=ep_size)
    assert np.any((counts_by_src_dst > 0) & (counts_by_src_dst % 64 != 0))

    key = jax.random.key(202)
    k_x, k_combine, k_w13, k_w2 = jax.random.split(key, 4)
    edge_value_scale = 0.2
    x_source = edge_value_scale * jax.random.normal(k_x, (ep_size, tokens_per_rank, hidden_dim), dtype=jnp.float32)
    combine_source = jax.nn.softmax(
        jax.random.normal(k_combine, (ep_size, tokens_per_rank, topk), dtype=jnp.float32),
        axis=-1,
    )
    w_gate_up_source = edge_value_scale * jax.random.normal(
        k_w13,
        (ep_size, experts_per_rank, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.float32,
    )
    w_down_source = edge_value_scale * jax.random.normal(
        k_w2,
        (ep_size, experts_per_rank, intermediate_dim, hidden_dim),
        dtype=jnp.float32,
    )

    mesh = Mesh(
        np.asarray(jax.devices()[:ep_size]),
        ("expert",),
        axis_types=(AxisType.Explicit,),
    )
    x, selected_experts, combine_weights, w_gate_up, w_down = _shard_public_ep_arrays(
        mesh,
        x_source,
        selected_source,
        combine_source,
        w_gate_up_source,
        w_down_source,
    )

    with jax.set_mesh(mesh):
        source_push_out, source_push_dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_gate_up,
            w_down,
            implementation="pallas_mgpu_source_push",
            mesh=mesh,
            capacity_factor=capacity_factor,
            report_capacity_overflow=True,
        )
        baselines = {
            implementation: moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_gate_up,
                w_down,
                implementation=implementation,
                mesh=mesh,
                capacity_factor=capacity_factor,
                report_capacity_overflow=True,
            )
            for implementation in ("ragged_all_to_all", "ring")
        }

    source_push_host = np.asarray(jax.device_get(source_push_out), dtype=np.float32)
    assert source_push_host.shape == (ep_size * tokens_per_rank, hidden_dim)
    assert int(jax.device_get(source_push_dropped)) == 0
    for baseline_out, baseline_dropped in baselines.values():
        baseline_host = np.asarray(jax.device_get(baseline_out), dtype=np.float32)
        diff = np.abs(source_push_host - baseline_host)
        assert int(jax.device_get(baseline_dropped)) == 0
        assert float(np.max(diff)) <= 0.03125
        assert float(np.mean(diff)) <= 0.002


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
