# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax._src import config as jax_config
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh
from haliax.nn.ragged_dot import ragged_dot

import levanter.grug.grug_moe as grug_moe
import levanter.grug._moe.ep_ragged_all_to_all as ep_ragged_a2a
from levanter.grug._moe.common import _prepare_moe_dispatch, _prepare_moe_dispatch_indices_with_assignment_ids
from levanter.grug._moe.ep_deepep import _pack_deepep_local_assignments
from levanter.grug._moe.ep_common import _compact_by_keep_mask_to_size
from levanter.grug._moe.ep_ragged_all_to_all import (
    _decode_slot_metadata,
    _encode_slot_metadata,
    _fixed_a2a_core,
    _fixed_dispatch_gather_reference,
    _fixed_dispatch_gather_sonic,
    _fixed_dispatch_gather_sonic_grad,
    _pack_sparse_clone_weights,
    _receiver_clipped_dispatch_metadata,
    _receiver_clipped_mnnvl_dispatch_metadata,
    _receiver_destination_compact_positions,
    _receiver_destination_pooled_dispatch_metadata,
    _round_robin_ppermute_all_to_all,
    _same_expert_clone_dispatch_metadata,
    _same_expert_compact_transport_metadata,
    _same_expert_echo_dispatch_metadata,
    _same_expert_echo_fixed_transport_metadata,
    _same_expert_hybridep_routing,
    _same_expert_pooled_dispatch_metadata,
    _sonic_unique_row_gather,
    _sparse_clone_weight_metadata,
)
from levanter.grug._moe.sonic import (
    sonic_capacity_refill,
    sonic_clone_weight_reduce,
    sonic_expert_local_rank,
    sonic_gather_sum,
    sonic_slot_weighted_grad,
    sonic_unique_row_scatter,
    sonic_unpermute_i32,
)
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


@pytest.mark.parametrize("payload_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_embedded_slot_metadata_round_trip(payload_dtype: jnp.dtype):
    slots = jnp.array([0, 1, 255, 256, 65535, 65536, 262144, (1 << 24) - 1], dtype=jnp.int32)

    words = _encode_slot_metadata(slots, payload_dtype)
    decoded = _decode_slot_metadata(words)

    assert words.shape == (slots.shape[0], 3)
    np.testing.assert_array_equal(np.asarray(decoded), np.asarray(slots))


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


def _reference_mnnvl_dispatch_gather(
    x_local: jax.Array,
    token_sources: jax.Array,
    dispatch_positions: jax.Array,
    keep: jax.Array,
    destination_ranks: jax.Array,
    destination_slots: jax.Array,
    receiver_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del dispatch_positions, keep
    sender_values = x_local[token_sources]
    all_values = jax.lax.all_gather(sender_values, "expert").reshape(-1, x_local.shape[1])
    all_destination_matrix = jax.lax.all_gather(destination_ranks, "expert")
    expert_shards = all_destination_matrix.shape[0]
    all_destinations = all_destination_matrix.reshape(-1)
    all_slots = jax.lax.all_gather(destination_slots, "expert").reshape(-1)
    sender_ranks = jnp.broadcast_to(
        jnp.arange(expert_shards, dtype=jnp.int32)[:, None],
        (expert_shards, token_sources.shape[0]),
    ).reshape(-1)
    sender_slots = jnp.broadcast_to(
        jnp.arange(token_sources.shape[0], dtype=jnp.int32)[None, :],
        (expert_shards, token_sources.shape[0]),
    ).reshape(-1)

    receiver_index = jax.lax.axis_index("expert")
    valid = jnp.logical_and(all_destinations == receiver_index, all_slots < receiver_capacity)
    scatter_slots = jnp.where(valid, all_slots, receiver_capacity)
    output = (
        jnp.zeros((receiver_capacity, x_local.shape[1]), dtype=x_local.dtype)
        .at[scatter_slots]
        .set(
            all_values,
            mode="drop",
        )
    )
    source_ranks = (
        jnp.full((receiver_capacity,), expert_shards, dtype=jnp.int32)
        .at[scatter_slots]
        .set(
            sender_ranks,
            mode="drop",
        )
    )
    source_slots = (
        jnp.full((receiver_capacity,), token_sources.shape[0], dtype=jnp.int32)
        .at[scatter_slots]
        .set(
            sender_slots,
            mode="drop",
        )
    )
    return output, source_ranks, source_slots


def _reference_mnnvl_combine(
    receiver_values: jax.Array,
    source_ranks: jax.Array,
    source_slots: jax.Array,
    dispatch_destination_ranks: jax.Array,
    dispatch_destination_slots: jax.Array,
    send_rows: int,
) -> jax.Array:
    del dispatch_destination_ranks, dispatch_destination_slots
    all_values = jax.lax.all_gather(receiver_values, "expert").reshape(-1, receiver_values.shape[1])
    all_source_ranks = jax.lax.all_gather(source_ranks, "expert").reshape(-1)
    all_source_slots = jax.lax.all_gather(source_slots, "expert").reshape(-1)
    sender_index = jax.lax.axis_index("expert")
    valid = jnp.logical_and(all_source_ranks == sender_index, all_source_slots < send_rows)
    scatter_slots = jnp.where(valid, all_source_slots, send_rows)
    return (
        jnp.zeros((send_rows, receiver_values.shape[1]), dtype=receiver_values.dtype)
        .at[scatter_slots]
        .set(
            all_values,
            mode="drop",
        )
    )


def _reference_sparse_clone_weight_exchange(
    local_weights: jax.Array,
    packed_local_experts: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    *,
    max_receiver_segments: int,
) -> jax.Array:
    del recv_sizes
    padded_local_weights = jnp.concatenate(
        [local_weights, jnp.zeros((1, *local_weights.shape[1:]), dtype=local_weights.dtype)],
        axis=0,
    )
    send_weights = padded_local_weights[packed_local_experts]
    all_send_weights = jax.lax.all_gather(send_weights, "expert")
    all_input_offsets = jax.lax.all_gather(input_offsets, "expert")
    all_send_sizes = jax.lax.all_gather(send_sizes, "expert")
    all_output_offsets = jax.lax.all_gather(output_offsets, "expert")
    receiver_index = jax.lax.axis_index("expert")
    receiver_count = all_send_weights.shape[0]
    output = jnp.zeros((max_receiver_segments, *local_weights.shape[1:]), dtype=local_weights.dtype)
    segment_positions = jnp.arange(max_receiver_segments, dtype=jnp.int32)
    for sender_index in range(receiver_count):
        segment_size = all_send_sizes[sender_index, receiver_index]
        input_position = all_input_offsets[sender_index, receiver_index] + segment_positions
        input_position = jnp.minimum(input_position, all_send_weights.shape[1] - 1)
        output_position = all_output_offsets[sender_index, receiver_index] + segment_positions
        output_position = jnp.where(segment_positions < segment_size, output_position, max_receiver_segments)
        output = output.at[output_position].set(all_send_weights[sender_index, input_position], mode="drop")
    return output


def _reference_echo_ragged_all_to_all(
    inputs: jax.Array,
    outputs: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
) -> jax.Array:
    del recv_sizes
    all_inputs = jax.lax.all_gather(inputs, "expert")
    all_input_offsets = jax.lax.all_gather(input_offsets, "expert")
    all_send_sizes = jax.lax.all_gather(send_sizes, "expert")
    all_output_offsets = jax.lax.all_gather(output_offsets, "expert")
    receiver_index = jax.lax.axis_index("expert")
    segment_positions = jnp.arange(outputs.shape[0], dtype=jnp.int32)
    for sender_index in range(all_inputs.shape[0]):
        segment_size = all_send_sizes[sender_index, receiver_index]
        input_position = all_input_offsets[sender_index, receiver_index] + segment_positions
        input_position = jnp.minimum(input_position, inputs.shape[0] - 1)
        output_position = all_output_offsets[sender_index, receiver_index] + segment_positions
        output_position = jnp.where(segment_positions < segment_size, output_position, outputs.shape[0])
        outputs = outputs.at[output_position].set(all_inputs[sender_index, input_position], mode="drop")
    return outputs


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


def _skip_without_quack_gpu_runtime() -> None:
    optional_modules = ("cutlass", "quack")
    if not all(importlib.util.find_spec(module) is not None for module in optional_modules):
        pytest.skip("QuACK/CuTe optional dependencies are not installed")
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("QuACK cutlass_call tests require a GPU")


def test_quack_grouped_wgrad_matches_dense_reference_on_gpu():
    _skip_without_quack_gpu_runtime()
    quack_grouped_wgrad = importlib.import_module("levanter.grug._moe.quack_moe_cute").quack_grouped_wgrad

    group_sizes = (96, 160, 64, 192)
    cu_seqlens = jnp.asarray(np.cumsum((0, *group_sizes)), dtype=jnp.int32)
    total_k = sum(group_sizes)
    lhs = jax.random.normal(jax.random.key(60), (total_k, 128), dtype=jnp.bfloat16) * 0.05
    rhs = jax.random.normal(jax.random.key(61), (total_k, 256), dtype=jnp.bfloat16) * 0.05

    actual = jax.jit(quack_grouped_wgrad)(lhs, rhs, cu_seqlens)
    reference_parts = []
    start = 0
    for size in group_sizes:
        reference_parts.append(lhs[start : start + size].T @ rhs[start : start + size])
        start += size
    reference = jnp.stack(reference_parts)
    jax.block_until_ready((actual, reference))

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(reference, dtype=np.float32),
        rtol=5e-2,
        atol=2e-2,
    )


def test_quack_expert_mlp_grouped_wgrad_matches_ragged_reference_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    _skip_without_quack_gpu_runtime()
    sonic_cute = importlib.import_module("levanter.grug._moe.sonic_cute")
    monkeypatch.setenv("SCALE_QUACK_GROUPED_WGRAD", "1")

    group_sizes = (96, 160, 64, 192)
    group_sizes_array = jnp.asarray(group_sizes, dtype=jnp.int32)
    cu_seqlens = jnp.asarray(np.cumsum((0, *group_sizes)), dtype=jnp.int32)
    total_tokens = sum(group_sizes)
    hidden_dim = 128
    intermediate_dim = 256
    dtype = jnp.bfloat16
    key_x, key_w13, key_w2, key_cotangent = jax.random.split(jax.random.key(62), 4)
    x = jax.random.normal(key_x, (total_tokens, hidden_dim), dtype=dtype) * 0.02
    w13_interleaved = (
        jax.random.normal(key_w13, (len(group_sizes), hidden_dim, 2 * intermediate_dim), dtype=dtype) * 0.02
    )
    w2 = jax.random.normal(key_w2, (len(group_sizes), intermediate_dim, hidden_dim), dtype=dtype) * 0.02
    cotangent = jax.random.normal(key_cotangent, (total_tokens, hidden_dim), dtype=dtype) * 0.02

    def quack_loss(x, w13_interleaved, w2):
        output = sonic_cute._expert_mlp(
            x,
            w13_interleaved,
            w2,
            group_sizes_array,
            cu_seqlens,
        )
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def reference_loss(x, w13_interleaved, w2):
        gate_up = ragged_dot(x, w13_interleaved, group_sizes_array)
        gate = gate_up[:, 0::2]
        up = gate_up[:, 1::2]
        output = ragged_dot(jax.nn.silu(gate) * up, w2, group_sizes_array)
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    quack_value, quack_grad = jax.jit(jax.value_and_grad(quack_loss, argnums=(0, 1, 2)))(
        x,
        w13_interleaved,
        w2,
    )
    reference_value, reference_grad = jax.jit(jax.value_and_grad(reference_loss, argnums=(0, 1, 2)))(
        x,
        w13_interleaved,
        w2,
    )
    jax.block_until_ready((quack_value, quack_grad, reference_value, reference_grad))

    np.testing.assert_allclose(
        np.asarray(quack_value),
        np.asarray(reference_value),
        rtol=5e-2,
        atol=2e-2,
    )
    jax.tree.map(
        lambda actual, reference: np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(reference, dtype=np.float32),
            rtol=5e-2,
            atol=2e-2,
        ),
        quack_grad,
        reference_grad,
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


def test_fixed_dispatch_gather_sonic_grad_matches_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    tokens = 16
    topk = 8
    hidden_dim = 5120
    send_size = tokens * topk
    key_x, key_grad = jax.random.split(jax.random.key(45))
    x = jax.random.normal(key_x, (tokens, hidden_dim), dtype=jnp.bfloat16)
    send_x_grad = jax.random.normal(key_grad, (send_size, hidden_dim), dtype=jnp.bfloat16)
    dispatch_positions = jnp.arange(send_size, dtype=jnp.int32).reshape(tokens, topk)
    dispatch_positions = dispatch_positions.at[-1, -2:].set(send_size)
    keep = dispatch_positions < send_size

    def sonic_loss(x):
        send_x = _fixed_dispatch_gather_sonic_grad(x, dispatch_positions, keep, send_size)
        return jnp.sum(send_x.astype(jnp.float32) * send_x_grad.astype(jnp.float32))

    def reference_loss(x):
        send_x = _fixed_dispatch_gather_reference(x, dispatch_positions, send_size=send_size)
        return jnp.sum(send_x.astype(jnp.float32) * send_x_grad.astype(jnp.float32))

    sonic_value, sonic_grad = jax.jit(jax.value_and_grad(sonic_loss))(x)
    reference_value, reference_grad = jax.jit(jax.value_and_grad(reference_loss))(x)
    sonic_value.block_until_ready()
    reference_value.block_until_ready()

    np.testing.assert_array_equal(np.asarray(sonic_value), np.asarray(reference_value))
    np.testing.assert_allclose(np.asarray(sonic_grad), np.asarray(reference_grad), rtol=1e-5, atol=1e-5)


def test_fixed_dispatch_gather_sonic_matches_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    tokens = 16
    topk = 8
    hidden_dim = 5120
    send_size = tokens * topk
    key_x, key_grad = jax.random.split(jax.random.key(47))
    x = jax.random.normal(key_x, (tokens, hidden_dim), dtype=jnp.bfloat16)
    send_x_grad = jax.random.normal(key_grad, (send_size, hidden_dim), dtype=jnp.bfloat16)
    dispatch_positions = jnp.arange(send_size, dtype=jnp.int32).reshape(tokens, topk)
    dispatch_positions = dispatch_positions.at[-1, -2:].set(send_size)
    keep = dispatch_positions < send_size

    def sonic_loss(x):
        send_x = _fixed_dispatch_gather_sonic(x, dispatch_positions, keep, send_size)
        return jnp.sum(send_x.astype(jnp.float32) * send_x_grad.astype(jnp.float32)), send_x

    def reference_loss(x):
        send_x = _fixed_dispatch_gather_reference(x, dispatch_positions, send_size=send_size)
        return jnp.sum(send_x.astype(jnp.float32) * send_x_grad.astype(jnp.float32)), send_x

    (sonic_value, sonic_out), sonic_grad = jax.jit(jax.value_and_grad(sonic_loss, has_aux=True))(x)
    (reference_value, reference_out), reference_grad = jax.jit(jax.value_and_grad(reference_loss, has_aux=True))(x)
    sonic_value.block_until_ready()
    reference_value.block_until_ready()

    np.testing.assert_array_equal(np.asarray(sonic_out), np.asarray(reference_out))
    np.testing.assert_array_equal(np.asarray(sonic_value), np.asarray(reference_value))
    np.testing.assert_array_equal(np.asarray(sonic_grad), np.asarray(reference_grad))


def test_sonic_slot_weighted_grad_matches_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    slots = 32
    hidden_dim = 5120
    key_dout, key_x, key_weights = jax.random.split(jax.random.key(49), 3)
    dout = jax.random.normal(key_dout, (slots, hidden_dim), dtype=jnp.bfloat16)
    x = jax.random.normal(key_x, (slots, hidden_dim), dtype=jnp.bfloat16)
    weights = jax.random.uniform(key_weights, (slots,), dtype=jnp.float32)

    actual_dx, actual_dw = jax.jit(sonic_slot_weighted_grad)(dout, x, weights)
    expected_dx = (dout.astype(jnp.float32) * weights[:, None]).astype(jnp.bfloat16)
    expected_dw = jnp.sum(dout.astype(jnp.float32) * x.astype(jnp.float32), axis=-1)
    jax.block_until_ready((actual_dx, actual_dw, expected_dx, expected_dw))

    np.testing.assert_array_equal(np.asarray(actual_dx), np.asarray(expected_dx))
    np.testing.assert_allclose(np.asarray(actual_dw), np.asarray(expected_dw), rtol=1e-5, atol=1e-3)


def test_sparse_clone_weight_pack_sonic_gradient_matches_reference_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    _skip_without_sonic_gpu_runtime()
    monkeypatch.setenv("SCALE_A2A_CLONE_WEIGHT_GRAD_BLOCK", "256")
    local_experts = 4
    packed_local_experts = jnp.asarray(
        [0, 1, 0, 3, 2, 1, 4, 3, 0, 4, 2, 1],
        dtype=jnp.int32,
    )
    weights = jax.random.normal(
        jax.random.key(63),
        (local_experts, 64, 128),
        dtype=jnp.bfloat16,
    )
    cotangent = jax.random.normal(
        jax.random.key(64),
        (packed_local_experts.shape[0], 64, 128),
        dtype=jnp.bfloat16,
    )

    def sonic_loss(weights):
        packed = _pack_sparse_clone_weights(weights, packed_local_experts, local_experts)
        return jnp.sum(packed.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def reference_loss(weights):
        padded_weights = jnp.concatenate(
            [weights, jnp.zeros((1, *weights.shape[1:]), dtype=weights.dtype)],
            axis=0,
        )
        packed = padded_weights[packed_local_experts]
        return jnp.sum(packed.astype(jnp.float32) * cotangent.astype(jnp.float32))

    sonic_value, sonic_grad = jax.jit(jax.value_and_grad(sonic_loss))(weights)
    reference_value, reference_grad = jax.jit(jax.value_and_grad(reference_loss))(weights)
    direct_grad = jax.jit(
        lambda clone_grads, packed: sonic_clone_weight_reduce(
            clone_grads,
            packed,
            local_experts=local_experts,
            block_features=256,
        )
    )(cotangent, packed_local_experts)
    jax.block_until_ready((sonic_value, sonic_grad, reference_value, reference_grad, direct_grad))

    np.testing.assert_allclose(np.asarray(sonic_value), np.asarray(reference_value), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(sonic_grad),
        np.asarray(reference_grad),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(direct_grad),
        np.asarray(reference_grad),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sonic_unique_row_gather_matches_reference_and_gradient_on_gpu():
    _skip_without_sonic_gpu_runtime()
    input_rows = 17
    output_rows = 23
    hidden_dim = 5120
    source_rows = jnp.asarray(
        [11, 3, 17, 14, 0, 8, 17, 5, 16, 1, 12, 17, 7, 4, 10, 17, 2, 13, 6, 17, 15, 9, 17],
        dtype=jnp.int32,
    )
    values = jax.random.normal(
        jax.random.key(65),
        (input_rows, hidden_dim),
        dtype=jnp.bfloat16,
    )
    cotangent = jax.random.normal(
        jax.random.key(66),
        (output_rows, hidden_dim),
        dtype=jnp.bfloat16,
    )

    def sonic_loss(values):
        gathered = _sonic_unique_row_gather(values, source_rows, input_rows)
        return jnp.sum(gathered.astype(jnp.float32) * cotangent.astype(jnp.float32)), gathered

    def reference_loss(values):
        padded_values = jnp.concatenate(
            [values, jnp.zeros((1, hidden_dim), dtype=values.dtype)],
            axis=0,
        )
        gathered = padded_values[source_rows]
        return jnp.sum(gathered.astype(jnp.float32) * cotangent.astype(jnp.float32)), gathered

    (sonic_value, sonic_output), sonic_grad = jax.jit(jax.value_and_grad(sonic_loss, has_aux=True))(values)
    (reference_value, reference_output), reference_grad = jax.jit(jax.value_and_grad(reference_loss, has_aux=True))(
        values
    )
    direct_grad = jax.jit(
        lambda rows, destinations: sonic_unique_row_scatter(
            rows,
            destinations,
            output_rows=input_rows,
        )
    )(cotangent, source_rows)
    jax.block_until_ready(
        (sonic_value, sonic_output, sonic_grad, reference_value, reference_output, reference_grad, direct_grad)
    )

    np.testing.assert_array_equal(np.asarray(sonic_output), np.asarray(reference_output))
    np.testing.assert_array_equal(np.asarray(sonic_value), np.asarray(reference_value))
    np.testing.assert_array_equal(np.asarray(sonic_grad), np.asarray(reference_grad))
    np.testing.assert_array_equal(np.asarray(direct_grad), np.asarray(reference_grad))


def test_sonic_unpermute_i32_matches_argsort_inverse_on_gpu():
    _skip_without_sonic_gpu_runtime()
    size = 524_288
    permutation = (jnp.arange(size, dtype=jnp.int32) * 8191) % size
    values = (jnp.arange(size, dtype=jnp.int32) * 17 + 3) % size

    actual = jax.jit(sonic_unpermute_i32)(values, permutation)
    expected = values[jnp.argsort(permutation)]
    jax.block_until_ready((actual, expected))

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_sonic_expert_local_rank_matches_stable_sort_on_gpu():
    _skip_without_sonic_gpu_runtime()
    assignments = 524_288
    num_experts = 256
    experts = jax.random.randint(
        jax.random.key(51),
        (assignments,),
        minval=0,
        maxval=num_experts,
        dtype=jnp.int32,
    )

    order = jnp.argsort(experts, stable=True)
    expert_counts = jnp.bincount(experts, length=num_experts).astype(jnp.int32)
    segment_starts = jnp.cumsum(expert_counts) - expert_counts
    sorted_ranks = jnp.arange(assignments, dtype=jnp.int32) - segment_starts[experts[order]]
    expected = jnp.zeros_like(sorted_ranks).at[order].set(sorted_ranks)
    actual = jax.jit(sonic_expert_local_rank, static_argnames=("num_experts",))(
        experts,
        num_experts=num_experts,
    )
    jax.block_until_ready((actual, expected))

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_sonic_capacity_refill_matches_stable_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    assignments = 524_288
    num_experts = 256
    capacity = assignments // num_experts
    experts = jax.random.randint(
        jax.random.key(52),
        (assignments,),
        minval=0,
        maxval=num_experts,
        dtype=jnp.int32,
    )
    experts = jnp.where(jnp.arange(assignments) % 4 == 0, 0, experts)

    order = jnp.argsort(experts, stable=True)
    expert_counts = jnp.bincount(experts, length=num_experts).astype(jnp.int32)
    segment_starts = jnp.cumsum(expert_counts) - expert_counts
    sorted_ranks = jnp.arange(assignments, dtype=jnp.int32) - segment_starts[experts[order]]
    local_ranks = jnp.zeros_like(sorted_ranks).at[order].set(sorted_ranks)
    keep = local_ranks < capacity
    occupied = jnp.minimum(expert_counts, capacity)

    vacancy_experts = jnp.broadcast_to(
        jnp.arange(num_experts, dtype=jnp.int32)[None, :],
        (capacity, num_experts),
    ).reshape(-1)
    vacancy_slots = jnp.broadcast_to(
        jnp.arange(capacity, dtype=jnp.int32)[:, None],
        (capacity, num_experts),
    ).reshape(-1)
    vacancy_mask = (jnp.arange(capacity, dtype=jnp.int32)[:, None] >= occupied[None, :]).reshape(-1)
    vacancy_rank = jnp.cumsum(vacancy_mask.astype(jnp.int32), dtype=jnp.int32) - 1
    vacancy_destination = jnp.where(vacancy_mask, vacancy_rank, assignments)
    compact_vacancy_experts = (
        jnp.full((assignments,), num_experts, dtype=jnp.int32)
        .at[vacancy_destination]
        .set(vacancy_experts, mode="drop")
    )
    compact_vacancy_slots = (
        jnp.full((assignments,), capacity, dtype=jnp.int32).at[vacancy_destination].set(vacancy_slots, mode="drop")
    )
    overflow_rank = jnp.cumsum(jnp.logical_not(keep).astype(jnp.int32), dtype=jnp.int32) - 1
    compact_index = jnp.maximum(overflow_rank, 0)
    expected_experts = jnp.where(keep, experts, compact_vacancy_experts[compact_index])
    expected_slots = jnp.where(keep, local_ranks, compact_vacancy_slots[compact_index])
    expected_replacements = jnp.sum(jnp.logical_not(keep), dtype=jnp.int32)

    actual_experts, actual_slots, actual_replacements = jax.jit(
        sonic_capacity_refill,
        static_argnames=("num_experts", "capacity"),
    )(
        experts,
        num_experts=num_experts,
        capacity=capacity,
    )
    jax.block_until_ready(
        (
            actual_experts,
            actual_slots,
            actual_replacements,
            expected_experts,
            expected_slots,
            expected_replacements,
        )
    )

    np.testing.assert_array_equal(np.asarray(actual_experts), np.asarray(expected_experts))
    np.testing.assert_array_equal(np.asarray(actual_slots), np.asarray(expected_slots))
    np.testing.assert_array_equal(np.asarray(actual_replacements), np.asarray(expected_replacements))


def test_sonic_gather_sum_h5120_value_and_gradients_match_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    tokens = 16
    topk = 8
    hidden_dim = 5120
    send_size = tokens * topk
    flat_positions = (jnp.arange(send_size, dtype=jnp.int32) * 31) % send_size
    dispatch_positions = flat_positions.reshape(tokens, topk)
    dispatch_positions = dispatch_positions.at[-1, -2:].set(send_size)
    keep = dispatch_positions < send_size

    output_rows = jnp.arange(send_size, dtype=jnp.int32) % 127
    output_columns = jnp.arange(hidden_dim, dtype=jnp.int32) % 31
    dispatch_output = (
        output_rows[:, None].astype(jnp.bfloat16) + output_columns[None, :].astype(jnp.bfloat16)
    ) / jnp.asarray(127, dtype=jnp.bfloat16)
    raw_weights = (jnp.arange(send_size, dtype=jnp.float32) % topk + 1).reshape(tokens, topk)
    combine_weights = jnp.where(keep, raw_weights / jnp.sum(raw_weights, axis=-1, keepdims=True), 0)
    cotangent_rows = jnp.arange(tokens, dtype=jnp.int32) % 61
    output_cotangent = (
        cotangent_rows[:, None].astype(jnp.bfloat16) + output_columns[None, :].astype(jnp.bfloat16)
    ) / jnp.asarray(61, dtype=jnp.bfloat16)

    def reference_combine(dispatch_output, combine_weights):
        padded_dispatch_output = jnp.concatenate(
            [dispatch_output, jnp.zeros((1, hidden_dim), dtype=dispatch_output.dtype)]
        )
        gathered = padded_dispatch_output[dispatch_positions]
        return jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(dispatch_output.dtype)

    def evaluate(combine, dispatch_output, combine_weights):
        output, pullback = jax.vjp(combine, dispatch_output, combine_weights)
        dispatch_grad, weights_grad = pullback(output_cotangent)
        return output, dispatch_grad, weights_grad

    sonic_result = jax.jit(
        lambda output, weights: evaluate(
            lambda output_arg, weights_arg: sonic_gather_sum(output_arg, dispatch_positions, weights_arg),
            output,
            weights,
        )
    )(dispatch_output, combine_weights)
    reference_result = jax.jit(lambda output, weights: evaluate(reference_combine, output, weights))(
        dispatch_output, combine_weights
    )
    jax.block_until_ready((sonic_result, reference_result))

    output_diff = np.abs(np.asarray(sonic_result[0], dtype=np.float32) - np.asarray(reference_result[0]))
    dispatch_grad_diff = np.abs(
        np.asarray(sonic_result[1], dtype=np.float32) - np.asarray(reference_result[1], dtype=np.float32)
    )
    weights_grad_diff = np.abs(np.asarray(sonic_result[2]) - np.asarray(reference_result[2]))
    assert float(np.max(output_diff)) <= 5e-3
    assert float(np.mean(output_diff)) <= 1e-4
    assert float(np.max(dispatch_grad_diff)) == 0.0
    assert float(np.mean(dispatch_grad_diff)) == 0.0
    assert float(np.max(weights_grad_diff)) <= 1e-3
    assert float(np.mean(weights_grad_diff)) <= 1e-4


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
    ("implementation", "fixed_a2a", "precomputed_slots"),
    [
        ("ring", False, False),
        ("ragged_all_to_all", False, False),
        ("ragged_all_to_all", True, False),
        ("ragged_all_to_all", True, True),
    ],
)
def test_moe_ep_path_lowers_on_abstract_mesh(
    implementation: MoeImplementation,
    fixed_a2a: bool,
    precomputed_slots: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    if fixed_a2a:
        monkeypatch.setenv("SCALE_A2A_FIXED", "1")
        monkeypatch.setenv("SCALE_A2A_CHUNKS", "1" if precomputed_slots else "2")
        monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")

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
        dispatch_slots = jax.ShapeDtypeStruct(
            shape=(tokens, topk),
            dtype=jnp.int32,
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

        def f(x, sel, cw, up_gate, down, slots):
            return moe_mlp(
                x,
                sel,
                cw,
                up_gate,
                down,
                activation=ActivationFunctionEnum.silu,
                implementation=implementation,
                mesh=mesh,
                dispatch_slots=slots if precomputed_slots else None,
            )

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = (
            jax.jit(f)
            .trace(x, selected_experts, combine_weights, w_up_gate, w_down, dispatch_slots)
            .lower(lowering_platforms=(platform,))
        )
        assert lowered is not None


def test_fixed_a2a_matches_dense_reference_on_one_expert_shard(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
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
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(41),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)

    def fixed_a2a(x, selected_experts, combine_weights, w_up_gate, w_down):
        return _fixed_a2a_core(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=2.0,
        )

    fixed_a2a_sharded = jax.shard_map(
        fixed_a2a,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    with jax.set_mesh(mesh):
        actual, dropped = fixed_a2a_sharded(x, selected_experts, combine_weights, w_up_gate, w_down)

    selected_w13 = w_up_gate[selected_experts]
    hidden = jnp.einsum("th,tkhi->tki", x, selected_w13)
    gate, up = jnp.split(hidden, [intermediate_dim], axis=-1)
    expert_output = jnp.einsum(
        "tki,tkih->tkh",
        jax.nn.silu(gate) * up,
        w_down[selected_experts],
    )
    expected = jnp.einsum("tkh,tk->th", expert_output, combine_weights)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert int(dropped) == 0


def test_fixed_a2a_precomputed_slots_match_internal_ranking(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
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
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(48),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    dispatch_slots = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32)[:, None], (tokens, topk))

    def fixed_a2a(x, selected_experts, combine_weights, w_up_gate, w_down, dispatch_slots):
        ranked = _fixed_a2a_core(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=1.0,
        )
        precomputed = _fixed_a2a_core(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            dispatch_slots,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=1.0,
        )
        return ranked, precomputed

    fixed_a2a_sharded = jax.shard_map(
        fixed_a2a,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P()),
        out_specs=((P(), P()), (P(), P())),
        check_vma=False,
    )
    with jax.set_mesh(mesh):
        ranked, precomputed = fixed_a2a_sharded(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            dispatch_slots,
        )

    np.testing.assert_array_equal(np.asarray(precomputed[0]), np.asarray(ranked[0]))
    np.testing.assert_array_equal(np.asarray(precomputed[1]), np.asarray(ranked[1]))


def test_fixed_a2a_gather_dispatch_is_bit_exact(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    mesh = Mesh(
        np.asarray([jax.devices()[0]]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )
    tokens = 5
    hidden_dim = 4
    intermediate_dim = 6
    num_experts = 2
    topk = 2
    inputs = _make_inputs(
        key=jax.random.key(42),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    inputs = (
        inputs[0],
        _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts),
        *inputs[2:],
    )

    def run_fixed_a2a(gather_dispatch: bool):
        if gather_dispatch:
            monkeypatch.setenv("SCALE_A2A_GATHER_DISPATCH", "1")
        else:
            monkeypatch.delenv("SCALE_A2A_GATHER_DISPATCH", raising=False)

        def fixed_a2a(x, selected_experts, combine_weights, w_up_gate, w_down):
            return _fixed_a2a_core(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=2.0,
            )

        sharded = jax.shard_map(
            fixed_a2a,
            mesh=mesh,
            in_specs=(P(), P(), P(), P(), P()),
            out_specs=(P(), P()),
            check_vma=False,
        )
        with jax.set_mesh(mesh):
            return jax.tree.map(np.asarray, sharded(*inputs))

    scatter_output, scatter_dropped = run_fixed_a2a(gather_dispatch=False)
    gather_output, gather_dropped = run_fixed_a2a(gather_dispatch=True)

    np.testing.assert_array_equal(gather_output, scatter_output)
    np.testing.assert_array_equal(gather_dropped, scatter_dropped)


def test_fixed_a2a_gather_dispatch_gradients_match(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    mesh = Mesh(
        np.asarray([jax.devices()[0]]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )
    tokens = 5
    hidden_dim = 4
    intermediate_dim = 6
    num_experts = 2
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(43),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    output_cotangent = jax.random.normal(jax.random.key(44), (tokens, hidden_dim))

    def run_fixed_a2a_grad(gather_dispatch: bool):
        if gather_dispatch:
            monkeypatch.setenv("SCALE_A2A_GATHER_DISPATCH", "1")
        else:
            monkeypatch.delenv("SCALE_A2A_GATHER_DISPATCH", raising=False)

        def fixed_a2a_loss(x, selected_experts, combine_weights, w_up_gate, w_down):
            output, _ = _fixed_a2a_core(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=2.0,
            )
            return jnp.sum(output * output_cotangent)

        sharded_loss = jax.shard_map(
            fixed_a2a_loss,
            mesh=mesh,
            in_specs=(P(), P(), P(), P(), P()),
            out_specs=P(),
            check_vma=False,
        )
        value_and_grad = jax.value_and_grad(sharded_loss, argnums=(0, 2, 3, 4))
        with jax.set_mesh(mesh):
            return jax.tree.map(
                np.asarray,
                value_and_grad(x, selected_experts, combine_weights, w_up_gate, w_down),
            )

    scatter_value, scatter_grad = run_fixed_a2a_grad(gather_dispatch=False)
    gather_value, gather_grad = run_fixed_a2a_grad(gather_dispatch=True)

    np.testing.assert_array_equal(gather_value, scatter_value)
    jax.tree.map(
        lambda gather, scatter: np.testing.assert_allclose(gather, scatter, rtol=1e-5, atol=1e-5),
        gather_grad,
        scatter_grad,
    )


@pytest.mark.parametrize(
    ("pack_dispatch", "pack_combine"),
    [(True, False), (False, True), (True, True)],
)
def test_fixed_a2a_packed_collectives_match_per_expert_collectives(
    monkeypatch: pytest.MonkeyPatch,
    pack_dispatch: bool,
    pack_combine: bool,
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices")

    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    monkeypatch.setenv("SCALE_A2A_GATHER_DISPATCH", "1")
    mesh = Mesh(
        np.asarray(devices[:2]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )
    tokens = 8
    hidden_dim = 4
    intermediate_dim = 6
    num_experts = 4
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(45),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    output_cotangent = jax.random.normal(jax.random.key(46), (tokens, hidden_dim))

    def run_fixed_a2a(*, use_packed_dispatch: bool, use_packed_combine: bool):
        if use_packed_dispatch:
            monkeypatch.setenv("SCALE_A2A_PACK_DISPATCH", "1")
        else:
            monkeypatch.delenv("SCALE_A2A_PACK_DISPATCH", raising=False)
        if use_packed_combine:
            monkeypatch.setenv("SCALE_A2A_PACK_COMBINE", "1")
        else:
            monkeypatch.delenv("SCALE_A2A_PACK_COMBINE", raising=False)

        def fixed_a2a_loss(x, selected_experts, combine_weights, w_up_gate, w_down, output_cotangent):
            output, _ = _fixed_a2a_core(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=2.0,
            )
            return jnp.sum(output * output_cotangent)

        sharded_loss = jax.shard_map(
            fixed_a2a_loss,
            mesh=mesh,
            in_specs=(P("expert"), P("expert"), P("expert"), P("expert"), P("expert"), P("expert")),
            out_specs=P(),
            check_vma=False,
        )
        value_and_grad = jax.value_and_grad(sharded_loss, argnums=(0, 2, 3, 4))
        with jax.set_mesh(mesh):
            sharding = NamedSharding(mesh, P("expert"))
            sharded_inputs = jax.tree.map(
                lambda array: jax.device_put(array, sharding),
                (x, selected_experts, combine_weights, w_up_gate, w_down, output_cotangent),
            )
            return jax.tree.map(
                np.asarray,
                value_and_grad(*sharded_inputs),
            )

    baseline_value, baseline_grad = run_fixed_a2a(use_packed_dispatch=False, use_packed_combine=False)
    packed_value, packed_grad = run_fixed_a2a(
        use_packed_dispatch=pack_dispatch,
        use_packed_combine=pack_combine,
    )

    np.testing.assert_array_equal(packed_value, baseline_value)
    jax.tree.map(np.testing.assert_array_equal, packed_grad, baseline_grad)


def test_fixed_a2a_custom_distributed_combine_matches_autodiff(monkeypatch: pytest.MonkeyPatch):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices")

    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    monkeypatch.setenv("SCALE_A2A_GATHER_DISPATCH", "1")
    monkeypatch.setenv("SCALE_A2A_PACK_DISPATCH", "1")
    monkeypatch.setenv("SCALE_A2A_PACK_COMBINE", "1")
    monkeypatch.delenv("SCALE_A2A_SONIC_COMBINE", raising=False)
    monkeypatch.delenv("SCALE_A2A_SONIC_DISTRIBUTED_COMBINE_GRAD", raising=False)
    mesh = Mesh(
        np.asarray(devices[:2]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )
    tokens = 8
    hidden_dim = 4
    intermediate_dim = 6
    num_experts = 4
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(47),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    output_cotangent = jax.random.normal(jax.random.key(48), (tokens, hidden_dim))

    def run_fixed_a2a(*, custom_distributed_combine: bool):
        if custom_distributed_combine:
            monkeypatch.setenv("SCALE_A2A_CUSTOM_DISTRIBUTED_COMBINE", "1")
        else:
            monkeypatch.delenv("SCALE_A2A_CUSTOM_DISTRIBUTED_COMBINE", raising=False)

        def fixed_a2a_loss(x, selected_experts, combine_weights, w_up_gate, w_down, output_cotangent):
            output, _ = _fixed_a2a_core(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=1.0,
            )
            return jnp.sum(output * output_cotangent)

        sharded_loss = jax.shard_map(
            fixed_a2a_loss,
            mesh=mesh,
            in_specs=(P("expert"), P("expert"), P("expert"), P("expert"), P("expert"), P("expert")),
            out_specs=P(),
            check_vma=False,
        )
        value_and_grad = jax.value_and_grad(sharded_loss, argnums=(0, 2, 3, 4))
        with jax.set_mesh(mesh):
            sharding = NamedSharding(mesh, P("expert"))
            sharded_inputs = jax.tree.map(
                lambda array: jax.device_put(array, sharding),
                (x, selected_experts, combine_weights, w_up_gate, w_down, output_cotangent),
            )
            return jax.tree.map(np.asarray, value_and_grad(*sharded_inputs))

    baseline_value, baseline_grad = run_fixed_a2a(custom_distributed_combine=False)
    custom_value, custom_grad = run_fixed_a2a(custom_distributed_combine=True)

    np.testing.assert_array_equal(custom_value, baseline_value)
    jax.tree.map(
        lambda custom, baseline: np.testing.assert_allclose(custom, baseline, rtol=1e-5, atol=1e-5),
        custom_grad,
        baseline_grad,
    )


def test_fixed_a2a_batched_expert_gemms_match_per_expert_gemms(monkeypatch: pytest.MonkeyPatch):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices")

    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    monkeypatch.setenv("SCALE_A2A_GATHER_DISPATCH", "1")
    monkeypatch.setenv("SCALE_A2A_PACK_DISPATCH", "1")
    monkeypatch.setenv("SCALE_A2A_PACK_COMBINE", "1")
    mesh = Mesh(
        np.asarray(devices[:2]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )
    tokens = 8
    hidden_dim = 4
    intermediate_dim = 6
    num_experts = 4
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(49),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    output_cotangent = jax.random.normal(jax.random.key(50), (tokens, hidden_dim))

    def run_fixed_a2a(*, batch_expert_gemms: bool, split_w13_gemms: bool = False):
        if batch_expert_gemms:
            monkeypatch.setenv("SCALE_A2A_BATCH_EXPERT_GEMMS", "1")
        else:
            monkeypatch.delenv("SCALE_A2A_BATCH_EXPERT_GEMMS", raising=False)
        if split_w13_gemms:
            monkeypatch.setenv("SCALE_A2A_SPLIT_W13_GEMMS", "1")
        else:
            monkeypatch.delenv("SCALE_A2A_SPLIT_W13_GEMMS", raising=False)

        def fixed_a2a_loss(x, selected_experts, combine_weights, w_up_gate, w_down, output_cotangent):
            output, _ = _fixed_a2a_core(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation_fn=jax.nn.silu,
                num_experts=num_experts,
                capacity_factor=1.0,
            )
            return jnp.sum(output * output_cotangent)

        sharded_loss = jax.shard_map(
            fixed_a2a_loss,
            mesh=mesh,
            in_specs=(P("expert"), P("expert"), P("expert"), P("expert"), P("expert"), P("expert")),
            out_specs=P(),
            check_vma=False,
        )
        value_and_grad = jax.value_and_grad(sharded_loss, argnums=(0, 2, 3, 4))
        with jax.set_mesh(mesh):
            sharding = NamedSharding(mesh, P("expert"))
            sharded_inputs = jax.tree.map(
                lambda array: jax.device_put(array, sharding),
                (x, selected_experts, combine_weights, w_up_gate, w_down, output_cotangent),
            )
            return jax.tree.map(np.asarray, value_and_grad(*sharded_inputs))

    baseline_value, baseline_grad = run_fixed_a2a(batch_expert_gemms=False)
    batched_value, batched_grad = run_fixed_a2a(batch_expert_gemms=True)
    split_value, split_grad = run_fixed_a2a(
        batch_expert_gemms=True,
        split_w13_gemms=True,
    )

    np.testing.assert_allclose(batched_value, baseline_value, rtol=1e-5, atol=1e-5)
    jax.tree.map(
        lambda batched, baseline: np.testing.assert_allclose(batched, baseline, rtol=1e-5, atol=1e-5),
        batched_grad,
        baseline_grad,
    )
    np.testing.assert_allclose(split_value, batched_value, rtol=1e-5, atol=1e-5)
    jax.tree.map(
        lambda split, batched: np.testing.assert_allclose(split, batched, rtol=1e-5, atol=1e-5),
        split_grad,
        batched_grad,
    )


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


@pytest.mark.parametrize(
    ("dense_experts", "mnnvl_transport"),
    [(False, False), (True, False), (False, True)],
)
def test_receiver_clipped_fixed_a2a_matches_ring_value_and_grad(
    monkeypatch: pytest.MonkeyPatch,
    dense_experts: bool,
    mnnvl_transport: bool,
):
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")
    monkeypatch.setenv("SCALE_A2A_FIXED", "1")
    monkeypatch.setenv("SCALE_A2A_RECEIVER_CLIP", "1")
    monkeypatch.setenv("SCALE_A2A_RECEIVER_SENDER_CAPACITY_FACTOR", "8")
    monkeypatch.setenv("SCALE_A2A_RECEIVER_RAGGED_FALLBACK", "0")
    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    if dense_experts:
        monkeypatch.setenv("SCALE_A2A_RECEIVER_DENSE_EXPERTS", "1")
    if mnnvl_transport:
        monkeypatch.setenv("SCALE_A2A_MNNVL_TRANSPORT", "1")
        monkeypatch.setattr(ep_ragged_a2a, "_mnnvl_dispatch_gather", _reference_mnnvl_dispatch_gather)
        monkeypatch.setattr(ep_ragged_a2a, "mnnvl_combine", _reference_mnnvl_combine)

    tokens = len(jax.devices()) * 8
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = 4
    topk = 2

    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
            key=jax.random.key(51),
            tokens=tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            topk=topk,
        )
        output_cotangent = jax.random.normal(jax.random.key(52), x.shape, dtype=x.dtype)

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        output_cotangent = jax.sharding.reshard(output_cotangent, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        def run(implementation: MoeImplementation):
            def loss(x, combine_weights, w_up_gate, w_down):
                output, dropped = moe_mlp(
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
                return jnp.sum(output * output_cotangent), dropped

            return jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(
                x,
                combine_weights,
                w_up_gate,
                w_down,
            )

        (ring_value, ring_dropped), ring_grad = run("ring")
        (fixed_value, fixed_dropped), fixed_grad = run("ragged_all_to_all")

    np.testing.assert_allclose(np.asarray(fixed_value), np.asarray(ring_value), rtol=1e-5, atol=1e-5)
    assert int(fixed_dropped) == int(ring_dropped)
    jax.tree.map(
        lambda fixed, ring: np.testing.assert_allclose(
            np.asarray(fixed),
            np.asarray(ring),
            rtol=1e-5,
            atol=1e-5,
        ),
        fixed_grad,
        ring_grad,
    )


def test_receiver_destination_pooled_a2a_matches_dense_value_and_grad(
    monkeypatch: pytest.MonkeyPatch,
):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices")
    mesh = Mesh(
        np.asarray(devices).reshape(1, len(devices), 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    monkeypatch.setenv("SCALE_A2A_FIXED", "1")
    monkeypatch.setenv("SCALE_A2A_RECEIVER_CLIP", "1")
    monkeypatch.setenv("SCALE_A2A_RECEIVER_DESTINATION_POOL", "1")
    monkeypatch.setenv("SCALE_A2A_RECEIVER_DESTINATION_CAPACITY_FACTOR", "1")
    monkeypatch.setenv("SCALE_A2A_RECEIVER_RAGGED_FALLBACK", "0")
    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    monkeypatch.setenv("RAGGED_DOT_IMPL", "xla")

    tokens = len(devices) * 8
    hidden_dim = 16
    intermediate_dim = 24
    num_experts = len(devices) * 2
    topk = 2
    x, _, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(53),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    x = x * 0.1
    w_up_gate = w_up_gate * 0.1
    w_down = w_down * 0.1
    selected_experts = jnp.arange(tokens * topk, dtype=jnp.int32).reshape(tokens, topk) % num_experts
    output_cotangent = 0.1 * jax.random.normal(jax.random.key(54), x.shape, dtype=x.dtype)

    def dense_loss(x, combine_weights, w_up_gate, w_down):
        selected_w13 = w_up_gate[selected_experts]
        hidden = jnp.einsum("th,tkhi->tki", x, selected_w13)
        gate, up = jnp.split(hidden, [intermediate_dim], axis=-1)
        expert_output = jnp.einsum(
            "tki,tkih->tkh",
            jax.nn.silu(gate) * up,
            w_down[selected_experts],
        )
        output = jnp.einsum("tkh,tk->th", expert_output, combine_weights)
        return jnp.sum(output * output_cotangent)

    dense_value, dense_grad = jax.value_and_grad(
        dense_loss,
        argnums=(0, 1, 2, 3),
    )(x, combine_weights, w_up_gate, w_down)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        sharded_x = jax.sharding.reshard(x, batch_sharding)
        sharded_selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        sharded_combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        sharded_output_cotangent = jax.sharding.reshard(output_cotangent, batch_sharding)
        sharded_w13 = jax.sharding.reshard(w_up_gate, expert_sharding)
        sharded_w2 = jax.sharding.reshard(w_down, expert_sharding)

        def fixed_loss(x, combine_weights, w_up_gate, w_down):
            output, dropped = moe_mlp(
                x,
                sharded_selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                implementation="ragged_all_to_all",
                mesh=None,
                report_capacity_overflow=True,
                capacity_factor=1.0,
            )
            return jnp.sum(output * sharded_output_cotangent), dropped

        (fixed_value, fixed_dropped), fixed_grad = jax.value_and_grad(
            fixed_loss,
            argnums=(0, 1, 2, 3),
            has_aux=True,
        )(sharded_x, sharded_combine_weights, sharded_w13, sharded_w2)

    np.testing.assert_allclose(np.asarray(fixed_value), np.asarray(dense_value), rtol=1e-5, atol=1e-5)
    assert int(fixed_dropped) == 0
    jax.tree.map(
        lambda fixed, dense: np.testing.assert_allclose(
            np.asarray(fixed),
            np.asarray(dense),
            rtol=1e-5,
            atol=1e-5,
        ),
        fixed_grad,
        dense_grad,
    )


def test_round_robin_ppermute_matches_all_to_all():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices")

    peer_count = len(devices)
    values = jnp.arange(
        peer_count * 2 * peer_count * 3,
        dtype=jnp.int32,
    ).reshape(peer_count, 2, peer_count, 3)

    baseline = jax.pmap(
        lambda x: jax.lax.all_to_all(
            x,
            "expert",
            split_axis=1,
            concat_axis=1,
            tiled=True,
        ),
        axis_name="expert",
    )(values)
    round_robin = jax.pmap(
        lambda x: _round_robin_ppermute_all_to_all(
            x,
            axis_name="expert",
            peer_axis=1,
        ),
        axis_name="expert",
    )(values)

    np.testing.assert_array_equal(np.asarray(round_robin), np.asarray(baseline))


def test_same_expert_clone_dispatch_metadata_is_dropless_and_preserves_experts():
    assignments = [
        jnp.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4], dtype=jnp.int32),
        jnp.asarray([0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=jnp.int32),
    ]
    expert_shards = len(assignments)
    num_experts = 5
    assignments_per_sender = assignments[0].size
    all_group_sizes = jnp.stack(
        [jnp.bincount(sender_assignments, length=num_experts) for sender_assignments in assignments]
    ).astype(jnp.int32)
    sender_destination_capacity = assignments_per_sender // expert_shards + num_experts
    receiver_capacity = assignments_per_sender + num_experts

    received: list[list[tuple[int, int]]] = [[] for _ in range(expert_shards)]
    expected_receiver_group_sizes = None
    for sender_index, flat_experts in enumerate(assignments):
        transport_position, receiver_slot, receiver_group_sizes, overflow = _same_expert_clone_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            jnp.asarray(sender_index, dtype=jnp.int32),
            sender_destination_capacity=sender_destination_capacity,
            receiver_capacity=receiver_capacity,
        )
        assert int(overflow) == 0
        assert len(np.unique(np.asarray(transport_position))) == assignments_per_sender
        expected_receiver_group_sizes = receiver_group_sizes
        for expert, position, slot in zip(
            np.asarray(flat_experts),
            np.asarray(transport_position),
            np.asarray(receiver_slot),
            strict=True,
        ):
            destination = int(position) // sender_destination_capacity
            received[destination].append((int(slot), int(expert)))

    assert expected_receiver_group_sizes is not None
    receiver_group_sizes = np.asarray(expected_receiver_group_sizes)
    receiver_group_offsets = np.cumsum(receiver_group_sizes, axis=1) - receiver_group_sizes
    for receiver_index, received_assignments in enumerate(received):
        slots = [slot for slot, _ in received_assignments]
        assert len(slots) == len(set(slots))
        assert len(slots) == int(np.sum(receiver_group_sizes[receiver_index]))
        for slot, expert in received_assignments:
            start = int(receiver_group_offsets[receiver_index, expert])
            end = start + int(receiver_group_sizes[receiver_index, expert])
            assert start <= slot < end


def test_same_expert_pooled_dispatch_metadata_fills_receivers_exactly():
    assignments = [
        jnp.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4], dtype=jnp.int32),
        jnp.asarray([0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=jnp.int32),
    ]
    expert_shards = len(assignments)
    num_experts = 5
    assignments_per_sender = assignments[0].size
    all_group_sizes = jnp.stack(
        [jnp.bincount(sender_assignments, length=num_experts) for sender_assignments in assignments]
    ).astype(jnp.int32)
    sender_destination_capacity = assignments_per_sender // expert_shards + num_experts

    received: list[list[tuple[int, int]]] = [[] for _ in range(expert_shards)]
    expected_receiver_group_sizes = None
    for sender_index, flat_experts in enumerate(assignments):
        transport_position, receiver_slot, receiver_group_sizes, overflow = _same_expert_pooled_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            jnp.asarray(sender_index, dtype=jnp.int32),
            sender_destination_capacity=sender_destination_capacity,
            receiver_capacity=assignments_per_sender,
        )
        assert int(overflow) == 0
        assert len(np.unique(np.asarray(transport_position))) == assignments_per_sender
        expected_receiver_group_sizes = receiver_group_sizes
        for expert, position, slot in zip(
            np.asarray(flat_experts),
            np.asarray(transport_position),
            np.asarray(receiver_slot),
            strict=True,
        ):
            destination = int(position) // sender_destination_capacity
            received[destination].append((int(slot), int(expert)))

    assert expected_receiver_group_sizes is not None
    receiver_group_sizes = np.asarray(expected_receiver_group_sizes)
    receiver_group_offsets = np.cumsum(receiver_group_sizes, axis=1) - receiver_group_sizes
    np.testing.assert_array_equal(
        np.sum(receiver_group_sizes, axis=1),
        np.full((expert_shards,), assignments_per_sender),
    )
    assert int(np.count_nonzero(receiver_group_sizes)) <= num_experts + expert_shards - 1
    for receiver_index, received_assignments in enumerate(received):
        slots = [slot for slot, _ in received_assignments]
        assert sorted(slots) == list(range(assignments_per_sender))
        for slot, expert in received_assignments:
            start = int(receiver_group_offsets[receiver_index, expert])
            end = start + int(receiver_group_sizes[receiver_index, expert])
            assert start <= slot < end


def test_same_expert_echo_dispatch_metadata_clones_only_hot_receiver_overflow():
    assignments = [
        jnp.asarray([0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5], dtype=jnp.int32),
        jnp.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5], dtype=jnp.int32),
        jnp.asarray([0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5], dtype=jnp.int32),
    ]
    expert_shards = len(assignments)
    num_experts = 6
    local_experts = num_experts // expert_shards
    receiver_capacity = assignments[0].size
    all_group_sizes = jnp.stack(
        [jnp.bincount(sender_assignments, length=num_experts) for sender_assignments in assignments]
    ).astype(jnp.int32)

    received: list[list[tuple[int, int]]] = [[] for _ in range(expert_shards)]
    expected_receiver_group_sizes = None
    for sender_index, flat_experts in enumerate(assignments):
        destination, receiver_slot, receiver_group_sizes, overflow = _same_expert_echo_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            jnp.asarray(sender_index, dtype=jnp.int32),
            receiver_capacity=receiver_capacity,
            max_receiver_segments=3,
        )
        assert int(overflow) == 0
        expected_receiver_group_sizes = receiver_group_sizes
        for expert, receiver, slot in zip(
            np.asarray(flat_experts),
            np.asarray(destination),
            np.asarray(receiver_slot),
            strict=True,
        ):
            received[int(receiver)].append((int(slot), int(expert)))

    assert expected_receiver_group_sizes is not None
    receiver_group_sizes = np.asarray(expected_receiver_group_sizes)
    np.testing.assert_array_equal(
        np.sum(receiver_group_sizes, axis=1),
        np.full((expert_shards,), receiver_capacity),
    )
    np.testing.assert_array_equal(
        np.count_nonzero(receiver_group_sizes, axis=1),
        np.array([2, 3, 3]),
    )

    global_group_sizes = np.sum(np.asarray(all_group_sizes), axis=0)
    retained_home = np.zeros((num_experts,), dtype=np.int32)
    receiver_group_offsets = np.cumsum(receiver_group_sizes, axis=1) - receiver_group_sizes
    for receiver_index, received_assignments in enumerate(received):
        slots = [slot for slot, _ in received_assignments]
        assert sorted(slots) == list(range(receiver_capacity))
        for slot, expert in received_assignments:
            start = int(receiver_group_offsets[receiver_index, expert])
            end = start + int(receiver_group_sizes[receiver_index, expert])
            assert start <= slot < end
            if receiver_index == expert // local_experts:
                retained_home[expert] += 1

    np.testing.assert_array_equal(retained_home, np.array([8, 4, 5, 3, 4, 4], dtype=np.int32))
    assert int(np.sum(global_group_sizes - retained_home)) == 8


def test_same_expert_echo_fixed_transport_metadata_counts_only_envelope_overflow():
    destination = jnp.array([0, 1, 0, 2, 1, 0], dtype=jnp.int32)

    transport_position, keep, envelope_overflow = _same_expert_echo_fixed_transport_metadata(
        destination,
        expert_shards=2,
        sender_destination_capacity=2,
    )

    np.testing.assert_array_equal(transport_position, np.array([0, 2, 1, 4, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(keep, np.array([True, True, True, False, True, False]))
    assert int(envelope_overflow) == 1


def test_same_expert_compact_transport_metadata_groups_valid_rows_by_destination():
    destination = jnp.asarray([2, 0, 1, 3, 0, 2, 1, 3], dtype=jnp.int32)
    receiver_slot = jnp.asarray([5, 1, 4, 8, 0, 3, 2, 8], dtype=jnp.int32)

    position, packed_destination, packed_slot, keep = _same_expert_compact_transport_metadata(
        destination,
        receiver_slot,
        expert_shards=3,
        receiver_capacity=8,
    )

    np.testing.assert_array_equal(position, np.array([4, 0, 2, 8, 1, 5, 3, 8], dtype=np.int32))
    np.testing.assert_array_equal(keep, np.array([True, True, True, False, True, True, True, False]))
    np.testing.assert_array_equal(
        packed_destination,
        np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        packed_slot,
        np.array([1, 0, 4, 2, 5, 3, 8, 8], dtype=np.int32),
    )


def test_same_expert_hybridep_routing_encodes_receiver_segments():
    flat_experts = jnp.asarray([0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5], dtype=jnp.int32)
    destination = jnp.asarray([0, 0, 0, 1, 0, 1, 1, 2, 1, 2, 2, 2], dtype=jnp.int32)
    receiver_group_sizes = jnp.asarray(
        [
            [2, 2, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 2, 2],
        ],
        dtype=jnp.int32,
    )
    combine_weights = jnp.arange(1, flat_experts.size + 1, dtype=jnp.float32).reshape(6, 2)
    routing_map, probabilities = _same_expert_hybridep_routing(
        destination,
        flat_experts,
        combine_weights,
        receiver_group_sizes,
        max_receiver_segments=3,
    )

    receiver_group_position = np.cumsum(np.asarray(receiver_group_sizes) > 0, axis=1) - 1
    expected_synthetic_expert = (
        np.asarray(destination) * 3
        + receiver_group_position[
            np.asarray(destination),
            np.asarray(flat_experts),
        ]
    )
    token_index = np.repeat(np.arange(6), 2)
    np.testing.assert_array_equal(np.sum(np.asarray(routing_map), axis=1), np.full((6,), 2))
    np.testing.assert_array_equal(
        np.asarray(routing_map)[token_index, expected_synthetic_expert],
        np.ones((flat_experts.size,), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        np.asarray(probabilities)[token_index, expected_synthetic_expert],
        np.asarray(combine_weights).reshape(-1),
    )


def test_same_expert_pooled_dispatch_metadata_caps_receiver_segments():
    assignments = [
        jnp.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4], dtype=jnp.int32),
        jnp.asarray([0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=jnp.int32),
    ]
    expert_shards = len(assignments)
    num_experts = 5
    assignments_per_sender = assignments[0].size
    all_group_sizes = jnp.stack(
        [jnp.bincount(sender_assignments, length=num_experts) for sender_assignments in assignments]
    ).astype(jnp.int32)

    total_overflow = 0
    expected_receiver_group_sizes = None
    for sender_index, flat_experts in enumerate(assignments):
        transport_position, _, receiver_group_sizes, overflow = _same_expert_pooled_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            jnp.asarray(sender_index, dtype=jnp.int32),
            sender_destination_capacity=assignments_per_sender,
            receiver_capacity=assignments_per_sender,
            max_receiver_segments=1,
        )
        total_overflow += int(overflow)
        expected_receiver_group_sizes = receiver_group_sizes
        assert int(jnp.sum(transport_position < expert_shards * assignments_per_sender)) == (
            assignments_per_sender - int(overflow)
        )

    assert expected_receiver_group_sizes is not None
    receiver_group_sizes = np.asarray(expected_receiver_group_sizes)
    np.testing.assert_array_less(np.count_nonzero(receiver_group_sizes, axis=1), 2)
    assert total_overflow == expert_shards * assignments_per_sender - int(np.sum(receiver_group_sizes))


def test_sparse_clone_weight_metadata_matches_sender_receiver_segments():
    receiver_group_sizes = jnp.asarray(
        [
            [3, 2, 0, 0, 4, 3],
            [0, 1, 4, 4, 2, 1],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=jnp.int32,
    )
    expert_shards = receiver_group_sizes.shape[0]
    local_experts = receiver_group_sizes.shape[1] // expert_shards
    send_matrix = np.zeros((expert_shards, expert_shards), dtype=np.int32)
    recv_matrix = np.zeros_like(send_matrix)

    for shard_index in range(expert_shards):
        (
            packed_local_experts,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            compact_group_sizes,
            overflow,
        ) = _sparse_clone_weight_metadata(
            receiver_group_sizes,
            jnp.asarray(shard_index, dtype=jnp.int32),
            local_experts=local_experts,
            max_receiver_segments=receiver_group_sizes.shape[1],
            topk=2,
        )
        del input_offsets, output_offsets
        assert int(overflow) == 0
        send_matrix[shard_index] = np.asarray(send_sizes)
        recv_matrix[shard_index] = np.asarray(recv_sizes)

        expected_groups = np.asarray(receiver_group_sizes[shard_index])
        expected_groups = expected_groups[expected_groups > 0]
        np.testing.assert_array_equal(
            np.asarray(compact_group_sizes)[: expected_groups.size],
            expected_groups,
        )
        np.testing.assert_array_equal(
            np.asarray(compact_group_sizes)[expected_groups.size :],
            0,
        )

        local_needed = np.asarray(
            receiver_group_sizes[:, shard_index * local_experts : (shard_index + 1) * local_experts] > 0
        )
        expected_local_experts = np.broadcast_to(np.arange(local_experts)[None, :], local_needed.shape)[local_needed]
        np.testing.assert_array_equal(
            np.asarray(packed_local_experts)[: expected_local_experts.size],
            expected_local_experts,
        )

    np.testing.assert_array_equal(send_matrix, recv_matrix.T)


@pytest.mark.parametrize(
    (
        "pooled_dispatch",
        "sparse_weights",
        "mnnvl_transport",
        "echo_ragged_transport",
        "echo_dispatch",
        "embedded_slot_metadata",
        "pipeline_chunks",
    ),
    [
        (False, False, False, False, False, False, 1),
        (True, False, False, False, False, False, 1),
        (True, True, False, False, False, False, 1),
        (True, True, False, False, True, False, 1),
        (True, True, False, False, True, False, 2),
        (True, True, False, False, True, True, 1),
        (True, True, True, False, True, False, 1),
        (True, True, False, True, True, False, 1),
    ],
)
def test_same_expert_cloned_fixed_a2a_matches_dense_value_and_grad(
    monkeypatch: pytest.MonkeyPatch,
    pooled_dispatch: bool,
    sparse_weights: bool,
    mnnvl_transport: bool,
    echo_ragged_transport: bool,
    echo_dispatch: bool,
    embedded_slot_metadata: bool,
    pipeline_chunks: int,
):
    mesh = _make_ep_mesh_or_none()
    if mesh is None:
        pytest.skip("requires an even number of >=2 devices")
    if sparse_weights and not mnnvl_transport and not any(device.platform == "gpu" for device in jax.devices()):
        monkeypatch.setattr(
            ep_ragged_a2a,
            "_sparse_clone_weight_exchange",
            _reference_sparse_clone_weight_exchange,
        )

    monkeypatch.setenv("SCALE_A2A_FIXED", "1")
    monkeypatch.setenv("SCALE_A2A_SAME_EXPERT_CLONES", "1")
    monkeypatch.setenv("SCALE_A2A_NO_BARRIER", "1")
    if pooled_dispatch:
        monkeypatch.setenv("SCALE_A2A_CLONE_POOLED", "1")
    if sparse_weights:
        monkeypatch.setenv("SCALE_A2A_CLONE_SPARSE_WEIGHTS", "1")
    if mnnvl_transport:
        monkeypatch.setenv("SCALE_A2A_MNNVL_TRANSPORT", "1")
        monkeypatch.setattr(ep_ragged_a2a, "_mnnvl_dispatch_gather", _reference_mnnvl_dispatch_gather)
        monkeypatch.setattr(ep_ragged_a2a, "mnnvl_combine", _reference_mnnvl_combine)
        monkeypatch.setattr(
            ep_ragged_a2a,
            "_sparse_clone_weight_exchange",
            _reference_sparse_clone_weight_exchange,
        )
    if echo_ragged_transport:
        monkeypatch.setenv("SCALE_A2A_ECHO_RAGGED_TRANSPORT", "1")
        if not any(device.platform == "gpu" for device in jax.devices()):
            monkeypatch.setattr(
                ep_ragged_a2a,
                "_echo_ragged_all_to_all",
                _reference_echo_ragged_all_to_all,
            )
    if echo_dispatch:
        monkeypatch.setenv("SCALE_A2A_ECHO_CLONES", "1")
    if embedded_slot_metadata:
        monkeypatch.setenv("SCALE_A2A_EMBED_SLOT_METADATA", "1")
        monkeypatch.setenv("SCALE_A2A_CLONE_TOKEN_PADDING_EXPERTS", "5")
    if pipeline_chunks > 1:
        monkeypatch.setenv("SCALE_A2A_CLONE_PIPELINE_CHUNKS", str(pipeline_chunks))
        monkeypatch.setenv("SCALE_A2A_CLONE_TOKEN_PADDING_EXPERTS", "2")
    tokens = len(jax.devices()) * 8
    hidden_dim = 8
    intermediate_dim = 12
    num_experts = 4
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(61),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    selected_experts = jnp.stack(
        [
            jnp.zeros((tokens,), dtype=jnp.int32),
            1 + jnp.arange(tokens, dtype=jnp.int32) % (num_experts - 1),
        ],
        axis=1,
    )
    output_cotangent = jax.random.normal(jax.random.key(62), x.shape, dtype=x.dtype)

    def dense_loss(x, combine_weights, w_up_gate, w_down):
        selected_w13 = w_up_gate[selected_experts]
        hidden = jnp.einsum("th,tkhi->tki", x, selected_w13)
        gate, up = jnp.split(hidden, [intermediate_dim], axis=-1)
        expert_output = jnp.einsum(
            "tki,tkih->tkh",
            jax.nn.silu(gate) * up,
            w_down[selected_experts],
        )
        output = jnp.einsum("tkh,tk->th", expert_output, combine_weights)
        return jnp.sum(output * output_cotangent)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        sharded_x = jax.sharding.reshard(x, batch_sharding)
        sharded_selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        sharded_combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        sharded_output_cotangent = jax.sharding.reshard(output_cotangent, batch_sharding)
        sharded_w13 = jax.sharding.reshard(w_up_gate, expert_sharding)
        sharded_w2 = jax.sharding.reshard(w_down, expert_sharding)

        def cloned_loss(x, combine_weights, w_up_gate, w_down):
            output, dropped = moe_mlp(
                x,
                sharded_selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                implementation="ragged_all_to_all",
                mesh=None,
                report_capacity_overflow=True,
                capacity_factor=1.0,
            )
            return jnp.sum(output * sharded_output_cotangent), dropped

        (cloned_value, dropped), cloned_grad = jax.value_and_grad(
            cloned_loss,
            argnums=(0, 1, 2, 3),
            has_aux=True,
        )(sharded_x, sharded_combine_weights, sharded_w13, sharded_w2)
        dense_value, dense_grad = jax.value_and_grad(
            dense_loss,
            argnums=(0, 1, 2, 3),
        )(x, combine_weights, w_up_gate, w_down)

    assert int(dropped) == 0
    np.testing.assert_allclose(np.asarray(cloned_value), np.asarray(dense_value), rtol=1e-5, atol=1e-5)
    jax.tree.map(
        lambda cloned, dense: np.testing.assert_allclose(
            np.asarray(cloned),
            np.asarray(dense),
            rtol=1e-5,
            atol=1e-5,
        ),
        cloned_grad,
        dense_grad,
    )


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


def test_bounded_compaction_and_expansion_value_and_grad():
    inputs = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    keep_mask = jnp.array([False, True, True, False, True, True])
    output_cotangent = jnp.arange(12, 24, dtype=jnp.float32).reshape(6, 2)

    def loss(inputs):
        compacted = _compact_by_keep_mask_to_size(inputs, keep_mask, output_size=3)
        expanded = _expand_from_keep_mask(compacted, keep_mask)
        return jnp.sum(expanded * output_cotangent), (compacted, expanded)

    (_, (compacted, expanded)), input_grad = jax.value_and_grad(loss, has_aux=True)(inputs)

    expected_compacted = jnp.stack([inputs[1], inputs[2], inputs[4]])
    expected_expanded = jnp.zeros_like(inputs).at[jnp.array([1, 2, 4])].set(expected_compacted)
    expected_grad = jnp.zeros_like(inputs).at[jnp.array([1, 2, 4])].set(output_cotangent[jnp.array([1, 2, 4])])
    np.testing.assert_array_equal(np.asarray(compacted), np.asarray(expected_compacted))
    np.testing.assert_array_equal(np.asarray(expanded), np.asarray(expected_expanded))
    np.testing.assert_array_equal(np.asarray(input_grad), np.asarray(expected_grad))
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


@pytest.mark.parametrize(
    ("flat_experts", "sender_index", "sender_expert_capacity", "expected_receiver_keep", "expected_transport_keep"),
    [
        ([0, 1, 0, 0], 0, 2, [True, False, True, True], [True, False, True, False]),
        (
            [2, 0, 2, 3, 2, 2, 0],
            1,
            4,
            [True, False, True, False, True, False, False],
            [True, False, True, False, True, False, False],
        ),
    ],
)
def test_receiver_clipped_fixed_dispatch_uses_ragged_sender_priority(
    flat_experts: list[int],
    sender_index: int,
    sender_expert_capacity: int,
    expected_receiver_keep: list[bool],
    expected_transport_keep: list[bool],
):
    all_group_sizes = jnp.array(
        [
            [3, 1, 0, 0],
            [2, 0, 4, 1],
        ],
        dtype=jnp.int32,
    )

    receiver_keep, transport_keep, clipped, receiver_dropped, envelope_overflow = _receiver_clipped_dispatch_metadata(
        jnp.asarray(flat_experts, dtype=jnp.int32),
        all_group_sizes,
        jnp.asarray(sender_index, dtype=jnp.int32),
        local_experts=2,
        receiver_capacity=3,
        sender_expert_capacity=sender_expert_capacity,
    )

    np.testing.assert_array_equal(np.asarray(receiver_keep), expected_receiver_keep)
    np.testing.assert_array_equal(np.asarray(transport_keep), expected_transport_keep)
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
    assert int(receiver_dropped) == len(flat_experts) - sum(expected_receiver_keep)
    assert int(envelope_overflow) == sum(expected_receiver_keep) - sum(expected_transport_keep)


def test_receiver_destination_pool_preserves_receiver_clipping_and_compacts_by_expert():
    sender_assignments = (
        jnp.asarray([0, 1, 0, 0], dtype=jnp.int32),
        jnp.asarray([2, 0, 2, 3, 2, 2, 0], dtype=jnp.int32),
    )
    all_group_sizes = jnp.asarray(
        [
            [3, 1, 0, 0],
            [2, 0, 4, 1],
        ],
        dtype=jnp.int32,
    )
    expected_positions = (
        np.asarray([0, 6, 1, 2], dtype=np.int32),
        np.asarray([3, 6, 4, 6, 5, 6, 6], dtype=np.int32),
    )

    for sender_index, (flat_experts, expected) in enumerate(zip(sender_assignments, expected_positions, strict=True)):
        keep, positions, clipped, receiver_dropped, envelope_overflow = _receiver_destination_pooled_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            jnp.asarray(sender_index, dtype=jnp.int32),
            local_experts=2,
            receiver_capacity=3,
            sender_destination_capacity=3,
        )
        np.testing.assert_array_equal(np.asarray(positions), expected)
        np.testing.assert_array_equal(np.asarray(keep), expected < 6)
        assert int(receiver_dropped) == flat_experts.size - int(jnp.sum(keep))
        assert int(envelope_overflow) == 0
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

    compact_position, valid, receiver_group_sizes = _receiver_destination_compact_positions(
        jnp.asarray([[2, 1], [1, 2]], dtype=jnp.int32),
        sender_destination_capacity=4,
        receiver_capacity=6,
    )
    np.testing.assert_array_equal(
        np.asarray(compact_position),
        np.asarray([[0, 1, 3, 6], [2, 4, 5, 6]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(valid),
        np.asarray([[True, True, True, False], [True, True, True, False]]),
    )
    np.testing.assert_array_equal(np.asarray(receiver_group_sizes), np.asarray([3, 3], dtype=np.int32))


def test_receiver_clipped_mnnvl_dispatch_uses_unique_expert_grouped_slots():
    sender_assignments = (
        jnp.asarray([0, 1, 0, 0], dtype=jnp.int32),
        jnp.asarray([2, 0, 2, 3, 2, 2, 0], dtype=jnp.int32),
    )
    all_group_sizes = jnp.asarray(
        [
            [3, 1, 0, 0],
            [2, 0, 4, 1],
        ],
        dtype=jnp.int32,
    )

    kept_destinations: list[np.ndarray] = []
    kept_slots: list[np.ndarray] = []
    for sender_index, flat_experts in enumerate(sender_assignments):
        keep, destination, receiver_slot, clipped, dropped = _receiver_clipped_mnnvl_dispatch_metadata(
            flat_experts,
            all_group_sizes,
            jnp.asarray(sender_index, dtype=jnp.int32),
            local_experts=2,
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
        assert int(dropped) == flat_experts.size - int(jnp.sum(keep))
        kept_destinations.append(np.asarray(destination[keep]))
        kept_slots.append(np.asarray(receiver_slot[keep]))

    np.testing.assert_array_equal(kept_destinations[0], np.asarray([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(kept_slots[0], np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(kept_destinations[1], np.asarray([1, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(kept_slots[1], np.asarray([0, 1, 2], dtype=np.int32))

    occupied = np.concatenate(
        [
            destinations.astype(np.int64) * 3 + slots
            for destinations, slots in zip(kept_destinations, kept_slots, strict=True)
        ]
    )
    assert np.unique(occupied).size == occupied.size
