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
from levanter.grug._moe.ep_ragged_all_to_all import (
    _fixed_a2a_core,
    _fixed_dispatch_gather_reference,
    _fixed_dispatch_gather_sonic,
    _fixed_dispatch_gather_sonic_grad,
    _receiver_clipped_dispatch_metadata,
    _round_robin_ppermute_all_to_all,
)
from levanter.grug._moe.sonic import (
    sonic_expert_local_rank,
    sonic_gather_sum,
    sonic_slot_weighted_grad,
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
    ("implementation", "fixed_a2a"),
    [
        ("ring", False),
        ("ragged_all_to_all", False),
        ("ragged_all_to_all", True),
    ],
)
def test_moe_ep_path_lowers_on_abstract_mesh(
    implementation: MoeImplementation, fixed_a2a: bool, monkeypatch: pytest.MonkeyPatch
):
    if fixed_a2a:
        monkeypatch.setenv("SCALE_A2A_FIXED", "1")
        monkeypatch.setenv("SCALE_A2A_CHUNKS", "2")
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


@pytest.mark.parametrize("dense_experts", [False, True])
def test_receiver_clipped_fixed_a2a_matches_ring_value_and_grad(
    monkeypatch: pytest.MonkeyPatch,
    dense_experts: bool,
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
