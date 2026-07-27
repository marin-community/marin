# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
from functools import partial

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
from levanter.grug._moe.ep_ring import (
    _ep_ring_two_chunk_fast_path_local,
    _moe_mlp_ep_ring_local,
    _moe_mlp_ep_ring_two_chunk_local,
    _validate_quack_bulk_ring_contract,
)
from levanter.grug._moe.ep_ring_fused import (
    _assignment_to_compact_rows,
    _moe_mlp_ep_ring_fused_local,
    ring_combine,
    ring_combine_triton,
    ring_dispatch_reference,
    ring_dispatch_triton,
)
from levanter.grug._moe.sonic import sonic_gather_sum
from levanter.grug._moe.sonic_quack import _require_quack, quack_mlp_varlen
from levanter.grug.grug_moe import (
    MoEExpertMlp,
    MoEExpertMlpPspecs,
    MoeImplementation,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _shard_a2a_params,
    moe_mlp,
    moe_mlp_accumulating_weight_gradient,
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


def _make_ep_ring_test_mesh_or_none() -> Mesh | None:
    devices = jax.devices()
    if len(devices) < 2:
        return None
    expert_size = 4 if len(devices) >= 4 else 2
    mesh_devices = np.array(devices[:expert_size]).reshape(1, expert_size, 1)
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


def _absolute_error_metrics(actual: jax.Array, expected: jax.Array) -> tuple[float, float]:
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    return float(np.max(difference)), float(np.mean(difference))


def _assert_allclose_with_error_metrics(actual: jax.Array, expected: jax.Array) -> tuple[float, float]:
    max_error, mean_error = _absolute_error_metrics(actual, expected)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"max_absolute_error={max_error}, mean_absolute_error={mean_error}",
    )
    return max_error, mean_error


def _optimized_hlo_rank_two_float_scatter_lines(lowered: jax.stages.Lowered) -> list[str]:
    hlo = lowered.compile().as_text()
    rank_two_float = re.compile(r"(?:bf16|f32)\[[^\]]*,[^\]]*\].*\bscatter\(")
    return [line.strip() for line in hlo.splitlines() if rank_two_float.search(line)]


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
    optional_modules = ("jax_triton", "triton", "quack", "jax_tvm_ffi")
    if not all(importlib.util.find_spec(module) is not None for module in optional_modules):
        pytest.skip("raw Sonic optional dependencies are not installed")
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("raw Sonic triton_call tests require a GPU")


def _skip_without_triton_gpu_runtime() -> None:
    if not all(importlib.util.find_spec(module) is not None for module in ("jax_triton", "triton")):
        pytest.skip("jax-triton and triton are not installed")
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("triton_call tests require a GPU")


def _skip_without_quack_gpu_runtime() -> None:
    try:
        _require_quack()
    except ImportError:
        pytest.skip("QuACK optional GPU dependencies are not installed")
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("QuACK tests require a GPU")


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
    optional_modules = ("jax_triton", "triton", "quack", "jax_tvm_ffi")
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

    with pytest.raises(ImportError, match="implementation='sonic' requires"):
        moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=None,
            implementation="sonic",
        )


@pytest.mark.parametrize(
    ("x_dtype", "w13_shape", "w2_shape", "activation_fn", "error", "match"),
    [
        (jnp.float32, (8, 2560, 2560), (8, 1280, 2560), jax.nn.silu, TypeError, "bfloat16"),
        (jnp.bfloat16, (8, 2560, 2558), (8, 1280, 2560), jax.nn.silu, ValueError, "twice"),
        (jnp.bfloat16, (8, 2560, 2560), (8, 1280, 2552), jax.nn.silu, ValueError, "hidden"),
        (jnp.bfloat16, (8, 2560, 2560), (8, 1280, 2560), jax.nn.gelu, ValueError, "SiLU/SwiGLU"),
    ],
)
def test_ep_ring_quack_rejects_unsupported_dtype_layout_and_activation(
    x_dtype, w13_shape, w2_shape, activation_fn, error, match
):
    x = jax.ShapeDtypeStruct((81920, 2560), x_dtype)
    w13 = jax.ShapeDtypeStruct(w13_shape, jnp.bfloat16)
    w2 = jax.ShapeDtypeStruct(w2_shape, jnp.bfloat16)

    with pytest.raises(error, match=match):
        _validate_quack_bulk_ring_contract(x, w13, w2, activation_fn=activation_fn)


def test_quack_mlp_varlen_matches_ragged_dot_output_and_vjp_on_gpu():
    _skip_without_quack_gpu_runtime()
    assignments = 256
    local_experts = 8
    hidden_dim = 128
    intermediate_dim = 128
    group_sizes = jnp.array([0, 17, 31, 64, 1, 80, 48, 15], dtype=jnp.int32)
    assert int(group_sizes.sum()) == assignments
    key_x, key_w13, key_w2, key_cotangent = jax.random.split(jax.random.key(58), 4)
    x = jax.random.normal(key_x, (assignments, hidden_dim), dtype=jnp.bfloat16)
    w13 = 0.02 * jax.random.normal(key_w13, (local_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.bfloat16)
    w2 = 0.02 * jax.random.normal(key_w2, (local_experts, intermediate_dim, hidden_dim), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(key_cotangent, x.shape, dtype=jnp.bfloat16)

    def reference(x, w13, w2):
        preactivation = ragged_dot(x, w13, group_sizes)
        gate, up = jnp.split(preactivation, 2, axis=-1)
        return ragged_dot(jax.nn.silu(gate) * up, w2, group_sizes)

    def loss(compute, x, w13, w2):
        return jnp.sum(compute(x, w13, w2).astype(jnp.float32) * cotangent.astype(jnp.float32))

    quack_out = jax.jit(quack_mlp_varlen)(x, w13, w2, group_sizes)
    reference_out = jax.jit(reference)(x, w13, w2)
    quack_grads = jax.jit(jax.grad(partial(loss, quack_mlp_varlen), argnums=(0, 1, 2)))(x, w13, w2)
    reference_grads = jax.jit(jax.grad(partial(loss, reference), argnums=(0, 1, 2)))(x, w13, w2)
    jax.block_until_ready((quack_out, reference_out, quack_grads, reference_grads))

    np.testing.assert_allclose(np.asarray(quack_out), np.asarray(reference_out), rtol=0.1, atol=2e-4)
    for quack_grad, reference_grad in zip(quack_grads, reference_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(quack_grad, dtype=np.float32),
            np.asarray(reference_grad, dtype=np.float32),
            rtol=0.1,
            atol=2e-4,
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


def test_moe_mlp_sonic_gradients_match_jax_reference_on_gpu():
    _skip_without_sonic_gpu_runtime()
    tokens = 64
    hidden_dim = 128
    intermediate_dim = 128
    num_experts = 8
    topk = 2
    k_x, k_logits, k_w13, k_w2 = jax.random.split(jax.random.key(33), 4)
    dtype = jnp.bfloat16
    x = jax.random.normal(k_x, (tokens, hidden_dim), dtype=dtype)
    selected_experts = _make_unique_topk_experts(tokens=tokens, topk=topk, num_experts=num_experts)
    combine_weights = jax.nn.softmax(
        jax.random.normal(k_logits, (tokens, topk), dtype=jnp.float32),
        axis=-1,
    )
    w_up_gate = 0.02 * jax.random.normal(
        k_w13,
        (num_experts, hidden_dim, 2 * intermediate_dim),
        dtype=dtype,
    )
    w_down = 0.02 * jax.random.normal(
        k_w2,
        (num_experts, intermediate_dim, hidden_dim),
        dtype=dtype,
    )

    def loss(x, combine_weights, w_up_gate, w_down, implementation):
        if implementation == "sonic":
            out = moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation=ActivationFunctionEnum.silu,
                implementation="sonic",
                mesh=None,
            )
        else:
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
            out = _gather_sum_reference(dispatch_out, dispatch_positions, combine_weights)
        return jnp.mean(jnp.square(out.astype(jnp.float32)))

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)), static_argnums=4)
    sonic_grads = grad_fn(x, combine_weights, w_up_gate, w_down, "sonic")
    reference_grads = grad_fn(x, combine_weights, w_up_gate, w_down, "reference")
    jax.block_until_ready((sonic_grads, reference_grads))

    for sonic_grad, reference_grad in zip(sonic_grads, reference_grads):
        np.testing.assert_allclose(
            np.asarray(sonic_grad, dtype=np.float32),
            np.asarray(reference_grad, dtype=np.float32),
            rtol=0.1,
            atol=2e-4,
        )


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


def test_moe_no_ep_fsdp_weights_match_unsharded_reference():
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(32),
        tokens=16 * len(jax.devices()),
        hidden_dim=16,
        intermediate_dim=24,
        num_experts=4,
        topk=2,
    )

    def run(x, selected_experts, combine_weights, w_up_gate, w_down, mesh):
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation="scatter",
            mesh=mesh,
        )

    reference = jax.jit(
        lambda x, selected_experts, combine_weights, w_up_gate, w_down: run(
            x, selected_experts, combine_weights, w_up_gate, w_down, None
        )
    )(x, selected_experts, combine_weights, w_up_gate, w_down)

    mesh = _make_dense_mesh()
    with jax.set_mesh(mesh):
        sharded_x = jax.device_put(x, NamedSharding(mesh, P("data", None)))
        sharded_selected_experts = jax.device_put(selected_experts, NamedSharding(mesh, P("data", None)))
        sharded_combine_weights = jax.device_put(combine_weights, NamedSharding(mesh, P("data", None)))
        sharded_w_up_gate = jax.device_put(w_up_gate, NamedSharding(mesh, P(None, "data", None)))
        sharded_w_down = jax.device_put(w_down, NamedSharding(mesh, P(None, None, "data")))
        actual = jax.jit(
            lambda x, selected_experts, combine_weights, w_up_gate, w_down: run(
                x, selected_experts, combine_weights, w_up_gate, w_down, mesh
            )
        )(
            sharded_x,
            sharded_selected_experts,
            sharded_combine_weights,
            sharded_w_up_gate,
            sharded_w_down,
        )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("implementation", ["ring", "ring_local_combine", "ring_ppermute", "ragged_all_to_all"])
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


@pytest.mark.parametrize("overflow", [False, True])
def test_moe_mlp_ring_ppermute_matches_ring_values_and_gradients(overflow: bool):
    mesh = _make_ep_ring_test_mesh_or_none()
    if mesh is None:
        pytest.skip("requires >=2 devices")

    expert_size = mesh.shape["expert"]
    tokens = expert_size * 4
    hidden_dim = 8
    intermediate_dim = 12
    num_experts = expert_size * 2
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(41),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    if overflow:
        selected_experts = jnp.zeros_like(selected_experts)
        combine_weights = jnp.full_like(combine_weights, 0.5)
    cotangent = jax.random.normal(jax.random.key(42), x.shape, dtype=x.dtype)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        cotangent = jax.sharding.reshard(cotangent, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        def run(implementation, x, combine_weights, w_up_gate, w_down):
            return moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                implementation=implementation,
                mesh=None,
                report_capacity_overflow=True,
            )

        ring_out, ring_dropped = run("ring", x, combine_weights, w_up_gate, w_down)
        streamed_out, streamed_dropped = run("ring_ppermute", x, combine_weights, w_up_gate, w_down)

        def loss(implementation, x, combine_weights, w_up_gate, w_down):
            out, _ = run(implementation, x, combine_weights, w_up_gate, w_down)
            return jnp.sum(out * cotangent)

        ring_grads = jax.grad(loss, argnums=(1, 2, 3, 4))("ring", x, combine_weights, w_up_gate, w_down)
        streamed_grads = jax.grad(loss, argnums=(1, 2, 3, 4))("ring_ppermute", x, combine_weights, w_up_gate, w_down)

    np.testing.assert_allclose(np.asarray(streamed_out), np.asarray(ring_out), rtol=1e-5, atol=1e-5)
    assert int(streamed_dropped) == int(ring_dropped)
    if overflow:
        assert int(streamed_dropped) > 0
    for streamed_grad, ring_grad in zip(streamed_grads, ring_grads, strict=True):
        np.testing.assert_allclose(np.asarray(streamed_grad), np.asarray(ring_grad), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("overflow", [False, True])
def test_moe_mlp_ring_local_combine_matches_ring_values_and_gradients(overflow: bool):
    mesh = _make_ep_ring_test_mesh_or_none()
    if mesh is None:
        pytest.skip("requires >=2 devices")

    expert_size = mesh.shape["expert"]
    tokens = expert_size * 4
    hidden_dim = 8
    intermediate_dim = 12
    num_experts = expert_size * 2
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(43),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    if overflow:
        selected_experts = jnp.zeros_like(selected_experts)
        combine_weights = jnp.full_like(combine_weights, 0.5)
    cotangent = jax.random.normal(jax.random.key(44), x.shape, dtype=x.dtype)

    with jax.set_mesh(mesh):
        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        cotangent = jax.sharding.reshard(cotangent, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        def run(implementation, x, combine_weights, w_up_gate, w_down):
            return moe_mlp(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                implementation=implementation,
                mesh=None,
                report_capacity_overflow=True,
            )

        ring_out, ring_dropped = run("ring", x, combine_weights, w_up_gate, w_down)
        local_out, local_dropped = run("ring_local_combine", x, combine_weights, w_up_gate, w_down)

        def loss(implementation, x, combine_weights, w_up_gate, w_down):
            out, _ = run(implementation, x, combine_weights, w_up_gate, w_down)
            return jnp.sum(out * cotangent)

        ring_grads = jax.grad(loss, argnums=(1, 2, 3, 4))("ring", x, combine_weights, w_up_gate, w_down)
        local_grads = jax.grad(loss, argnums=(1, 2, 3, 4))("ring_local_combine", x, combine_weights, w_up_gate, w_down)

    np.testing.assert_allclose(np.asarray(local_out), np.asarray(ring_out), rtol=1e-5, atol=1e-5)
    assert int(local_dropped) == int(ring_dropped)
    if overflow:
        assert int(local_dropped) > 0
    for local_grad, ring_grad in zip(local_grads, ring_grads, strict=True):
        np.testing.assert_allclose(np.asarray(local_grad), np.asarray(ring_grad), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("overflow", [False, True])
def test_moe_mlp_ring_fused_matches_ring_values_and_full_gradients(overflow: bool):
    mesh = _make_ep_ring_test_mesh_or_none()
    if mesh is None:
        pytest.skip("requires >=2 devices")

    expert_size = mesh.shape["expert"]
    tokens = expert_size * 4
    hidden_dim = 8
    intermediate_dim = 12
    num_experts = expert_size * 2
    topk = 2
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
        key=jax.random.key(45),
        tokens=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        topk=topk,
    )
    if overflow:
        selected_experts = jnp.zeros_like(selected_experts)
        combine_weights = jnp.full_like(combine_weights, 0.5)
    cotangent = jax.random.normal(jax.random.key(46), x.shape, dtype=x.dtype)

    with jax.set_mesh(mesh):
        batch_spec = P(("data", "expert"), None)
        expert_spec = P("expert", None, None)
        batch_sharding = NamedSharding(mesh, batch_spec)
        expert_sharding = NamedSharding(mesh, expert_spec)
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        cotangent = jax.sharding.reshard(cotangent, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        def shard_runner(local_fn, **kwargs):
            return jax.shard_map(
                partial(
                    local_fn,
                    activation_fn=jax.nn.silu,
                    num_experts=num_experts,
                    capacity_factor=1.0,
                    **kwargs,
                ),
                mesh=mesh,
                in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
                out_specs=(batch_spec, P()),
                check_vma=False,
            )

        ring = shard_runner(_moe_mlp_ep_ring_local)
        fused = shard_runner(_moe_mlp_ep_ring_fused_local, routing_implementation="reference")

        def run(runner, x, combine_weights, w_up_gate, w_down):
            return runner(x, selected_experts, combine_weights, w_up_gate, w_down)

        ring_out, ring_dropped = run(ring, x, combine_weights, w_up_gate, w_down)
        fused_out, fused_dropped = run(fused, x, combine_weights, w_up_gate, w_down)

        def loss(runner, x, combine_weights, w_up_gate, w_down):
            out, _ = run(runner, x, combine_weights, w_up_gate, w_down)
            return jnp.sum(out * cotangent)

        ring_grads = jax.grad(loss, argnums=(1, 2, 3, 4))(ring, x, combine_weights, w_up_gate, w_down)
        fused_grads = jax.grad(loss, argnums=(1, 2, 3, 4))(fused, x, combine_weights, w_up_gate, w_down)

    _assert_allclose_with_error_metrics(fused_out, ring_out)
    assert int(fused_dropped) == int(ring_dropped)
    if overflow:
        assert int(fused_dropped) > 0
    for fused_grad, ring_grad in zip(fused_grads, ring_grads, strict=True):
        _assert_allclose_with_error_metrics(fused_grad, ring_grad)


def _two_chunk_routing_case(case: str, expert_size: int) -> tuple[jax.Array, int, float, bool]:
    if case == "balanced":
        tokens_per_shard = 4
        num_experts = expert_size
        capacity_factor = 1.0
        routes = np.fromfunction(
            lambda source, token: (source + token) % expert_size,
            (expert_size, tokens_per_shard),
            dtype=int,
        )
        expect_drops = False
    elif case == "overflow":
        tokens_per_shard = 3
        num_experts = 2 * expert_size
        capacity_factor = 2.0 / 3.0
        routes = np.empty((expert_size, tokens_per_shard), dtype=np.int32)
        for source in range(expert_size):
            routes[source] = (2 * source + 1, 2 * source + 1, 2 * source)
        expect_drops = True
    elif case == "boundary_spanning":
        tokens_per_shard = 4
        num_experts = expert_size
        capacity_factor = float(expert_size)
        routes = np.zeros((expert_size, tokens_per_shard), dtype=np.int32)
        expect_drops = False
    elif case == "odd_capacity":
        tokens_per_shard = 5
        num_experts = expert_size
        capacity_factor = 1.0
        routes = np.fromfunction(
            lambda source, token: (source + token) % expert_size,
            (expert_size, tokens_per_shard),
            dtype=int,
        )
        expect_drops = False
    elif case == "one_half_fallback":
        tokens_per_shard = 6
        num_experts = expert_size
        capacity_factor = 2.0 / 3.0
        routes = np.zeros((expert_size, tokens_per_shard), dtype=np.int32)
        expect_drops = True
    else:
        raise ValueError(f"unknown two-chunk routing case: {case}")
    return jnp.asarray(routes.reshape(-1, 1)), num_experts, capacity_factor, expect_drops


@pytest.mark.parametrize(
    "case",
    ["balanced", "overflow", "boundary_spanning", "odd_capacity", "one_half_fallback"],
)
def test_moe_mlp_ring_two_chunk_matches_bulk_output_drop_and_vjp(case: str):
    mesh = _make_ep_ring_test_mesh_or_none()
    if mesh is None:
        pytest.skip("requires >=2 devices")

    expert_size = mesh.shape["expert"]
    selected_experts, num_experts, capacity_factor, expect_drops = _two_chunk_routing_case(case, expert_size)
    tokens = selected_experts.shape[0]
    hidden_dim = 4
    intermediate_dim = 6
    key_x, key_weights, key_w13, key_w2, key_cotangent = jax.random.split(jax.random.key(53), 5)
    x = jax.random.normal(key_x, (tokens, hidden_dim), dtype=jnp.bfloat16)
    combine_weights = jax.nn.sigmoid(jax.random.normal(key_weights, selected_experts.shape, dtype=jnp.float32))
    w_up_gate = 0.02 * jax.random.normal(key_w13, (num_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.bfloat16)
    w_down = 0.02 * jax.random.normal(key_w2, (num_experts, intermediate_dim, hidden_dim), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(key_cotangent, x.shape, dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        batch_spec = P(("data", "expert"), None)
        expert_spec = P("expert", None, None)
        batch_sharding = NamedSharding(mesh, batch_spec)
        expert_sharding = NamedSharding(mesh, expert_spec)
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)
        cotangent = jax.sharding.reshard(cotangent, batch_sharding)

        def shard_runner(local_fn):
            return jax.shard_map(
                partial(
                    local_fn,
                    activation_fn=jax.nn.silu,
                    num_experts=num_experts,
                    capacity_factor=capacity_factor,
                ),
                mesh=mesh,
                in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
                out_specs=(batch_spec, P()),
                check_vma=False,
            )

        bulk = shard_runner(_moe_mlp_ep_ring_local)
        two_chunk = shard_runner(_moe_mlp_ep_ring_two_chunk_local)
        fast_path_gate = jax.shard_map(
            partial(
                _ep_ring_two_chunk_fast_path_local,
                local_experts=num_experts // expert_size,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            ),
            mesh=mesh,
            in_specs=(batch_spec,),
            out_specs=P(),
            check_vma=False,
        )

        bulk_out, bulk_dropped = bulk(x, selected_experts, combine_weights, w_up_gate, w_down)
        chunked_out, chunked_dropped = two_chunk(x, selected_experts, combine_weights, w_up_gate, w_down)
        use_fast_path = fast_path_gate(selected_experts)

        def loss(runner, x, combine_weights, w_up_gate, w_down):
            out, _ = runner(x, selected_experts, combine_weights, w_up_gate, w_down)
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

        bulk_grads = jax.grad(loss, argnums=(1, 2, 3, 4))(bulk, x, combine_weights, w_up_gate, w_down)
        chunked_grads = jax.grad(loss, argnums=(1, 2, 3, 4))(two_chunk, x, combine_weights, w_up_gate, w_down)

    np.testing.assert_allclose(
        np.asarray(chunked_out, dtype=np.float32),
        np.asarray(bulk_out, dtype=np.float32),
        rtol=0.1,
        atol=2e-4,
    )
    assert int(chunked_dropped) == int(bulk_dropped)
    assert (int(chunked_dropped) > 0) == expect_drops
    assert bool(use_fast_path) == (case != "one_half_fallback")
    for chunked_grad, bulk_grad in zip(chunked_grads, bulk_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(chunked_grad, dtype=np.float32),
            np.asarray(bulk_grad, dtype=np.float32),
            rtol=0.1,
            atol=2e-4,
        )


def test_ring_fused_triton_routing_matches_references_on_gpu():
    _skip_without_triton_gpu_runtime()
    tokens = 32
    topk = 4
    hidden_dim = 128
    compact_rows_with_sentinel = 33
    assignments = tokens * topk
    local_assignment_indices = jax.random.permutation(jax.random.key(47), assignments)[:compact_rows_with_sentinel]
    valid = jnp.arange(compact_rows_with_sentinel) < compact_rows_with_sentinel - 1
    dispatch_rows = _assignment_to_compact_rows(
        local_assignment_indices,
        valid,
        tokens=tokens,
        topk=topk,
    )
    x_global = jax.random.normal(jax.random.key(48), (tokens, hidden_dim), dtype=jnp.float32)
    out_dispatch = (
        jax.random.normal(
            jax.random.key(49),
            (compact_rows_with_sentinel, hidden_dim),
            dtype=jnp.float32,
        )
        .at[-1]
        .set(0)
    )
    assignment_weights = jax.random.normal(jax.random.key(50), (assignments,), dtype=jnp.float32)
    dispatch_cotangent = jax.random.normal(
        jax.random.key(51),
        out_dispatch.shape,
        dtype=jnp.float32,
    )
    combine_cotangent = jax.random.normal(
        jax.random.key(52),
        x_global.shape,
        dtype=jnp.float32,
    )

    reference_dispatch, reference_dispatch_pullback = jax.vjp(
        partial(
            ring_dispatch_reference,
            local_assignment_indices=local_assignment_indices,
            valid=valid,
            topk=topk,
        ),
        x_global,
    )
    triton_dispatch, triton_dispatch_pullback = jax.vjp(
        partial(
            ring_dispatch_triton,
            local_assignment_indices=local_assignment_indices,
            valid=valid,
            dispatch_rows=dispatch_rows,
        ),
        x_global,
    )

    def combine(implementation, out_dispatch, assignment_weights):
        return ring_combine(
            out_dispatch,
            local_assignment_indices,
            valid,
            assignment_weights,
            tokens=tokens,
            topk=topk,
            implementation=implementation,
        )

    reference_combine, reference_combine_pullback = jax.vjp(
        partial(combine, "reference"),
        out_dispatch,
        assignment_weights,
    )

    def triton_combine_fn(out_dispatch, assignment_weights):
        return ring_combine_triton(
            out_dispatch,
            local_assignment_indices,
            valid,
            assignment_weights,
            tokens=tokens,
            topk=topk,
        )

    triton_combine, triton_combine_pullback = jax.vjp(
        triton_combine_fn,
        out_dispatch,
        assignment_weights,
    )

    _assert_allclose_with_error_metrics(triton_dispatch, reference_dispatch)
    _assert_allclose_with_error_metrics(
        triton_dispatch_pullback(dispatch_cotangent)[0],
        reference_dispatch_pullback(dispatch_cotangent)[0],
    )
    _assert_allclose_with_error_metrics(triton_combine, reference_combine)
    for triton_grad, reference_grad in zip(
        triton_combine_pullback(combine_cotangent),
        reference_combine_pullback(combine_cotangent),
        strict=True,
    ):
        _assert_allclose_with_error_metrics(triton_grad, reference_grad)

    def dispatch_loss(x_global):
        dispatched = ring_dispatch_triton(
            x_global,
            local_assignment_indices,
            valid,
            dispatch_rows,
        )
        return jnp.sum(dispatched * dispatch_cotangent)

    def combine_loss(out_dispatch, assignment_weights):
        combined = ring_combine_triton(
            out_dispatch,
            local_assignment_indices,
            valid,
            assignment_weights,
            tokens=tokens,
            topk=topk,
        )
        return jnp.sum(combined * combine_cotangent)

    dispatch_backward = jax.jit(jax.grad(dispatch_loss)).lower(x_global)
    combine_backward = jax.jit(jax.grad(combine_loss, argnums=(0, 1))).lower(out_dispatch, assignment_weights)
    assert _optimized_hlo_rank_two_float_scatter_lines(dispatch_backward) == []
    assert _optimized_hlo_rank_two_float_scatter_lines(combine_backward) == []


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


def test_moe_mlp_accumulating_weight_gradient_adds_prior_only_to_weight_cotangents(monkeypatch):
    mesh = _make_ep_ring_test_mesh_or_none()
    if mesh is None:
        pytest.skip("requires at least two devices")

    ragged_dot_module = __import__("haliax.nn.ragged_dot", fromlist=[""])

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
        *,
        output_dtype=None,
    ):
        output = jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )
        return output if output_dtype is None else output.astype(output_dtype)

    def fake_accumulating_pallas_call(lhs, rhs, group_sizes, accumulator, accumulation_scale):
        fresh_gradient = fake_triton_pallas_call(
            lhs,
            rhs,
            group_sizes,
            ragged_dot_module._DRHS_DIM_NUMS,
            output_dtype=jnp.float32,
        )
        return fresh_gradient + accumulation_scale * accumulator

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)
    monkeypatch.setattr(
        ragged_dot_module,
        "_triton_ragged_contracting_dim_accumulating_pallas_call",
        fake_accumulating_pallas_call,
    )

    expert_size = mesh.shape["expert"]
    num_experts = 2 * expert_size
    tokens = expert_size * 4
    hidden_dim = 4
    intermediate_dim = 3
    topk = 2
    keys = jax.random.split(jax.random.key(91), 6)
    x = jax.random.normal(keys[0], (tokens, hidden_dim), dtype=jnp.bfloat16)
    selected_experts = jnp.arange(tokens * topk, dtype=jnp.int32).reshape(tokens, topk) % num_experts
    combine_weights = jax.nn.softmax(
        jax.random.normal(keys[1], (tokens, topk), dtype=jnp.float32),
        axis=-1,
    ).astype(jnp.bfloat16)
    w13 = jax.random.normal(
        keys[2],
        (num_experts, hidden_dim, 2 * intermediate_dim),
        dtype=jnp.bfloat16,
    )
    w2 = jax.random.normal(
        keys[3],
        (num_experts, intermediate_dim, hidden_dim),
        dtype=jnp.bfloat16,
    )
    w13_prior = jax.random.normal(keys[4], w13.shape, dtype=jnp.float32)
    w2_prior = jax.random.normal(keys[5], w2.shape, dtype=jnp.float32)
    output_cotangent = jnp.arange(tokens * hidden_dim, dtype=jnp.float32).reshape(tokens, hidden_dim) / 100

    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))

    def sharded(value, sharding):
        return jax.device_put(value, sharding)

    x = sharded(x, batch_sharding)
    selected_experts = sharded(selected_experts, batch_sharding)
    combine_weights = sharded(combine_weights, batch_sharding)
    output_cotangent = sharded(output_cotangent, batch_sharding)
    w13 = sharded(w13, expert_sharding)
    w2 = sharded(w2, expert_sharding)
    w13_prior = sharded(w13_prior, expert_sharding)
    w2_prior = sharded(w2_prior, expert_sharding)
    zero_w13 = sharded(jnp.zeros(w13.shape, dtype=jnp.float32), expert_sharding)
    zero_w2 = sharded(jnp.zeros(w2.shape, dtype=jnp.float32), expert_sharding)

    def loss(x, combine_weights, w13, w2, w13_accumulator, w2_accumulator):
        output, dropped, token = moe_mlp_accumulating_weight_gradient(
            x,
            selected_experts,
            combine_weights,
            w13,
            w2,
            w13_accumulator,
            w2_accumulator,
            implementation="ring",
            mesh=mesh,
            capacity_factor=1.0,
        )
        value = jnp.sum(output.astype(jnp.float32) * output_cotangent) + token
        return value, (output, dropped, token)

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)
    with jax.set_mesh(mesh):
        (zero_value, zero_aux), zero_gradients = value_and_grad(
            x,
            combine_weights,
            w13,
            w2,
            zero_w13,
            zero_w2,
        )
        (prior_value, prior_aux), prior_gradients = value_and_grad(
            x,
            combine_weights,
            w13,
            w2,
            w13_prior,
            w2_prior,
        )

    np.testing.assert_array_equal(np.asarray(prior_aux[0]), np.asarray(zero_aux[0]))
    assert int(prior_aux[1]) == int(zero_aux[1])
    assert float(prior_aux[2]) == 0.0
    assert float(prior_value) == float(zero_value)
    np.testing.assert_array_equal(np.asarray(prior_gradients[0]), np.asarray(zero_gradients[0]))
    np.testing.assert_array_equal(np.asarray(prior_gradients[1]), np.asarray(zero_gradients[1]))
    np.testing.assert_allclose(
        np.asarray(prior_gradients[2]),
        np.asarray(zero_gradients[2] + w13_prior),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(prior_gradients[3]),
        np.asarray(zero_gradients[3] + w2_prior),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(np.asarray(prior_gradients[4]), np.zeros(w13.shape, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(prior_gradients[5]), np.zeros(w2.shape, dtype=np.float32))


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
