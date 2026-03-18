# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX-only TPU repro for issue #3718.

This is intentionally stripped to the smallest interaction we have confirmed:
- RMSNorm
- router/top-k expert selection
- ragged_all_to_all expert dispatch
- checkpointed block input

Attention is intentionally omitted because it is not required to reproduce.
This version frames the bug in terms of backward inconsistency:
- compute the same step-0 loss and gradients with checkpoint offload disabled
- compute them again with checkpoint offload enabled
- check whether the two gradient pytrees are allclose
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, reshard

BATCH = int(os.environ.get("REPRO_BATCH", "2"))
SEQ_LEN = int(os.environ.get("REPRO_SEQ_LEN", "128"))
HIDDEN = int(os.environ.get("REPRO_HIDDEN", "2"))
NUM_LAYERS = int(os.environ.get("REPRO_NUM_LAYERS", "2"))
NUM_EXPERTS = int(os.environ.get("REPRO_NUM_EXPERTS", "4"))
TOP_K = int(os.environ.get("REPRO_TOPK", "1"))
MLP_DIM = int(os.environ.get("REPRO_MLP_DIM", "1"))
INIT_STD = float(os.environ.get("REPRO_INIT_STD", "0.02"))
SEED = int(os.environ.get("REPRO_SEED", "0"))
ATOL = float(os.environ.get("REPRO_ATOL", "1e-6"))
RTOL = float(os.environ.get("REPRO_RTOL", "1e-2"))

CAPACITY_FACTOR = 1.25
EXPERT_AXIS = 2
MODEL_AXIS = 2

if TOP_K > NUM_EXPERTS:
    raise ValueError("TOP_K must be <= NUM_EXPERTS")
if any(v <= 0 for v in (BATCH, SEQ_LEN, HIDDEN, NUM_LAYERS, NUM_EXPERTS, TOP_K, MLP_DIM)):
    raise ValueError("all integer config values must be > 0")
if CAPACITY_FACTOR <= 0.0:
    raise ValueError("CAPACITY_FACTOR must be > 0")
if ATOL < 0.0 or RTOL < 0.0:
    raise ValueError("REPRO_ATOL and REPRO_RTOL must be >= 0")


def _make_mesh() -> Mesh:
    devices = jax.devices()
    expected = EXPERT_AXIS * MODEL_AXIS
    if len(devices) != expected:
        raise ValueError(f"Expected {expected} devices for v5p-8 mesh, got {len(devices)}")
    grid = [[[devices[r * MODEL_AXIS + c]] for c in range(MODEL_AXIS)] for r in range(EXPERT_AXIS)]
    return Mesh(
        grid,
        ("expert", "model", "data"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _batch_spec() -> P:
    return P(("data", "expert"), None, None)


def _token_spec() -> P:
    return P(("data", "expert"), None)


def _batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, _batch_spec())


def _token_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, _token_spec())


def _expert_weight_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P("expert", None, None))


def _replicated_2d_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P(None, None))


def _offload_policy():
    return jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=(),
        names_which_can_be_offloaded=("block_input",),
        offload_src="device",
        offload_dst="pinned_host",
    )


def _rms_norm(x: jax.Array, weight: jax.Array) -> jax.Array:
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    out = x32 * jax.lax.rsqrt(var + 1e-5)
    return (out * weight.astype(jnp.float32)).astype(x.dtype)


def _init_params(key: jax.Array) -> tuple[dict[str, jax.Array], ...]:
    # Preserve the original key positions from the larger repro so the tiny
    # failing initialization remains aligned even though attention is deleted.
    keys = jax.random.split(key, NUM_LAYERS * 7)

    def layer_params(i: int) -> dict[str, jax.Array]:
        k = keys[7 * i : 7 * i + 7]
        return {
            "rms": jnp.ones((HIDDEN,), dtype=jnp.float32),
            "router": INIT_STD * jax.random.truncated_normal(k[4], -3, 3, (HIDDEN, NUM_EXPERTS), dtype=jnp.float32),
            "w_up_gate": (
                INIT_STD
                * jax.random.truncated_normal(k[5], -3, 3, (NUM_EXPERTS, HIDDEN, 2 * MLP_DIM), dtype=jnp.float32)
            ),
            "w2": INIT_STD * jax.random.truncated_normal(k[6], -3, 3, (NUM_EXPERTS, MLP_DIM, HIDDEN), dtype=jnp.float32),
        }

    return tuple(layer_params(i) for i in range(NUM_LAYERS))


def _prefix_offsets(sizes: jax.Array) -> jax.Array:
    zero = jnp.zeros((1,), dtype=jnp.int32)
    return jnp.concatenate([zero, jnp.cumsum(sizes[:-1], dtype=jnp.int32)], axis=0)


@jax.custom_vjp
def _permute_activations(inputs: jax.Array, permutation: jax.Array) -> jax.Array:
    return inputs[permutation, ...]


def _permute_activations_fwd(inputs: jax.Array, permutation: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _permute_activations(inputs, permutation), permutation


def _permute_activations_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
    permutation = residuals
    return _permute_activations(grads, jnp.argsort(permutation)), None


_permute_activations.defvjp(_permute_activations_fwd, _permute_activations_bwd)


def _expert_ffn_for_assignments(
    x_rows: jax.Array,
    local_expert_ids: jax.Array,
    w_up_gate_local: jax.Array,
    w2_local: jax.Array,
) -> jax.Array:
    w_up_gate_take = jnp.take(w_up_gate_local, local_expert_ids, axis=0)
    w2_take = jnp.take(w2_local, local_expert_ids, axis=0)
    up_gate = jnp.einsum("td,tdm->tm", x_rows, w_up_gate_take)
    gate, up = jnp.split(up_gate, [MLP_DIM], axis=-1)
    hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(x_rows.dtype) * up
    return jnp.einsum("tm,tmd->td", hidden, w2_take)


def _moe_ragged_all_to_all(
    mesh: Mesh,
    x_flat: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w2: jax.Array,
) -> jax.Array:
    w_up_gate = jax.lax.optimization_barrier(reshard(w_up_gate, _expert_weight_sharding(mesh)))
    w2 = jax.lax.optimization_barrier(reshard(w2, _expert_weight_sharding(mesh)))

    def _local(x_local, selected_local, combine_local, w_up_gate_local, w2_local):
        local_experts = w_up_gate_local.shape[0]
        num_experts = local_experts * jax.lax.psum(1, "expert")
        ep_size = num_experts // local_experts

        tokens_local = x_local.shape[0]
        assignments_local = tokens_local * TOP_K
        local_capacity = math.ceil(CAPACITY_FACTOR * assignments_local)
        local_capacity = max(local_experts, local_capacity)

        expert_flat = selected_local.reshape(assignments_local)
        weight_flat = combine_local.reshape(assignments_local).astype(x_local.dtype)
        token_local = jnp.floor_divide(jnp.arange(assignments_local, dtype=jnp.int32), TOP_K)

        dest_shard = jnp.floor_divide(expert_flat, local_experts)
        local_expert = expert_flat - dest_shard * local_experts
        flat_pos = jnp.arange(assignments_local, dtype=jnp.int32)
        sort_key = dest_shard * assignments_local + flat_pos
        sort_idx = jnp.argsort(sort_key, axis=0)

        dest_sorted = jnp.take(dest_shard, sort_idx, axis=0)
        local_expert_sorted = jnp.take(local_expert, sort_idx, axis=0)
        weight_sorted = jnp.take(weight_flat, sort_idx, axis=0)
        token_sorted = jnp.take(token_local, sort_idx, axis=0)
        x_send = jnp.take(x_local, token_sorted, axis=0)

        send_sizes_orig = jnp.bincount(dest_sorted, length=ep_size).astype(jnp.int32)
        input_offsets = _prefix_offsets(send_sizes_orig)
        all_send_sizes_orig = jax.lax.all_gather(send_sizes_orig, "expert")

        expert_axis = jax.lax.axis_index("expert")
        sender_oh = jax.nn.one_hot(expert_axis, ep_size, dtype=jnp.int32)
        prefix_inclusive = jnp.cumsum(all_send_sizes_orig, axis=0, dtype=jnp.int32)
        prefix_before = jnp.sum(prefix_inclusive * sender_oh[:, None], axis=0, dtype=jnp.int32) - send_sizes_orig
        send_sizes = jnp.minimum(send_sizes_orig, jnp.maximum(local_capacity - prefix_before, 0))

        segment_starts = jnp.take(input_offsets, dest_sorted, axis=0)
        segment_pos = jnp.arange(assignments_local, dtype=jnp.int32) - segment_starts
        keep_limit = jnp.take(send_sizes, dest_sorted, axis=0)
        keep_mask = segment_pos < keep_limit

        x_send = jnp.where(keep_mask[:, None], x_send, jnp.zeros_like(x_send)).astype(jnp.float32)
        local_expert_send = jnp.where(keep_mask, local_expert_sorted, 0)

        all_send_sizes = jax.lax.all_gather(send_sizes, "expert")
        recv_sizes = jnp.sum(all_send_sizes * sender_oh[None, :], axis=1, dtype=jnp.int32)
        output_offsets = _prefix_offsets(recv_sizes)
        recv_total = jnp.sum(recv_sizes, dtype=jnp.int32)

        x_recv = jax.lax.ragged_all_to_all(
            x_send,
            jnp.zeros((local_capacity, x_local.shape[1]), dtype=jnp.float32),
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
        local_expert_recv = jax.lax.ragged_all_to_all(
            local_expert_send,
            jnp.zeros((local_capacity,), dtype=jnp.int32),
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )

        valid_recv = jnp.arange(local_capacity, dtype=jnp.int32) < recv_total
        local_expert_recv = jnp.where(valid_recv, local_expert_recv, 0)

        max_key = local_experts * local_capacity + jnp.arange(local_capacity, dtype=jnp.int32)
        perm_key = local_expert_recv * local_capacity + jnp.arange(local_capacity, dtype=jnp.int32)
        perm_key = jnp.where(valid_recv, perm_key, max_key)
        local_perm = jnp.argsort(perm_key, axis=0)
        x_dispatch = _permute_activations(x_recv, local_perm)
        expert_dispatch = jnp.take(local_expert_recv, local_perm, axis=0)

        out_dispatch = _expert_ffn_for_assignments(
            x_dispatch.astype(x_local.dtype), expert_dispatch, w_up_gate_local, w2_local
        )
        inv_local_perm = jnp.argsort(local_perm, axis=0)
        out_recv = _permute_activations(out_dispatch.astype(jnp.float32), inv_local_perm)

        out_sorted = jax.lax.ragged_all_to_all(
            out_recv,
            jnp.zeros((assignments_local, x_local.shape[1]), dtype=jnp.float32),
            output_offsets,
            recv_sizes,
            input_offsets,
            send_sizes,
            axis_name="expert",
        )
        out_sorted = jnp.where(keep_mask[:, None], out_sorted, jnp.zeros_like(out_sorted)).astype(x_local.dtype)
        return (
            jnp.zeros_like(x_local)
            .at[token_sorted]
            .add(out_sorted * weight_sorted[:, None].astype(x_local.dtype), mode="drop")
        )

    fn = shard_map(
        _local,
        mesh=mesh,
        in_specs=(_token_spec(), _token_spec(), _token_spec(), P("expert", None, None), P("expert", None, None)),
        out_specs=_token_spec(),
    )
    return fn(x_flat, selected_experts, combine_weights, w_up_gate, w2)


def _moe(layer: dict[str, jax.Array], x: jax.Array, mesh: Mesh) -> jax.Array:
    x_flat = reshard(jnp.reshape(x, (BATCH * SEQ_LEN, HIDDEN)), _token_sharding(mesh))
    x_for_router = reshard(x_flat, _replicated_2d_sharding(mesh))

    router = reshard(layer["router"], _replicated_2d_sharding(mesh)).astype(x_flat.dtype)
    logits = jnp.einsum("td,de->te", x_for_router, router)
    logits = reshard(logits, _token_sharding(mesh))

    topk_logits, selected_experts = jax.lax.top_k(logits, TOP_K)
    combine_weights = jax.nn.softmax(topk_logits.astype(jnp.float32), axis=-1).astype(x_flat.dtype)

    routed = _moe_ragged_all_to_all(
        mesh,
        x_flat,
        selected_experts.astype(jnp.int32),
        combine_weights,
        layer["w_up_gate"],
        layer["w2"],
    )
    return reshard(jnp.reshape(routed, (BATCH, SEQ_LEN, HIDDEN)), _batch_sharding(mesh))


def _forward(params: tuple[dict[str, jax.Array], ...], x: jax.Array, mesh: Mesh, use_offload: bool) -> jax.Array:
    policy = _offload_policy() if use_offload else None

    def block_fn(layer, h):
        x_in = checkpoint_name(h, "block_input")
        return x_in + _moe(layer, _rms_norm(x_in, layer["rms"]), mesh)

    block = jax.checkpoint(block_fn, policy=policy)

    h = x
    for layer in params:
        h = block(layer, h)
    return h


def _leaf_name(path) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return "/".join(parts)


def _run_grads(mesh: Mesh, x: jax.Array, target: jax.Array, params, use_offload: bool):
    @jax.jit
    def grad_fn(params_t, x_t, target_t):
        def loss_fn(p):
            out = _forward(p, x_t, mesh, use_offload=use_offload)
            return jnp.mean(jnp.square(out.astype(jnp.float32) - target_t.astype(jnp.float32)))

        loss, grads = jax.value_and_grad(loss_fn)(params_t)
        return loss, grads

    return grad_fn(params, x, target)


def _compare_grads(mesh: Mesh, x: jax.Array, target: jax.Array, key: jax.Array) -> None:
    params = _init_params(key)
    loss_no_offload, grads_no_offload = _run_grads(mesh, x, target, params, use_offload=False)
    loss_offload, grads_offload = _run_grads(mesh, x, target, params, use_offload=True)

    loss_no_offload_value = float(jax.device_get(loss_no_offload))
    loss_offload_value = float(jax.device_get(loss_offload))
    print(f"loss_offload_0={loss_no_offload_value}", flush=True)
    print(f"loss_offload_1={loss_offload_value}", flush=True)
    print(f"loss_abs_diff={abs(loss_no_offload_value - loss_offload_value)}", flush=True)

    allclose = True
    for (path_no, grad_no), (path_off, grad_off) in zip(
        jax.tree_util.tree_leaves_with_path(grads_no_offload),
        jax.tree_util.tree_leaves_with_path(grads_offload),
        strict=True,
    ):
        if path_no != path_off:
            raise ValueError("gradient trees do not have matching paths")
        name = _leaf_name(path_no)
        grad_no_32 = jax.device_get(grad_no).astype(jnp.float32)
        grad_off_32 = jax.device_get(grad_off).astype(jnp.float32)
        is_allclose = bool(jnp.allclose(grad_no_32, grad_off_32, rtol=RTOL, atol=ATOL))
        max_abs_diff = float(jnp.max(jnp.abs(grad_no_32 - grad_off_32)))
        denom = jnp.maximum(jnp.abs(grad_no_32), jnp.abs(grad_off_32))
        max_rel_diff = float(jnp.max(jnp.abs(grad_no_32 - grad_off_32) / jnp.where(denom > 0, denom, 1.0)))
        finite = bool(jnp.all(jnp.isfinite(grad_no_32)) and jnp.all(jnp.isfinite(grad_off_32)))
        print(
            f"grad_leaf={name} finite={finite} allclose={is_allclose} "
            f"max_abs_diff={max_abs_diff} max_rel_diff={max_rel_diff}",
            flush=True,
        )
        allclose = allclose and is_allclose

    if allclose:
        print("result=grads_allclose", flush=True)
    else:
        print("result=grads_not_allclose", flush=True)


def main() -> None:
    mesh = _make_mesh()
    print(
        "config: "
        f"batch={BATCH} seq={SEQ_LEN} hidden={HIDDEN} "
        f"layers={NUM_LAYERS} experts={NUM_EXPERTS} topk={TOP_K} mlp={MLP_DIM} "
        f"init_std={INIT_STD} seed={SEED} "
        f"mesh={mesh.shape} cap_factor={CAPACITY_FACTOR} "
        f"atol={ATOL} rtol={RTOL}",
        flush=True,
    )

    key = jax.random.PRNGKey(SEED)
    key, xk, tk, pk = jax.random.split(key, 4)

    with mesh:
        batch_sharding = _batch_sharding(mesh)
        x = jax.device_put(jax.random.normal(xk, (BATCH, SEQ_LEN, HIDDEN), dtype=jnp.bfloat16), batch_sharding)
        target = jax.device_put(jax.random.normal(tk, (BATCH, SEQ_LEN, HIDDEN), dtype=jnp.bfloat16), batch_sharding)
        _compare_grads(mesh, x, target, pk)


if __name__ == "__main__":
    main()
