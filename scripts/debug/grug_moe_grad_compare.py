#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from iris.runtime.jax_init import initialize_jax
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


def _make_ep_mesh() -> Mesh:
    devices = jax.devices()
    if len(devices) < 2 or len(devices) % 2 != 0:
        raise RuntimeError(f"Need an even number of devices >= 2, got {len(devices)}")
    mesh_devices = np.array(devices).reshape(len(devices) // 2, 2, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _make_inputs(
    *,
    key: jax.Array,
    tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
    overflow: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    k_x, k_sel, k_logits, k_w13, k_w2 = jax.random.split(key, 5)
    x = jax.random.normal(k_x, (tokens, hidden_dim), dtype=jnp.float32)
    if overflow:
        selected_experts = jnp.zeros((tokens, topk), dtype=jnp.int32)
        combine_weights = jnp.full((tokens, topk), 1.0 / topk, dtype=jnp.float32)
    else:
        selected_experts = jax.random.randint(k_sel, (tokens, topk), 0, num_experts, dtype=jnp.int32)
        combine_logits = jax.random.normal(k_logits, (tokens, topk), dtype=jnp.float32)
        combine_weights = jax.nn.softmax(combine_logits, axis=-1)
    w_up_gate = jax.random.normal(k_w13, (num_experts, hidden_dim, 2 * intermediate_dim), dtype=jnp.float32)
    w_down = jax.random.normal(k_w2, (num_experts, intermediate_dim, hidden_dim), dtype=jnp.float32)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def _tree_diff_stats(a, b) -> dict[str, float]:
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    max_abs = 0.0
    max_rel = 0.0
    l2_sq = 0.0
    ref_l2_sq = 0.0
    for xa, xb in zip(leaves_a, leaves_b, strict=True):
        da = np.asarray(xa)
        db = np.asarray(xb)
        diff = np.abs(da - db)
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        denom = np.maximum(np.abs(db), 1e-12)
        max_rel = max(max_rel, float((diff / denom).max(initial=0.0)))
        l2_sq += float(np.sum((da - db) ** 2))
        ref_l2_sq += float(np.sum(db**2))
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "l2": l2_sq**0.5,
        "ref_l2": ref_l2_sq**0.5,
        "rel_l2": (l2_sq**0.5) / max(ref_l2_sq**0.5, 1e-12),
    }


def _host_array(x: jax.Array) -> np.ndarray:
    if jax.process_count() > 1 and getattr(x, "ndim", 0) > 0:
        x = multihost_utils.process_allgather(x, tiled=True)
    return np.asarray(x)


def _host_scalar(x: jax.Array) -> float:
    return float(np.asarray(x))


def _run_case(mesh: Mesh, *, overflow: bool) -> dict[str, object]:
    hidden_dim = 128
    intermediate_dim = 256
    num_experts = 8
    topk = 4
    tokens = max(len(jax.devices()) * 16, 64)

    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(
            key=jax.random.key(17 if overflow else 7),
            tokens=tokens,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            topk=topk,
            overflow=overflow,
        )

        batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.sharding.reshard(x, batch_sharding)
        selected_experts = jax.sharding.reshard(selected_experts, batch_sharding)
        combine_weights = jax.sharding.reshard(combine_weights, batch_sharding)
        w_up_gate = jax.sharding.reshard(w_up_gate, expert_sharding)
        w_down = jax.sharding.reshard(w_down, expert_sharding)

        def run_impl(implementation: str):
            def loss_and_drop(
                x_arg,
                selected_experts_arg,
                combine_weights_arg,
                w_up_gate_arg,
                w_down_arg,
            ):
                out, dropped = moe_mlp(
                    x_arg,
                    selected_experts_arg,
                    combine_weights_arg,
                    w_up_gate_arg,
                    w_down_arg,
                    activation=ActivationFunctionEnum.silu,
                    implementation=implementation,
                    mesh=None,
                    report_capacity_overflow=True,
                    capacity_factor=1.0,
                )
                loss = jnp.mean(out.astype(jnp.float32) ** 2)
                return loss, (out, dropped)

            fn = jax.jit(jax.value_and_grad(loss_and_drop, has_aux=True, argnums=(0, 3, 4)))
            (loss, (out, dropped)), grads = fn(x, selected_experts, combine_weights, w_up_gate, w_down)
            return loss, out, dropped, grads

        ring_loss, ring_out, ring_dropped, ring_grads = run_impl("ring")
        ragged_loss, ragged_out, ragged_dropped, ragged_grads = run_impl("ragged_all_to_all")

        ring_loss = _host_scalar(ring_loss)
        ragged_loss = _host_scalar(ragged_loss)
        ring_out_np = _host_array(ring_out)
        ragged_out_np = _host_array(ragged_out)
        ring_grad_x = _host_array(ring_grads[0])
        ragged_grad_x = _host_array(ragged_grads[0])
        ring_grad_w_up_gate = _host_array(ring_grads[1])
        ragged_grad_w_up_gate = _host_array(ragged_grads[1])
        ring_grad_w_down = _host_array(ring_grads[2])
        ragged_grad_w_down = _host_array(ragged_grads[2])

        return {
            "overflow": overflow,
            "tokens": tokens,
            "num_devices": len(jax.devices()),
            "num_processes": jax.process_count(),
            "ring_loss": ring_loss,
            "ragged_loss": ragged_loss,
            "loss_delta": ring_loss - ragged_loss,
            "ring_dropped": int(np.asarray(ring_dropped)),
            "ragged_dropped": int(np.asarray(ragged_dropped)),
            "output_diff": _tree_diff_stats(ring_out_np, ragged_out_np),
            "grad_x_diff": _tree_diff_stats(ring_grad_x, ragged_grad_x),
            "grad_w_up_gate_diff": _tree_diff_stats(ring_grad_w_up_gate, ragged_grad_w_up_gate),
            "grad_w_down_diff": _tree_diff_stats(ring_grad_w_down, ragged_grad_w_down),
        }


def main() -> None:
    initialize_jax()
    mesh = _make_ep_mesh()
    normal = _run_case(mesh, overflow=False)
    overflow = _run_case(mesh, overflow=True)
    if jax.process_index() == 0:
        print(json.dumps({"normal": normal, "overflow": overflow}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
