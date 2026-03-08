# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import os
import sys
import time

from haliax.nn.ragged_dot import ragged_dot
import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import levanter.tracker
from levanter.callbacks import profile_ctx
from levanter.grug.grug_moe import _prefix_cap_counts, _take_rows_impl, moe_mlp
from levanter.grug.grug_moe_sparsecore import (
    sparsecore_row_gather_bf16_prebitcast,
    sparsecore_row_scatter_add_transpose,
)
from levanter.tracker import NoopTracker
from levanter.utils.activation import ActivationFunctionEnum


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _append_xla_flag(flag: str) -> None:
    current = os.environ.get("XLA_FLAGS", "")
    parts = current.split()
    if flag in parts:
        return
    os.environ["XLA_FLAGS"] = (current + " " + flag).strip()


def _configure_xla_dump_dir(xla_dump_dir: str | None) -> str | None:
    if xla_dump_dir is None:
        return None
    resolved = os.path.abspath(xla_dump_dir)
    os.makedirs(resolved, exist_ok=True)
    _append_xla_flag(f"--xla_dump_to={resolved}")
    _append_xla_flag("--xla_dump_hlo_as_text")
    return resolved


@contextmanager
def _tee_stdio(log_path: str | None):
    if log_path is None:
        yield
        return
    resolved = os.path.abspath(log_path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(resolved, "a", encoding="utf-8") as handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_stdout, handle)
        sys.stderr = _TeeStream(old_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@dataclass(frozen=True)
class BenchCfg:
    batch: int
    seq: int
    hidden: int
    intermediate: int
    experts: int
    topk: int
    warmup: int
    iters: int
    dtype: jnp.dtype
    implementation: str
    profile_dir: str | None
    expert_axis_size: int
    capacity_factor: float
    chunk_experts: int
    barrier: bool
    scatter_implementation: str


def _make_mesh(devices: list[jax.Device], expert_axis_size: int) -> Mesh:
    if expert_axis_size <= 1:
        arr = np.array(devices).reshape(len(devices), 1)
        return Mesh(
            arr,
            axis_names=("data", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )

    if len(devices) % expert_axis_size != 0:
        raise ValueError(
            f"Need device count divisible by expert_axis_size, got devices={len(devices)}, "
            f"expert_axis_size={expert_axis_size}"
        )
    data = len(devices) // expert_axis_size
    arr = np.array(devices).reshape(data, expert_axis_size, 1)
    return Mesh(
        arr,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _parse_dtype(name: str) -> jnp.dtype:
    mapping = {
        "bf16": jnp.bfloat16,
        "f32": jnp.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}") from exc


def _profile_match_preset(devices: list[jax.Device]) -> dict[str, int | float]:
    target_local_tokens = 40960
    seq = 4096
    local_sequences = target_local_tokens // seq
    if local_sequences * seq != target_local_tokens:
        raise ValueError(f"Preset local token target {target_local_tokens} must be divisible by seq {seq}")
    return {
        "batch": local_sequences * len(devices),
        "seq": seq,
        "hidden": 2048,
        "intermediate": 1536,
        "experts": 128,
        "topk": 4,
        "expert_axis_size": 4,
        "capacity_factor": 1.0,
        "chunk_experts": 8,
    }


def _make_inputs(cfg: BenchCfg) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    tokens = cfg.batch * cfg.seq
    k_x, k_sel, k_logits, k_w13, k_w2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(k_x, (tokens, cfg.hidden), dtype=cfg.dtype)
    selected_experts = jax.random.randint(k_sel, (tokens, cfg.topk), 0, cfg.experts, dtype=jnp.int32)
    combine_logits = jax.random.normal(k_logits, (tokens, cfg.topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1).astype(jnp.float32)
    w_up_gate = jax.random.normal(k_w13, (cfg.experts, cfg.hidden, 2 * cfg.intermediate), dtype=cfg.dtype)
    w_down = jax.random.normal(k_w2, (cfg.experts, cfg.intermediate, cfg.hidden), dtype=cfg.dtype)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def _batch_spec(mesh: Mesh) -> P:
    if "expert" in mesh.shape and int(mesh.shape["expert"]) > 1:
        return P(("data", "expert"))
    return P(("data",))


def _scatter_add_rows(
    out_global: jax.Array,
    token_ids: jax.Array,
    values: jax.Array,
    *,
    implementation: str,
) -> jax.Array:
    if implementation == "naive":
        return out_global.at[token_ids].add(values, mode="drop")

    if implementation == "sparsecore_transpose":
        return sparsecore_row_scatter_add_transpose(values, token_ids, num_rows=int(out_global.shape[0]))

    if implementation != "coalesced":
        raise ValueError(f"Unknown scatter implementation {implementation!r}")

    chunk_capacity = int(token_ids.shape[0])
    output_tokens = int(out_global.shape[0])
    order = jnp.argsort(token_ids, stable=True)
    sorted_token_ids = token_ids[order]
    sorted_values = values[order]

    is_first = jnp.concatenate(
        (
            jnp.array([True], dtype=jnp.bool_),
            sorted_token_ids[1:] != sorted_token_ids[:-1],
        )
    )
    segment_ids = jnp.cumsum(is_first.astype(jnp.int32)) - 1
    num_unique = jnp.sum(is_first.astype(jnp.int32), dtype=jnp.int32)

    first_positions = jnp.where(is_first, size=chunk_capacity, fill_value=0)[0]
    unique_token_ids = sorted_token_ids[first_positions]
    coalesced_values = jax.ops.segment_sum(
        sorted_values,
        segment_ids,
        num_segments=chunk_capacity,
        indices_are_sorted=True,
    )

    valid = jnp.arange(chunk_capacity, dtype=jnp.int32) < num_unique
    dropped_token_id = jnp.full_like(unique_token_ids, output_tokens)
    unique_token_ids = jnp.where(valid, unique_token_ids, dropped_token_id)
    coalesced_values = jnp.where(valid[:, None], coalesced_values, jnp.zeros_like(coalesced_values))
    return out_global.at[unique_token_ids].add(coalesced_values, mode="drop")


def _coalesce_rows(
    token_ids: jax.Array,
    values: jax.Array,
    *,
    output_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    capacity = int(token_ids.shape[0])
    order = jnp.argsort(token_ids, stable=True)
    sorted_token_ids = token_ids[order]
    sorted_values = values[order]

    is_first = jnp.concatenate(
        (
            jnp.array([True], dtype=jnp.bool_),
            sorted_token_ids[1:] != sorted_token_ids[:-1],
        )
    )
    segment_ids = jnp.cumsum(is_first.astype(jnp.int32)) - 1
    num_unique = jnp.sum(is_first.astype(jnp.int32), dtype=jnp.int32)

    first_positions = jnp.where(is_first, size=capacity, fill_value=0)[0]
    unique_token_ids = sorted_token_ids[first_positions]
    coalesced_values = jax.ops.segment_sum(
        sorted_values,
        segment_ids,
        num_segments=capacity,
        indices_are_sorted=True,
    )

    valid = jnp.arange(capacity, dtype=jnp.int32) < num_unique
    dropped_token_id = jnp.full_like(unique_token_ids, output_tokens)
    unique_token_ids = jnp.where(valid, unique_token_ids, dropped_token_id)
    coalesced_values = jnp.where(valid[:, None], coalesced_values, jnp.zeros_like(coalesced_values))
    return unique_token_ids, coalesced_values


def _finalize_deferred_scatter(
    token_ids_chunks: list[jax.Array],
    value_chunks: list[jax.Array],
    *,
    output_tokens: int,
    implementation: str,
    dtype: jnp.dtype,
) -> jax.Array:
    with jax.named_scope("scatter_finalize"):
        all_token_ids = jnp.concatenate(token_ids_chunks, axis=0)
        all_values = jnp.concatenate(value_chunks, axis=0)

        if implementation == "deferred_segment_sum":
            order = jnp.argsort(all_token_ids, stable=True)
            sorted_token_ids = all_token_ids[order]
            sorted_values = all_values[order]
            return jax.ops.segment_sum(
                sorted_values,
                sorted_token_ids,
                num_segments=output_tokens,
                indices_are_sorted=True,
            )

        if implementation != "deferred_coalesced_set":
            raise ValueError(f"Unknown deferred scatter implementation {implementation!r}")

        unique_token_ids, coalesced_values = _coalesce_rows(
            all_token_ids,
            all_values,
            output_tokens=output_tokens,
        )
        out_global = jnp.zeros((output_tokens, all_values.shape[1]), dtype=dtype)
        return out_global.at[unique_token_ids].set(coalesced_values, mode="drop")


def _gather_rows_dedup_expand(
    x: jax.Array,
    token_ids: jax.Array,
    *,
    implementation: str,
) -> jax.Array:
    capacity = int(token_ids.shape[0])
    order = jnp.argsort(token_ids, stable=True)
    sorted_token_ids = token_ids[order]
    is_first = jnp.concatenate(
        (
            jnp.array([True], dtype=jnp.bool_),
            sorted_token_ids[1:] != sorted_token_ids[:-1],
        )
    )
    segment_ids = jnp.cumsum(is_first.astype(jnp.int32)) - 1
    first_positions = jnp.where(is_first, size=capacity, fill_value=0)[0]
    unique_token_ids = sorted_token_ids[first_positions]
    inverse = jnp.zeros((capacity,), dtype=jnp.int32).at[order].set(segment_ids)
    x_unique = _take_rows_impl(x, unique_token_ids, implementation=implementation)
    return x_unique[inverse]


def _ep_ring_pipeline_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn,
    num_experts: int,
    capacity_factor: float,
    chunk_experts: int,
    barrier: bool,
    scatter_implementation: str,
    gather_implementation: str,
) -> tuple[jax.Array, jax.Array]:
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

        local_experts = int(moe_w13_local.shape[0])
        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )
        if local_experts % chunk_experts != 0:
            raise ValueError(
                f"chunk_experts={chunk_experts} must divide local expert count={local_experts} for the benchmark path"
            )

        ep_size = num_experts // local_experts
        local_capacity = int(math.ceil(capacity_factor * assignments / ep_size))
        local_capacity = max(local_experts, local_capacity)

        expert_axis = jax.lax.axis_index("expert")
        expert_start = expert_axis * local_experts
        local_expert = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)
        local_expert = jnp.where(local_mask, local_expert, 0)

        expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
        local_mask_i32 = local_mask.astype(jnp.int32)
        counts = jnp.sum(
            (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
            axis=0,
            dtype=jnp.int32,
        )
        accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
        accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
        dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total

        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
        _, local_idx = jax.lax.top_k(selection_key, local_capacity)

        token_local = jnp.floor_divide(local_idx, topk)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

    use_deferred_scatter = scatter_implementation in {"deferred_coalesced_set", "deferred_segment_sum"}
    out_global = jnp.zeros_like(x_global)
    x_global_sc = None
    if gather_implementation == "sparsecore_prebitcast_reuse":
        if x_global.dtype != jnp.bfloat16 or x_global.shape[1] % 2 != 0:
            raise ValueError(
                "sparsecore_prebitcast_reuse requires bf16 activations with an even hidden size; "
                f"got dtype={x_global.dtype}, shape={x_global.shape}"
            )
        x_global_sc = x_global.view(jnp.int32)
    expert_offsets = jnp.cumsum(accepted_counts, dtype=jnp.int32) - accepted_counts
    chunk_capacity = math.ceil(local_capacity * chunk_experts / local_experts)
    token_local_padded = jnp.pad(token_local, ((0, chunk_capacity),), constant_values=0)
    weight_local_padded = jnp.pad(weight_local, ((0, chunk_capacity),), constant_values=0)
    num_chunks = local_experts // chunk_experts
    deferred_token_ids: list[jax.Array] = []
    deferred_values: list[jax.Array] = []

    def _gather_chunk(chunk_idx: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        expert_start_local = chunk_idx * chunk_experts
        chunk_group_sizes = accepted_counts[expert_start_local : expert_start_local + chunk_experts]
        chunk_total = jnp.sum(chunk_group_sizes, dtype=jnp.int32)
        chunk_offset = expert_offsets[expert_start_local]
        valid = jnp.arange(chunk_capacity, dtype=jnp.int32) < chunk_total
        chunk_token_ids = jax.lax.dynamic_slice(token_local_padded, (chunk_offset,), (chunk_capacity,))
        chunk_weights = jax.lax.dynamic_slice(weight_local_padded, (chunk_offset,), (chunk_capacity,))
        with jax.named_scope(f"gather_chunk_{chunk_idx}"):
            if gather_implementation == "sparsecore_prebitcast_reuse":
                assert x_global_sc is not None
                x_chunk = sparsecore_row_gather_bf16_prebitcast(x_global, x_global_sc, chunk_token_ids)
            elif gather_implementation == "sparsecore_dedup_expand":
                x_chunk = _gather_rows_dedup_expand(x_global, chunk_token_ids, implementation="sparsecore")
            elif gather_implementation == "xla_dedup_expand":
                x_chunk = _gather_rows_dedup_expand(x_global, chunk_token_ids, implementation="xla")
            else:
                x_chunk = _take_rows_impl(x_global, chunk_token_ids, implementation=gather_implementation)
            x_chunk = jnp.where(valid[:, None], x_chunk, jnp.zeros_like(x_chunk))
            chunk_weights = jnp.where(valid, chunk_weights, jnp.zeros_like(chunk_weights))
        padded_group_sizes = chunk_group_sizes.at[-1].add(chunk_capacity - chunk_total)
        return chunk_token_ids, chunk_weights, x_chunk, padded_group_sizes

    current_token_ids, current_weights, current_x_chunk, current_group_sizes = _gather_chunk(0)

    for chunk_idx in range(num_chunks - 1):
        next_token_ids, next_weights, next_x_chunk, next_group_sizes = _gather_chunk(chunk_idx + 1)

        if barrier:
            current_x_chunk = jax.lax.optimization_barrier(current_x_chunk)
            current_weights = jax.lax.optimization_barrier(current_weights)

        expert_start_local = chunk_idx * chunk_experts
        expert_stop_local = expert_start_local + chunk_experts
        moe_w13_chunk = moe_w13_local[expert_start_local:expert_stop_local]
        moe_w2_chunk = moe_w2_local[expert_start_local:expert_stop_local]

        with jax.named_scope(f"mlp_chunk_{chunk_idx}"):
            w13_out = ragged_dot(current_x_chunk, moe_w13_chunk, current_group_sizes)
            moe_dim = moe_w2_chunk.shape[1]
            gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
            out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_chunk, current_group_sizes)

        if barrier:
            out_dispatch = jax.lax.optimization_barrier(out_dispatch)

        with jax.named_scope(f"scatter_chunk_{chunk_idx}"):
            weighted_dispatch = out_dispatch * current_weights[:, None]
            if use_deferred_scatter:
                deferred_token_ids.append(current_token_ids)
                deferred_values.append(weighted_dispatch)
            else:
                out_global = _scatter_add_rows(
                    out_global,
                    current_token_ids,
                    weighted_dispatch,
                    implementation=scatter_implementation,
                )

        current_token_ids = next_token_ids
        current_weights = next_weights
        current_x_chunk = next_x_chunk
        current_group_sizes = next_group_sizes

    if barrier:
        current_x_chunk = jax.lax.optimization_barrier(current_x_chunk)
        current_weights = jax.lax.optimization_barrier(current_weights)

    expert_start_local = (num_chunks - 1) * chunk_experts
    moe_w13_chunk = moe_w13_local[expert_start_local : expert_start_local + chunk_experts]
    moe_w2_chunk = moe_w2_local[expert_start_local : expert_start_local + chunk_experts]

    with jax.named_scope(f"mlp_chunk_{num_chunks - 1}"):
        w13_out = ragged_dot(current_x_chunk, moe_w13_chunk, current_group_sizes)
        moe_dim = moe_w2_chunk.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_chunk, current_group_sizes)

    if barrier:
        out_dispatch = jax.lax.optimization_barrier(out_dispatch)

    with jax.named_scope(f"scatter_chunk_{num_chunks - 1}"):
        weighted_dispatch = out_dispatch * current_weights[:, None]
        if use_deferred_scatter:
            deferred_token_ids.append(current_token_ids)
            deferred_values.append(weighted_dispatch)
        else:
            out_global = _scatter_add_rows(
                out_global,
                current_token_ids,
                weighted_dispatch,
                implementation=scatter_implementation,
            )

    if use_deferred_scatter:
        out_global = _finalize_deferred_scatter(
            deferred_token_ids,
            deferred_values,
            output_tokens=int(x_global.shape[0]),
            implementation=scatter_implementation,
            dtype=x_global.dtype,
        )

    with jax.named_scope("scatter"):
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local, dropped_total


def _bench_one(cfg: BenchCfg, mesh: Mesh) -> tuple[float, float, float]:
    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(cfg)
        batch_axis = ("data", "expert") if "expert" in mesh.shape and int(mesh.shape["expert"]) > 1 else ("data",)
        batch_sharding = NamedSharding(mesh, P(batch_axis, None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        x = jax.device_put(x, batch_sharding)
        selected_experts = jax.device_put(selected_experts, batch_sharding)
        combine_weights = jax.device_put(combine_weights, batch_sharding)
        w_up_gate = jax.device_put(w_up_gate, expert_sharding)
        w_down = jax.device_put(w_down, expert_sharding)

        activation_fn = ActivationFunctionEnum.silu.to_jax_fn()

        if cfg.implementation in {
            "pipeline_sc_ep",
            "pipeline_xla_ep",
            "pipeline_sc_reusebitcast_ep",
            "pipeline_sc_dedup_ep",
            "pipeline_xla_dedup_ep",
        }:
            gather_implementation = {
                "pipeline_sc_ep": "sparsecore",
                "pipeline_xla_ep": "xla",
                "pipeline_sc_reusebitcast_ep": "sparsecore_prebitcast_reuse",
                "pipeline_sc_dedup_ep": "sparsecore_dedup_expand",
                "pipeline_xla_dedup_ep": "xla_dedup_expand",
            }[cfg.implementation]
            shard_fn = shard_map(
                lambda x_local, selected_local, weights_local, up_local, down_local: _ep_ring_pipeline_local(
                    x_local,
                    selected_local,
                    weights_local,
                    up_local,
                    down_local,
                    activation_fn=activation_fn,
                    num_experts=cfg.experts,
                    capacity_factor=cfg.capacity_factor,
                    chunk_experts=cfg.chunk_experts,
                    barrier=cfg.barrier,
                    scatter_implementation=cfg.scatter_implementation,
                    gather_implementation=gather_implementation,
                ),
                mesh=mesh,
                in_specs=(
                    _batch_spec(mesh),
                    _batch_spec(mesh),
                    _batch_spec(mesh),
                    P("expert", None, None),
                    P("expert", None, None),
                ),
                out_specs=(_batch_spec(mesh), P()),
                check_vma=False,
            )

            def loss_fn(x_in, up_in, down_in):
                out, _ = shard_fn(x_in, selected_experts, combine_weights, up_in, down_in)
                return jnp.mean(jnp.square(out.astype(jnp.float32)))

        else:
            dispatch_impl = "xla" if cfg.implementation == "xla_ep" else "sparsecore"

            def loss_fn(x_in, up_in, down_in):
                out = moe_mlp(
                    x_in,
                    selected_experts,
                    combine_weights,
                    up_in,
                    down_in,
                    mesh=mesh,
                    activation=ActivationFunctionEnum.silu,
                    capacity_factor=cfg.capacity_factor,
                    dispatch_implementation=dispatch_impl,
                )
                return jnp.mean(jnp.square(out.astype(jnp.float32)))

        step = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))

        start = time.perf_counter()
        loss, grads = step(x, w_up_gate, w_down)
        jax.block_until_ready((loss, grads))
        compile_time = time.perf_counter() - start

        for _ in range(cfg.warmup):
            loss, grads = step(x, w_up_gate, w_down)
            jax.block_until_ready((loss, grads))

        prof_ctx = nullcontext()
        if cfg.profile_dir is not None:
            prof_ctx = profile_ctx(cfg.profile_dir, create_perfetto_link=False)

        with levanter.tracker.current_tracker(NoopTracker()):
            with prof_ctx:
                start = time.perf_counter()
                for _ in range(cfg.iters):
                    loss, grads = step(x, w_up_gate, w_down)
                    jax.block_until_ready((loss, grads))
                steady_time = (time.perf_counter() - start) / cfg.iters

    tokens = cfg.batch * cfg.seq
    return compile_time, steady_time, tokens / steady_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("qwen3-32b-ep4-profile",), default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument(
        "--implementation",
        choices=(
            "xla_ep",
            "sparsecore_ep",
            "pipeline_sc_ep",
            "pipeline_xla_ep",
            "pipeline_sc_reusebitcast_ep",
            "pipeline_sc_dedup_ep",
            "pipeline_xla_dedup_ep",
        ),
        required=True,
    )
    parser.add_argument("--profile-dir", type=str, default=None)
    parser.add_argument("--xla-dump-dir", type=str, default=None)
    parser.add_argument("--compiler-log-path", type=str, default=None)
    parser.add_argument("--expert-axis-size", type=int, default=None)
    parser.add_argument("--capacity-factor", type=float, default=None)
    parser.add_argument("--chunk-experts", type=int, default=None)
    parser.add_argument("--barrier", action="store_true")
    parser.add_argument(
        "--scatter-implementation",
        choices=(
            "naive",
            "coalesced",
            "sparsecore_transpose",
            "deferred_coalesced_set",
            "deferred_segment_sum",
        ),
        default="naive",
    )
    args = parser.parse_args()

    preset = _profile_match_preset(jax.devices()) if args.preset == "qwen3-32b-ep4-profile" else {}
    batch = args.batch if args.batch is not None else int(preset.get("batch", 40))
    seq = args.seq if args.seq is not None else int(preset.get("seq", 4096))
    hidden = args.hidden if args.hidden is not None else int(preset.get("hidden", 2048))
    intermediate = args.intermediate if args.intermediate is not None else int(preset.get("intermediate", 1536))
    experts = args.experts if args.experts is not None else int(preset.get("experts", 128))
    topk = args.topk if args.topk is not None else int(preset.get("topk", 4))
    expert_axis_size = (
        args.expert_axis_size if args.expert_axis_size is not None else int(preset.get("expert_axis_size", 4))
    )
    capacity_factor = (
        args.capacity_factor if args.capacity_factor is not None else float(preset.get("capacity_factor", 1.0))
    )
    chunk_experts = args.chunk_experts if args.chunk_experts is not None else int(preset.get("chunk_experts", 8))

    cfg = BenchCfg(
        batch=batch,
        seq=seq,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
        warmup=args.warmup,
        iters=args.iters,
        dtype=_parse_dtype(args.dtype),
        implementation=args.implementation,
        profile_dir=args.profile_dir,
        expert_axis_size=expert_axis_size,
        capacity_factor=capacity_factor,
        chunk_experts=chunk_experts,
        barrier=args.barrier,
        scatter_implementation=args.scatter_implementation,
    )

    dump_dir = _configure_xla_dump_dir(args.xla_dump_dir)

    with _tee_stdio(args.compiler_log_path):
        print("devices", jax.devices())
        print("cfg", cfg)
        print("preset", args.preset)
        print("LIBTPU_INIT_ARGS", os.environ.get("LIBTPU_INIT_ARGS", ""))
        print("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))
        if dump_dir is not None:
            print("xla_dump_dir", dump_dir)

        mesh = _make_mesh(jax.devices(), cfg.expert_axis_size)
        print("mesh", mesh)
        compile_s, steady_s, tokens_per_s = _bench_one(cfg, mesh)
        print(
            f"{cfg.implementation},compile_s={compile_s:.6f},"
            f"steady_s={steady_s:.6f},tokens_per_s={tokens_per_s:.2f}"
        )


if __name__ == "__main__":
    main()
