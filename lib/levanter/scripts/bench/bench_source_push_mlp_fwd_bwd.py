# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# pyrefly: ignore-errors

"""Benchmark source-push MoE MLP forward and forward+backward paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict
from statistics import median
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import levanter.grug._moe.source_push_mlp as source_push_mlp
from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_STAGED_DEVICE_SYNC,
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    FORWARD_STAGE_COMBINE,
    FORWARD_STAGE_TOTAL,
    FORWARD_STAGE_W13,
    FORWARD_STAGE_W2_RETURN,
    FORWARD_STAGES,
    SourcePushForwardDeviceInputs,
    _shard_source_push_forward_inputs,
    _sharded_source_combine_kernel,
    _time_staged_source_push_forward,
    device_source_push_forward_inputs_from_plan,
    make_source_push_forward_inputs,
    make_source_push_forward_source_plan_raw_inputs,
    source_push_forward_with_h_from_plan,
)
from levanter.grug._moe.source_push_inbox import (
    AXIS,
    BYTES_PER_BF16,
    PushInboxConfig,
    _block_until_ready,
    _sharded_raw_token_w13_h_kernel,
)
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILES, source_push_profile_defaults
from levanter.grug._moe.source_push_mlp import (
    SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED,
    SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
    source_push_mlp_route_table_from_plan,
    source_push_moe_mlp_from_plan,
)
from levanter.grug._moe.source_push_plan import (
    source_push_recv_route_weights_jax,
)
from levanter.grug._moe.source_push_w2_return import _sharded_w2_from_h_return_direct_to_source_kernel
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


KERNEL_NAME = "source_push_mlp_fwd_bwd"
MODE_FORWARD = "forward"
MODE_FORWARD_BACKWARD = "forward_backward"
MODE_FORWARD_DECOMPOSED = "forward_decomposed"
MODE_FORWARD_DECOMPOSED_RAW_TOKENS = "forward_decomposed_raw_tokens"
MODE_BACKWARD_DECOMPOSED = "backward_decomposed"
FORWARD_DECOMPOSED_STAGE_PACK_INPUTS = "pack_inputs"
FORWARD_DECOMPOSED_STAGE_PREPARE_INPUTS = "prepare_inputs"
BACKWARD_STAGE_TOTAL = "backward_total"
BACKWARD_STAGE_FORWARD_H = "forward_h"
BACKWARD_STAGE_DY_ROUTE = "dy_route"
BACKWARD_STAGE_ACTIVATION = "activation"
BACKWARD_STAGE_W2 = "w2_backward"
BACKWARD_STAGE_SWIGLU = "swiglu_backward"
BACKWARD_STAGE_X_REMAT = "x_rematerialization"
BACKWARD_STAGE_W13 = "w13_backward"
BACKWARD_STAGE_DX_COMBINE = "dx_return_combine"
BACKWARD_STAGES = (
    BACKWARD_STAGE_DY_ROUTE,
    BACKWARD_STAGE_ACTIVATION,
    BACKWARD_STAGE_W2,
    BACKWARD_STAGE_SWIGLU,
    BACKWARD_STAGE_X_REMAT,
    BACKWARD_STAGE_W13,
    BACKWARD_STAGE_DX_COMBINE,
)
MODES = (
    MODE_FORWARD,
    MODE_FORWARD_BACKWARD,
    MODE_FORWARD_DECOMPOSED,
    MODE_FORWARD_DECOMPOSED_RAW_TOKENS,
    MODE_BACKWARD_DECOMPOSED,
)
BACKEND_RING = "ring"
BACKEND_RAGGED_A2A = "ragged_all_to_all"
BACKEND_PUBLIC_SOURCE_PUSH = "public_source_push"
BACKEND_PUBLIC_SOURCE_PUSH_BLACKWELL = "public_source_push_blackwell"
BACKEND_SOURCE_PUSH_REFERENCE = "source_push_reference"
BACKEND_SOURCE_PUSH_PALLAS = "source_push_pallas_mgpu"
BACKEND_SOURCE_PUSH_BLACKWELL = "source_push_blackwell_staged"
BACKENDS = (
    BACKEND_RING,
    BACKEND_RAGGED_A2A,
    BACKEND_PUBLIC_SOURCE_PUSH,
    BACKEND_PUBLIC_SOURCE_PUSH_BLACKWELL,
    BACKEND_SOURCE_PUSH_REFERENCE,
    BACKEND_SOURCE_PUSH_PALLAS,
    BACKEND_SOURCE_PUSH_BLACKWELL,
)
PUBLIC_BACKEND_TO_IMPLEMENTATION = {
    BACKEND_RING: "ring",
    BACKEND_RAGGED_A2A: "ragged_all_to_all",
    BACKEND_PUBLIC_SOURCE_PUSH: "pallas_mgpu_source_push",
    BACKEND_PUBLIC_SOURCE_PUSH_BLACKWELL: "pallas_mgpu_source_push_blackwell",
}
SOURCE_PUSH_BACKEND_TO_IMPLEMENTATION = {
    BACKEND_SOURCE_PUSH_REFERENCE: SOURCE_PUSH_MLP_IMPLEMENTATION_REFERENCE,
    BACKEND_SOURCE_PUSH_PALLAS: SOURCE_PUSH_MLP_IMPLEMENTATION_PALLAS_MGPU,
    BACKEND_SOURCE_PUSH_BLACKWELL: SOURCE_PUSH_MLP_IMPLEMENTATION_BLACKWELL_STAGED,
}
SOURCE_PUSH_BACKEND_TO_EXECUTION_MODE = {
    BACKEND_SOURCE_PUSH_REFERENCE: FORWARD_EXECUTION_STAGED_HOST_SYNC,
    BACKEND_SOURCE_PUSH_PALLAS: FORWARD_EXECUTION_STAGED_HOST_SYNC,
    BACKEND_SOURCE_PUSH_BLACKWELL: FORWARD_EXECUTION_STAGED_DEVICE_SYNC,
}
OUTER_JIT_CHOICES = ("auto", "true", "false")
SUMMARY_METRICS = (
    "steady_state_time",
    "compile_time",
    "lower_compile_time",
    "first_run_time",
    "first_call_time",
    "useful_forward_tflops_per_rank",
    "rounded_forward_tflops_per_rank",
    "useful_fwd_bwd_tflops_per_rank",
    "rounded_fwd_bwd_tflops_per_rank",
    "useful_backward_tflops_per_rank",
    "rounded_backward_tflops_per_rank",
    "useful_tflops_per_rank",
    "rounded_tflops_per_rank",
    "dropped_routes",
)


class MlpTiming(NamedTuple):
    """Timing result for one MLP benchmark callable."""

    compile_time: float | None
    lower_compile_time: float | None
    first_run_time: float | None
    first_call_time: float
    steady_state_times: list[float]
    output: Any


class InputPackTiming(NamedTuple):
    """Timing result for dynamic source-push input packing and sharding."""

    first_call_time: float
    steady_state_times: list[float]
    output: Any


class RawTokenForwardInputs(NamedTuple):
    """Device inputs for the raw-token W13-H source-push decomposition."""

    x: jax.Array
    token_ids: jax.Array
    send_meta: jax.Array
    recv_meta: jax.Array
    expert_base: jax.Array
    src_base_by_expert: jax.Array
    w_gate_up: jax.Array
    recv_route_weights: jax.Array
    w_down: jax.Array
    queue_dst_ord: jax.Array
    queue_entry: jax.Array
    queue_row: jax.Array
    route_combine_weights: jax.Array
    route_valid_mask: jax.Array
    use_exact_expert_major: bool


class RawTokenForwardTiming(NamedTuple):
    """Timing result for raw-token staged source-push forward."""

    compile_time: float
    steady_state_times: list[float]
    output: Any
    stage_steady_state_times: dict[str, list[float]]
    stage_compile_times: dict[str, float]


class BackwardDecomposedTiming(NamedTuple):
    """Timing result for staged source-push MLP backward diagnostics."""

    first_call_time: float
    steady_state_times: list[float]
    output: Any
    stage_steady_state_times: dict[str, list[float]]


def _profile_defaults(argv: Sequence[str] | None = None) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    args, _ = pre_parser.parse_known_args(argv)
    return source_push_profile_defaults(args.source_push_profile)


def parse_source_push_mlp_fwd_bwd_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the source-push MLP forward/backward benchmark arguments."""

    profile_defaults = _profile_defaults(argv)

    def default(name: str, fallback: Any) -> Any:
        return profile_defaults.get(name, fallback)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    parser.add_argument("--ep-size", type=int, default=default("ep_size", 8))
    parser.add_argument("--entries-per-rank", type=int, default=default("entries_per_rank", 2))
    parser.add_argument("--inbox-slots", type=int, default=default("inbox_slots", 2))
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 2560))
    parser.add_argument("--intermediate-dim", type=int, default=default("intermediate_dim", 1280))
    parser.add_argument("--block-m", type=int, default=default("block_m", 64))
    parser.add_argument("--block-n", type=int, default=default("block_n", 128))
    parser.add_argument("--block-k", type=int, default=default("block_k", 128))
    parser.add_argument("--n-group", type=int, default=default("n_group", 1))
    parser.add_argument("--n-groups-per-job", type=int, default=default("n_groups_per_job", 1))
    parser.add_argument("--experts-per-rank", type=int, default=default("experts_per_rank", 32))
    parser.add_argument(
        "--send-worker-programs-per-peer",
        type=int,
        default=default("send_worker_programs_per_peer", 4),
    )
    parser.add_argument(
        "--worker-programs-per-peer",
        type=int,
        default=default("worker_programs_per_peer", 16),
    )
    parser.add_argument("--send-pipeline-depth", type=int, default=default("send_pipeline_depth", 1))
    parser.add_argument("--routing", type=str, default=default("routing", "balanced"))
    parser.add_argument("--tokens-per-rank", type=int, default=default("tokens_per_rank", 32768))
    parser.add_argument("--topk", type=int, default=default("topk", 4))
    parser.add_argument("--routing-seed", type=int, default=default("routing_seed", 0))
    parser.add_argument("--capacity-factor", type=float, default=default("capacity_factor", 1.25))
    parser.add_argument("--warmup", type=int, default=default("warmup", 1))
    parser.add_argument("--steps", type=int, default=default("steps", 3))
    parser.add_argument("--repeat-runs", type=int, default=default("repeat_runs", 1))
    parser.add_argument("--backends", default=BACKEND_SOURCE_PUSH_PALLAS)
    parser.add_argument("--modes", default=f"{MODE_FORWARD},{MODE_FORWARD_BACKWARD}")
    parser.add_argument(
        "--outer-jit",
        choices=OUTER_JIT_CHOICES,
        default="auto",
        help="Use an outer jax.jit around the measured callable. Auto jit-compiles ring/ragged/reference only.",
    )
    parser.add_argument("--separate-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_source_push_mlp_fwd_bwd_args(argv)
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    config = PushInboxConfig(
        ep_size=args.ep_size,
        entries_per_rank=args.entries_per_rank,
        inbox_slots=args.inbox_slots,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        n_group=args.n_group,
        n_groups_per_job=args.n_groups_per_job,
        experts_per_rank=args.experts_per_rank,
        send_worker_programs_per_peer=args.send_worker_programs_per_peer,
        worker_programs_per_peer=args.worker_programs_per_peer,
        send_pipeline_depth=args.send_pipeline_depth,
        routing=args.routing,
        tokens_per_rank=args.tokens_per_rank,
        topk=args.topk,
        routing_seed=args.routing_seed,
        capacity_factor=args.capacity_factor,
    )
    rows = run_source_push_mlp_fwd_bwd(
        config,
        backends=_parse_csv_choices(args.backends, BACKENDS, flag="--backends"),
        modes=_parse_csv_choices(args.modes, MODES, flag="--modes"),
        warmup=args.warmup,
        steps=args.steps,
        repeat_runs=args.repeat_runs,
        outer_jit=args.outer_jit,
        separate_compile=args.separate_compile,
        debug_exceptions=args.debug_exceptions,
    )
    for row in rows:
        if args.git_sha is not None:
            row["git_sha"] = args.git_sha
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                print(line, file=f, flush=True)


def run_source_push_mlp_fwd_bwd(
    config: PushInboxConfig,
    *,
    backends: Sequence[str],
    modes: Sequence[str],
    warmup: int,
    steps: int,
    repeat_runs: int,
    outer_jit: str,
    separate_compile: bool,
    debug_exceptions: bool = False,
) -> list[dict[str, Any]]:
    """Run public/preplanned MLP forward and forward+backward timings."""

    rows = []
    for backend in backends:
        for mode in modes:
            rows.extend(
                _run_one(
                    config,
                    backend=backend,
                    mode=mode,
                    warmup=warmup,
                    steps=steps,
                    repeat_runs=repeat_runs,
                    outer_jit=outer_jit,
                    separate_compile=separate_compile,
                    debug_exceptions=debug_exceptions,
                )
            )
    return rows


def _run_one(
    config: PushInboxConfig,
    *,
    backend: str,
    mode: str,
    warmup: int,
    steps: int,
    repeat_runs: int,
    outer_jit: str,
    separate_compile: bool,
    debug_exceptions: bool,
) -> list[dict[str, Any]]:
    try:
        config.validate()
        if repeat_runs <= 0:
            raise ValueError(f"repeat_runs must be positive, got {repeat_runs}")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        mesh = _make_public_ep_mesh(config.ep_size)
        raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
        host_inputs = make_source_push_forward_inputs(
            config,
            raw_inputs.x,
            raw_inputs.selected_experts,
            raw_inputs.combine_weights,
            raw_inputs.w_gate_up,
            raw_inputs.w_down,
        )
        route_table = source_push_mlp_route_table_from_plan(
            host_inputs.plan,
            src_base_by_expert=host_inputs.src_base_by_expert,
        )
        inputs = _device_benchmark_inputs(config, raw_inputs, mesh)
        use_outer_jit = _resolve_outer_jit(backend, outer_jit)
        if mode == MODE_BACKWARD_DECOMPOSED:
            if backend != BACKEND_SOURCE_PUSH_PALLAS:
                raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
            return _run_source_push_backward_decomposed(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                route_table=route_table,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
            )
        if mode in (MODE_FORWARD_DECOMPOSED, MODE_FORWARD_DECOMPOSED_RAW_TOKENS):
            if backend not in (BACKEND_SOURCE_PUSH_PALLAS, BACKEND_SOURCE_PUSH_BLACKWELL):
                raise ValueError(
                    f"{mode!r} only supports backend="
                    f"{(BACKEND_SOURCE_PUSH_PALLAS, BACKEND_SOURCE_PUSH_BLACKWELL)!r}"
                )
            if mode == MODE_FORWARD_DECOMPOSED_RAW_TOKENS:
                if backend != BACKEND_SOURCE_PUSH_PALLAS:
                    raise ValueError(f"{mode!r} only supports backend={BACKEND_SOURCE_PUSH_PALLAS!r}")
                return _run_source_push_forward_raw_token_decomposed(
                    config,
                    mesh=mesh,
                    host_inputs=host_inputs,
                    inputs=inputs,
                    warmup=warmup,
                    steps=steps,
                    repeat_runs=repeat_runs,
                )
            return _run_source_push_forward_decomposed(
                config,
                mesh=mesh,
                host_inputs=host_inputs,
                inputs=inputs,
                warmup=warmup,
                steps=steps,
                repeat_runs=repeat_runs,
                implementation=SOURCE_PUSH_BACKEND_TO_IMPLEMENTATION[backend],
                execution_mode=SOURCE_PUSH_BACKEND_TO_EXECUTION_MODE[backend],
            )
        fn, call_args = _make_benchmark_callable(
            config,
            backend=backend,
            mode=mode,
            mesh=mesh,
            host_inputs=host_inputs,
            route_table=route_table,
            inputs=inputs,
        )
        timing = _time_callable(
            fn,
            *call_args,
            mesh=mesh,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
            use_outer_jit=use_outer_jit,
            separate_compile=separate_compile,
        )
        return _timing_rows(
            config,
            backend=backend,
            mode=mode,
            timing=timing,
            queue_stats=host_inputs.queue_stats,
            repeat_runs=repeat_runs,
            outer_jit=use_outer_jit,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
        if debug_exceptions:
            raise
        return [
            {
                "kernel": KERNEL_NAME,
                "implementation": backend,
                "backend": backend,
                "mode": mode,
                "row_type": "error",
                "config": asdict(config),
                "outer_jit": _outer_jit_error_value(backend, outer_jit),
                "repeat_run": None,
                "repeat_runs": repeat_runs,
                "steady_state_time": None,
                "compile_time": None,
                "lower_compile_time": None,
                "first_run_time": None,
                "first_call_time": None,
                "dropped_routes": None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ]
    finally:
        jax.clear_caches()


def _run_source_push_forward_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
    implementation: str,
    execution_mode: str,
) -> list[dict[str, Any]]:
    pack_timing = _time_source_push_input_pack(
        config,
        mesh=mesh,
        host_inputs=host_inputs,
        x=inputs["x_source"],
        route_weights=inputs["combine_source"],
        w13=inputs["w13_source"],
        w2=inputs["w2_source"],
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    packed = pack_timing.output
    staged_timing = _time_staged_source_push_forward(
        mesh,
        config,
        packed.x,
        packed.send_meta,
        packed.recv_meta,
        packed.expert_base,
        packed.src_base_by_expert,
        packed.h_group_sizes,
        packed.w_gate_up,
        packed.h_route_weights,
        packed.w_down,
        packed.queue_dst_ord,
        packed.queue_entry,
        packed.queue_row,
        packed.route_combine_weights,
        packed.route_valid_mask,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        use_exact_expert_major=packed.use_exact_expert_major,
        implementation=implementation,
        execution_mode=execution_mode,
    )
    return _decomposed_forward_rows(
        config,
        pack_timing=pack_timing,
        staged_timing=staged_timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        mode=MODE_FORWARD_DECOMPOSED,
        input_stage=FORWARD_DECOMPOSED_STAGE_PACK_INPUTS,
    )


def _run_source_push_forward_raw_token_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    prepare_timing = _time_source_push_raw_token_input_prepare(
        config,
        mesh=mesh,
        host_inputs=host_inputs,
        x=inputs["x_source"],
        route_weights=inputs["combine_source"],
        w13=inputs["w13_source"],
        w2=inputs["w2_source"],
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    raw_inputs = prepare_timing.output
    staged_timing = _time_staged_source_push_forward_raw_tokens(
        mesh,
        config,
        raw_inputs,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
    )
    return _decomposed_forward_rows(
        config,
        pack_timing=prepare_timing,
        staged_timing=staged_timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        mode=MODE_FORWARD_DECOMPOSED_RAW_TOKENS,
        input_stage=FORWARD_DECOMPOSED_STAGE_PREPARE_INPUTS,
    )


def _run_source_push_backward_decomposed(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    with jax.set_mesh(mesh):
        forward_start = time.perf_counter()
        out, h_flat, dropped_routes = source_push_forward_with_h_from_plan(
            config,
            host_inputs,
            inputs["x_source"],
            inputs["combine_source"],
            inputs["w13_source"],
            inputs["w2_source"],
            execution_mode=FORWARD_EXECUTION_STAGED_HOST_SYNC,
            mesh=mesh,
        )
        _block_until_ready((out, h_flat, dropped_routes))
        forward_h_time = time.perf_counter() - forward_start

    dy = jnp.ones_like(out, dtype=jnp.float32)
    with jax.set_mesh(mesh):
        timing = _time_source_push_backward_decomposed(
            route_table,
            jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
            inputs["x_source"],
            inputs["combine_source"],
            inputs["w13_source"],
            inputs["w2_source"],
            h_flat,
            dy,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
        )
    return _decomposed_backward_rows(
        config,
        timing=timing,
        queue_stats=host_inputs.queue_stats,
        repeat_runs=repeat_runs,
        dropped_routes=int(jax.device_get(dropped_routes)),
        forward_h_time=forward_h_time,
    )


def _time_source_push_backward_decomposed(
    route_table,
    expert_base: jax.Array,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    h_flat: jax.Array,
    dy: jax.Array,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> BackwardDecomposedTiming:
    def call_backward(*, record_stage_times: bool = False):
        stage_times = {stage: 0.0 for stage in BACKWARD_STAGES}
        dx = jnp.zeros_like(x, dtype=jnp.float32)
        d_route_weights = jnp.zeros_like(route_weights, dtype=jnp.float32)
        dw13 = jnp.zeros_like(w13, dtype=jnp.float32)
        dw2 = jnp.zeros_like(w2, dtype=jnp.float32)

        for expert in range(route_table.experts_per_rank):
            _, safe_src, safe_token, safe_slot, valid_f = source_push_mlp._source_push_mlp_expert_route_indices(
                route_table,
                expert,
            )

            stage_start = time.perf_counter()
            h_block = source_push_mlp._source_push_mlp_h_flat_for_expert(route_table, expert_base, h_flat, expert)
            h_block = h_block.astype(jnp.float32) * valid_f[..., None]
            dy_block = source_push_mlp._source_push_mlp_dy_to_expert_major(
                dy,
                safe_src,
                safe_token,
                valid_f,
            )
            weights = source_push_mlp._source_push_mlp_route_weights_to_expert_major(
                route_weights,
                safe_src,
                safe_token,
                safe_slot,
                valid_f,
            )
            _block_until_ready((h_block, dy_block, weights))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_DY_ROUTE] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            gate, up, silu_gate, activation = source_push_mlp._source_push_mlp_activation_from_h(h_block)
            weighted_activation = source_push_mlp._source_push_mlp_weight_activation(activation, weights)
            _block_until_ready((gate, up, silu_gate, activation, weighted_activation))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_ACTIVATION] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            w2_block = w2[:, expert].astype(jnp.float32)
            d_weighted_activation, d_route_block, dw2_block = source_push_mlp._source_push_mlp_w2_backward_for_expert(
                dy_block,
                activation,
                weighted_activation,
                w2_block,
                valid_f,
            )
            _block_until_ready((d_weighted_activation, d_route_block, dw2_block))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_W2] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            d_h_block = source_push_mlp._source_push_mlp_swiglu_backward_from_h(
                d_weighted_activation,
                weights,
                gate,
                up,
                silu_gate,
            )
            _block_until_ready((d_route_block, d_h_block))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_SWIGLU] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            x_block = source_push_mlp._source_push_mlp_x_to_expert_major(x, safe_src, safe_token, valid_f)
            _block_until_ready(x_block)
            if record_stage_times:
                stage_times[BACKWARD_STAGE_X_REMAT] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            w13_block = w13[:, expert].astype(jnp.float32)
            dx_block, dw13_block = source_push_mlp._source_push_mlp_w13_backward_for_expert(
                x_block,
                d_h_block,
                w13_block,
            )
            _block_until_ready((dx_block, dw13_block))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_W13] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            dx, d_route_weights, dw13, dw2 = source_push_mlp._source_push_mlp_accumulate_expert_backward_outputs(
                dx,
                d_route_weights,
                dw13,
                dw2,
                expert,
                safe_src,
                safe_token,
                safe_slot,
                dx_block,
                d_route_block,
                dw13_block,
                dw2_block,
            )
            _block_until_ready((dx, d_route_weights, dw13, dw2))
            if record_stage_times:
                stage_times[BACKWARD_STAGE_DX_COMBINE] += time.perf_counter() - stage_start

        output = (
            dx.astype(x.dtype),
            d_route_weights.astype(route_weights.dtype),
            dw13.astype(w13.dtype),
            dw2.astype(w2.dtype),
        )
        _block_until_ready(output)
        return output, stage_times

    start = time.perf_counter()
    output, _ = call_backward(record_stage_times=False)
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output, _ = call_backward(record_stage_times=False)

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in BACKWARD_STAGES}
    for _ in range(repeat_runs):
        total_elapsed = 0.0
        stage_elapsed = {stage: 0.0 for stage in BACKWARD_STAGES}
        for _ in range(steps):
            start = time.perf_counter()
            output, step_stage_times = call_backward(record_stage_times=True)
            total_elapsed += time.perf_counter() - start
            for stage in BACKWARD_STAGES:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append(total_elapsed / steps)
        for stage in BACKWARD_STAGES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return BackwardDecomposedTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
        stage_steady_state_times=stage_steady_state_times,
    )


def _time_source_push_input_pack(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> InputPackTiming:
    def pack_inputs() -> SourcePushForwardDeviceInputs:
        with jax.set_mesh(mesh):
            packed = device_source_push_forward_inputs_from_plan(config, host_inputs, x, route_weights, w13, w2)
            packed = _shard_source_push_forward_inputs(mesh, packed)
        return _block_source_push_forward_device_inputs(packed)

    start = time.perf_counter()
    output = pack_inputs()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = pack_inputs()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = pack_inputs()
        steady_state_times.append((time.perf_counter() - start) / steps)

    return InputPackTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
    )


def _time_source_push_raw_token_input_prepare(
    config: PushInboxConfig,
    *,
    mesh: Mesh,
    host_inputs,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> InputPackTiming:
    def prepare_inputs() -> RawTokenForwardInputs:
        with jax.set_mesh(mesh):
            recv_route_weights = source_push_recv_route_weights_jax(route_weights, host_inputs.plan)
            prepared = RawTokenForwardInputs(
                x=x.astype(jnp.bfloat16),
                token_ids=jnp.asarray(host_inputs.plan.token_ids, dtype=jnp.int32),
                send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
                recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
                expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
                src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
                w_gate_up=jnp.asarray(w13, dtype=jnp.bfloat16),
                recv_route_weights=recv_route_weights.astype(jnp.bfloat16),
                w_down=jnp.asarray(w2, dtype=jnp.bfloat16),
                queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
                queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
                queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
                route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
                route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
                use_exact_expert_major=host_inputs.use_exact_expert_major,
            )
            prepared = _shard_raw_token_forward_inputs(mesh, prepared)
        return _block_raw_token_forward_inputs(prepared)

    start = time.perf_counter()
    output = prepare_inputs()
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        output = prepare_inputs()

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            output = prepare_inputs()
        steady_state_times.append((time.perf_counter() - start) / steps)

    return InputPackTiming(
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
    )


def _shard_raw_token_forward_inputs(mesh: Mesh, inputs: RawTokenForwardInputs) -> RawTokenForwardInputs:
    return RawTokenForwardInputs(
        x=jax.device_put(inputs.x, NamedSharding(mesh, P(AXIS, None, None))),
        token_ids=jax.device_put(inputs.token_ids, NamedSharding(mesh, P(AXIS, None, None, None))),
        send_meta=jax.device_put(inputs.send_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        recv_meta=jax.device_put(inputs.recv_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        expert_base=jax.device_put(inputs.expert_base, NamedSharding(mesh, P(AXIS, None))),
        src_base_by_expert=jax.device_put(inputs.src_base_by_expert, NamedSharding(mesh, P(AXIS, None, None))),
        w_gate_up=jax.device_put(inputs.w_gate_up, NamedSharding(mesh, P(AXIS, None, None, None))),
        recv_route_weights=jax.device_put(inputs.recv_route_weights, NamedSharding(mesh, P(AXIS, None, None, None))),
        w_down=jax.device_put(inputs.w_down, NamedSharding(mesh, P(AXIS, None, None, None))),
        queue_dst_ord=jax.device_put(inputs.queue_dst_ord, NamedSharding(mesh, P(AXIS, None, None))),
        queue_entry=jax.device_put(inputs.queue_entry, NamedSharding(mesh, P(AXIS, None, None))),
        queue_row=jax.device_put(inputs.queue_row, NamedSharding(mesh, P(AXIS, None, None))),
        route_combine_weights=jax.device_put(inputs.route_combine_weights, NamedSharding(mesh, P(AXIS, None, None))),
        route_valid_mask=jax.device_put(inputs.route_valid_mask, NamedSharding(mesh, P(AXIS, None, None))),
        use_exact_expert_major=inputs.use_exact_expert_major,
    )


def _block_raw_token_forward_inputs(inputs: RawTokenForwardInputs) -> RawTokenForwardInputs:
    _block_until_ready(tuple(value for value in inputs[:-1]))
    return inputs


def _time_staged_source_push_forward_raw_tokens(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: RawTokenForwardInputs,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> RawTokenForwardTiming:
    w13_h_fn = jax.jit(
        _sharded_raw_token_w13_h_kernel(
            mesh,
            config,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    w2_from_h_return_fn = jax.jit(
        _sharded_w2_from_h_return_direct_to_source_kernel(
            mesh,
            config,
            use_exact_expert_major=inputs.use_exact_expert_major,
        )
    )
    combine_fn = jax.jit(_sharded_source_combine_kernel(mesh, config))

    def call_stages(*, record_stage_times: bool = False):
        stage_times: dict[str, float] = {}

        stage_start = time.perf_counter()
        _, h = w13_h_fn(
            inputs.x,
            inputs.token_ids,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
        )
        _block_until_ready(h)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W13] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        source_return = w2_from_h_return_fn(
            h,
            inputs.recv_route_weights,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_down,
        )
        _block_until_ready(source_return)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W2_RETURN] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        out = combine_fn(
            source_return,
            inputs.queue_dst_ord,
            inputs.queue_entry,
            inputs.queue_row,
            jnp.ones_like(inputs.route_combine_weights),
            inputs.route_valid_mask,
        )
        _block_until_ready(out)
        if record_stage_times:
            stage_times[FORWARD_STAGE_COMBINE] = time.perf_counter() - stage_start
        return out, stage_times

    start = time.perf_counter()
    out, stage_compile_times = call_stages(record_stage_times=True)
    compile_time = time.perf_counter() - start

    for _ in range(warmup):
        out, _ = call_stages()

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in FORWARD_STAGES}
    for _ in range(repeat_runs):
        start = time.perf_counter()
        stage_elapsed = {stage: 0.0 for stage in FORWARD_STAGES}
        for _ in range(steps):
            out, step_stage_times = call_stages(record_stage_times=True)
            for stage in FORWARD_STAGES:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append((time.perf_counter() - start) / steps)
        for stage in FORWARD_STAGES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    return RawTokenForwardTiming(
        compile_time=compile_time,
        steady_state_times=steady_state_times,
        output=out,
        stage_steady_state_times=stage_steady_state_times,
        stage_compile_times=stage_compile_times,
    )


def _block_source_push_forward_device_inputs(inputs: SourcePushForwardDeviceInputs) -> SourcePushForwardDeviceInputs:
    _block_until_ready(
        (
            inputs.x,
            inputs.source_x,
            inputs.token_ids,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
            inputs.w_down,
            inputs.queue_dst_ord,
            inputs.queue_entry,
            inputs.queue_row,
            inputs.h_route_weights,
            inputs.route_combine_weights,
            inputs.route_valid_mask,
        )
    )
    return inputs


def _decomposed_forward_rows(
    config: PushInboxConfig,
    *,
    pack_timing: InputPackTiming,
    staged_timing,
    queue_stats: dict[str, Any],
    repeat_runs: int,
    mode: str,
    input_stage: str,
) -> list[dict[str, Any]]:
    if staged_timing.stage_steady_state_times is None:
        raise ValueError("decomposed forward requires staged source-push timing")

    useful_forward_flops, rounded_forward_flops = _forward_flops_per_rank(config, queue_stats)
    bytes_per_rank = _forward_bytes_per_rank(config, queue_stats)
    dropped_routes = int(jax.device_get(queue_stats["dropped_routes"]))
    rows = []

    for repeat_run, pack_time in enumerate(pack_timing.steady_state_times):
        staged_total_time = staged_timing.steady_state_times[repeat_run]
        total_time = pack_time + staged_total_time
        rows.append(
            _decomposed_forward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=FORWARD_STAGE_TOTAL,
                steady_state_time=total_time,
                first_call_time=pack_timing.first_call_time + staged_timing.compile_time,
                compile_time=staged_timing.compile_time,
                bytes_per_rank=bytes_per_rank,
                useful_forward_flops=useful_forward_flops,
                rounded_forward_flops=rounded_forward_flops,
                dropped_routes=dropped_routes,
                mode=mode,
            )
        )
        rows.append(
            _decomposed_forward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=input_stage,
                steady_state_time=pack_time,
                first_call_time=pack_timing.first_call_time,
                compile_time=None,
                bytes_per_rank=bytes_per_rank,
                useful_forward_flops=None,
                rounded_forward_flops=None,
                dropped_routes=dropped_routes,
                mode=mode,
            )
        )
        for stage in FORWARD_STAGES:
            stage_time = staged_timing.stage_steady_state_times[stage][repeat_run]
            rows.append(
                _decomposed_forward_row(
                    config,
                    queue_stats=queue_stats,
                    repeat_run=repeat_run,
                    repeat_runs=repeat_runs,
                    stage=stage,
                    steady_state_time=stage_time,
                    first_call_time=None,
                    compile_time=(
                        None if staged_timing.stage_compile_times is None else staged_timing.stage_compile_times[stage]
                    ),
                    bytes_per_rank=_stage_bytes_per_rank(config, queue_stats, stage),
                    useful_forward_flops=_stage_useful_flops_per_rank(config, queue_stats, stage),
                    rounded_forward_flops=_stage_rounded_flops_per_rank(config, queue_stats, stage),
                    dropped_routes=dropped_routes,
                    mode=mode,
                )
            )

    grouped_rows = []
    for stage in (
        FORWARD_STAGE_TOTAL,
        input_stage,
        FORWARD_STAGE_W13,
        FORWARD_STAGE_W2_RETURN,
        FORWARD_STAGE_COMBINE,
    ):
        stage_rows = [row for row in rows if row["stage"] == stage]
        grouped_rows.extend(stage_rows)
        grouped_rows.append(_summary_row(stage_rows))
    return grouped_rows


def _decomposed_forward_row(
    config: PushInboxConfig,
    *,
    queue_stats: dict[str, Any],
    repeat_run: int,
    repeat_runs: int,
    stage: str,
    steady_state_time: float,
    first_call_time: float | None,
    compile_time: float | None,
    bytes_per_rank: float | None,
    useful_forward_flops: float | None,
    rounded_forward_flops: float | None,
    dropped_routes: int,
    mode: str,
) -> dict[str, Any]:
    useful_tflops = None if useful_forward_flops is None else useful_forward_flops / steady_state_time / 1e12
    rounded_tflops = None if rounded_forward_flops is None else rounded_forward_flops / steady_state_time / 1e12
    return {
        "kernel": KERNEL_NAME,
        "implementation": (
            BACKEND_SOURCE_PUSH_PALLAS if stage == FORWARD_STAGE_TOTAL else f"{BACKEND_SOURCE_PUSH_PALLAS}_{stage}"
        ),
        "backend": BACKEND_SOURCE_PUSH_PALLAS,
        "mode": mode,
        "stage": stage,
        "row_type": "repeat",
        "config": asdict(config),
        "queue_stats": queue_stats,
        **queue_stats,
        "outer_jit": False,
        "compile_time": compile_time,
        "lower_compile_time": None,
        "first_run_time": None,
        "first_call_time": first_call_time,
        "repeat_run": repeat_run,
        "repeat_runs": repeat_runs,
        "steady_state_time": steady_state_time,
        "bytes_per_rank": bytes_per_rank,
        "forward_gbps_per_rank": None if bytes_per_rank is None else bytes_per_rank / steady_state_time / 1e9,
        "useful_forward_tflops_per_rank": useful_tflops,
        "rounded_forward_tflops_per_rank": rounded_tflops,
        "useful_fwd_bwd_tflops_per_rank": None,
        "rounded_fwd_bwd_tflops_per_rank": None,
        "useful_backward_tflops_per_rank": None,
        "rounded_backward_tflops_per_rank": None,
        "useful_tflops_per_rank": useful_tflops,
        "rounded_tflops_per_rank": rounded_tflops,
        "dropped_routes": dropped_routes,
        "error": None,
        "error_type": None,
        "error_message": None,
    }


def _decomposed_backward_rows(
    config: PushInboxConfig,
    *,
    timing: BackwardDecomposedTiming,
    queue_stats: dict[str, Any],
    repeat_runs: int,
    dropped_routes: int,
    forward_h_time: float,
) -> list[dict[str, Any]]:
    useful_backward_flops, rounded_backward_flops = _backward_flops_per_rank(config, queue_stats)
    rows = []
    for repeat_run, backward_time in enumerate(timing.steady_state_times):
        rows.append(
            _decomposed_backward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=BACKWARD_STAGE_TOTAL,
                steady_state_time=backward_time,
                first_call_time=timing.first_call_time,
                useful_backward_flops=useful_backward_flops,
                rounded_backward_flops=rounded_backward_flops,
                dropped_routes=dropped_routes,
            )
        )
        rows.append(
            _decomposed_backward_row(
                config,
                queue_stats=queue_stats,
                repeat_run=repeat_run,
                repeat_runs=repeat_runs,
                stage=BACKWARD_STAGE_FORWARD_H,
                steady_state_time=forward_h_time,
                first_call_time=forward_h_time,
                useful_backward_flops=None,
                rounded_backward_flops=None,
                dropped_routes=dropped_routes,
            )
        )
        for stage in BACKWARD_STAGES:
            stage_time = timing.stage_steady_state_times[stage][repeat_run]
            stage_useful_flops, stage_rounded_flops = _backward_stage_flops_per_rank(config, queue_stats, stage)
            rows.append(
                _decomposed_backward_row(
                    config,
                    queue_stats=queue_stats,
                    repeat_run=repeat_run,
                    repeat_runs=repeat_runs,
                    stage=stage,
                    steady_state_time=stage_time,
                    first_call_time=None,
                    useful_backward_flops=stage_useful_flops,
                    rounded_backward_flops=stage_rounded_flops,
                    dropped_routes=dropped_routes,
                )
            )

    grouped_rows = []
    for stage in (
        BACKWARD_STAGE_FORWARD_H,
        BACKWARD_STAGE_TOTAL,
        *BACKWARD_STAGES,
    ):
        stage_rows = [row for row in rows if row["stage"] == stage]
        grouped_rows.extend(stage_rows)
        grouped_rows.append(_summary_row(stage_rows))
    return grouped_rows


def _decomposed_backward_row(
    config: PushInboxConfig,
    *,
    queue_stats: dict[str, Any],
    repeat_run: int,
    repeat_runs: int,
    stage: str,
    steady_state_time: float,
    first_call_time: float | None,
    useful_backward_flops: float | None,
    rounded_backward_flops: float | None,
    dropped_routes: int,
) -> dict[str, Any]:
    useful_tflops = None if useful_backward_flops is None else useful_backward_flops / steady_state_time / 1e12
    rounded_tflops = None if rounded_backward_flops is None else rounded_backward_flops / steady_state_time / 1e12
    return {
        "kernel": KERNEL_NAME,
        "implementation": (
            BACKEND_SOURCE_PUSH_PALLAS if stage == BACKWARD_STAGE_TOTAL else f"{BACKEND_SOURCE_PUSH_PALLAS}_{stage}"
        ),
        "backend": BACKEND_SOURCE_PUSH_PALLAS,
        "mode": MODE_BACKWARD_DECOMPOSED,
        "stage": stage,
        "row_type": "repeat",
        "config": asdict(config),
        "queue_stats": queue_stats,
        **queue_stats,
        "outer_jit": False,
        "compile_time": None,
        "lower_compile_time": None,
        "first_run_time": None,
        "first_call_time": first_call_time,
        "repeat_run": repeat_run,
        "repeat_runs": repeat_runs,
        "steady_state_time": steady_state_time,
        "bytes_per_rank": None,
        "forward_gbps_per_rank": None,
        "useful_forward_tflops_per_rank": None,
        "rounded_forward_tflops_per_rank": None,
        "useful_fwd_bwd_tflops_per_rank": None,
        "rounded_fwd_bwd_tflops_per_rank": None,
        "useful_backward_tflops_per_rank": useful_tflops,
        "rounded_backward_tflops_per_rank": rounded_tflops,
        "useful_tflops_per_rank": useful_tflops,
        "rounded_tflops_per_rank": rounded_tflops,
        "dropped_routes": dropped_routes,
        "error": None,
        "error_type": None,
        "error_message": None,
    }


def _make_benchmark_callable(
    config: PushInboxConfig,
    *,
    backend: str,
    mode: str,
    mesh: Mesh,
    host_inputs,
    route_table,
    inputs: dict[str, jax.Array],
) -> tuple[Callable[..., Any], tuple[jax.Array, ...]]:
    if backend in PUBLIC_BACKEND_TO_IMPLEMENTATION:
        implementation = PUBLIC_BACKEND_TO_IMPLEMENTATION[backend]
        if mode == MODE_FORWARD:
            return (
                lambda x, selected, combine, w13, w2: _public_moe_forward(
                    config, mesh, implementation, x, selected, combine, w13, w2
                ),
                (
                    inputs["x_public"],
                    inputs["selected_public"],
                    inputs["combine_public"],
                    inputs["w13_public"],
                    inputs["w2_public"],
                ),
            )
        if mode == MODE_FORWARD_BACKWARD:
            return (
                jax.value_and_grad(
                    lambda x, selected, combine, w13, w2: _public_moe_loss_aux(
                        config, mesh, implementation, x, selected, combine, w13, w2
                    ),
                    argnums=(0, 2, 3, 4),
                    has_aux=True,
                ),
                (
                    inputs["x_public"],
                    inputs["selected_public"],
                    inputs["combine_public"],
                    inputs["w13_public"],
                    inputs["w2_public"],
                ),
            )
    if backend in SOURCE_PUSH_BACKEND_TO_IMPLEMENTATION:
        implementation = SOURCE_PUSH_BACKEND_TO_IMPLEMENTATION[backend]
        execution_mode = SOURCE_PUSH_BACKEND_TO_EXECUTION_MODE[backend]
        if mode == MODE_FORWARD:
            return (
                lambda x, combine, w13, w2: _preplanned_source_push_forward(
                    config,
                    mesh,
                    host_inputs,
                    route_table,
                    implementation,
                    execution_mode,
                    x,
                    combine,
                    w13,
                    w2,
                ),
                (inputs["x_source"], inputs["combine_source"], inputs["w13_source"], inputs["w2_source"]),
            )
        if mode == MODE_FORWARD_BACKWARD:
            return (
                jax.value_and_grad(
                    lambda x, combine, w13, w2: _preplanned_source_push_loss_aux(
                        config,
                        mesh,
                        host_inputs,
                        route_table,
                        implementation,
                        execution_mode,
                        x,
                        combine,
                        w13,
                        w2,
                    ),
                    argnums=(0, 1, 2, 3),
                    has_aux=True,
                ),
                (inputs["x_source"], inputs["combine_source"], inputs["w13_source"], inputs["w2_source"]),
            )
    raise ValueError(f"unsupported backend={backend!r} mode={mode!r}")


def _public_moe_forward(config: PushInboxConfig, mesh: Mesh, implementation: str, x, selected, combine, w13, w2):
    return moe_mlp(
        x,
        selected,
        combine,
        w13,
        w2,
        activation=ActivationFunctionEnum.silu,
        implementation=implementation,
        mesh=mesh,
        capacity_factor=config.capacity_factor,
        report_capacity_overflow=True,
    )


def _public_moe_loss_aux(config: PushInboxConfig, mesh: Mesh, implementation: str, x, selected, combine, w13, w2):
    out, dropped = _public_moe_forward(config, mesh, implementation, x, selected, combine, w13, w2)
    return jnp.sum(out.astype(jnp.float32)), dropped


def _preplanned_source_push_forward(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    route_table,
    implementation: str,
    execution_mode: str,
    x,
    combine,
    w13,
    w2,
):
    return source_push_moe_mlp_from_plan(
        config,
        host_inputs,
        route_table,
        x,
        combine,
        w13,
        w2,
        implementation=implementation,
        execution_mode=execution_mode,
        mesh=mesh,
    )


def _preplanned_source_push_loss_aux(
    config: PushInboxConfig,
    mesh: Mesh,
    host_inputs,
    route_table,
    implementation: str,
    execution_mode: str,
    x,
    combine,
    w13,
    w2,
):
    out, dropped = _preplanned_source_push_forward(
        config,
        mesh,
        host_inputs,
        route_table,
        implementation,
        execution_mode,
        x,
        combine,
        w13,
        w2,
    )
    return jnp.sum(out.astype(jnp.float32)), dropped


def _time_callable(
    fn: Callable[..., Any],
    *args,
    mesh: Mesh,
    warmup: int,
    steps: int,
    repeat_runs: int,
    use_outer_jit: bool,
    separate_compile: bool,
) -> MlpTiming:
    call = jax.jit(fn) if use_outer_jit else fn
    lower_compile_time = None
    first_run_time = None

    if use_outer_jit and separate_compile:
        with jax.set_mesh(mesh):
            lowered = call.lower(*args)
            start = time.perf_counter()
            compiled = lowered.compile()
            lower_compile_time = time.perf_counter() - start
            start = time.perf_counter()
            output = compiled(*args)
        _block_until_ready(output)
        first_run_time = time.perf_counter() - start
        first_call_time = lower_compile_time + first_run_time
        timed_call = compiled
        compile_time = first_call_time
    else:
        start = time.perf_counter()
        with jax.set_mesh(mesh):
            output = call(*args)
        _block_until_ready(output)
        first_call_time = time.perf_counter() - start
        timed_call = call
        compile_time = first_call_time if use_outer_jit else None
        first_run_time = first_call_time

    for _ in range(warmup):
        with jax.set_mesh(mesh):
            output = timed_call(*args)
        _block_until_ready(output)

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            with jax.set_mesh(mesh):
                output = timed_call(*args)
            _block_until_ready(output)
        steady_state_times.append((time.perf_counter() - start) / steps)

    return MlpTiming(
        compile_time=compile_time,
        lower_compile_time=lower_compile_time,
        first_run_time=first_run_time,
        first_call_time=first_call_time,
        steady_state_times=steady_state_times,
        output=output,
    )


def _timing_rows(
    config: PushInboxConfig,
    *,
    backend: str,
    mode: str,
    timing: MlpTiming,
    queue_stats: dict[str, Any],
    repeat_runs: int,
    outer_jit: bool,
) -> list[dict[str, Any]]:
    useful_forward_flops, rounded_forward_flops = _forward_flops_per_rank(config, queue_stats)
    useful_fwd_bwd_flops = useful_forward_flops * 3
    rounded_fwd_bwd_flops = rounded_forward_flops * 3
    useful_mode_flops = useful_fwd_bwd_flops if mode == MODE_FORWARD_BACKWARD else useful_forward_flops
    rounded_mode_flops = rounded_fwd_bwd_flops if mode == MODE_FORWARD_BACKWARD else rounded_forward_flops
    bytes_per_rank = _forward_bytes_per_rank(config, queue_stats)
    dropped_routes = _dropped_routes_from_output(mode, timing.output)

    rows = []
    for repeat_run, steady_state_time in enumerate(timing.steady_state_times):
        row = {
            "kernel": KERNEL_NAME,
            "implementation": backend,
            "backend": backend,
            "mode": mode,
            "row_type": "repeat",
            "config": asdict(config),
            "queue_stats": queue_stats,
            **queue_stats,
            "outer_jit": outer_jit,
            "compile_time": timing.compile_time,
            "lower_compile_time": timing.lower_compile_time,
            "first_run_time": timing.first_run_time,
            "first_call_time": timing.first_call_time,
            "repeat_run": repeat_run,
            "repeat_runs": repeat_runs,
            "steady_state_time": steady_state_time,
            "bytes_per_rank": bytes_per_rank,
            "forward_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
            "useful_forward_tflops_per_rank": useful_forward_flops / steady_state_time / 1e12,
            "rounded_forward_tflops_per_rank": rounded_forward_flops / steady_state_time / 1e12,
            "useful_fwd_bwd_tflops_per_rank": useful_fwd_bwd_flops / steady_state_time / 1e12,
            "rounded_fwd_bwd_tflops_per_rank": rounded_fwd_bwd_flops / steady_state_time / 1e12,
            "useful_backward_tflops_per_rank": None,
            "rounded_backward_tflops_per_rank": None,
            "useful_tflops_per_rank": useful_mode_flops / steady_state_time / 1e12,
            "rounded_tflops_per_rank": rounded_mode_flops / steady_state_time / 1e12,
            "dropped_routes": dropped_routes,
            "error": None,
            "error_type": None,
            "error_message": None,
        }
        rows.append(row)
    return [*rows, _summary_row(rows)]


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    summary = {
        "kernel": KERNEL_NAME,
        "implementation": first["implementation"],
        "backend": first["backend"],
        "mode": first["mode"],
        "row_type": "summary",
        "config": first["config"],
        "queue_stats": first["queue_stats"],
        "outer_jit": first["outer_jit"],
        "repeat_runs": first["repeat_runs"],
        "repeat_rows": len(rows),
        "error": None,
        "error_type": None,
        "error_message": None,
        "min_steady_state_time": min(row["steady_state_time"] for row in rows),
        "max_steady_state_time": max(row["steady_state_time"] for row in rows),
        "p90_steady_state_time": _percentile(rows, "steady_state_time", 0.90),
        "p95_steady_state_time": _percentile(rows, "steady_state_time", 0.95),
    }
    if "stage" in first:
        summary["stage"] = first["stage"]
    summary.update(first["queue_stats"])
    for metric in SUMMARY_METRICS:
        summary[f"median_{metric}"] = _median(rows, metric)
    return summary


def _device_benchmark_inputs(config: PushInboxConfig, raw_inputs, mesh: Mesh) -> dict[str, jax.Array]:
    x_source = jnp.asarray(raw_inputs.x, dtype=jnp.bfloat16)
    selected_source = jnp.asarray(raw_inputs.selected_experts, dtype=jnp.int32)
    combine_source = jnp.asarray(raw_inputs.combine_weights, dtype=jnp.bfloat16)
    w13_source = jnp.asarray(raw_inputs.w_gate_up, dtype=jnp.bfloat16)
    w2_source = jnp.asarray(raw_inputs.w_down, dtype=jnp.bfloat16)
    return {
        "x_source": jax.device_put(x_source, NamedSharding(mesh, P(AXIS, None, None))),
        "selected_source": jax.device_put(selected_source, NamedSharding(mesh, P(AXIS, None, None))),
        "combine_source": jax.device_put(combine_source, NamedSharding(mesh, P(AXIS, None, None))),
        "w13_source": jax.device_put(w13_source, NamedSharding(mesh, P(AXIS, None, None, None))),
        "w2_source": jax.device_put(w2_source, NamedSharding(mesh, P(AXIS, None, None, None))),
        "x_public": jax.device_put(
            x_source.reshape(config.ep_size * config.tokens_per_rank, config.hidden_dim),
            NamedSharding(mesh, P(AXIS, None)),
        ),
        "selected_public": jax.device_put(
            selected_source.reshape(config.ep_size * config.tokens_per_rank, config.topk),
            NamedSharding(mesh, P(AXIS, None)),
        ),
        "combine_public": jax.device_put(
            combine_source.reshape(config.ep_size * config.tokens_per_rank, config.topk),
            NamedSharding(mesh, P(AXIS, None)),
        ),
        "w13_public": jax.device_put(
            w13_source.reshape(
                config.ep_size * config.experts_per_rank,
                config.hidden_dim,
                2 * config.intermediate_dim,
            ),
            NamedSharding(mesh, P(AXIS, None, None)),
        ),
        "w2_public": jax.device_put(
            w2_source.reshape(
                config.ep_size * config.experts_per_rank,
                config.intermediate_dim,
                config.hidden_dim,
            ),
            NamedSharding(mesh, P(AXIS, None, None)),
        ),
    }


def _make_public_ep_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices()[:ep_size])
    if devices.size < ep_size:
        raise RuntimeError(f"Need {ep_size} visible JAX devices, got {devices.size}")
    return Mesh(devices, (AXIS,), axis_types=(AxisType.Explicit,))


def _resolve_outer_jit(backend: str, outer_jit: str) -> bool:
    if outer_jit == "true":
        return True
    if outer_jit == "false":
        return False
    return backend in (BACKEND_RING, BACKEND_RAGGED_A2A, BACKEND_SOURCE_PUSH_REFERENCE)


def _outer_jit_error_value(backend: str, outer_jit: str) -> bool | str:
    if outer_jit in ("true", "false"):
        return outer_jit == "true"
    if backend not in BACKENDS:
        return "auto"
    return _resolve_outer_jit(backend, outer_jit)


def _forward_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> tuple[float, float]:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    useful = useful_rows * config.hidden_dim * config.intermediate_dim * 6
    rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 6
    return float(useful), float(rounded)


def _backward_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> tuple[float, float]:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    useful = useful_rows * config.hidden_dim * config.intermediate_dim * 12
    rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 12
    return float(useful), float(rounded)


def _backward_stage_flops_per_rank(
    config: PushInboxConfig, queue_stats: dict[str, Any], stage: str
) -> tuple[float | None, float | None]:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    if stage == BACKWARD_STAGE_W2:
        useful = useful_rows * config.hidden_dim * config.intermediate_dim * 4
        rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 4
        return float(useful), float(rounded)
    if stage == BACKWARD_STAGE_W13:
        useful = useful_rows * config.hidden_dim * config.intermediate_dim * 8
        rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * 8
        return float(useful), float(rounded)
    return None, None


def _forward_bytes_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> float:
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    send_bytes = queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16
    w2_bytes = rounded_rows * (config.intermediate_dim + config.hidden_dim) * BYTES_PER_BF16
    output_bytes = config.tokens_per_rank * config.hidden_dim * BYTES_PER_BF16
    return float(send_bytes + w2_bytes + output_bytes)


def _stage_bytes_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any], stage: str) -> float | None:
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    if stage == FORWARD_STAGE_W13:
        return float(queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16)
    if stage == FORWARD_STAGE_W2_RETURN:
        return float(rounded_rows * (config.intermediate_dim + config.hidden_dim) * BYTES_PER_BF16)
    if stage == FORWARD_STAGE_COMBINE:
        return float(config.tokens_per_rank * config.hidden_dim * BYTES_PER_BF16)
    return None


def _stage_useful_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any], stage: str) -> float | None:
    useful_rows = queue_stats["valid_rows_per_rank_mean"]
    if stage == FORWARD_STAGE_W13:
        return float(useful_rows * config.hidden_dim * config.intermediate_dim * 4)
    if stage == FORWARD_STAGE_W2_RETURN:
        return float(useful_rows * config.hidden_dim * config.intermediate_dim * 2)
    return None


def _stage_rounded_flops_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any], stage: str) -> float | None:
    rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
    if stage == FORWARD_STAGE_W13:
        return float(rounded_rows * config.hidden_dim * config.intermediate_dim * 4)
    if stage == FORWARD_STAGE_W2_RETURN:
        return float(rounded_rows * config.hidden_dim * config.intermediate_dim * 2)
    return None


def _dropped_routes_from_output(mode: str, output: Any) -> int:
    if mode == MODE_FORWARD:
        dropped = output[1]
    else:
        dropped = output[0][1]
    return int(jax.device_get(dropped))


def _median(rows: list[dict[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return median(values)


def _percentile(rows: list[dict[str, Any]], field: str, percentile: float) -> float | int | None:
    values = sorted(row[field] for row in rows if row.get(field) is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _parse_csv_choices(value: str, choices: Sequence[str], *, flag: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{flag} must include at least one value")
    invalid = [item for item in parsed if item not in choices]
    if invalid:
        raise ValueError(f"{flag} has unsupported values {invalid}; expected choices from {tuple(choices)}")
    return parsed


if __name__ == "__main__":
    main()
