# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the staged Blackwell source-push forward path end to end."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_MODES,
    FORWARD_EXECUTION_STAGED_DEVICE_SYNC,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
    _shard_source_push_forward_inputs,
    _staged_source_push_forward_callables,
    device_source_push_forward_inputs_from_plan,
    make_source_push_forward_inputs,
    make_source_push_forward_source_plan_raw_inputs,
    prepare_source_push_forward_device_inputs_from_plan,
    reference_source_push_forward_with_h,
    source_push_forward_with_h_from_prepared_device_inputs,
    source_push_forward_with_h_from_plan,
)
from levanter.grug._moe.source_push_inbox import AXIS, PushInboxConfig


BYTES_PER_BF16 = 2
STAGE_INPUT_PREPARE = "input_prepare"
STAGE_DESTINATION_X = "destination_x_transport"
STAGE_W13 = "w13"
STAGE_W2 = "w2"
STAGE_RETURN = "return_transport"
STAGE_COMBINE = "combine"
STAGE_NAMES = (
    STAGE_INPUT_PREPARE,
    STAGE_DESTINATION_X,
    STAGE_W13,
    STAGE_W2,
    STAGE_RETURN,
    STAGE_COMBINE,
)


def _require_blackwell_gpus(ep_size: int) -> str:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Blackwell forward smoke requires a GPU backend, got {jax.default_backend()!r}")
    devices = jax.devices("gpu")
    if len(devices) < ep_size:
        raise RuntimeError(f"Blackwell forward smoke requires {ep_size} visible GPUs, got {len(devices)}")
    device_kind = getattr(devices[0], "device_kind", "")
    if any(name in device_kind for name in ("B200", "B300", "GB200", "GB300")):
        return device_kind
    compute_capability = getattr(devices[0], "compute_capability", None)
    if compute_capability is not None:
        try:
            if float(compute_capability) >= 10.0:
                return device_kind
        except (TypeError, ValueError):
            pass
    raise RuntimeError(f"Blackwell forward smoke requires Blackwell GPUs, got {device_kind!r}")


def _config(args: argparse.Namespace) -> PushInboxConfig:
    return PushInboxConfig(
        ep_size=args.ep_size,
        entries_per_rank=args.entries_per_rank,
        inbox_slots=args.inbox_slots,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        block_m=args.block_m,
        block_k=args.block_k,
        block_n=args.block_n,
        experts_per_rank=args.experts_per_rank,
        send_worker_programs_per_peer=args.send_worker_programs_per_peer,
        worker_programs_per_peer=args.worker_programs_per_peer,
        send_pipeline_depth=args.send_pipeline_depth,
        n_group=args.n_group,
        n_groups_per_job=args.n_groups_per_job,
        routing=args.routing,
        tokens_per_rank=args.tokens_per_rank,
        topk=args.topk,
        capacity_factor=args.capacity_factor,
    )


def _group_row_mask(group_sizes: np.ndarray, rows_per_rank: int) -> np.ndarray:
    mask = np.zeros((group_sizes.shape[0], rows_per_rank), dtype=np.bool_)
    for dst in range(group_sizes.shape[0]):
        row_start = 0
        for size in group_sizes[dst]:
            row_end = row_start + int(size)
            mask[dst, row_start:row_end] = True
            row_start = row_end
    return mask


def _flops_per_rank(config: PushInboxConfig, h_group_rows_total: int) -> dict[str, float]:
    live_rows_per_rank = h_group_rows_total / config.ep_size
    useful_w13 = 2.0 * live_rows_per_rank * config.hidden_dim * (2 * config.intermediate_dim)
    useful_w2 = 2.0 * live_rows_per_rank * config.intermediate_dim * config.hidden_dim
    return {
        "useful_w13_flops_per_rank": useful_w13,
        "useful_w2_flops_per_rank": useful_w2,
        "useful_forward_flops_per_rank": useful_w13 + useful_w2,
    }


def _bytes_per_rank(config: PushInboxConfig, h_group_rows_total: int) -> dict[str, float]:
    live_rows_per_rank = h_group_rows_total / config.ep_size
    source_push_input_bytes = live_rows_per_rank * config.hidden_dim * BYTES_PER_BF16
    return {
        "source_push_input_bytes_per_rank": source_push_input_bytes,
    }


def _block_stage_inputs(inputs: Any) -> None:
    jax.block_until_ready(
        (
            inputs.x,
            inputs.source_x,
            inputs.token_ids,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.h_group_sizes,
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


def _time_blackwell_stages(
    config: PushInboxConfig,
    host_inputs: Any,
    mesh: Mesh,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    *,
    execution_mode: str,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> dict[str, Any]:
    stage_fns = _staged_source_push_forward_callables(
        mesh,
        config,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
        implementation=SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
        execution_mode=execution_mode,
    )
    if stage_fns.destination_local_x_fn is None or stage_fns.w2_y_fn is None or stage_fns.return_fn is None:
        raise AssertionError("Blackwell stage callables were not initialized")

    def make_inputs():
        return _shard_source_push_forward_inputs(
            mesh,
            device_source_push_forward_inputs_from_plan(
                config,
                host_inputs,
                x,
                route_weights,
                w13,
                w2,
                pack_source_tokens=False,
            ),
        )

    def run_step(*, record: bool) -> tuple[jax.Array, jax.Array, dict[str, float]]:
        stage_times: dict[str, float] = {}

        start = time.perf_counter()
        inputs = make_inputs()
        _block_stage_inputs(inputs)
        if record:
            stage_times[STAGE_INPUT_PREPARE] = time.perf_counter() - start

        start = time.perf_counter()
        destination_transport_expert_base = jax.device_put(inputs.expert_base, NamedSharding(mesh, P(None, None)))
        destination_transport_src_base_by_expert = jax.device_put(
            inputs.src_base_by_expert,
            NamedSharding(mesh, P(None, None, None)),
        )
        if inputs.source_x is not None and inputs.token_ids is not None:
            if stage_fns.raw_destination_local_x_fn is None:
                raise AssertionError("Blackwell raw-token destination callable was not initialized")
            destination_x = stage_fns.raw_destination_local_x_fn(
                inputs.source_x,
                inputs.token_ids,
                inputs.send_meta,
                destination_transport_expert_base,
                destination_transport_src_base_by_expert,
            )
        else:
            if inputs.x is None:
                raise ValueError("Blackwell staged transport requires either raw source tokens or packed x inputs")
            destination_x = stage_fns.destination_local_x_fn(
                inputs.x,
                inputs.send_meta,
                destination_transport_expert_base,
                destination_transport_src_base_by_expert,
            )
        if stage_fns.destination_x_barrier_fn is not None:
            destination_x = stage_fns.destination_x_barrier_fn(destination_x)
        jax.block_until_ready(destination_x)
        if record:
            stage_times[STAGE_DESTINATION_X] = time.perf_counter() - start

        start = time.perf_counter()
        h = stage_fns.w13_h_fn(destination_x, inputs.w_gate_up, inputs.h_group_sizes)
        jax.block_until_ready(h)
        if record:
            stage_times[STAGE_W13] = time.perf_counter() - start

        start = time.perf_counter()
        y = stage_fns.w2_y_fn(
            h,
            inputs.h_route_weights,
            inputs.w_down,
            inputs.h_group_sizes,
        )
        jax.block_until_ready(y)
        if record:
            stage_times[STAGE_W2] = time.perf_counter() - start

        start = time.perf_counter()
        source_return = stage_fns.return_fn(
            y,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
        )
        if stage_fns.source_return_barrier_fn is not None:
            source_return = stage_fns.source_return_barrier_fn(source_return)
        jax.block_until_ready(source_return)
        if record:
            stage_times[STAGE_RETURN] = time.perf_counter() - start

        start = time.perf_counter()
        out = stage_fns.combine_fn(
            source_return,
            inputs.queue_dst_ord,
            inputs.queue_entry,
            inputs.queue_row,
            jnp.ones_like(inputs.route_combine_weights),
            inputs.route_valid_mask,
        )
        jax.block_until_ready((out, h))
        if record:
            stage_times[STAGE_COMBINE] = time.perf_counter() - start
        return out, h, stage_times

    start = time.perf_counter()
    out, h, stage_first_call_times = run_step(record=True)
    stage_first_call_total = time.perf_counter() - start
    for _ in range(warmup):
        out, h, _ = run_step(record=False)

    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in STAGE_NAMES}
    stage_total_times = []
    for _ in range(repeat_runs):
        stage_elapsed = {stage: 0.0 for stage in STAGE_NAMES}
        start = time.perf_counter()
        for _ in range(steps):
            out, h, step_stage_times = run_step(record=True)
            for stage in STAGE_NAMES:
                stage_elapsed[stage] += step_stage_times[stage]
        stage_total_times.append((time.perf_counter() - start) / steps)
        for stage in STAGE_NAMES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)

    stage_median_times = {stage: float(np.median(times)) for stage, times in stage_steady_state_times.items()}
    stage_sum_median = float(sum(stage_median_times.values()))
    return {
        "stage_first_call_total": stage_first_call_total,
        "stage_first_call_times": stage_first_call_times,
        "stage_total_times": stage_total_times,
        "stage_median_total_time": float(np.median(stage_total_times)) if stage_total_times else None,
        "stage_steady_state_times": stage_steady_state_times,
        "stage_median_times": stage_median_times,
        "stage_sum_median_time": stage_sum_median,
        "stage_fraction_of_sum": {
            stage: (time_value / stage_sum_median if stage_sum_median else None)
            for stage, time_value in stage_median_times.items()
        },
        "stage_output_shape": list(out.shape),
        "stage_h_shape": list(h.shape),
    }


def _time_prepared_inputs(
    config: PushInboxConfig,
    host_inputs: Any,
    mesh: Mesh,
    x: jax.Array,
    route_weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    *,
    execution_mode: str,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> dict[str, Any]:
    prepare_start = time.perf_counter()
    prepared = prepare_source_push_forward_device_inputs_from_plan(
        config,
        host_inputs,
        x,
        route_weights,
        w13,
        w2,
        mesh=mesh,
    )
    _block_stage_inputs(prepared.inputs)
    prepare_time = time.perf_counter() - prepare_start

    def run_prepared():
        return source_push_forward_with_h_from_prepared_device_inputs(
            config,
            prepared,
            implementation=SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
            execution_mode=execution_mode,
        )

    start = time.perf_counter()
    out, h = run_prepared()
    jax.block_until_ready((out, h))
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        out, h = run_prepared()
        jax.block_until_ready((out, h))

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            out, h = run_prepared()
            jax.block_until_ready((out, h))
        steady_state_times.append((time.perf_counter() - start) / steps)

    return {
        "prepared_input_prepare_time": prepare_time,
        "prepared_input_first_call_time": first_call_time,
        "prepared_input_warmup": warmup,
        "prepared_input_steps": steps,
        "prepared_input_repeat_runs": repeat_runs,
        "prepared_input_steady_state_times": steady_state_times,
        "prepared_input_median_steady_state_time": (
            float(np.median(steady_state_times)) if steady_state_times else None
        ),
        "prepared_input_output_shape": list(out.shape),
        "prepared_input_h_shape": list(h.shape),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device_kind = _require_blackwell_gpus(args.ep_size)
    config = _config(args)
    config.validate()
    raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
    host_inputs = make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="source_push_plan",
    )
    devices = np.asarray(jax.devices("gpu")[: config.ep_size])
    mesh = Mesh(devices, (AXIS,), axis_types=(AxisType.Explicit,))

    x = jnp.asarray(raw_inputs.x, dtype=jnp.bfloat16)
    route_weights = jnp.asarray(raw_inputs.combine_weights, dtype=jnp.float32)
    w13 = jnp.asarray(raw_inputs.w_gate_up, dtype=jnp.bfloat16)
    w2 = jnp.asarray(raw_inputs.w_down, dtype=jnp.bfloat16)

    def run_observed():
        return source_push_forward_with_h_from_plan(
            config,
            host_inputs,
            x,
            route_weights,
            w13,
            w2,
            implementation=SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
            execution_mode=args.execution_mode,
            mesh=mesh,
        )

    start = time.perf_counter()
    observed_out, observed_h, dropped_routes = run_observed()
    jax.block_until_ready((observed_out, observed_h))
    first_call_time = time.perf_counter() - start

    for _ in range(args.warmup):
        observed_out, observed_h, _ = run_observed()
        jax.block_until_ready((observed_out, observed_h))
    steady_state_times = []
    for _ in range(args.repeat_runs):
        start = time.perf_counter()
        for _ in range(args.steps):
            observed_out, observed_h, _ = run_observed()
            jax.block_until_ready((observed_out, observed_h))
        steady_state_times.append((time.perf_counter() - start) / args.steps)

    h_mask = _group_row_mask(host_inputs.h_group_sizes, config.hidden_rows_per_rank)
    h_group_rows_total = int(np.count_nonzero(h_mask))
    output_max_abs_diff = None
    output_mean_abs_diff = None
    h_group_max_abs_diff = None
    h_group_mean_abs_diff = None
    if args.check:
        expected_out, expected_h = reference_source_push_forward_with_h(config, host_inputs)
        observed_out_host = np.asarray(jax.device_get(observed_out), dtype=np.float32)
        expected_out_host = np.asarray(jax.device_get(expected_out), dtype=np.float32)
        observed_h_host = np.asarray(jax.device_get(observed_h), dtype=np.float32)
        expected_h_host = np.asarray(jax.device_get(expected_h), dtype=np.float32)

        output_diff = np.abs(observed_out_host - expected_out_host)
        h_diff = np.abs(observed_h_host[h_mask] - expected_h_host[h_mask])
        output_max_abs_diff = float(np.max(output_diff)) if output_diff.size else 0.0
        output_mean_abs_diff = float(np.mean(output_diff)) if output_diff.size else 0.0
        h_group_max_abs_diff = float(np.max(h_diff)) if h_diff.size else 0.0
        h_group_mean_abs_diff = float(np.mean(h_diff)) if h_diff.size else 0.0

    flop_metrics = _flops_per_rank(config, h_group_rows_total)
    byte_metrics = _bytes_per_rank(config, h_group_rows_total)
    median_steady_state_time = float(np.median(steady_state_times)) if steady_state_times else None
    row = {
        "suite": "blackwell_source_push_forward_smoke",
        "device_kind": device_kind,
        "config": asdict(config),
        "check": args.check,
        "execution_mode": args.execution_mode,
        "first_call_time": first_call_time,
        "warmup": args.warmup,
        "steps": args.steps,
        "repeat_runs": args.repeat_runs,
        "steady_state_times": steady_state_times,
        "median_steady_state_time": median_steady_state_time,
        "output_max_abs_diff": output_max_abs_diff,
        "output_mean_abs_diff": output_mean_abs_diff,
        "h_group_max_abs_diff": h_group_max_abs_diff,
        "h_group_mean_abs_diff": h_group_mean_abs_diff,
        "h_group_rows_total": h_group_rows_total,
        "dropped_routes": int(jax.device_get(dropped_routes)),
        "observed_output_shape": list(observed_out.shape),
        "observed_h_shape": list(observed_h.shape),
        **flop_metrics,
        **byte_metrics,
    }
    if median_steady_state_time:
        row["useful_forward_tflops_per_rank"] = flop_metrics["useful_forward_flops_per_rank"] / (
            median_steady_state_time * 1e12
        )
        row["useful_w13_tflops_per_rank"] = flop_metrics["useful_w13_flops_per_rank"] / (
            median_steady_state_time * 1e12
        )
        row["useful_w2_tflops_per_rank"] = flop_metrics["useful_w2_flops_per_rank"] / (median_steady_state_time * 1e12)
        row["source_push_input_gbps_per_rank"] = byte_metrics["source_push_input_bytes_per_rank"] / (
            median_steady_state_time * 1e9
        )
    if args.stage_timing:
        row.update(
            _time_blackwell_stages(
                config,
                host_inputs,
                mesh,
                x,
                route_weights,
                w13,
                w2,
                execution_mode=args.execution_mode,
                warmup=args.stage_warmup,
                steps=args.stage_steps,
                repeat_runs=args.stage_repeat_runs,
            )
        )
    if args.prepared_input_timing:
        prepared_metrics = _time_prepared_inputs(
            config,
            host_inputs,
            mesh,
            x,
            route_weights,
            w13,
            w2,
            execution_mode=args.execution_mode,
            warmup=args.prepared_input_warmup,
            steps=args.prepared_input_steps,
            repeat_runs=args.prepared_input_repeat_runs,
        )
        prepared_median = prepared_metrics["prepared_input_median_steady_state_time"]
        if prepared_median:
            prepared_metrics["prepared_input_useful_forward_tflops_per_rank"] = flop_metrics[
                "useful_forward_flops_per_rank"
            ] / (prepared_median * 1e12)
            prepared_metrics["prepared_input_useful_w13_tflops_per_rank"] = flop_metrics[
                "useful_w13_flops_per_rank"
            ] / (prepared_median * 1e12)
            prepared_metrics["prepared_input_useful_w2_tflops_per_rank"] = flop_metrics["useful_w2_flops_per_rank"] / (
                prepared_median * 1e12
            )
            prepared_metrics["prepared_input_source_push_input_gbps_per_rank"] = byte_metrics[
                "source_push_input_bytes_per_rank"
            ] / (prepared_median * 1e9)
        row.update(prepared_metrics)
    print(json.dumps(row, sort_keys=True), flush=True)
    if args.check and row["output_max_abs_diff"] > args.output_max_abs_tolerance:
        raise RuntimeError(f"output mismatch exceeds tolerance: {row}")
    if args.check and row["h_group_max_abs_diff"] > args.h_max_abs_tolerance:
        raise RuntimeError(f"H mismatch exceeds tolerance: {row}")
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--entries-per-rank", type=int, default=4)
    parser.add_argument("--inbox-slots", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--intermediate-dim", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--send-worker-programs-per-peer", type=int, default=1)
    parser.add_argument("--worker-programs-per-peer", type=int, default=4)
    parser.add_argument("--send-pipeline-depth", type=int, default=1)
    parser.add_argument("--n-group", type=int, default=1)
    parser.add_argument("--n-groups-per-job", type=int, default=1)
    parser.add_argument("--routing", choices=("balanced", "roughly_balanced"), default="balanced")
    parser.add_argument("--tokens-per-rank", type=int, default=64)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--output-max-abs-tolerance", type=float, default=2.0)
    parser.add_argument("--h-max-abs-tolerance", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--execution-mode", choices=FORWARD_EXECUTION_MODES, default=FORWARD_EXECUTION_STAGED_DEVICE_SYNC
    )
    parser.add_argument("--stage-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stage-warmup", type=int, default=1)
    parser.add_argument("--stage-steps", type=int, default=1)
    parser.add_argument("--stage-repeat-runs", type=int, default=3)
    parser.add_argument("--prepared-input-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prepared-input-warmup", type=int, default=1)
    parser.add_argument("--prepared-input-steps", type=int, default=1)
    parser.add_argument("--prepared-input-repeat-runs", type=int, default=3)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_smoke(parse_args(argv))


if __name__ == "__main__":
    main()
