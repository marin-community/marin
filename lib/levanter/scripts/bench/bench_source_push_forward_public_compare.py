# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare package-private source-push full forward against public EP MoE backends."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.source_push_forward import (
    FORWARD_EXECUTION_MODES,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE,
    SourcePushForwardImplementation,
    make_source_push_forward_source_plan_raw_inputs,
    source_push_forward,
)
from levanter.grug._moe.source_push_inbox import AXIS, ROUTING_MODES, PushInboxConfig, _make_mesh
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILES, source_push_profile_defaults
from levanter.grug._moe.source_push_public import (
    moe_mlp_ep_source_push_mgpu_from_plan,
    prepare_moe_mlp_ep_source_push_mgpu_plan,
)
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


PUBLIC_EP_BACKENDS = ("ring", "ragged_all_to_all", "pallas_mgpu_source_push", "pallas_mgpu_source_push_blackwell")
KERNEL_NAME = "source_push_forward_public_compare"
TIMING_KERNEL_NAME = "source_push_forward_public_timing"
BYTES_PER_BF16 = 2


def _profile_defaults(argv: Sequence[str] | None = None) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    args, _ = pre_parser.parse_known_args(argv)
    return source_push_profile_defaults(args.source_push_profile)


def parse_source_push_forward_public_compare_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source-push public-backend comparison arguments."""

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
    parser.add_argument("--routing", choices=ROUTING_MODES, default=default("routing", "balanced"))
    parser.add_argument("--tokens-per-rank", type=int, default=default("tokens_per_rank", 32768))
    parser.add_argument("--topk", type=int, default=default("topk", 4))
    parser.add_argument("--routing-seed", type=int, default=default("routing_seed", 0))
    parser.add_argument("--capacity-factor", type=float, default=default("capacity_factor", 1.25))
    parser.add_argument(
        "--source-push-implementation",
        choices=(
            SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE,
            SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
            SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
        ),
        default=SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    )
    parser.add_argument(
        "--source-push-execution-mode",
        choices=FORWARD_EXECUTION_MODES,
        default=default("execution_mode", "staged_host_sync"),
    )
    parser.add_argument(
        "--public-implementations",
        default="ragged_all_to_all",
        help="Comma-separated public EP implementations to compare against.",
    )
    parser.add_argument(
        "--public-timing",
        action="store_true",
        help="Time public implementations without diffing output.",
    )
    parser.add_argument("--public-call-mode", choices=("direct", "preplanned"), default="direct")
    parser.add_argument("--warmup", type=int, default=default("warmup", 1))
    parser.add_argument("--steps", type=int, default=default("steps", 1))
    parser.add_argument("--repeat-runs", type=int, default=default("repeat_runs", 3))
    parser.add_argument("--separate-compile", action="store_true", default=default("separate_compile", False))
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_source_push_forward_public_compare_args(argv)
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
    public_implementations = _parse_public_implementations(args.public_implementations)
    if args.public_timing:
        rows = run_source_push_forward_public_timing(
            config,
            public_implementations=public_implementations,
            warmup=args.warmup,
            steps=args.steps,
            repeat_runs=args.repeat_runs,
            separate_compile=args.separate_compile,
            public_call_mode=args.public_call_mode,
        )
    else:
        rows = run_source_push_forward_public_compare(
            config,
            source_push_implementation=args.source_push_implementation,
            source_push_execution_mode=args.source_push_execution_mode,
            public_implementations=public_implementations,
            public_call_mode=args.public_call_mode,
        )
    for row in rows:
        if args.git_sha is not None:
            row["git_sha"] = args.git_sha
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                print(line, file=f, flush=True)


def run_source_push_forward_public_compare(
    config: PushInboxConfig,
    *,
    source_push_implementation: SourcePushForwardImplementation,
    source_push_execution_mode: str,
    public_implementations: Sequence[str],
    public_call_mode: str = "direct",
) -> list[dict[str, Any]]:
    """Compare source-push full forward with public EP backends on one generated input."""

    try:
        if public_call_mode not in ("direct", "preplanned"):
            raise ValueError(f"unknown public_call_mode={public_call_mode!r}")
        config.validate()
        source_push_mesh = _make_mesh(config.ep_size)
        public_mesh = _make_public_ep_mesh(config.ep_size)
        raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
        source_push_out, source_push_dropped = source_push_forward(
            config,
            raw_inputs.x,
            raw_inputs.selected_experts,
            raw_inputs.combine_weights,
            raw_inputs.w_gate_up,
            raw_inputs.w_down,
            implementation=source_push_implementation,
            execution_mode=source_push_execution_mode,
            mesh=source_push_mesh,
        )
        source_push_host = np.asarray(jax.device_get(source_push_out), dtype=np.float32)
        source_push_dropped_host = int(jax.device_get(source_push_dropped))
        rows = []
        for public_implementation in public_implementations:
            public_inputs = _make_public_moe_inputs(config, raw_inputs, public_mesh)
            public_plan = None
            if public_call_mode == "preplanned":
                public_plan = _prepare_public_moe_plan(config, public_inputs, public_mesh, public_implementation)
            public_out, public_dropped = _call_public_moe_with_mode(
                config,
                public_inputs,
                public_mesh,
                public_implementation,
                public_call_mode=public_call_mode,
                public_plan=public_plan,
            )
            public_host = np.asarray(jax.device_get(public_out), dtype=np.float32).reshape(source_push_host.shape)
            public_dropped_host = int(jax.device_get(public_dropped))
            diff = np.abs(source_push_host - public_host)
            rows.append(
                {
                    "kernel": KERNEL_NAME,
                    "implementation": KERNEL_NAME,
                    "source_push_implementation": source_push_implementation,
                    "source_push_execution_mode": source_push_execution_mode,
                    "public_implementation": public_implementation,
                    "public_call_mode": public_call_mode,
                    "config": asdict(config),
                    "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
                    "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
                    "source_push_dropped_routes": source_push_dropped_host,
                    "public_dropped_routes": public_dropped_host,
                    "dropped_route_delta": source_push_dropped_host - public_dropped_host,
                    "output_shape": list(source_push_host.shape),
                    "error": None,
                    "error_type": None,
                    "error_message": None,
                }
            )
        return rows
    except Exception as exc:  # noqa: BLE001 - comparison scripts should emit structured failure rows.
        return [
            {
                "kernel": KERNEL_NAME,
                "implementation": KERNEL_NAME,
                "source_push_implementation": source_push_implementation,
                "source_push_execution_mode": source_push_execution_mode,
                "public_implementation": ",".join(public_implementations),
                "public_call_mode": public_call_mode,
                "config": asdict(config),
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "source_push_dropped_routes": None,
                "public_dropped_routes": None,
                "dropped_route_delta": None,
                "output_shape": None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ]
    finally:
        jax.clear_caches()


def run_source_push_forward_public_timing(
    config: PushInboxConfig,
    *,
    public_implementations: Sequence[str],
    warmup: int,
    steps: int,
    repeat_runs: int,
    separate_compile: bool,
    public_call_mode: str,
) -> list[dict[str, Any]]:
    """Time public EP MoE implementations on one generated source-push input."""

    if warmup < 0 or steps <= 0 or repeat_runs <= 0:
        raise ValueError(f"expected warmup>=0, steps>0, repeat_runs>0; got {warmup=}, {steps=}, {repeat_runs=}")
    if public_call_mode not in ("direct", "preplanned"):
        raise ValueError(f"unknown public_call_mode={public_call_mode!r}")
    rows = []
    try:
        config.validate()
        public_mesh = _make_public_ep_mesh(config.ep_size)
        raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
        public_inputs = _make_public_moe_inputs(config, raw_inputs, public_mesh)
        devices = np.asarray(public_mesh.devices, dtype=object).reshape(-1)
        device_type = getattr(devices[0], "device_kind", None) if devices.size else None
        backend = jax.default_backend()
        shape = {
            "tokens_per_rank": config.tokens_per_rank,
            "hidden_dim": config.hidden_dim,
            "intermediate_dim": config.intermediate_dim,
            "experts_per_rank": config.experts_per_rank,
            "ep_size": config.ep_size,
            "topk": config.topk,
        }
        block_sizes = {
            "block_m": config.block_m,
            "block_n": config.block_n,
            "block_k": config.block_k,
            "inbox_slots": config.inbox_slots,
            "entries_per_rank": config.entries_per_rank,
            "send_worker_programs_per_peer": config.send_worker_programs_per_peer,
            "worker_programs_per_peer": config.worker_programs_per_peer,
            "send_pipeline_depth": config.send_pipeline_depth,
        }
        useful_rows_per_rank = config.tokens_per_rank * config.topk
        useful_forward_flops_per_rank = useful_rows_per_rank * config.hidden_dim * config.intermediate_dim * 6
        source_push_input_bytes_per_rank = useful_rows_per_rank * config.hidden_dim * BYTES_PER_BF16

        for public_implementation in public_implementations:
            public_plan = None
            if public_call_mode == "preplanned":
                public_plan = _prepare_public_moe_plan(config, public_inputs, public_mesh, public_implementation)
            compile_time = None
            dropped_routes = None
            if separate_compile:
                compile_start = time.perf_counter()
                out, dropped = _call_public_moe_with_mode(
                    config,
                    public_inputs,
                    public_mesh,
                    public_implementation,
                    public_call_mode=public_call_mode,
                    public_plan=public_plan,
                )
                jax.block_until_ready((out, dropped))
                compile_time = time.perf_counter() - compile_start
                dropped_routes = int(jax.device_get(dropped))
            for _ in range(warmup):
                out, dropped = _call_public_moe_with_mode(
                    config,
                    public_inputs,
                    public_mesh,
                    public_implementation,
                    public_call_mode=public_call_mode,
                    public_plan=public_plan,
                )
                jax.block_until_ready((out, dropped))
                dropped_routes = int(jax.device_get(dropped))

            steady_state_times = []
            for _ in range(repeat_runs):
                start = time.perf_counter()
                for _ in range(steps):
                    out, dropped = _call_public_moe_with_mode(
                        config,
                        public_inputs,
                        public_mesh,
                        public_implementation,
                        public_call_mode=public_call_mode,
                        public_plan=public_plan,
                    )
                    jax.block_until_ready((out, dropped))
                elapsed = (time.perf_counter() - start) / steps
                steady_state_times.append(elapsed)
                dropped_routes = int(jax.device_get(dropped))

            steady_state_time = float(np.median(np.asarray(steady_state_times, dtype=np.float64)))
            rows.append(
                {
                    "kernel": TIMING_KERNEL_NAME,
                    "implementation": public_implementation,
                    "shape": shape,
                    "dtype": "bfloat16",
                    "backend": backend,
                    "device_type": device_type,
                    "device_count": int(devices.size),
                    "block_sizes": block_sizes,
                    "config": asdict(config),
                    "compile_time": compile_time,
                    "steady_state_time": steady_state_time,
                    "steady_state_times": steady_state_times,
                    "warmup": warmup,
                    "steps": steps,
                    "repeat_runs": repeat_runs,
                    "separate_compile": separate_compile,
                    "public_call_mode": public_call_mode,
                    "dropped_routes": dropped_routes,
                    "useful_forward_flops_per_rank": useful_forward_flops_per_rank,
                    "useful_forward_tflops_per_rank": useful_forward_flops_per_rank / steady_state_time / 1e12,
                    "source_push_input_bytes_per_rank": source_push_input_bytes_per_rank,
                    "source_push_input_gbps_per_rank": source_push_input_bytes_per_rank / steady_state_time / 1e9,
                    "xla_flags": os.environ.get("XLA_FLAGS"),
                    "backend_env": {
                        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                        "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
                    },
                    "error": None,
                    "error_type": None,
                    "error_message": None,
                }
            )
        return rows
    except Exception as exc:  # noqa: BLE001 - benchmark scripts should emit structured failure rows.
        return [
            {
                "kernel": TIMING_KERNEL_NAME,
                "implementation": ",".join(public_implementations),
                "shape": {
                    "tokens_per_rank": config.tokens_per_rank,
                    "hidden_dim": config.hidden_dim,
                    "intermediate_dim": config.intermediate_dim,
                    "experts_per_rank": config.experts_per_rank,
                    "ep_size": config.ep_size,
                    "topk": config.topk,
                },
                "dtype": "bfloat16",
                "backend": jax.default_backend(),
                "device_type": None,
                "device_count": None,
                "block_sizes": None,
                "config": asdict(config),
                "compile_time": None,
                "steady_state_time": None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ]
    finally:
        jax.clear_caches()


def _make_public_ep_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices()[:ep_size])
    if devices.size < ep_size:
        raise RuntimeError(f"Need {ep_size} visible JAX devices, got {devices.size}")
    return Mesh(devices, (AXIS,), axis_types=(AxisType.Explicit,))


def _make_public_moe_inputs(config: PushInboxConfig, raw_inputs, mesh):
    x = jnp.asarray(
        raw_inputs.x.reshape(config.ep_size * config.tokens_per_rank, config.hidden_dim), dtype=jnp.bfloat16
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
    x = jax.device_put(x, NamedSharding(mesh, P(AXIS, None)))
    selected_experts = jax.device_put(selected_experts, NamedSharding(mesh, P(AXIS, None)))
    combine_weights = jax.device_put(combine_weights, NamedSharding(mesh, P(AXIS, None)))
    w_gate_up = jax.device_put(w_gate_up, NamedSharding(mesh, P(AXIS, None, None)))
    w_down = jax.device_put(w_down, NamedSharding(mesh, P(AXIS, None, None)))
    return x, selected_experts, combine_weights, w_gate_up, w_down


def _call_public_moe(config: PushInboxConfig, public_inputs, mesh, implementation: str):
    x, selected_experts, combine_weights, w_gate_up, w_down = public_inputs
    with jax.set_mesh(mesh):
        return moe_mlp(
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


def _prepare_public_moe_plan(config: PushInboxConfig, public_inputs, mesh, implementation: str):
    x, selected_experts, combine_weights, w_gate_up, w_down = public_inputs
    return prepare_moe_mlp_ep_source_push_mgpu_plan(
        x,
        selected_experts,
        combine_weights,
        w_gate_up,
        w_down,
        activation=ActivationFunctionEnum.silu,
        mesh=mesh,
        batch_spec=P(AXIS, None),
        capacity_factor=config.capacity_factor,
        implementation=implementation,
    )


def _call_public_moe_with_mode(
    config: PushInboxConfig,
    public_inputs,
    mesh,
    implementation: str,
    *,
    public_call_mode: str,
    public_plan,
):
    if public_call_mode == "direct":
        return _call_public_moe(config, public_inputs, mesh, implementation)
    if public_call_mode == "preplanned":
        if public_plan is None:
            raise ValueError("preplanned public call mode requires a prepared plan")
        x, _selected_experts, combine_weights, w_gate_up, w_down = public_inputs
        return moe_mlp_ep_source_push_mgpu_from_plan(public_plan, x, combine_weights, w_gate_up, w_down)
    raise ValueError(f"unknown public_call_mode={public_call_mode!r}")


def _parse_public_implementations(value: str) -> tuple[str, ...]:
    implementations = tuple(part.strip() for part in value.split(",") if part.strip())
    if not implementations:
        raise ValueError("--public-implementations must include at least one implementation")
    invalid = [implementation for implementation in implementations if implementation not in PUBLIC_EP_BACKENDS]
    if invalid:
        raise ValueError(
            f"unsupported public EP implementations {invalid}; expected choices from {PUBLIC_EP_BACKENDS}"
        )
    return implementations


if __name__ == "__main__":
    main()
