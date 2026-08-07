# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan and benchmark deterministic bounded KV-major slot-wave attention.

The CLI works without GPU dependencies in planning mode. ``--gpu`` executes
the intentionally coarse one-launch-per-slot candidate and records the global
FP32 state cost, raw timings, and deterministic output hash. It does not claim
KV staging reuse or overlap yet; the point is to execute RelationPlan's generic
relation in KV-major order without FSA's per-edge partial materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_metadata import command_record, nvidia_smi_snapshot, toolchain_snapshot
from kv_major_slot_waves import SlotWaveSchedule, build_slot_wave_schedule, execute_slot_wave_reference, schedule_record

from tile_lifetime import (
    RelationPlan,
    RoutedAttentionPlanConfig,
    build_routed_attention_relation,
    compile_bounded_kv_major_candidate,
    make_causal_block_relation,
    routed_attention_reference,
)

try:
    import torch
    import triton
    from triton_kv_major_slot_waves import device_slot_waves, execute_slot_waves
except ImportError:
    torch = None
    triton = None
    device_slot_waves = None
    execute_slot_waves = None


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _dense_relation_sha256(relation: RelationPlan) -> str:
    dense = np.zeros((relation.source_item_count, relation.destination_count), dtype=np.bool_)
    valid_routes = np.flatnonzero(relation.edge_valid.reshape(-1))
    dense[relation.source_item[valid_routes], relation.destination_item[valid_routes]] = True
    return _array_sha256(dense)


def _toolchain_record() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {"nvcc": {"status": "unavailable"}}
    return toolchain_snapshot(nvcc)


def _planning_result(args: argparse.Namespace) -> tuple[RelationPlan, SlotWaveSchedule, dict[str, Any]]:
    routing_started = time.perf_counter()
    selected, edge_valid = make_causal_block_relation(
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        selected_blocks=args.selected_blocks,
    )
    routing_ms = (time.perf_counter() - routing_started) * 1_000
    relation_started = time.perf_counter()
    relation = build_routed_attention_relation(selected, edge_valid=edge_valid)
    relation_ms = (time.perf_counter() - relation_started) * 1_000
    schedule_started = time.perf_counter()
    schedule = build_slot_wave_schedule(relation, edge_order=args.wave_order)
    schedule_ms = (time.perf_counter() - schedule_started) * 1_000
    physical = compile_bounded_kv_major_candidate(
        relation,
        RoutedAttentionPlanConfig(
            query_block_size=args.block_size,
            key_value_block_size=args.block_size,
            query_heads=args.query_heads,
            key_value_heads=args.key_value_heads,
            head_dimension=args.head_dimension,
            value_dimension=args.head_dimension,
            buffer_depth=2,
            transfer_workers=1,
            matrix_workers=4,
            reduction_workers=1,
        ),
    )
    record = {
        "selected_sha256": _array_sha256(selected),
        "edge_valid_sha256": _array_sha256(edge_valid),
        "dense_boolean_relation_sha256": _dense_relation_sha256(relation),
        "routing_ms": routing_ms,
        "relation_planning_ms": relation_ms,
        "slot_wave_planning_ms": schedule_ms,
        "relation_edges": relation.route_count,
        "slot_waves": schedule_record(schedule),
        "physical_plan": physical.dump(),
    }
    return relation, schedule, record


def _cpu_check() -> dict[str, Any]:
    sequence_length = 96
    block_size = 32
    selected, edge_valid = make_causal_block_relation(
        sequence_length=sequence_length,
        block_size=block_size,
        selected_blocks=3,
    )
    relation = build_routed_attention_relation(selected, edge_valid=edge_valid)
    schedule = build_slot_wave_schedule(relation)
    rng = np.random.default_rng(20260807)
    query = rng.normal(size=(3, block_size, 4, 8)).astype(np.float32)
    key = rng.normal(size=(3, block_size, 2, 8)).astype(np.float32)
    value = rng.normal(size=(3, block_size, 2, 8)).astype(np.float32)
    scale = 1.0 / math.sqrt(query.shape[-1])
    actual = execute_slot_wave_reference(
        query,
        key,
        value,
        schedule,
        scale=scale,
        causal=True,
        sequence_length=sequence_length,
    )
    expected = routed_attention_reference(
        query,
        key,
        value,
        selected,
        edge_valid=edge_valid,
        scale=scale,
        causal=True,
        sequence_length=sequence_length,
    )
    absolute = np.abs(actual - expected)
    return {
        "maximum_absolute_error": float(np.max(absolute)),
        "mean_absolute_error": float(np.mean(absolute)),
        "allclose_rtol_2e-6_atol_2e-6": bool(np.allclose(actual, expected, rtol=2e-6, atol=2e-6)),
        "output_sha256": _array_sha256(actual),
    }


def _cuda_samples(operation: Callable[[], object], torch: Any, *, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _torch_bf16_sha256(value: Any) -> str:
    return _array_sha256(value.detach().contiguous().view(torch.uint16).cpu().numpy())


def _selected_attention_reference(
    torch_module: Any,
    query: Any,
    key: Any,
    value: Any,
    relation: RelationPlan,
    *,
    block_size: int,
    scale: float,
    checked_query_blocks: np.ndarray,
) -> tuple[Any, Any]:
    head_map = torch_module.arange(query.shape[1], device=query.device) // (query.shape[1] // key.shape[1])
    token_indices = []
    outputs = []
    for query_block in checked_query_blocks.tolist():
        query_start = query_block * block_size
        query_stop = query_start + block_size
        query_positions = torch_module.arange(query_start, query_stop, device=query.device)
        flat_start = query_block * relation.route_slots
        slots = np.flatnonzero(relation.edge_valid[query_block])
        selected_blocks = relation.destination_item[flat_start + slots]
        key_indices = torch_module.cat(
            [
                torch_module.arange(
                    int(key_value_block) * block_size,
                    (int(key_value_block) + 1) * block_size,
                    device=query.device,
                )
                for key_value_block in selected_blocks
            ]
        )
        expanded_key = key[key_indices][:, head_map].float()
        scores = torch_module.einsum("qhd,khd->qhk", query[query_start:query_stop].float(), expanded_key) * scale
        scores.masked_fill_(query_positions[:, None, None] < key_indices[None, None, :], -torch_module.inf)
        probabilities = torch_module.softmax(scores, dim=-1)
        expanded_value = value[key_indices][:, head_map].float()
        outputs.append(torch_module.einsum("qhk,khv->qhv", probabilities, expanded_value))
        token_indices.append(query_positions)
    return torch_module.cat(token_indices), torch_module.cat(outputs)


def _gpu_result(args: argparse.Namespace, relation: RelationPlan, schedule: SlotWaveSchedule) -> dict[str, Any]:
    if torch is None or triton is None or device_slot_waves is None or execute_slot_waves is None:
        raise RuntimeError("--gpu requires PyTorch and Triton")
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu requires a CUDA device")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    query = torch.randn(
        (args.sequence_length, args.query_heads, args.head_dimension),
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        (args.sequence_length, args.key_value_heads, args.head_dimension),
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    device_waves = device_slot_waves(schedule, device)
    scale = 1.0 / math.sqrt(args.head_dimension)

    def operation() -> Any:
        return execute_slot_waves(
            query,
            key,
            value,
            device_waves,
            block_size=args.block_size,
            query_tile_size=args.query_tile_size,
            scale=scale,
        )

    first_started = time.perf_counter()
    first_output = operation()
    torch.cuda.synchronize()
    compile_and_first_run_seconds = time.perf_counter() - first_started
    samples = _cuda_samples(operation, torch, warmups=args.warmups, repeats=args.repeats)
    output = operation()
    torch.cuda.synchronize()
    block_count = relation.source_item_count
    checked_query_blocks = np.unique(
        np.linspace(0, block_count - 1, min(args.correctness_blocks, block_count), dtype=np.int32)
    )
    checked_tokens, reference = _selected_attention_reference(
        torch,
        query,
        key,
        value,
        relation,
        block_size=args.block_size,
        scale=scale,
        checked_query_blocks=checked_query_blocks,
    )
    absolute = (output[checked_tokens].float() - reference).abs()
    qk_pv_flops_per_edge = 4 * args.query_heads * args.block_size * args.block_size * args.head_dimension
    selected_work_flops = relation.route_count * qk_pv_flops_per_edge
    return {
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "latency_ms": {
            "samples": samples,
            "median": statistics.median(samples),
            "mean": statistics.fmean(samples),
            "minimum": min(samples),
            "maximum": max(samples),
        },
        "selected_qk_pv_flops": selected_work_flops,
        "selected_qk_pv_tflops_at_median": selected_work_flops / statistics.median(samples) / 1.0e9,
        "hashes": {
            "query_sha256": _torch_bf16_sha256(query),
            "key_sha256": _torch_bf16_sha256(key),
            "value_sha256": _torch_bf16_sha256(value),
            "first_output_sha256": _torch_bf16_sha256(first_output),
            "timed_output_sha256": _torch_bf16_sha256(output),
        },
        "deterministic_repeated_output": bool(torch.equal(first_output, output)),
        "correctness": {
            "checked_query_blocks": checked_query_blocks.tolist(),
            "maximum_absolute_error": float(absolute.max()),
            "mean_absolute_error": float(absolute.mean()),
            "p99_absolute_error": float(torch.quantile(absolute, 0.99)),
            "allclose_atol_0.03_rtol_0.03": bool(
                torch.allclose(output[checked_tokens].float(), reference, atol=0.03, rtol=0.03)
            ),
            "nan_count": int(torch.isnan(output).sum()),
            "infinity_count": int(torch.isinf(output).sum()),
        },
        "online_state_bytes": (
            args.sequence_length * args.query_heads * (args.head_dimension + 2) * np.dtype(np.float32).itemsize
        ),
        "edge_partial_materialization_bytes": 0,
        "sequence_squared_materialization_bytes": 0,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu_telemetry": {"initial": nvidia_smi_snapshot(), "final": nvidia_smi_snapshot()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--selected-blocks", type=int, default=16)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--key-value-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--query-tile-size", type=int, default=16)
    parser.add_argument("--wave-order", choices=("kv_major", "source"), default="kv_major")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--correctness-blocks", type=int, default=4)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.sequence_length % args.block_size:
        raise ValueError("the first slot-wave candidate requires a block-aligned sequence")
    if args.query_heads % args.key_value_heads:
        raise ValueError("query heads must map evenly onto KV heads")

    relation, schedule, planning = _planning_result(args)
    result = {
        "schema_version": 1,
        "benchmark": "shuttle_h100_kv_major_slot_waves",
        "status": "gpu_executed" if args.gpu else "planning_only",
        "shape": {
            "sequence_length": args.sequence_length,
            "block_size": args.block_size,
            "selected_blocks": args.selected_blocks,
            "query_heads": args.query_heads,
            "key_value_heads": args.key_value_heads,
            "head_dimension": args.head_dimension,
            "query_tile_size": args.query_tile_size,
            "wave_order": args.wave_order,
            "dtype": "bfloat16",
            "causal": True,
        },
        "planning": planning,
        "cpu_check": _cpu_check(),
        "gpu": _gpu_result(args, relation, schedule) if args.gpu else None,
        "numerical_policy": "ascending selected-slot FP32 online updates; no atomics",
        "limitations": [
            "Each selected-slot wave is a separate kernel launch and global synchronization point.",
            "Destination sorting targets cache locality but does not yet stage one KV block for multiple query CTAs.",
            "The FP32 online state is materialized globally between waves.",
            "Each edge/head update is split into independent query-row tiles; the default physical M tile is 16.",
            "The first Triton kernel supports BF16, causal self-attention, equal Q/K/V dimensions, and D=64 or 128.",
            (
                "Kernel timings include FP32 state allocation/initialization and finalization, "
                "but exclude relation planning."
            ),
        ],
        "environment": {
            "command": command_record(),
            "toolchain": _toolchain_record(),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
