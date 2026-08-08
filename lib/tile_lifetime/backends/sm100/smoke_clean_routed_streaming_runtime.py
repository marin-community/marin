# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual GB200 smoke for the clean routed streaming runtime."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from clean_routed_streaming_emitter import PartialMergeScheduleKind, PartialValueDType  # noqa: E402
from clean_routed_streaming_runtime import compile_routed_streaming_callable  # noqa: E402

from tile_lifetime import DType, StreamingTileSchedule, build_attention_tensor_program  # noqa: E402
from tile_lifetime.relation import build_relation_plan  # noqa: E402
from tile_lifetime.sm100_routed_lowering import (  # noqa: E402
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
)
from tile_lifetime.streaming_attention import derive_streaming_attention, scaled_score_map  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa-root", type=Path, required=True)
    parser.add_argument("--query-length", type=int, default=128)
    parser.add_argument("--key-length", type=int, default=512)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int, default=2)
    parser.add_argument("--selected-count", type=int, default=4)
    parser.add_argument("--partial-merge-threads", type=int, choices=(128, 256), default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument(
        "--partial-value-dtype",
        choices=tuple(value.value for value in PartialValueDType),
        default=PartialValueDType.BF16.value,
    )
    parser.add_argument(
        "--partial-merge-schedule",
        choices=tuple(value.value for value in PartialMergeScheduleKind),
        default=PartialMergeScheduleKind.ROW_BLOCK.value,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _relation_indices(
    *,
    query_length: int,
    key_value_heads: int,
    key_block_count: int,
    selected_count: int,
    seed: int,
) -> np.ndarray:
    if selected_count > key_block_count:
        raise ValueError("selected count cannot exceed the number of KV blocks")
    generator = np.random.default_rng(seed)
    indices = np.empty((key_value_heads, query_length, selected_count), dtype=np.int32)
    for head, query in np.ndindex(key_value_heads, query_length):
        # Independent rows deliberately make right-side fanout non-monotone.
        indices[head, query] = np.sort(generator.choice(key_block_count, size=selected_count, replace=False))
    return indices


def _lowering(arguments: argparse.Namespace, q2k: np.ndarray):
    score_map = scaled_score_map(128**-0.5)
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=arguments.query_length,
        key_length=arguments.key_length,
        query_heads=arguments.query_heads,
        key_value_heads=arguments.key_value_heads,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    program = derive_streaming_attention(
        tensor_program,
        schedule=StreamingTileSchedule(
            query_tile_size=128,
            key_value_tile_size=128,
            pipeline_depth=2,
        ),
    )
    destinations = np.transpose(q2k, (1, 0, 2)).reshape(arguments.query_length, -1)
    relation = build_relation_plan(
        destinations,
        np.ones(destinations.shape, dtype=np.float32),
        destination_rank_by_item=np.zeros(arguments.key_length // 128, dtype=np.int32),
        destination_local_item_by_item=np.arange(arguments.key_length // 128, dtype=np.int32),
        padding_quantum=1,
    )
    schedule = replace(
        default_sm100_routed_schedules()[1],
        partial_merge_threads=arguments.partial_merge_threads,
    )
    return lower_sm100_routed_streaming_program(
        program,
        relation,
        schedule,
    )


def _reference(q2k: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    query_count, query_heads, head_dim = q.shape
    key_value_heads = k.shape[1]
    query_heads_per_key_value_head = query_heads // key_value_heads
    block_size = 128
    token_offsets = torch.arange(block_size, device=q.device)
    result_by_head = []
    q_by_head = q.float().reshape(query_count, key_value_heads, query_heads_per_key_value_head, head_dim)
    k_by_head = k.float().permute(1, 0, 2)
    v_by_head = v.float().permute(1, 0, 2)
    for key_value_head in range(key_value_heads):
        token_indices = q2k[key_value_head, :, :, None] * block_size + token_offsets
        token_indices = token_indices.reshape(query_count, -1).long()
        selected_k = k_by_head[key_value_head][token_indices]
        selected_v = v_by_head[key_value_head][token_indices]
        scores = torch.einsum(
            "qgd,qld->qgl",
            q_by_head[:, key_value_head],
            selected_k,
        ) * (head_dim**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        result_by_head.append(torch.einsum("qgl,qld->qgd", probabilities, selected_v))
    return torch.stack(result_by_head, dim=1).reshape(query_count, query_heads, head_dim)


def main() -> None:
    arguments = _arguments()
    if arguments.key_length % 128:
        raise ValueError("key length must be divisible by 128")
    torch.manual_seed(arguments.seed)
    q2k_host = _relation_indices(
        query_length=arguments.query_length,
        key_value_heads=arguments.key_value_heads,
        key_block_count=arguments.key_length // 128,
        selected_count=arguments.selected_count,
        seed=arguments.seed,
    )
    lowering = _lowering(arguments, q2k_host)
    device = torch.device("cuda")
    q2k = torch.from_numpy(q2k_host).to(device)
    q = torch.randn(
        arguments.query_length,
        arguments.query_heads,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        arguments.key_length,
        arguments.key_value_heads,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    operation = compile_routed_streaming_callable(
        arguments.msa_root,
        lowering,
        partial_value_dtype=PartialValueDType(arguments.partial_value_dtype),
        partial_merge_schedule=PartialMergeScheduleKind(arguments.partial_merge_schedule),
        build_directory=arguments.build_directory,
    )

    first = operation(q2k, q, k, v)
    torch.cuda.synchronize()
    record: dict[str, object] = {
        "lowering": lowering.dump(),
        "generated_source_sha256": operation.generated_sources.generated_source_sha256,
        "output_shape": list(first.output.shape),
        "partial_value_dtype": arguments.partial_value_dtype,
        "partial_merge_schedule": arguments.partial_merge_schedule,
        "partial_merge_threads": arguments.partial_merge_threads,
        "work_count": int(first.relation_runtime.work_count.cpu().item()),
        "split_count_min": int(first.relation_runtime.split_counts.min().cpu().item()),
        "split_count_max": int(first.relation_runtime.split_counts.max().cpu().item()),
    }
    if not arguments.skip_reference:
        expected = _reference(q2k, q, k, v)
        difference = (first.output.float() - expected).abs()
        record["maximum_absolute_error"] = float(difference.max().item())
        record["mean_absolute_error"] = float(difference.mean().item())

    for _ in range(arguments.warmups):
        operation(q2k, q, k, v)
    torch.cuda.synchronize()
    samples = []
    for _ in range(arguments.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation(q2k, q, k, v)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    record["samples_ms"] = samples
    record["median_ms"] = float(np.median(samples))
    record["finite"] = bool(torch.isfinite(first.output).all().item())
    record["deterministic"] = bool(torch.equal(first.output, operation(q2k, q, k, v).output))
    record["effective_selected_tokens"] = arguments.selected_count * 128
    record["dense_fraction"] = arguments.selected_count * 128 / arguments.key_length
    record["theoretical_score_elements"] = (
        arguments.query_length * arguments.query_heads * arguments.selected_count * 128
    )
    record["score_scale"] = math.pow(128, -0.5)
    record["timestamp_seconds"] = time.time()
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
