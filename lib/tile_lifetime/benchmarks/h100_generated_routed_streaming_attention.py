# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark RelationPlan-driven SM90 sparse streaming from tensor semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from backends.h100.cute_routed_streaming_emitter import compile_h100_routed_streaming_program

from shuttle.ir import DType
from tile_lifetime import (
    RoutedAttentionOrientation,
    RoutedAttentionPlanConfig,
    StreamingTileSchedule,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    build_routed_attention_relation,
    compile_routed_streaming_attention_candidates,
    derive_streaming_attention,
    make_causal_block_relation,
    query_major_block_index_plan,
    scaled_score_map,
)


def _non_monotone_relation(sequence: int, block: int, slots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    block_count = sequence // block
    selected = np.full((block_count, slots), -1, dtype=np.int32)
    valid = np.zeros_like(selected, dtype=np.bool_)
    for query_block in range(block_count):
        count = min(slots, query_block + 1)
        generator = np.random.default_rng(seed + query_block * 104_729)
        chosen = generator.choice(query_block + 1, size=count, replace=False).astype(np.int32)
        generator.shuffle(chosen)
        selected[query_block, :count] = chosen
        valid[query_block, :count] = True
    return selected, valid


def _measure(operation, warmups: int, repeats: int) -> list[float]:
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


def _sampled_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected: np.ndarray,
    valid: np.ndarray,
    query_blocks: tuple[int, ...],
    block: int,
    scale: float,
    softcap: float | None,
) -> dict[int, torch.Tensor]:
    group_size = query.shape[2] // key.shape[2]
    reference = {}
    for query_block in query_blocks:
        q_start = query_block * block
        q = query[0, q_start : q_start + block].float()
        blocks = selected[query_block, valid[query_block]].tolist()
        k = torch.cat([key[0, item * block : (item + 1) * block] for item in blocks]).float()
        v = torch.cat([value[0, item * block : (item + 1) * block] for item in blocks]).float()
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
        scores = torch.einsum("mhd,nhd->hmn", q, k) * scale
        if softcap is not None:
            scores = softcap * torch.tanh(scores / softcap)
        query_position = torch.arange(q_start, q_start + block, device=query.device)
        key_position = torch.cat(
            [torch.arange(item * block, (item + 1) * block, device=query.device) for item in blocks]
        )
        scores.masked_fill_(key_position[None, None, :] > query_position[None, :, None], -torch.inf)
        probability = torch.softmax(scores, dim=-1)
        reference[query_block] = torch.einsum("hmn,nhv->mhv", probability, v)
    return reference


def _hardware_record() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--relation-pattern", choices=("non_monotone", "historical"), default="non_monotone")
    parser.add_argument("--scale", type=float, default=2**-3.5)
    parser.add_argument("--softcap", type=float)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.sequence % args.block:
        raise ValueError("sequence must be block aligned")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    query = torch.randn(1, args.sequence, 32, 128, dtype=torch.bfloat16, device=device)
    key = torch.randn(1, args.sequence, 8, 128, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    log_sum_exp = torch.empty(1, 32, args.sequence, dtype=torch.float32, device=device)

    if args.relation_pattern == "historical":
        selected, edge_valid = make_causal_block_relation(
            sequence_length=args.sequence,
            block_size=args.block,
            selected_blocks=args.slots,
        )
    else:
        selected, edge_valid = _non_monotone_relation(args.sequence, args.block, args.slots, args.seed)
    relation = build_routed_attention_relation(selected, edge_valid=edge_valid)
    score_map = apply_causal_score_mask(scaled_score_map(args.scale))
    if args.softcap is not None:
        score_map = apply_tanh_softcap(score_map, args.softcap)
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=args.sequence,
        key_length=args.sequence,
        query_heads=32,
        key_value_heads=8,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    streaming_program = derive_streaming_attention(
        tensor_program,
        schedule=StreamingTileSchedule(
            query_tile_size=args.block,
            key_value_tile_size=args.block,
            pipeline_depth=3,
        ),
    )
    config = RoutedAttentionPlanConfig(
        query_block_size=args.block,
        key_value_block_size=args.block,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
        value_dimension=128,
        buffer_depth=3,
        transfer_workers=1,
        matrix_workers=2,
        reduction_workers=1,
    )
    compilation = compile_routed_streaming_attention_candidates(streaming_program, relation, config)
    compile_started = time.perf_counter()
    executable = compile_h100_routed_streaming_program(
        compilation,
        orientation=RoutedAttentionOrientation.QUERY_MAJOR,
        query=query,
        key=key,
        value=value,
        output=output,
        log_sum_exp=log_sum_exp,
    )
    compile_seconds = time.perf_counter() - compile_started

    def operation() -> None:
        executable(query, key, value, output, log_sum_exp)

    operation()
    torch.cuda.synchronize()
    first_hash = hashlib.sha256(output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    operation()
    torch.cuda.synchronize()
    repeated_hash = hashlib.sha256(output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()

    sampled_blocks = tuple(dict.fromkeys((0, args.sequence // args.block // 2, args.sequence // args.block - 1)))
    reference = _sampled_reference(
        query,
        key,
        value,
        selected,
        edge_valid,
        sampled_blocks,
        args.block,
        args.scale,
        args.softcap,
    )
    errors = []
    for query_block, expected in reference.items():
        start = query_block * args.block
        actual = output[0, start : start + args.block].float()
        errors.append((actual - expected).abs().flatten())
    error = torch.cat(errors)
    samples = _measure(operation, args.warmups, args.repeats)

    index_plan = query_major_block_index_plan(relation)
    relation_bytes = np.concatenate((index_plan.block_count[:, None], index_plan.block_index), axis=1).tobytes()
    query_major, kv_major = compilation.candidates
    record = {
        "benchmark": "shuttle_generated_routed_streaming_attention_sm90",
        "source": "Contract/Map/Fold plus generic RelationPlan",
        "orientation_executed": query_major.orientation.value,
        "orientations_compiled": [item.orientation.value for item in compilation.candidates],
        "configuration": vars(args) | {"json_output": str(args.json_output)},
        "hardware": _hardware_record(),
        "torch": torch.__version__,
        "relation_sha256": hashlib.sha256(relation_bytes).hexdigest(),
        "relation_edges": relation.route_count,
        "relation_non_monotone_rows": int(
            sum(np.any(np.diff(row[mask]) < 0) for row, mask in zip(selected, edge_valid, strict=True))
        ),
        "compile_seconds": compile_seconds,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "mean_ms": statistics.mean(samples),
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "deterministic": first_hash == repeated_hash,
        "output_sha256": first_hash,
        "query_major_plan": query_major.dump(),
        "kv_major_plan": kv_major.dump(),
        "physical_staging": {
            "transport": "TMA global-to-shared K/V",
            "buffer": "three-stage circular shared-memory K/V pipeline",
            "cluster_multicast": False,
            "cluster_note": "SM90 candidate uses cluster task semantics but no TMA multicast",
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
