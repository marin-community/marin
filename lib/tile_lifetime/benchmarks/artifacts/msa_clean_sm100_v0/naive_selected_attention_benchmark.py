"""Measure a literal eager PyTorch reference for the pinned 16K MSA payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch


BLOCK_SIZE = 128
HEAD_DIMENSION = 128
SELECTED_BLOCKS = 16


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def payload_reference(
    q2k: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_chunk_size: int,
) -> torch.Tensor:
    """Execute gather-QK-softmax-PV literally with bounded query chunks."""
    query_count, query_heads, head_dimension = query.shape
    key_value_heads = key.shape[1]
    heads_per_group = query_heads // key_value_heads
    offsets = torch.arange(BLOCK_SIZE, device=query.device)
    query_grouped = query.float().reshape(query_count, key_value_heads, heads_per_group, head_dimension)
    outputs = []
    for group in range(key_value_heads):
        group_outputs = []
        key_group = key[:, group].float()
        value_group = value[:, group].float()
        for query_begin in range(0, query_count, query_chunk_size):
            query_end = min(query_begin + query_chunk_size, query_count)
            chunk_count = query_end - query_begin
            valid_blocks = q2k[group, query_begin:query_end] >= 0
            safe_blocks = q2k[group, query_begin:query_end].clamp_min(0)
            token_indices = (safe_blocks[:, :, None] * BLOCK_SIZE + offsets).reshape(chunk_count, -1).long()
            valid_tokens = valid_blocks[:, :, None].expand(-1, -1, BLOCK_SIZE).reshape(chunk_count, -1)
            selected_key = key_group[token_indices]
            selected_value = value_group[token_indices]
            scores = torch.einsum(
                "qgd,qld->qgl", query_grouped[query_begin:query_end, group], selected_key
            ) * (head_dimension**-0.5)
            query_position = torch.arange(query_begin, query_end, device=query.device) + key.shape[0] - query_count
            score_valid = valid_tokens[:, None, :] & (token_indices[:, None, :] <= query_position[:, None, None])
            scores.masked_fill_(~score_valid, -math.inf)
            probabilities = torch.softmax(scores, dim=-1)
            group_outputs.append(torch.einsum("qgl,qld->qgd", probabilities, selected_value))
        outputs.append(torch.cat(group_outputs, dim=0))
    return torch.stack(outputs, dim=1).reshape(query_count, query_heads, head_dimension).to(torch.bfloat16)


def causal_relation(query_count: int, key_value_heads: int, key_count: int, device: torch.device) -> torch.Tensor:
    """Build a deterministic valid relation for timing when no saved q2k tensor is supplied."""
    block_count = key_count // BLOCK_SIZE
    query_positions = torch.arange(query_count, device=device) + key_count - query_count
    local_blocks = torch.div(query_positions, BLOCK_SIZE, rounding_mode="floor")
    slot = torch.arange(SELECTED_BLOCKS, device=device)
    selected = local_blocks[:, None] - slot[None, :]
    selected = selected.masked_fill(selected < 0, -1).to(torch.int32)
    return selected[None].expand(key_value_heads, -1, -1).contiguous()


def materialized_relation(
    query_hidden: torch.Tensor,
    key_value_hidden: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    *,
    key_value_heads: int,
) -> torch.Tensor:
    """Execute the natural dense score/block-max/top-k semantics outside the timed payload."""
    query_count = query_hidden.shape[0]
    key_count = key_value_hidden.shape[0]
    left = torch.matmul(query_hidden.float(), left_weight).to(torch.bfloat16).reshape(query_count, key_value_heads, 128)
    right = torch.matmul(key_value_hidden.float(), right_weight).to(torch.bfloat16)
    scores = torch.matmul(left.float(), right.float().transpose(0, 1)) * (128**-0.5)
    query_positions = torch.arange(query_count, device=scores.device) + key_count - query_count
    key_positions = torch.arange(key_count, device=scores.device)
    scores.masked_fill_(key_positions[None, None, :] > query_positions[:, None, None], -math.inf)
    block_scores = scores.reshape(query_count, key_value_heads, key_count // BLOCK_SIZE, BLOCK_SIZE).amax(-1)
    local_blocks = torch.div(query_positions, BLOCK_SIZE, rounding_mode="floor")
    block_scores.scatter_(2, local_blocks[:, None, None].expand(-1, key_value_heads, 1), math.inf)
    selected_scores, selected = torch.topk(block_scores, SELECTED_BLOCKS, dim=-1, sorted=False)
    selected = torch.where(torch.isfinite(selected_scores) | torch.isposinf(selected_scores), selected, -1)
    sentinel = torch.full_like(selected, key_count // BLOCK_SIZE)
    selected = torch.where(selected >= 0, selected, sentinel).sort(-1).values
    selected = selected.masked_fill(selected == key_count // BLOCK_SIZE, -1).to(torch.int32)
    return selected.permute(1, 0, 2).contiguous()


def elapsed_milliseconds(callable_, *, warmups: int, repeats: int) -> tuple[list[float], torch.Tensor, int]:
    for _ in range(warmups):
        output = callable_()
    torch.cuda.synchronize()
    del output
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    samples = []
    final_output = None
    peak_delta = 0
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = callable_()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        peak_delta = max(peak_delta, torch.cuda.max_memory_allocated() - baseline)
        final_output = output
        del output
    assert final_output is not None
    return samples, final_output, peak_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-count", type=int, default=16384)
    parser.add_argument("--key-count", type=int, default=16384)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--q2k")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(20260808)
    # Preserve the exact random-stream position of the pinned natural fixture.
    query_hidden = torch.randn(args.query_count, 128, device=device, dtype=torch.bfloat16)
    key_value_hidden = torch.randn(args.key_count, 128, device=device, dtype=torch.bfloat16)
    left_weight = torch.randn(128, args.key_value_heads * 128, device=device, dtype=torch.float32)
    right_weight = torch.randn(128, 128, device=device, dtype=torch.float32)
    query = torch.randn(args.query_count, args.query_heads, HEAD_DIMENSION, device=device, dtype=torch.bfloat16)
    key = torch.randn(args.key_count, args.key_value_heads, HEAD_DIMENSION, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    q2k = torch.load(args.q2k, map_location=device, weights_only=True) if args.q2k else materialized_relation(
        query_hidden,
        key_value_hidden,
        left_weight,
        right_weight,
        key_value_heads=args.key_value_heads,
    )
    del query_hidden, key_value_hidden, left_weight, right_weight

    samples, output, peak_delta = elapsed_milliseconds(
        lambda: payload_reference(q2k, query, key, value, query_chunk_size=args.query_chunk_size),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    repeated = payload_reference(q2k, query, key, value, query_chunk_size=args.query_chunk_size)
    torch.cuda.synchronize()
    shape = {
        "query_count": args.query_count,
        "key_count": args.key_count,
        "query_heads": args.query_heads,
        "key_value_heads": args.key_value_heads,
        "head_dimension": HEAD_DIMENSION,
        "block_size": BLOCK_SIZE,
        "selected_blocks": SELECTED_BLOCKS,
        "causal": True,
    }
    record = {
        "kind": "naive_eager_selected_attention",
        "shape": shape,
        "boundary": {
            "included": [
                "selected K/V gather",
                "FP32 QK Contract",
                "causal token mask",
                "materialized FP32 softmax",
                "FP32 PV Contract",
                "BF16 output cast",
            ],
            "excluded": ["route construction", "index projections", "top-k selection"],
            "query_chunk_size": args.query_chunk_size,
        },
        "relation": {
            "source": "saved exact q2k" if args.q2k else "materialized natural score/block-max/top-k relation",
            "sha256": tensor_sha256(q2k),
        },
        "timing": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "raw_milliseconds": samples,
            "median_milliseconds": statistics.median(samples),
            "mean_milliseconds": statistics.mean(samples),
            "minimum_milliseconds": min(samples),
            "maximum_milliseconds": max(samples),
        },
        "memory": {
            "peak_allocated_delta_bytes": peak_delta,
            "peak_allocated_delta_gib": peak_delta / 2**30,
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "correctness": {
            "finite": bool(torch.isfinite(output).all()),
            "repeat_bitwise": bool(torch.equal(output, repeated)),
            "output_sha256": tensor_sha256(output),
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
