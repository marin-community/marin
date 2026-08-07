# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a RelationPlan-driven query-major block-sparse attention candidate."""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark_metadata import (
    canonical_json_sha256,
    command_output,
    command_record,
    framed_tensor_sha256,
    nvidia_smi_snapshot,
    toolchain_snapshot,
)

from tile_lifetime import (
    RoutedAttentionPlanConfig,
    build_routed_attention_relation,
    compile_routed_attention_candidates,
    make_causal_block_relation,
)

try:
    from block_sparse_attn import block_sparse_attn_func
except ImportError:
    block_sparse_attn_func = None


PINNED_BLOCK_SPARSE_ATTENTION_REVISION = "49d6c39e4dc0303442cda3bb758b3925d4399c49"


def _measure(function: Callable[[], torch.Tensor], *, warmups: int, repeats: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def _relation_planning_samples(
    *, sequence_length: int, block_size: int, selected_blocks: int, repeats: int
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    routing_samples = []
    planning_samples = []
    selected = np.empty((0, 0), dtype=np.int32)
    edge_valid = np.empty((0, 0), dtype=np.bool_)
    for _ in range(repeats):
        start = time.perf_counter()
        selected, edge_valid = make_causal_block_relation(
            sequence_length=sequence_length,
            block_size=block_size,
            selected_blocks=selected_blocks,
        )
        routing_samples.append((time.perf_counter() - start) * 1_000)
        start = time.perf_counter()
        build_routed_attention_relation(selected, edge_valid=edge_valid)
        planning_samples.append((time.perf_counter() - start) * 1_000)
    return selected, edge_valid, routing_samples, planning_samples


def _block_mask(selected: np.ndarray, edge_valid: np.ndarray, *, query_heads: int, device: torch.device) -> torch.Tensor:
    query_blocks = selected.shape[0]
    key_value_blocks = int(np.max(selected[edge_valid])) + 1
    mask = torch.zeros(
        (1, query_heads, query_blocks, key_value_blocks),
        dtype=torch.bool,
        device=device,
    )
    for query_block in range(query_blocks):
        destinations = selected[query_block, edge_valid[query_block]].tolist()
        mask[0, :, query_block, destinations] = True
    return mask


def _block_sparse_candidate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    sequence_length: int,
    query_heads: int,
    scale: float,
) -> Callable[[], torch.Tensor]:
    if block_sparse_attn_func is None:
        raise RuntimeError("block_sparse_attn is not installed")
    device = query.device
    cumulative_sequence = torch.tensor([0, sequence_length], dtype=torch.int32, device=device)
    head_mask_type = torch.ones(query_heads, dtype=torch.int32, device=device)
    streaming_info = torch.zeros(2 * query_heads, dtype=torch.int32, device=device)

    def run() -> torch.Tensor:
        return block_sparse_attn_func(
            query,
            key,
            value,
            cumulative_sequence,
            cumulative_sequence,
            head_mask_type,
            streaming_info,
            block_mask,
            sequence_length,
            sequence_length,
            0.0,
            deterministic=True,
            softmax_scale=scale,
            is_causal=True,
            exact_streaming=False,
            return_attn_probs=False,
        )

    return run


def _dense_candidate(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1)[None],
            key.transpose(0, 1)[None],
            value.transpose(0, 1)[None],
            is_causal=True,
            enable_gqa=True,
        )[0].transpose(0, 1)

    return run


def _selected_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected: np.ndarray,
    edge_valid: np.ndarray,
    *,
    block_size: int,
    scale: float,
    checked_query_blocks: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_map = torch.arange(query.shape[1], device=query.device) // (query.shape[1] // key.shape[1])
    outputs = []
    token_indices = []
    for query_block in checked_query_blocks.tolist():
        query_start = query_block * block_size
        query_stop = min(query_start + block_size, query.shape[0])
        query_positions = torch.arange(query_start, query_stop, device=query.device)
        chosen = selected[query_block, edge_valid[query_block]]
        key_indices = torch.cat(
            [
                torch.arange(
                    int(block) * block_size,
                    min((int(block) + 1) * block_size, key.shape[0]),
                    device=query.device,
                )
                for block in chosen
            ]
        )
        expanded_key = key[key_indices][:, head_map].float()
        scores = torch.einsum("qhd,khd->qhk", query[query_start:query_stop].float(), expanded_key) * scale
        scores.masked_fill_(query_positions[:, None, None] < key_indices[None, None, :], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        expanded_value = value[key_indices][:, head_map].float()
        outputs.append(torch.einsum("qhk,khv->qhv", probabilities, expanded_value))
        token_indices.append(query_positions)
    return torch.cat(token_indices), torch.cat(outputs)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    payload = (
        contiguous.view(torch.uint16).numpy().tobytes(order="C")
        if contiguous.dtype == torch.bfloat16
        else contiguous.numpy().tobytes(order="C")
    )
    return framed_tensor_sha256(str(contiguous.dtype), tuple(contiguous.shape), payload)


def _relation_sha256(selected: np.ndarray, edge_valid: np.ndarray) -> str:
    return canonical_json_sha256(
        {
            "selected": framed_tensor_sha256(str(selected.dtype), selected.shape, selected.tobytes(order="C")),
            "edge_valid": framed_tensor_sha256(str(edge_valid.dtype), edge_valid.shape, edge_valid.tobytes(order="C")),
        }
    )


def _dense_relation_sha256(selected: np.ndarray, edge_valid: np.ndarray) -> str:
    block_count = selected.shape[0]
    dense_relation = np.zeros((block_count, block_count), dtype=np.bool_)
    for query_block in range(block_count):
        dense_relation[query_block, selected[query_block, edge_valid[query_block]]] = True
    return hashlib.sha256(dense_relation.tobytes(order="C")).hexdigest()


def _source_record(root: Path | None, declared_revision: str) -> dict[str, Any]:
    record: dict[str, Any] = {"declared_revision": declared_revision}
    if root is not None:
        record.update(
            {
                "root": str(root.resolve()),
                "git_head": command_output(["git", "-C", str(root.resolve()), "rev-parse", "HEAD"]),
                "git_status": command_output(["git", "-C", str(root.resolve()), "status", "--short"]),
            }
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--selected-blocks", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--key-value-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--planning-repeats", type=int, default=20)
    parser.add_argument("--correctness-blocks", type=int, default=8)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--block-sparse-root", type=Path)
    parser.add_argument(
        "--block-sparse-revision",
        default=PINNED_BLOCK_SPARSE_ATTENTION_REVISION,
    )
    parser.add_argument("--clock-policy", default="cluster_default_unpinned")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    if args.sequence_length % args.block_size:
        raise ValueError("the first H100 benchmark requires sequence length divisible by block size")
    if args.query_heads % args.key_value_heads:
        raise ValueError("query heads must map evenly onto key/value heads")
    if block_sparse_attn_func is None:
        raise RuntimeError("install the pinned Block-Sparse-Attention checkout before running")

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    scale = 1.0 / math.sqrt(args.head_dimension)
    selected, edge_valid, routing_samples, planning_samples = _relation_planning_samples(
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        selected_blocks=args.selected_blocks,
        repeats=args.planning_repeats,
    )
    relation = build_routed_attention_relation(selected, edge_valid=edge_valid)
    query_major, kv_major = compile_routed_attention_candidates(
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
    query = torch.randn(
        args.sequence_length,
        args.query_heads,
        args.head_dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        args.sequence_length,
        args.key_value_heads,
        args.head_dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    block_mask = _block_mask(selected, edge_valid, query_heads=args.query_heads, device=device)
    candidates: list[tuple[str, Callable[[], torch.Tensor], str]] = [
        (
            "shuttle_query_major_block_sparse_attention",
            _block_sparse_candidate(
                query,
                key,
                value,
                block_mask,
                sequence_length=args.sequence_length,
                query_heads=args.query_heads,
                scale=scale,
            ),
            query_major.dump(),
        )
    ]
    if args.include_dense:
        candidates.append(("torch_dense_flash_sdpa", _dense_candidate(query, key, value), "dense causal attention"))

    block_count = selected.shape[0]
    checked_query_blocks = np.unique(
        np.linspace(0, block_count - 1, min(args.correctness_blocks, block_count), dtype=np.int32)
    )
    checked_tokens, reference = _selected_attention_reference(
        query,
        key,
        value,
        selected,
        edge_valid,
        block_size=args.block_size,
        scale=scale,
        checked_query_blocks=checked_query_blocks,
    )
    telemetry_initial = nvidia_smi_snapshot()
    candidate_results = []
    for name, function, plan_dump in candidates:
        output = function()
        torch.cuda.synchronize()
        timing = _measure(function, warmups=args.warmups, repeats=args.repeats, iterations=args.iterations)
        if name.startswith("shuttle_query_major"):
            difference = output[checked_tokens].float() - reference
            correctness = {
                "checked_query_blocks": checked_query_blocks.tolist(),
                "maximum_absolute_error": difference.abs().max().item(),
                "mean_absolute_error": difference.abs().mean().item(),
                "nan_count": int(torch.isnan(output).sum().item()),
                "infinity_count": int(torch.isinf(output).sum().item()),
            }
        else:
            correctness = {"note": "dense semantics intentionally differ from selected-block attention"}
        candidate_results.append(
            {
                "name": name,
                "orientation": "query_major" if name.startswith("shuttle_query_major") else "dense",
                "plan_dump": plan_dump,
                "timing": timing,
                "correctness": correctness,
                "output_sha256": _tensor_sha256(output),
            }
        )

    result = {
        "schema_version": 1,
        "benchmark": "shuttle_routed_sparse_attention_h100",
        "status": "ok",
        "source": {
            "shuttle_revision": args.shuttle_revision,
            "block_sparse_attention": _source_record(args.block_sparse_root, args.block_sparse_revision),
        },
        "shape": {
            "sequence_length": args.sequence_length,
            "block_size": args.block_size,
            "selected_blocks": args.selected_blocks,
            "query_heads": args.query_heads,
            "key_value_heads": args.key_value_heads,
            "head_dimension": args.head_dimension,
            "causal": True,
            "dtype": "bfloat16",
        },
        "relation": {
            "fingerprint_sha256": _relation_sha256(selected, edge_valid),
            "dense_boolean_sha256": _dense_relation_sha256(selected, edge_valid),
            "valid_edges": relation.route_count,
            "source_degrees": np.count_nonzero(edge_valid, axis=1).tolist(),
            "destination_degrees": relation.group_count.tolist(),
            "synthetic_routing_samples_ms": routing_samples,
            "synthetic_routing_median_ms": statistics.median(routing_samples),
            "planning_samples_ms": planning_samples,
            "planning_median_ms": statistics.median(planning_samples),
        },
        "candidate_set": {
            "query_major": query_major.dump(),
            "kv_major": kv_major.dump(),
            "kv_major_status": "structurally generated but not executable through the query-major oracle adapter",
        },
        "candidates": candidate_results,
        "protocol": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations": args.iterations,
            "selection_metric": "median CUDA-event milliseconds",
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "clock_policy": args.clock_policy,
            "command": command_record(),
            "toolchain": toolchain_snapshot("nvcc"),
            "gpu_telemetry": {"initial": telemetry_initial, "final": nvidia_smi_snapshot()},
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
