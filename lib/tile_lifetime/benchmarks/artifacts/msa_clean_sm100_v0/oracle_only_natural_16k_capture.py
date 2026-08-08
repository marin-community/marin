#!/usr/bin/env python3
"""Isolated MiniMax MSA natural-route oracle capture for the 16K Shuttle case.

This process loads only the pinned public MSA score/top-k path and the pinned
MSA sparse-attention payload adapter. It deliberately does not import any
generated Shuttle selector or payload backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


SEED = 20260808
BLOCK_SIZE = 128
TOP_K = 16
HEAD_DIMENSION = 128


def _load_module(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _sha256(value: torch.Tensor) -> str:
    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _measure(operation: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _samples(operation: Callable[[], Any], *, warmups: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    values = [_measure(operation) for _ in range(repeats)]
    return {
        "samples_ms": values,
        "minimum_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "maximum_ms": max(values),
    }


def _project_indices(
    query_hidden: torch.Tensor,
    key_value_hidden: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    *,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.matmul(query_hidden.float(), left_weight).to(torch.bfloat16)
    right = torch.matmul(key_value_hidden.float(), right_weight).to(torch.bfloat16)
    return left.reshape(query_hidden.shape[0], group_count, HEAD_DIMENSION), right


def _canonical_q2k(selection: torch.Tensor) -> torch.Tensor:
    return selection.permute(1, 0, 2).contiguous()


def _selection_reference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_count, group_count, _ = left.shape
    right_count = right.shape[0]
    score = torch.matmul(left.float(), right.float().transpose(0, 1)) * (HEAD_DIMENSION**-0.5)
    query_position = torch.arange(query_count, device=left.device, dtype=torch.int64) + offset
    key_position = torch.arange(right_count, device=left.device, dtype=torch.int64)
    score.masked_fill_(key_position[None, None, :] > query_position[:, None, None], -math.inf)
    block_score = score.reshape(query_count, group_count, right_count // BLOCK_SIZE, BLOCK_SIZE).amax(-1)
    local = torch.div(query_position, BLOCK_SIZE, rounding_mode="floor")
    block_score.scatter_(2, local[:, None, None].expand(-1, group_count, 1), math.inf)
    ranked_score = torch.topk(block_score, min(TOP_K + 1, block_score.shape[-1]), dim=-1, sorted=True).values
    margins = ranked_score[..., TOP_K - 1] - ranked_score[..., TOP_K]
    selected_score, selected = torch.topk(block_score, TOP_K, dim=-1, sorted=False)
    selected = torch.where(torch.isfinite(selected_score) | torch.isposinf(selected_score), selected, -1)
    sentinel = torch.full_like(selected, right_count // BLOCK_SIZE)
    canonical = torch.where(selected >= 0, selected, sentinel).sort(-1).values
    q2k = _canonical_q2k(canonical.masked_fill(canonical == right_count // BLOCK_SIZE, -1).to(torch.int32))
    return q2k, margins


def _margin_record(margins: torch.Tensor, mismatch_rows: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(margins)
    finite_all = margins[finite]
    finite_mismatch = margins[mismatch_rows & finite]

    def quantiles(values: torch.Tensor) -> dict[str, float | int | None]:
        if not values.numel():
            return {"count": 0, "minimum": None, "p01": None, "p10": None, "median": None, "mean": None, "maximum": None}
        values = values.float()
        q = torch.quantile(values, torch.tensor([0.01, 0.10, 0.50], device=values.device))
        return {
            "count": int(values.numel()),
            "minimum": float(values.min().item()),
            "p01": float(q[0].item()),
            "p10": float(q[1].item()),
            "median": float(q[2].item()),
            "mean": float(values.mean().item()),
            "maximum": float(values.max().item()),
        }

    return {
        "all_finite_rows": quantiles(finite_all),
        "mismatch_finite_rows": quantiles(finite_mismatch),
        "mismatch_row_count": int(mismatch_rows.sum().item()),
        "nonfinite_margin_count": int((~finite).sum().item()),
    }


def _payload_reference(
    q2k: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
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
        for query_begin in range(0, query_count, 256):
            query_end = min(query_begin + 256, query_count)
            chunk_count = query_end - query_begin
            valid_blocks = q2k[group, query_begin:query_end] >= 0
            safe_blocks = q2k[group, query_begin:query_end].clamp_min(0)
            token_indices = (safe_blocks[:, :, None] * BLOCK_SIZE + offsets).reshape(chunk_count, -1).long()
            valid_tokens = valid_blocks[:, :, None].expand(-1, -1, BLOCK_SIZE).reshape(chunk_count, -1)
            selected_key = key_group[token_indices]
            selected_value = value_group[token_indices]
            scores = torch.einsum(
                "qgd,qld->qgl",
                query_grouped[query_begin:query_end, group],
                selected_key,
            ) * (head_dimension**-0.5)
            query_position = torch.arange(query_begin, query_end, device=query.device) + key.shape[0] - query_count
            score_valid = valid_tokens[:, None, :] & (token_indices[:, None, :] <= query_position[:, None, None])
            scores.masked_fill_(~score_valid, -math.inf)
            probabilities = torch.softmax(scores, dim=-1)
            group_outputs.append(torch.einsum("qgl,qld->qgd", probabilities, selected_value))
        outputs.append(torch.cat(group_outputs, dim=0))
    return torch.stack(outputs, dim=1).reshape(query_count, query_heads, head_dimension).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa-root", type=Path, required=True)
    parser.add_argument("--oracle-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--query-count", type=int, default=16_384)
    parser.add_argument("--right-count", type=int, default=16_384)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--key-value-heads", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.msa_root / "python"))
    from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select

    oracle_adapter = _load_module("isolated_sm100_minimax_msa_oracle", args.oracle_adapter)
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    offset = args.right_count - args.query_count
    pages = args.right_count // BLOCK_SIZE

    query_hidden = torch.randn(args.query_count, HEAD_DIMENSION, device=device, dtype=torch.bfloat16)
    key_value_hidden = torch.randn(args.right_count, HEAD_DIMENSION, device=device, dtype=torch.bfloat16)
    left_weight = torch.randn(
        HEAD_DIMENSION,
        args.key_value_heads * HEAD_DIMENSION,
        device=device,
        dtype=torch.float32,
    )
    right_weight = torch.randn(HEAD_DIMENSION, HEAD_DIMENSION, device=device, dtype=torch.float32)
    query = torch.randn(
        args.query_count,
        args.query_heads,
        HEAD_DIMENSION,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        args.right_count,
        args.key_value_heads,
        HEAD_DIMENSION,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    query_lengths = torch.tensor([args.query_count], dtype=torch.int32)
    key_lengths = torch.tensor([args.right_count], dtype=torch.int32)
    query_offset = torch.tensor([offset], dtype=torch.int32)
    score_plan = fmha_sm100_plan(
        query_lengths,
        key_lengths,
        args.key_value_heads,
        causal=True,
        qo_offset=query_offset,
        page_size=BLOCK_SIZE,
        output_maxscore=True,
        num_kv_heads=1,
    )
    key_value_indices = torch.arange(pages, dtype=torch.int32, device=device)
    query_position = torch.arange(args.query_count, dtype=torch.int64, device=device) + offset
    local_block = torch.div(query_position, BLOCK_SIZE, rounding_mode="floor")
    group_index = torch.arange(args.key_value_heads, dtype=torch.int64, device=device)[:, None]
    query_index = torch.arange(args.query_count, dtype=torch.int64, device=device)[None, :]

    projected_left, projected_right = _project_indices(
        query_hidden,
        key_value_hidden,
        left_weight,
        right_weight,
        group_count=args.key_value_heads,
    )

    def selection_core(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        key_pages = right.contiguous().reshape(pages, 1, BLOCK_SIZE, HEAD_DIMENSION)
        _, scores = fmha_sm100(
            left.contiguous(),
            key_pages,
            key_pages,
            score_plan,
            sm_scale=HEAD_DIMENSION**-0.5,
            kv_indices=key_value_indices,
            output_o=False,
            output_maxscore=True,
        )
        if scores is None:
            raise RuntimeError("MSA score oracle returned no block scores")
        scores[group_index, local_block[None, :], query_index] = math.inf
        return _canonical_q2k(
            sparse_topk_select(scores, TOP_K, num_valid_pages=pages)
        )

    def project() -> tuple[torch.Tensor, torch.Tensor]:
        return _project_indices(
            query_hidden,
            key_value_hidden,
            left_weight,
            right_weight,
            group_count=args.key_value_heads,
        )

    def natural_selection() -> torch.Tensor:
        left, right = project()
        return selection_core(left, right)

    oracle_route = natural_selection().clone()
    torch.cuda.synchronize()
    reference_route, margins = _selection_reference(projected_left, projected_right, offset=offset)
    mismatch = oracle_route != reference_route
    mismatch_rows = mismatch.any(dim=-1).permute(1, 0).contiguous()

    key_pages = key.reshape(pages, BLOCK_SIZE, args.key_value_heads, HEAD_DIMENSION).permute(0, 2, 1, 3).contiguous()
    value_pages = value.reshape(pages, BLOCK_SIZE, args.key_value_heads, HEAD_DIMENSION).permute(0, 2, 1, 3).contiguous()
    workload = oracle_adapter.MsaOracleWorkload(
        query=query,
        key_pages=key_pages,
        value_pages=value_pages,
        query_segment_lengths=query_lengths,
        key_value_segment_lengths=key_lengths,
        key_value_page_indices=key_value_indices,
        query_offsets=query_offset,
        block_size=BLOCK_SIZE,
        top_k=TOP_K,
        softmax_scale=HEAD_DIMENSION**-0.5,
        partial_dtype=torch.bfloat16,
    )
    payload, manifest = oracle_adapter.build_minimax_msa_payload(args.msa_root, workload)

    def full() -> torch.Tensor:
        return payload(natural_selection())

    oracle_output = full().clone()
    torch.cuda.synchronize()
    reference_output = _payload_reference(reference_route, query, key, value)
    difference = (oracle_output.float() - reference_output.float()).abs()
    repeated_route = natural_selection()
    repeated_output = full()
    torch.cuda.synchronize()

    result = {
        "kind": "isolated_minimax_msa_natural_route_oracle_sm100",
        "source_revision": manifest["source"],
        "shape": {
            "query_hidden": list(query_hidden.shape),
            "key_value_hidden": list(key_value_hidden.shape),
            "left_index_weight": list(left_weight.shape),
            "right_index_weight": list(right_weight.shape),
            "query": list(query.shape),
            "key_value": list(key.shape),
            "block_size": BLOCK_SIZE,
            "top_k": TOP_K,
            "query_position_offset": offset,
        },
        "boundary": {
            "included": [
                "BF16 hidden inputs",
                "FP32-accumulating index Q/K Contracts",
                "BF16 projection casts",
                "official MSA score Contract/block-max Fold/top-k Selection",
                "q2k-to-k2q relation scheduling",
                "official sparse QK/normalized-exp/PV/combine payload",
            ],
            "excluded": ["main QKV projection", "output projection"],
        },
        "timings": {
            "projected_selection_core": _samples(
                lambda: selection_core(projected_left, projected_right),
                warmups=args.warmups,
                repeats=args.repeats,
            ),
            "natural_projection_and_selection": _samples(
                natural_selection,
                warmups=args.warmups,
                repeats=args.repeats,
            ),
            "natural_projection_selection_and_payload": _samples(
                full,
                warmups=args.warmups,
                repeats=args.repeats,
            ),
        },
        "correctness": {
            "route_reference_mismatch_count": int(mismatch.sum().item()),
            "route_reference_mismatch_row_count": int(mismatch_rows.sum().item()),
            "cutoff_margin_distribution": _margin_record(margins, mismatch_rows),
            "route_sha256": _sha256(oracle_route),
            "reference_route_sha256": _sha256(reference_route),
            "route_repeat_bitwise": bool(torch.equal(oracle_route, repeated_route)),
            "output_maximum_absolute_error": float(difference.max().item()),
            "output_mean_absolute_error": float(difference.mean().item()),
            "output_sha256": _sha256(oracle_output),
            "reference_output_sha256": _sha256(reference_output),
            "output_repeat_bitwise": bool(torch.equal(oracle_output, repeated_output)),
        },
        "oracle_manifest": manifest,
        "process_isolation": {
            "generated_selector_loaded": False,
            "generated_payload_loaded": False,
            "complete_workload_oracle_only": True,
        },
        "measurement": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "raw_samples_preserved": True,
            "timing": "one CUDA event interval per operation",
        },
        "input_hashes": {
            "query_hidden": _sha256(query_hidden),
            "key_value_hidden": _sha256(key_value_hidden),
            "left_weight": _sha256(left_weight),
            "right_weight": _sha256(right_weight),
            "query": _sha256(query),
            "key": _sha256(key),
            "value": _sha256(value),
        },
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
