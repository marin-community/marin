#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark direct-generated MSA routing and payload against the pinned oracle.

The acceptance boundary begins from ordinary BF16 hidden states and FP32 index
projection weights. Both sides independently execute the identical generic
Contracts and BF16 casts before their respective selection and payload paths.
Main QKV and output projections remain outside the comparison.
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

import numpy as np
import torch

from tile_lifetime import DType, StreamingTileSchedule, build_attention_tensor_program
from tile_lifetime.relation import build_relation_plan
from tile_lifetime.routed_attention import IndexDomainRestriction, ProjectedBlockSelectionProgram
from tile_lifetime.sm100_routed_lowering import (
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
)
from tile_lifetime.sm100_selection_lowering import (
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
)
from tile_lifetime.streaming_attention import apply_causal_score_mask, derive_streaming_attention, scaled_score_map

BACKEND = Path(__file__).parents[1] / "backends" / "sm100"
ORACLE_BACKEND = Path(__file__).parent / "backends"
for module_root in (BACKEND, ORACLE_BACKEND):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import backend from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _sha256(value: torch.Tensor) -> str:
    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _measure_one(operation: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _counterbalanced_samples(
    generated: Callable[[], Any],
    oracle: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, list[float]], list[tuple[str, str]]]:
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be a positive even number")
    operations = {"generated_shuttle": generated, "matched_msa_oracle": oracle}
    labels = tuple(operations)
    for index in range(warmups):
        order = labels if index % 2 == 0 else tuple(reversed(labels))
        for label in order:
            operations[label]()
    torch.cuda.synchronize()
    samples = {label: [] for label in labels}
    orders = []
    for index in range(repeats):
        order = labels if index % 2 == 0 else tuple(reversed(labels))
        orders.append(order)
        for label in order:
            samples[label].append(_measure_one(operations[label]))
    return samples, orders


def _sample_record(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples,
        "minimum_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "maximum_ms": max(samples),
    }


def _boundary_record(
    generated: Callable[[], Any],
    oracle: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    included: tuple[str, ...],
    excluded: tuple[str, ...],
) -> dict[str, Any]:
    samples, orders = _counterbalanced_samples(
        generated,
        oracle,
        warmups=warmups,
        repeats=repeats,
    )
    generated_record = _sample_record(samples["generated_shuttle"])
    oracle_record = _sample_record(samples["matched_msa_oracle"])
    return {
        "generated_shuttle": generated_record,
        "matched_msa_oracle": oracle_record,
        "generated_to_oracle_ratio": generated_record["median_ms"] / oracle_record["median_ms"],
        "sample_launch_orders": orders,
        "sample_pairing": "samples at the same index form one Shuttle/MSA pair",
        "included": included,
        "excluded": excluded,
    }


def _generated_boundary_record(
    operation: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    included: tuple[str, ...],
    excluded: tuple[str, ...],
) -> dict[str, Any]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = [_measure_one(operation) for _ in range(repeats)]
    return {
        "generated_shuttle": _sample_record(samples),
        "matched_msa_oracle": None,
        "generated_to_oracle_ratio": None,
        "sample_launch_orders": [("generated_shuttle",)] * repeats,
        "sample_pairing": "generated-only run; compare the isolated oracle artifact separately",
        "included": included,
        "excluded": excluded,
    }


def _project_indices(
    query_hidden: torch.Tensor,
    key_value_hidden: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    *,
    group_count: int,
    feature_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the natural FP32-accumulating Contracts and exported BF16 casts."""
    left = torch.matmul(query_hidden.float(), left_weight).to(torch.bfloat16)
    right = torch.matmul(key_value_hidden.float(), right_weight).to(torch.bfloat16)
    return left.reshape(query_hidden.shape[0], group_count, feature_count), right


def _canonical_q2k(selection: torch.Tensor) -> torch.Tensor:
    """Convert ``[query,group,slot]`` Selection output to MSA's q2k layout."""
    return selection.permute(1, 0, 2).contiguous()


def _selection_reference(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    scale: float,
    offset: int,
    block_size: int,
    top_k: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    query_count, group_count, _ = left.shape
    right_count = right.shape[0]
    score = torch.matmul(left.float(), right.float().transpose(0, 1)) * scale
    query_position = torch.arange(query_count, device=left.device, dtype=torch.int64) + offset
    key_position = torch.arange(right_count, device=left.device, dtype=torch.int64)
    score.masked_fill_(key_position[None, None, :] > query_position[:, None, None], -math.inf)
    block_score = score.reshape(query_count, group_count, right_count // block_size, block_size).amax(-1)
    local = torch.div(query_position, block_size, rounding_mode="floor")
    block_score.scatter_(2, local[:, None, None].expand(-1, group_count, 1), math.inf)
    ranked_score, _ = torch.topk(block_score, min(top_k + 1, block_score.shape[-1]), dim=-1, sorted=True)
    finite_margin = torch.isfinite(ranked_score[..., top_k - 1]) & torch.isfinite(ranked_score[..., top_k])
    margins = ranked_score[..., top_k - 1] - ranked_score[..., top_k]
    selected_score, selected = torch.topk(block_score, top_k, dim=-1, sorted=False)
    selected = torch.where(torch.isfinite(selected_score) | torch.isposinf(selected_score), selected, -1)
    sentinel = torch.full_like(selected, right_count // block_size)
    canonical = torch.where(selected >= 0, selected, sentinel).sort(-1).values
    result = _canonical_q2k(canonical.masked_fill(canonical == right_count // block_size, -1).to(torch.int32))
    finite_margins = margins[finite_margin]
    margin_record: dict[str, float | int] = {
        "finite_cutoff_count": int(finite_margins.numel()),
        "nonfinite_cutoff_count": int(margins.numel() - finite_margins.numel()),
        "minimum_finite_cutoff_margin": float(finite_margins.min().item()) if finite_margins.numel() else math.inf,
        "mean_finite_cutoff_margin": float(finite_margins.mean().item()) if finite_margins.numel() else math.inf,
    }
    return result, margin_record


def _payload_reference(q2k: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    query_count, query_heads, head_dim = query.shape
    key_value_heads = key.shape[1]
    heads_per_group = query_heads // key_value_heads
    block_size = 128
    query_chunk_size = 256
    offsets = torch.arange(block_size, device=query.device)
    query_grouped = query.float().reshape(query_count, key_value_heads, heads_per_group, head_dim)
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
            token_indices = (safe_blocks[:, :, None] * block_size + offsets).reshape(chunk_count, -1).long()
            valid_tokens = valid_blocks[:, :, None].expand(-1, -1, block_size).reshape(chunk_count, -1)
            selected_key = key_group[token_indices]
            selected_value = value_group[token_indices]
            scores = torch.einsum("qgd,qld->qgl", query_grouped[query_begin:query_end, group], selected_key) * (
                head_dim**-0.5
            )
            query_position = torch.arange(query_begin, query_end, device=query.device) + key.shape[0] - query_count
            score_valid = valid_tokens[:, None, :] & (token_indices[:, None, :] <= query_position[:, None, None])
            scores.masked_fill_(~score_valid, -math.inf)
            probabilities = torch.softmax(scores, dim=-1)
            group_outputs.append(torch.einsum("qgl,qld->qgd", probabilities, selected_value))
        outputs.append(torch.cat(group_outputs, dim=0))
    return torch.stack(outputs, dim=1).reshape(query_count, query_heads, head_dim).to(torch.bfloat16)


def _payload_lowering(
    *,
    query_count: int,
    right_count: int,
    query_heads: int,
    key_value_heads: int,
    selected_count: int,
):
    tensor_program = build_attention_tensor_program(
        batch_size=1,
        query_length=query_count,
        key_length=right_count,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        key_dimension=128,
        value_dimension=128,
        score_map=apply_causal_score_mask(scaled_score_map(128**-0.5)),
        input_dtype=DType.BF16,
    )
    streaming_program = derive_streaming_attention(
        tensor_program,
        schedule=StreamingTileSchedule(query_tile_size=128, key_value_tile_size=128, pipeline_depth=2),
    )
    seed_destinations = np.tile(
        np.arange(selected_count, dtype=np.int32),
        (query_count, key_value_heads),
    ).reshape(query_count, key_value_heads * selected_count)
    relation = build_relation_plan(
        seed_destinations,
        np.ones(seed_destinations.shape, dtype=np.float32),
        destination_rank_by_item=np.zeros(right_count // 128, dtype=np.int32),
        destination_local_item_by_item=np.arange(right_count // 128, dtype=np.int32),
        padding_quantum=1,
    )
    return lower_sm100_routed_streaming_program(
        streaming_program,
        relation,
        default_sm100_routed_schedules()[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--query-count", type=int, default=16384)
    parser.add_argument("--right-count", type=int, default=16384)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument(
        "--numerical-policy",
        choices=("source_ordered", "real_algebra_equivalent"),
        default="real_algebra_equivalent",
    )
    parser.add_argument("--maximum-absolute-error", type=float, default=0.01)
    parser.add_argument("--mean-absolute-error", type=float, default=0.001)
    parser.add_argument(
        "--execution-mode",
        choices=("paired", "generated_only"),
        default="generated_only",
    )
    parser.add_argument("--skip-payload", action="store_true")
    args = parser.parse_args()

    backend = _load_module("shuttle_fused_projected_selection", BACKEND / "fused_projected_selection.py")
    payload_backend = _load_module("shuttle_clean_routed_streaming", BACKEND / "clean_routed_streaming_runtime.py")
    oracle_backend = None
    selection_oracle = None
    if args.execution_mode == "paired":
        oracle_backend = _load_module(
            "shuttle_minimax_msa_oracle",
            ORACLE_BACKEND / "sm100_minimax_msa_oracle.py",
        )
        selection_oracle = _load_module(
            "shuttle_projected_selection_oracle",
            ORACLE_BACKEND / "sm100_projected_selection_oracle.py",
        )
    torch.manual_seed(20260808)
    query_count = args.query_count
    right_count = args.right_count
    query_heads = args.query_heads
    group_count = args.key_value_heads
    hidden_count = 128
    feature_count = 128
    block_size = 128
    top_k = 16
    offset = right_count - query_count
    scale = feature_count**-0.5
    device = torch.device("cuda")
    if query_count <= 0 or right_count < query_count:
        raise ValueError("bottom-right prefill requires 0 < query-count <= right-count")
    if query_heads <= 0 or group_count <= 0 or query_heads % group_count:
        raise ValueError("query heads must be a positive multiple of key-value heads")

    query_hidden = torch.randn(query_count, hidden_count, device=device, dtype=torch.bfloat16)
    key_value_hidden = torch.randn(right_count, hidden_count, device=device, dtype=torch.bfloat16)
    left_weight = torch.randn(hidden_count, group_count * feature_count, device=device, dtype=torch.float32)
    right_weight = torch.randn(hidden_count, feature_count, device=device, dtype=torch.float32)
    query = torch.randn(query_count, query_heads, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(right_count, group_count, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)

    selection_program = ProjectedBlockSelectionProgram(
        source_input="query_hidden",
        left_weight_input="left_index_weight",
        right_weight_input="right_index_weight",
        source_count=query_count,
        source_feature_count=hidden_count,
        group_count=group_count,
        relation_feature_count=feature_count,
        right_block_size=block_size,
        selected_count=top_k,
        score_scale=scale,
        token_restriction=IndexDomainRestriction(
            left_axis="query_position",
            right_axis="key_position",
            predicate="left_greater_equal_right",
        ),
        force_local_block=True,
        right_source_input="key_value_hidden",
        right_source_feature_count=hidden_count,
        right_count=right_count,
        left_position_offset=offset,
    )
    selection_lowering = lower_sm100_projected_selection(
        selection_program,
        default_sm100_selection_schedules()[1],
    )
    generated_selection = backend.prepare_fused_projected_selection(
        selection_lowering,
        msa_root=args.msa_root,
        device=device,
    )
    if generated_selection.generated_sources is None:
        raise RuntimeError("accepted generated selection did not retain its generated-source audit")
    direct_sources = generated_selection.generated_sources
    oracle_selection = None
    if selection_oracle is not None:
        oracle_selection = selection_oracle.prepare_public_projected_selection_oracle(
            selection_lowering,
            msa_root=args.msa_root,
            device=device,
        )

    def project() -> tuple[torch.Tensor, torch.Tensor]:
        return _project_indices(
            query_hidden,
            key_value_hidden,
            left_weight,
            right_weight,
            group_count=group_count,
            feature_count=feature_count,
        )

    projected_left, projected_right = project()

    def generated_selection_core() -> torch.Tensor:
        return _canonical_q2k(generated_selection(projected_left, projected_right))

    def oracle_selection_core() -> torch.Tensor:
        if oracle_selection is None:
            raise RuntimeError("the private oracle is disabled in generated-only mode")
        return _canonical_q2k(oracle_selection(projected_left, projected_right))

    def generated_natural_selection() -> torch.Tensor:
        left, right = project()
        return _canonical_q2k(generated_selection(left, right))

    def oracle_natural_selection() -> torch.Tensor:
        if oracle_selection is None:
            raise RuntimeError("the private oracle is disabled in generated-only mode")
        left, right = project()
        return _canonical_q2k(oracle_selection(left, right))

    generated_route = generated_natural_selection().clone()
    # Localize asynchronous launch failures to the generated boundary before
    # invoking the independent oracle runner.
    torch.cuda.synchronize()
    oracle_route = oracle_natural_selection().clone() if oracle_selection is not None else None
    if oracle_route is not None:
        torch.cuda.synchronize()
    reference_route, cutoff_margins = _selection_reference(
        projected_left,
        projected_right,
        scale=scale,
        offset=offset,
        block_size=block_size,
        top_k=top_k,
    )
    generated_reference_mismatch = generated_route != reference_route
    oracle_reference_mismatch = oracle_route != reference_route if oracle_route is not None else None
    if args.numerical_policy == "source_ordered" and (
        bool(generated_reference_mismatch.any())
        or (oracle_reference_mismatch is not None and bool(oracle_reference_mismatch.any()))
    ):
        raise ValueError("source_ordered selection must exactly match the materialized semantic reference")

    selection_core_boundary = {
        "warmups": args.warmups,
        "repeats": args.repeats,
        "included": ("score Contract", "block maximum Fold", "top-k Selection"),
        "excluded": ("index Q Contract", "index K Contract", "sparse payload"),
    }
    natural_selection_boundary = {
        "warmups": args.warmups,
        "repeats": args.repeats,
        "included": (
            "BF16 hidden inputs",
            "FP32-accumulating index Q/K Contracts",
            "BF16 projection casts",
            "score Contract",
            "block maximum Fold",
            "top-k Selection",
        ),
        "excluded": ("main QKV projection", "sparse payload", "output projection"),
    }
    if oracle_selection is None:
        boundaries: dict[str, Any] = {
            "projected_selection_core": _generated_boundary_record(
                generated_selection_core,
                **selection_core_boundary,
            ),
            "natural_projection_and_selection": _generated_boundary_record(
                generated_natural_selection,
                **natural_selection_boundary,
            ),
        }
    else:
        boundaries = {
            "projected_selection_core": _boundary_record(
                generated_selection_core,
                oracle_selection_core,
                **selection_core_boundary,
            ),
            "natural_projection_and_selection": _boundary_record(
                generated_natural_selection,
                oracle_natural_selection,
                **natural_selection_boundary,
            ),
        }

    payload_correctness: dict[str, Any] | None = None
    oracle_manifest: dict[str, Any] | None = None
    payload_source_record: dict[str, Any] | None = None
    if not args.skip_payload:
        payload_lowering = _payload_lowering(
            query_count=query_count,
            right_count=right_count,
            query_heads=query_heads,
            key_value_heads=group_count,
            selected_count=top_k,
        )
        generated_payload = payload_backend.compile_routed_streaming_callable(
            args.msa_root,
            payload_lowering,
            partial_value_dtype=payload_backend.PartialValueDType.BF16,
            partial_merge_schedule=payload_backend.PartialMergeScheduleKind.WARP_ROWS,
            build_directory=args.build_directory,
        )
        payload_sources = generated_payload.generated_sources
        payload_source_record = {
            "physical_audit_clean": payload_sources.physical_audit.clean,
            "semantic_audit_clean": payload_sources.semantic_audit.clean,
            "relation_builder_audit_clean": payload_sources.relation_builder_audit.clean,
            "scheduler_audit_clean": payload_sources.scheduler_audit.clean,
            "generated_source_sha256": payload_sources.generated_source_sha256,
            "external_semantic_kernels": list(payload_sources.emitter_plan.external_semantic_kernels),
            "partial_value_dtype": payload_sources.emitter_plan.partial_value_dtype.value,
            "partial_merge_schedule": payload_sources.emitter_plan.partial_merge.schedule_kind.value,
        }
        oracle_payload = None
        if oracle_backend is not None:
            key_pages = (
                key.reshape(right_count // block_size, block_size, group_count, 128).permute(0, 2, 1, 3).contiguous()
            )
            value_pages = (
                value.reshape(right_count // block_size, block_size, group_count, 128).permute(0, 2, 1, 3).contiguous()
            )
            workload = oracle_backend.MsaOracleWorkload(
                query=query,
                key_pages=key_pages,
                value_pages=value_pages,
                query_segment_lengths=torch.tensor([query_count], dtype=torch.int32),
                key_value_segment_lengths=torch.tensor([right_count], dtype=torch.int32),
                key_value_page_indices=torch.arange(right_count // block_size, dtype=torch.int32, device=device),
                query_offsets=torch.tensor([offset], dtype=torch.int32),
                block_size=block_size,
                top_k=top_k,
                softmax_scale=128**-0.5,
                partial_dtype=torch.bfloat16,
            )
            oracle_payload, oracle_manifest = oracle_backend.build_minimax_msa_payload(args.msa_root, workload)

        def generated_full() -> torch.Tensor:
            left, right = project()
            route = _canonical_q2k(generated_selection(left, right))
            return generated_payload(route, query, key, value).output

        def oracle_full() -> torch.Tensor:
            if oracle_selection is None or oracle_payload is None:
                raise RuntimeError("the private oracle is disabled in generated-only mode")
            left, right = project()
            route = _canonical_q2k(oracle_selection(left, right))
            return oracle_payload(route)

        generated_output = generated_full().clone()
        oracle_output = oracle_full().clone() if oracle_payload is not None else None
        reference_output = _payload_reference(reference_route, query, key, value)
        torch.cuda.synchronize()
        generated_difference = (generated_output.float() - reference_output.float()).abs()
        oracle_difference = (
            (oracle_output.float() - reference_output.float()).abs() if oracle_output is not None else None
        )
        payload_correctness = {
            "generated_maximum_absolute_error": float(generated_difference.max().item()),
            "generated_mean_absolute_error": float(generated_difference.mean().item()),
            "oracle_maximum_absolute_error": (
                float(oracle_difference.max().item()) if oracle_difference is not None else None
            ),
            "oracle_mean_absolute_error": (
                float(oracle_difference.mean().item()) if oracle_difference is not None else None
            ),
            "generated_sha256": _sha256(generated_output),
            "oracle_sha256": _sha256(oracle_output) if oracle_output is not None else None,
            "reference_sha256": _sha256(reference_output),
            "generated_repeat_bitwise": bool(torch.equal(generated_output, generated_full())),
            "oracle_repeat_bitwise": (
                bool(torch.equal(oracle_output, oracle_full())) if oracle_output is not None else None
            ),
            "declared_tolerance": {
                "maximum_absolute_error": args.maximum_absolute_error,
                "mean_absolute_error": args.mean_absolute_error,
            },
            "generated_within_tolerance": bool(
                generated_difference.max().item() <= args.maximum_absolute_error
                and generated_difference.mean().item() <= args.mean_absolute_error
            ),
            "oracle_within_tolerance": (
                bool(
                    oracle_difference.max().item() <= args.maximum_absolute_error
                    and oracle_difference.mean().item() <= args.mean_absolute_error
                )
                if oracle_difference is not None
                else None
            ),
        }
        full_boundary = {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "included": (
                "BF16 hidden inputs",
                "FP32-accumulating index Q/K Contracts",
                "BF16 projection casts",
                "selection",
                "relation scheduling",
                "sparse QK/normalized-exp/PV",
                "deterministic partial-state merge",
            ),
            "excluded": ("main QKV projection", "output projection"),
        }
        if oracle_payload is None:
            boundaries["natural_projection_selection_and_payload"] = _generated_boundary_record(
                generated_full,
                **full_boundary,
            )
        else:
            boundaries["natural_projection_selection_and_payload"] = _boundary_record(
                generated_full,
                oracle_full,
                **full_boundary,
            )

    result = {
        "kind": "direct_generated_msa_natural_route_sm100",
        "acceptance_boundary": "natural_projection_selection_and_payload",
        "source_revision": backend.MINIMAX_MSA_COMMIT,
        "shape": {
            "query_hidden": list(query_hidden.shape),
            "key_value_hidden": list(key_value_hidden.shape),
            "left_index_weight": list(left_weight.shape),
            "right_index_weight": list(right_weight.shape),
            "query": list(query.shape),
            "key_value": list(key.shape),
            "block_size": block_size,
            "top_k": top_k,
            "query_position_offset": offset,
        },
        "physical_source": {
            "generated": generated_selection.physical_source_classification,
            "oracle": oracle_selection.physical_source_classification if oracle_selection is not None else None,
            "direct_source_clean": bool(direct_sources.clean),
            "direct_source_sha256": direct_sources.source_sha256,
            "external_semantic_kernels": list(generated_selection.adapter_plan.external_semantic_kernels),
            "accepted_runtime_audit_clean": backend.audit_fused_projected_selection_source().clean,
            "oracle_adapter_classification": (
                "oracle_derived_private_variant_manager_contaminated" if oracle_selection is not None else None
            ),
            "payload": payload_source_record,
        },
        "selection_correctness": {
            "generated_matches_reference": bool(torch.equal(generated_route, reference_route)),
            "oracle_matches_reference": (
                bool(torch.equal(oracle_route, reference_route)) if oracle_route is not None else None
            ),
            "generated_repeat_bitwise": bool(torch.equal(generated_route, generated_natural_selection())),
            "oracle_repeat_bitwise": (
                bool(torch.equal(oracle_route, oracle_natural_selection())) if oracle_route is not None else None
            ),
            "route_sha256": _sha256(reference_route),
            "numerical_policy": args.numerical_policy,
            "generated_reference_mismatch_count": int(generated_reference_mismatch.sum().item()),
            "oracle_reference_mismatch_count": (
                int(oracle_reference_mismatch.sum().item()) if oracle_reference_mismatch is not None else None
            ),
            "generated_oracle_mismatch_count": (
                int((generated_route != oracle_route).sum().item()) if oracle_route is not None else None
            ),
            "cutoff_margins": cutoff_margins,
        },
        "payload_correctness": payload_correctness,
        "oracle_manifest": oracle_manifest,
        "boundaries": boundaries,
        "target": {
            "maximum_generated_to_oracle_ratio": 1.2,
            "paired_oracle_derived": args.execution_mode == "paired",
            "natural_route_absolute_target_ms": (
                1.2 * boundaries["natural_projection_selection_and_payload"]["matched_msa_oracle"]["median_ms"]
                if "natural_projection_selection_and_payload" in boundaries
                and boundaries["natural_projection_selection_and_payload"]["matched_msa_oracle"] is not None
                else None
            ),
        },
        "execution_mode": args.execution_mode,
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "raw_samples_preserved": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
