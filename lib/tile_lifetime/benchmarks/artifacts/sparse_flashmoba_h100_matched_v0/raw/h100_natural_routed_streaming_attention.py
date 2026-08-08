# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark natural JAX routing through generated sparse SM90 streaming."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from backends.h100.cuda_kv_reuse_emitter import compile_h100_bounded_kv_reuse_program
from backends.h100.cute_routed_streaming_emitter import compile_h100_routed_streaming_program

from tile_lifetime import (
    RoutedAttentionDebugConfig,
    RoutedAttentionOrientation,
    RoutedAttentionPlanConfig,
    StreamingTileSchedule,
    compile_stablehlo_routed_attention_program,
    export_debug_routed_attention,
)
from tile_lifetime.routed_attention_frontend import ROUTED_ATTENTION_INPUT_NAMES

try:
    from block_sparse_attn import block_sparse_attn_func
except ImportError:
    block_sparse_attn_func = None

try:
    from flash_moba import flash_moba_attn_varlen_func
except ImportError:
    flash_moba_attn_varlen_func = None


PINNED_BLOCK_SPARSE_ATTENTION_REVISION = "49d6c39e4dc0303442cda3bb758b3925d4399c49"
PINNED_FLASH_MOBA_REVISION = "39d9ac043b271d046a2181a9991e99a26b67bca1"


def _measure(operation: Callable[[], object], warmups: int, repeats: int) -> list[float]:
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


def _alternating_launch_orders(labels: tuple[str, str], rounds: int) -> list[tuple[str, str]]:
    return [labels if round_index % 2 == 0 else tuple(reversed(labels)) for round_index in range(rounds)]


def _measure_counterbalanced_pair(
    generated: Callable[[], object],
    oracle: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, list[float]], dict[str, object]]:
    """Measure a pair while alternating which implementation launches first."""
    if repeats % 2 != 0:
        raise ValueError("counterbalanced paired measurement requires an even repeat count")

    operations = {
        "generated": generated,
        "matched_expert_oracle": oracle,
    }
    labels = ("generated", "matched_expert_oracle")
    warmup_launch_orders = _alternating_launch_orders(labels, warmups)
    for order in warmup_launch_orders:
        for label in order:
            operations[label]()
    torch.cuda.synchronize()

    samples = {label: [] for label in labels}
    sample_launch_orders = _alternating_launch_orders(labels, repeats)
    for order in sample_launch_orders:
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[label]()
            end.record()
            end.synchronize()
            samples[label].append(float(start.elapsed_time(end)))

    protocol = {
        "name": "alternating_counterbalanced_pairs",
        "variants": labels,
        "warmup_launch_orders": warmup_launch_orders,
        "sample_launch_orders": sample_launch_orders,
        "sample_pairing": "samples at the same index form one generated/oracle pair",
        "timing": "separate CUDA events per launch; each end event is synchronized before the next launch",
    }
    return samples, protocol


def _hardware_record() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _copy_runtime_relation(executable, selected: torch.Tensor, count: torch.Tensor) -> None:
    """Forward generic top-k output into the relation index plane on device."""
    sparse = executable.block_sparse_tensors

    def copy_shared_relation(target: torch.Tensor, source: torch.Tensor) -> None:
        # CuTe's normalization broadcasts one relation across batch and heads
        # with zero-stride views.  Write the unique backing slice rather than
        # asking PyTorch to copy into overlapping aliases.
        unique_slice = tuple(0 if stride == 0 else slice(None) for stride in target.stride())
        destination = target[unique_slice]
        destination.copy_(source.to(dtype=target.dtype).reshape(destination.shape))

    copy_shared_relation(sparse.mask_block_idx, selected)
    copy_shared_relation(sparse.mask_block_cnt, count)


def _sampled_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected: torch.Tensor,
    count: torch.Tensor,
    *,
    query_blocks: tuple[int, ...],
    block_size: int,
    scale: float,
) -> dict[int, torch.Tensor]:
    group_size = query.shape[2] // key.shape[2]
    reference = {}
    for query_block in query_blocks:
        start = query_block * block_size
        q = query[0, start : start + block_size].float()
        blocks = selected[query_block, : int(count[query_block])].tolist()
        k = torch.cat([key[0, item * block_size : (item + 1) * block_size] for item in blocks]).float()
        v = torch.cat([value[0, item * block_size : (item + 1) * block_size] for item in blocks]).float()
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
        scores = torch.einsum("mhd,nhd->hmn", q, k) * scale
        query_position = torch.arange(start, start + block_size, device=query.device)
        key_position = torch.cat(
            [torch.arange(item * block_size, (item + 1) * block_size, device=query.device) for item in blocks]
        )
        scores.masked_fill_(key_position[None, None, :] > query_position[None, :, None], -torch.inf)
        probability = torch.softmax(scores, dim=-1)
        reference[query_block] = torch.einsum("hmn,nhv->mhv", probability, v)
    return reference


def _matched_block_sparse_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_metadata: torch.Tensor,
    key_value_metadata: torch.Tensor,
    router_valid: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    slots: int,
    scale: float,
):
    """Build the same natural boundary around an optional expert kernel."""
    if block_sparse_attn_func is None:
        raise RuntimeError("install the pinned Block-Sparse-Attention checkout to run the matched expert oracle")
    block_count = query_metadata.shape[0]
    query_heads = query.shape[2]
    dense_mask = torch.zeros(1, query_heads, block_count, block_count, dtype=torch.bool, device=query.device)
    cumulative_sequence = torch.tensor([0, query.shape[1]], dtype=torch.int32, device=query.device)
    head_mask_type = torch.ones(query_heads, dtype=torch.int32, device=query.device)
    streaming_info = torch.zeros(2 * query_heads, dtype=torch.int32, device=query.device)
    slot = torch.arange(slots, device=query.device)[None, :]

    def operation() -> torch.Tensor:
        router_score = query_metadata @ key_value_metadata.T
        router_score.masked_fill_(~router_valid, -torch.inf)
        selected = torch.topk(router_score, slots, dim=-1, sorted=True).indices
        edge_valid = slot < valid_count[:, None]
        dense_mask.zero_()
        dense_mask.scatter_(
            -1,
            selected[None, None].expand(1, query_heads, block_count, slots),
            edge_valid[None, None].expand(1, query_heads, block_count, slots),
        )
        return block_sparse_attn_func(
            query[0],
            key[0],
            value[0],
            cumulative_sequence,
            cumulative_sequence,
            head_mask_type,
            streaming_info,
            dense_mask,
            query.shape[1],
            query.shape[1],
            0.0,
            deterministic=True,
            softmax_scale=scale,
            is_causal=True,
            exact_streaming=False,
            return_attn_probs=False,
        )

    return operation


def _flash_moba_column_relation(
    selected: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    query_heads: int,
    query_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorient a shared block relation into FlashMoBA's column-major row lists.

    The transformation uses only generic relation fields: source block,
    destination block, and edge validity. FlashMoBA stores one sorted query-row
    list per ``(head, destination block)``. The accepted Shuttle workload uses
    one relation shared by every query head, so the final row lists are repeated
    across heads without changing the semantic relation.
    """
    if selected.ndim != 2 or valid_count.shape != selected.shape[:1]:
        raise ValueError("selected must be [query_block, slot] and valid_count must be [query_block]")
    if query_heads <= 0 or query_block_size <= 0:
        raise ValueError("query_heads and query_block_size must be positive")

    block_count, route_slots = selected.shape
    slot = torch.arange(route_slots, device=selected.device)[None, :]
    edge_valid = slot < valid_count[:, None]
    block_relation = torch.zeros(block_count, block_count, dtype=torch.bool, device=selected.device)
    block_relation.scatter_(1, selected, edge_valid)

    # nonzero() is lexicographic in the logical destination-major view, so the
    # source blocks and the expanded query rows are sorted within every column.
    destination_source = block_relation.T.contiguous().nonzero()
    source_block = destination_source[:, 1]
    row_in_block = torch.arange(query_block_size, device=selected.device)
    query_rows = (source_block[:, None] * query_block_size + row_in_block[None, :]).reshape(-1)
    row_indices = query_rows.repeat(query_heads).to(torch.int32)

    rows_per_destination = block_relation.sum(dim=0, dtype=torch.int64) * query_block_size
    col_nnz = rows_per_destination.to(torch.int32)[None, None, :].expand(1, query_heads, -1).contiguous()
    flat_counts = col_nnz.reshape(-1).to(torch.int64)
    flat_offsets = torch.cumsum(flat_counts, dim=0) - flat_counts
    col_offsets = flat_offsets.reshape(1, query_heads, block_count).contiguous()
    return col_offsets, col_nnz, row_indices


def _matched_flash_moba_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_metadata: torch.Tensor,
    key_value_metadata: torch.Tensor,
    router_valid: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    block_size: int,
    query_group_size: int,
    slots: int,
    scale: float,
) -> tuple[
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], torch.Tensor],
    Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """Build matched full-boundary and attention-only FlashMoBA operations."""
    if flash_moba_attn_varlen_func is None:
        raise RuntimeError("install the pinned FlashMoBA checkout to run the primary expert oracle")
    sequence = query.shape[1]
    cumulative_sequence = torch.tensor([0, sequence], dtype=torch.int32, device=query.device)
    slot = torch.arange(slots, device=query.device)[None, :]
    cached: dict[str, torch.Tensor] = {}

    def select() -> torch.Tensor:
        router_score = query_metadata @ key_value_metadata.T
        router_score.masked_fill_(~router_valid, -torch.inf)
        selected = torch.topk(router_score, slots, dim=-1, sorted=True).indices
        cached["selected"] = selected
        return selected

    def reorient(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_valid = slot < valid_count[:, None]
        # Preserve the selected set and validity as an explicit semantic check;
        # FlashMoBA's physical relation is destination-major and erases slot order.
        cached["edge_valid"] = edge_valid
        return _flash_moba_column_relation(
            selected,
            valid_count,
            query_heads=query.shape[2],
            query_block_size=block_size,
        )

    initial_selected = select()
    initial_relation = reorient(initial_selected)
    cached["col_offsets"], cached["col_nnz"], cached["row_indices"] = initial_relation

    def attend(
        col_offsets: torch.Tensor,
        col_nnz: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            output = flash_moba_attn_varlen_func(
                query[0],
                key[0],
                value[0],
                cumulative_sequence,
                cumulative_sequence,
                sequence,
                sequence,
                col_offsets,
                col_nnz,
                row_indices,
                query_group_size,
                block_size,
                0.0,
                softmax_scale=scale,
                causal=True,
                deterministic=True,
                return_attn_probs=False,
            )
        return output

    def full_operation() -> torch.Tensor:
        col_offsets, col_nnz, row_indices = reorient(select())
        return attend(col_offsets, col_nnz, row_indices)

    def attention_only_operation() -> torch.Tensor:
        return attend(cached["col_offsets"], cached["col_nnz"], cached["row_indices"])

    def relation_only_operation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return reorient(initial_selected)

    return full_operation, attention_only_operation, select, relation_only_operation, cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=16_384)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--router-dimension", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--include-block-sparse-oracle", action="store_true")
    parser.add_argument("--include-flash-moba-oracle", action="store_true")
    parser.add_argument("--flash-moba-query-group", type=int, default=768)
    parser.add_argument("--include-kv-major", action="store_true")
    parser.add_argument("--kv-query-capacity", type=int, default=2)
    parser.add_argument("--include-kv-capacity-mutation", action="store_true")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.include_block_sparse_oracle and args.include_flash_moba_oracle:
        raise ValueError("select only one matched expert oracle per counterbalanced run")
    config = RoutedAttentionDebugConfig(
        sequence=args.sequence,
        block_size=args.block,
        selected_blocks=args.slots,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
        router_dimension=args.router_dimension,
    )

    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    query = torch.randn(1, args.sequence, 32, 128, dtype=torch.bfloat16, device=device)
    key = torch.randn(1, args.sequence, 8, 128, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    query_metadata_np = generator.normal(size=(config.block_count, config.router_dimension)).astype(np.float32)
    key_value_metadata_np = generator.normal(size=query_metadata_np.shape).astype(np.float32)
    query_metadata = torch.as_tensor(query_metadata_np, device=device)
    key_value_metadata = torch.as_tensor(key_value_metadata_np, device=device)
    block_id = torch.arange(config.block_count, device=device)
    router_valid = block_id[None, :] <= block_id[:, None]
    valid_count = torch.minimum(block_id + 1, torch.full_like(block_id, args.slots))

    artifact = export_debug_routed_attention(config)
    schedule = StreamingTileSchedule(
        query_tile_size=args.block,
        key_value_tile_size=args.block,
        pipeline_depth=3,
    )
    physical_config = RoutedAttentionPlanConfig(
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
    natural = compile_stablehlo_routed_attention_program(
        artifact,
        input_names=ROUTED_ATTENTION_INPUT_NAMES,
        runtime_inputs={
            "query_metadata": query_metadata_np,
            "key_value_metadata": key_value_metadata_np,
        },
        schedule=schedule,
        config=physical_config,
    )
    output = torch.empty_like(query)
    log_sum_exp = torch.empty(1, 32, args.sequence, dtype=torch.float32, device=device)
    compile_started = time.perf_counter()
    executable = compile_h100_routed_streaming_program(
        natural.scheduled,
        orientation=RoutedAttentionOrientation.QUERY_MAJOR,
        query=query,
        key=key,
        value=value,
        output=output,
        log_sum_exp=log_sum_exp,
    )
    compile_seconds = time.perf_counter() - compile_started

    kv_output = torch.empty_like(query)
    kv_executable = None
    kv_compile_seconds = None
    if args.include_kv_major:
        kv_compile_started = time.perf_counter()
        kv_executable = compile_h100_bounded_kv_reuse_program(
            natural.scheduled,
            query=query,
            key=key,
            value=value,
            output=kv_output,
            query_capacity_per_task=args.kv_query_capacity,
        )
        kv_compile_seconds = time.perf_counter() - kv_compile_started

    last_selected = torch.empty(config.block_count, args.slots, dtype=torch.int64, device=device)

    def operation() -> None:
        router_score = query_metadata @ key_value_metadata.T
        router_score.masked_fill_(~router_valid, -torch.inf)
        selected = torch.topk(router_score, args.slots, dim=-1, sorted=True).indices
        last_selected.copy_(selected)
        _copy_runtime_relation(executable, selected, valid_count)
        executable(query, key, value, output, log_sum_exp)

    def kv_operation() -> None:
        if kv_executable is None:
            raise RuntimeError("KV-major execution was not requested")
        # Keep the natural router/top-k boundary in the physical comparison.
        # The current prototype specializes bounded task groups to this
        # RelationPlan after compilation; dynamic task regrouping remains a
        # separate index-plane optimization.
        router_score = query_metadata @ key_value_metadata.T
        router_score.masked_fill_(~router_valid, -torch.inf)
        torch.topk(router_score, args.slots, dim=-1, sorted=True)
        kv_executable(query, key, value, kv_output)

    operation()
    torch.cuda.synchronize()
    first_hash = hashlib.sha256(output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    selected_cpu = last_selected.cpu()
    count_cpu = valid_count.cpu()
    current_block = torch.arange(config.block_count)[:, None]
    valid_slot = torch.arange(args.slots)[None, :] < count_cpu[:, None]
    includes_current = ((selected_cpu == current_block) & valid_slot).any(dim=1)
    omitted_current = torch.nonzero(~includes_current & (torch.arange(config.block_count) > 0)).flatten()
    omitted_sample = int(omitted_current[-1]) if omitted_current.numel() else config.block_count - 1
    sampled_blocks = tuple(dict.fromkeys((0, config.block_count // 2, config.block_count - 1, omitted_sample)))
    reference = _sampled_reference(
        query,
        key,
        value,
        selected_cpu,
        count_cpu,
        query_blocks=sampled_blocks,
        block_size=args.block,
        scale=config.scale,
    )
    errors = []
    for query_block, expected in reference.items():
        start = query_block * args.block
        errors.append((output[0, start : start + args.block].float() - expected).abs().flatten())
    error = torch.cat(errors)
    operation()
    torch.cuda.synchronize()
    repeated_hash = hashlib.sha256(output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()

    oracle_record = None
    measurement_protocol: dict[str, object]
    oracle_attention_only = None
    oracle_adapter_record = None
    if args.include_flash_moba_oracle:
        (
            oracle,
            oracle_attention_only,
            flash_moba_router_only,
            flash_moba_relation_only,
            flash_moba_relation,
        ) = _matched_flash_moba_oracle(
            query,
            key,
            value,
            query_metadata,
            key_value_metadata,
            router_valid,
            valid_count,
            block_size=args.block,
            query_group_size=args.flash_moba_query_group,
            slots=args.slots,
            scale=config.scale,
        )
        oracle_implementation = "pinned FlashMoBA precomputed-relation expert oracle"
        oracle_revision = PINNED_FLASH_MOBA_REVISION
        oracle_adapter_record = {
            "source_relation": "query block -> selected KV block, shared across query heads",
            "physical_relation": "destination KV block -> sorted query-token rows, repeated across query heads",
            "col_offsets_shape": tuple(flash_moba_relation["col_offsets"].shape),
            "col_nnz_shape": tuple(flash_moba_relation["col_nnz"].shape),
            "row_indices_shape": tuple(flash_moba_relation["row_indices"].shape),
            "physical_query_group_size": args.flash_moba_query_group,
            "slot_order_policy": "erased to a selected set before exact normalized-exponential Fold",
            "routing_cost_in_full_boundary": True,
            "query_blocks_omitting_current_block": int(omitted_current.numel()),
            "sampled_omitted_current_block": omitted_sample,
        }
    elif args.include_block_sparse_oracle:
        oracle = _matched_block_sparse_oracle(
            query,
            key,
            value,
            query_metadata,
            key_value_metadata,
            router_valid,
            valid_count,
            slots=args.slots,
            scale=config.scale,
        )
        oracle_implementation = "pinned Block-Sparse-Attention expert oracle"
        oracle_revision = PINNED_BLOCK_SPARSE_ATTENTION_REVISION
    else:
        oracle = None

    if oracle is not None:
        oracle_output = oracle()
        torch.cuda.synchronize()
        oracle_hash = hashlib.sha256(oracle_output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
        repeated_oracle_output = oracle()
        torch.cuda.synchronize()
        repeated_oracle_hash = hashlib.sha256(
            repeated_oracle_output.view(torch.int16).cpu().numpy().tobytes()
        ).hexdigest()
        paired_samples, measurement_protocol = _measure_counterbalanced_pair(
            operation,
            oracle,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        samples = paired_samples["generated"]
        oracle_samples = paired_samples["matched_expert_oracle"]
        oracle_error = oracle_output.float() - output[0].float()
        oracle_median = statistics.median(oracle_samples)
        attention_only_record = None
        if oracle_attention_only is not None:
            attention_only_samples = _measure(oracle_attention_only, args.warmups, args.repeats)
            router_only_samples = _measure(flash_moba_router_only, args.warmups, args.repeats)
            relation_only_samples = _measure(flash_moba_relation_only, args.warmups, args.repeats)
            attention_only_record = {
                "samples_ms": attention_only_samples,
                "median_ms": statistics.median(attention_only_samples),
                "router_only_samples_ms": router_only_samples,
                "router_only_median_ms": statistics.median(router_only_samples),
                "relation_reorientation_only_samples_ms": relation_only_samples,
                "relation_reorientation_only_median_ms": statistics.median(relation_only_samples),
                "excluded": ("router Contract", "top-k", "relation reorientation"),
                "acceptance_denominator": False,
            }
        oracle_record = {
            "implementation": oracle_implementation,
            "revision": oracle_revision,
            "samples_ms": oracle_samples,
            "median_ms": oracle_median,
            "maximum_absolute_difference_from_generated": float(oracle_error.abs().max()),
            "mean_absolute_difference_from_generated": float(oracle_error.abs().mean()),
            "deterministic": oracle_hash == repeated_oracle_hash,
            "output_sha256": oracle_hash,
            "generated_to_oracle_ratio": statistics.median(samples) / oracle_median,
            "matched_boundary": True,
            "attention_only": attention_only_record,
            "relation_adapter": oracle_adapter_record,
        }
    else:
        samples = _measure(operation, args.warmups, args.repeats)
        measurement_protocol = {
            "name": "single_implementation_sequential",
            "variants": ("generated",),
            "warmup_launch_orders": [("generated",)] * args.warmups,
            "sample_launch_orders": [("generated",)] * args.repeats,
            "timing": "separate CUDA events per launch; each end event is synchronized before the next launch",
        }

    kv_record = None
    if kv_executable is not None:
        kv_operation()
        torch.cuda.synchronize()
        kv_first_hash = hashlib.sha256(kv_output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
        kv_samples = _measure(kv_operation, args.warmups, args.repeats)
        kv_operation()
        torch.cuda.synchronize()
        kv_repeated_hash = hashlib.sha256(kv_output.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
        kv_difference = kv_output.float() - output.float()
        mutation = None
        if args.include_kv_capacity_mutation:
            mutation_capacity = 1 if args.kv_query_capacity != 1 else 2
            mutation_output = torch.empty_like(output)
            mutation_executable = compile_h100_bounded_kv_reuse_program(
                natural.scheduled,
                query=query,
                key=key,
                value=value,
                output=mutation_output,
                query_capacity_per_task=mutation_capacity,
            )

            def mutation_operation() -> None:
                mutation_executable(query, key, value, mutation_output)

            mutation_samples = _measure(mutation_operation, args.warmups, args.repeats)
            mutation_operation()
            torch.cuda.synchronize()
            mutation_difference = mutation_output.float() - kv_output.float()
            mutation = {
                "query_capacity_per_task": mutation_capacity,
                "task_count": mutation_executable.reuse_plan.task_count,
                "samples_ms": mutation_samples,
                "median_ms": statistics.median(mutation_samples),
                "maximum_absolute_difference": float(mutation_difference.abs().max()),
                "mean_absolute_difference": float(mutation_difference.abs().mean()),
                "output_sha256": hashlib.sha256(mutation_output.view(torch.int16).cpu().numpy().tobytes()).hexdigest(),
                "same_generated_source": (
                    mutation_executable.generated_source_sha256 == kv_executable.generated_source_sha256
                ),
            }
        kv_record = {
            "orientation": "kv_major_slot_waves",
            "query_capacity_per_task": args.kv_query_capacity,
            "task_count": kv_executable.reuse_plan.task_count,
            "edge_count": kv_executable.reuse_plan.edge_count,
            "wave_task_counts": [wave.task_count for wave in kv_executable.reuse_plan.waves],
            "samples_ms": kv_samples,
            "median_ms": statistics.median(kv_samples),
            "minimum_ms": min(kv_samples),
            "maximum_ms": max(kv_samples),
            "mean_ms": statistics.mean(kv_samples),
            "query_major_to_kv_major_ratio": statistics.median(samples) / statistics.median(kv_samples),
            "maximum_absolute_difference_from_query_major": float(kv_difference.abs().max()),
            "mean_absolute_difference_from_query_major": float(kv_difference.abs().mean()),
            "deterministic": kv_first_hash == kv_repeated_hash,
            "output_sha256": kv_first_hash,
            "generated_source_sha256": kv_executable.generated_source_sha256,
            "physical_plan": kv_executable.physical_plan.dump(),
            "partial_state_materialization_bytes": kv_executable.physical_plan.partial_state_materialization_bytes,
            "online_state_materialization_bytes": kv_executable.physical_plan.online_state_materialization_bytes,
            "shared_kv_bytes_per_cta": 2 * args.block * config.head_dimension * query.element_size(),
            "source_order": "ascending selected slot; one query-state writer per wave; no atomics",
            "right_resource_reuse": (
                "one CTA stages one KV-head block in dynamic shared memory and reuses it for a bounded query group"
            ),
            "mutation": mutation,
        }

    report = natural.recovered.semantic_erasure_report
    generated_launches = (
        "generic router Contract",
        "generic top-k Relation index generation",
        "generated SM90 QK/online-Fold/PV streaming skeleton",
    )
    record = {
        "benchmark": "shuttle_natural_routed_streaming_attention_sm90",
        "source": "ordinary JAX -> StableHLO -> Relation/Contract/Map/DomainRestriction/Fold",
        "configuration": vars(args) | {"json_output": str(args.json_output)},
        "hardware": _hardware_record(),
        "torch": torch.__version__,
        "compile_seconds": compile_seconds,
        "kv_major_compile_seconds": kv_compile_seconds,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "mean_ms": statistics.mean(samples),
        "measurement_protocol": measurement_protocol,
        "kv_major": kv_record,
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "deterministic": first_hash == repeated_hash,
        "output_sha256": first_hash,
        "relation_sha256": hashlib.sha256(selected_cpu.numpy().tobytes()).hexdigest(),
        "semantic_erasure": {
            "source_semantics": report.source_semantics,
            "lowering_steps": [
                {"source_semantic": step.source_semantic, "generic_primitives": step.generic_primitives}
                for step in report.lowering_steps
            ],
            "scheduling_keys": report.scheduling_keys,
            "validation_errors": report.validation_errors,
        },
        "launch_manifest": {
            "generated_path": generated_launches,
            "expert_oracle_semantic_kernels": [],
        },
        "benchmark_boundary": {
            "included": ("router Contract", "top-k/index plane", "selected exact attention"),
            "excluded": ("QKV projection", "output projection"),
            "oracle_requirement": "must include the identical router Contract and top-k/index plane",
            "legacy_seer_2_388208_ms_is_matched": False,
        },
        "matched_expert_oracle": oracle_record,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
