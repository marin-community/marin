# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark pinned Flash Sparse Attention through a generic RelationPlan adapter.

FSA is an expert KV-major oracle, not a Shuttle backend. Its public selected-
attention API accepts token-level top-k block indices and reconstructs its own
block-to-token relation inside every call. This benchmark keeps that seam
visible: Shuttle builds the generic block relation, the adapter expands it to
FSA's token-level input, and the timed FSA call includes FSA's private relation
planning, partial computation, and reduction.

The script can run with ``--planning-only`` on a CPU host without PyTorch or
FSA installed. The full benchmark requires a checkout of FSA at the pinned
revision and an Ampere or Hopper GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_metadata import (
    canonical_json_sha256,
    command_output,
    command_record,
    framed_tensor_sha256,
    nvidia_smi_snapshot,
    toolchain_snapshot,
)

from tile_lifetime import (
    RelationPlan,
    RoutedAttentionPlanConfig,
    build_routed_attention_relation,
    compile_routed_attention_candidates,
    make_causal_block_relation,
)

PINNED_FSA_REVISION = "7ff144fd7ff485dc4220d439f31cc1708b64fef3"
FSA_SUPPORTED_BLOCK_SIZES = frozenset({32, 64, 128, 256})


def fsa_indices_from_relation(
    relation: RelationPlan,
    *,
    sequence_length: int,
    query_block_size: int,
    key_value_heads: int,
) -> np.ndarray:
    """Expand a block-shared RelationPlan into FSA's token-level index tensor.

    This uses only generic relation fields. No attention-specific route-plan
    representation is introduced. FSA requires selected block IDs to be sorted
    by block ID; rejecting another source order preserves Shuttle's declared
    selected-slot order instead of silently changing its numerical contract.
    """
    expected_query_blocks = math.ceil(sequence_length / query_block_size)
    if relation.source_item_count != expected_query_blocks:
        raise ValueError(
            "RelationPlan source count must equal ceil(sequence length / query block size): "
            f"{relation.source_item_count} != {expected_query_blocks}"
        )
    if sequence_length <= 0 or query_block_size <= 0 or key_value_heads <= 0:
        raise ValueError("sequence length, query block size, and KV-head count must be positive")

    block_indices = np.full(
        (relation.source_item_count, relation.route_slots),
        -1,
        dtype=np.int32,
    )
    flat_valid = relation.edge_valid.reshape(-1)
    valid_routes = np.flatnonzero(flat_valid)
    block_indices[
        relation.source_item[valid_routes],
        relation.route_slot[valid_routes],
    ] = relation.destination_item[valid_routes]

    for source_item in range(relation.source_item_count):
        selected = block_indices[source_item, relation.edge_valid[source_item]]
        if selected.size == 0:
            raise ValueError(f"FSA requires at least one selected KV block for query block {source_item}")
        if np.any(selected[1:] <= selected[:-1]):
            raise ValueError(
                "FSA requires strictly increasing selected KV blocks; changing slot order would violate "
                f"the source-ordered merge contract for query block {source_item}"
            )

    token_indices = np.repeat(block_indices, query_block_size, axis=0)[:sequence_length]
    return np.broadcast_to(
        token_indices[None, :, :],
        (key_value_heads, sequence_length, relation.route_slots),
    ).copy()


def _timed_cpu_samples(operation: Callable[[], Any], repeats: int) -> tuple[Any, list[float]]:
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        samples.append((time.perf_counter() - started) * 1_000)
    assert result is not None
    return result, samples


def _sample_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return framed_tensor_sha256(str(contiguous.dtype), contiguous.shape, contiguous.tobytes(order="C"))


def _relation_sha256(relation: RelationPlan) -> str:
    fields = {
        "edge_valid": relation.edge_valid,
        "source_item": relation.source_item,
        "route_slot": relation.route_slot,
        "destination_item": relation.destination_item,
        "route_to_destination_row": relation.route_to_destination_row,
        "destination_row_to_route": relation.destination_row_to_route,
        "group_destination_item": relation.group_destination_item,
        "group_count": relation.group_count,
        "group_offset": relation.group_offset,
    }
    return canonical_json_sha256({name: _array_sha256(value) for name, value in fields.items()})


def _semantic_relation_mask_sha256(relation: RelationPlan) -> str:
    mask = np.zeros((relation.source_item_count, relation.destination_count), dtype=np.bool_)
    valid_routes = np.flatnonzero(relation.edge_valid.reshape(-1))
    mask[relation.source_item[valid_routes], relation.destination_item[valid_routes]] = True
    return hashlib.sha256(mask.tobytes(order="C")).hexdigest()


def _selected_qk_pv_flops(
    relation: RelationPlan,
    *,
    sequence_length: int,
    block_size: int,
    query_heads: int,
    head_dimension: int,
) -> int:
    valid_routes = np.flatnonzero(relation.edge_valid.reshape(-1))
    query_tokens = np.minimum(
        block_size,
        sequence_length - relation.source_item[valid_routes].astype(np.int64) * block_size,
    )
    key_tokens = np.minimum(
        block_size,
        sequence_length - relation.destination_item[valid_routes].astype(np.int64) * block_size,
    )
    token_pairs = int(np.sum(query_tokens * key_tokens, dtype=np.int64))
    return 4 * token_pairs * query_heads * head_dimension


def _fsa_buffer_bytes(
    fsa_indices: np.ndarray,
    *,
    sequence_length: int,
    block_size: int,
    head_dimension: int,
) -> dict[str, Any]:
    """Reproduce the source-visible forward-buffer shapes at the pinned FSA revision."""
    key_value_heads, _, top_k = fsa_indices.shape
    real_blocks = math.ceil(sequence_length / block_size)
    allocated_blocks = max(real_blocks, top_k)
    destination_degrees = np.zeros((key_value_heads, allocated_blocks), dtype=np.int64)
    for head in range(key_value_heads):
        valid = fsa_indices[head][fsa_indices[head] >= 0]
        destination_degrees[head] = np.bincount(valid, minlength=allocated_blocks)
    if allocated_blocks > 1:
        global_max_valid_tokens = int(np.max(destination_degrees[:, 1:]))
    else:
        global_max_valid_tokens = int(np.max(destination_degrees))
    rest_items = (allocated_blocks - 1) * global_max_valid_tokens

    weighted_value_bytes = 2 * head_dimension * (sequence_length + rest_items)
    denominator_bytes = 4 * (sequence_length + rest_items)
    output_rescale_bytes = denominator_bytes
    # cummax materializes m_ij_tiles separately from m_i_cur_tiles.
    maxima_bytes = 2 * 4 * allocated_blocks * sequence_length
    partial_state_bytes = weighted_value_bytes + denominator_bytes + output_rescale_bytes + maxima_bytes

    dense_inverse_index_bytes = 2 * 4 * allocated_blocks * sequence_length
    valid_lens_bytes = 4 * key_value_heads * allocated_blocks
    compact_selected_tokens_bytes = 4 * int(np.count_nonzero(fsa_indices >= 0))
    valid_start_indices_bytes = 4 * key_value_heads * allocated_blocks
    internal_index_bytes = (
        dense_inverse_index_bytes + valid_lens_bytes + compact_selected_tokens_bytes + valid_start_indices_bytes
    )
    return {
        "allocated_kv_blocks": allocated_blocks,
        "global_max_selected_tokens_per_nonzero_kv_block": global_max_valid_tokens,
        "weighted_value_partial_bytes": weighted_value_bytes,
        "denominator_partial_bytes": denominator_bytes,
        "output_rescale_partial_bytes": output_rescale_bytes,
        "maxima_working_bytes": maxima_bytes,
        "partial_and_statistics_bytes": partial_state_bytes,
        "internal_inverse_index_bytes": internal_index_bytes,
        "adapter_input_index_bytes": int(fsa_indices.nbytes),
        "accounting_scope": (
            "Pinned FSA forward source allocations for one sequence. Partial buffers have head_tile=1 and are "
            "reused serially across query heads. Output, LSE, Q/K/V, allocator fragmentation, and temporary "
            "PyTorch indexing/cummax workspace are excluded."
        ),
    }


def _semantic_mismatches() -> list[dict[str, str]]:
    return [
        {
            "seam": "relation_granularity",
            "mismatch": (
                "Shuttle has one edge per query-block/selected-block pair; FSA accepts one selected-block list "
                "per query token and KV head."
            ),
            "adapter_effect": "Each generic edge is repeated across its query-block tokens and all KV heads.",
        },
        {
            "seam": "orientation_input",
            "mismatch": (
                "FSA cannot consume RelationPlan grouped_route_indices, destination offsets, or inverse mapping."
            ),
            "adapter_effect": (
                "The timed FSA call reconstructs a private block-to-token relation, so it is a KV-major expert "
                "oracle rather than an execution of Shuttle's physical schedule."
            ),
        },
        {
            "seam": "head_dependent_routing",
            "mismatch": "FSA permits a different selected relation per KV head; the Shuttle experiment is block-shared.",
            "adapter_effect": "The identical relation is duplicated across KV heads.",
        },
        {
            "seam": "merge_order",
            "mismatch": (
                "Pinned FSA's online maxima are accumulated in ascending physical KV-block order and its input "
                "generator sorts top-k indices."
            ),
            "adapter_effect": (
                "The adapter rejects non-increasing selected-slot order instead of silently changing Shuttle's "
                "source-ordered floating-point merge contract."
            ),
        },
        {
            "seam": "causal_and_sequence_shape",
            "mismatch": (
                "The public FSA entry point uses one shared Q/K/V cumulative-length vector and always runs its "
                "causal forward path."
            ),
            "adapter_effect": "This benchmark is restricted to causal self-attention with equal Q and KV lengths.",
        },
        {
            "seam": "timing_decomposition",
            "mismatch": (
                "The public FSA call does not expose separate inverse-plan, QK/PV, partial-state, or reduction timings."
            ),
            "adapter_effect": (
                "Shuttle relation planning and adapter expansion are timed separately; FSA's internal planning, "
                "allocations, compute, and merge remain combined in the kernel sample."
            ),
        },
    ]


def _source_record(root: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {"declared_revision": PINNED_FSA_REVISION}
    if root is None:
        return record
    resolved = root.resolve()
    head = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(resolved), "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(
        ["git", "-C", str(resolved), "diff", "--"],
        check=False,
        capture_output=True,
        text=True,
    )
    record.update(
        {
            "root": str(resolved),
            "git_head": head.stdout.strip(),
            "git_head_returncode": head.returncode,
            "git_status": status.stdout.splitlines(),
            "git_status_returncode": status.returncode,
            "git_diff": diff.stdout,
            "git_diff_returncode": diff.returncode,
        }
    )
    return record


def _validate_fsa_source(root: Path) -> None:
    source = _source_record(root)
    if source["git_head_returncode"] != 0:
        raise ValueError(f"FSA root is not a readable Git checkout: {root}")
    if source["git_head"] != PINNED_FSA_REVISION:
        raise ValueError(f"FSA must be checked out at {PINNED_FSA_REVISION}; found {source['git_head']}")


def _import_torch_and_fsa(root: Path) -> tuple[Any, Callable[..., Any]]:
    _validate_fsa_source(root)
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("triton")
    except ImportError as error:
        raise RuntimeError(
            "The executable benchmark requires the pinned FSA dependencies (PyTorch and Triton); "
            "use --planning-only to exercise the adapter without them"
        ) from error
    sys.path.insert(0, str(root.resolve()))
    try:
        module = importlib.import_module("fsa.ops.FSA_topk_sparse_attention")
    except ImportError as error:
        raise RuntimeError(
            "Could not import pinned FSA. Install its requirements in the benchmark environment or use "
            "--planning-only. No FSA source is copied into Shuttle."
        ) from error
    return torch, module.FSA_topk_sparse_attention


def _torch_tensor_sha256(torch: Any, tensor: Any) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    if contiguous.dtype == torch.bfloat16:
        payload = contiguous.view(torch.uint16).numpy().tobytes(order="C")
    else:
        payload = contiguous.numpy().tobytes(order="C")
    return framed_tensor_sha256(str(contiguous.dtype), tuple(contiguous.shape), payload)


def _toolchain_snapshot() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return toolchain_snapshot(nvcc)
    ptxas = shutil.which("ptxas")
    return {
        "python": sys.version,
        "platform": command_output(["uname", "-a"]),
        "nvcc": {
            "arguments": ["nvcc", "--version"],
            "returncode": 127,
            "stdout": "",
            "stderr": "nvcc is not installed; FSA is Triton JIT and does not require it",
        },
        "ptxas": (
            command_output([ptxas, "--version"])
            if ptxas is not None
            else {
                "arguments": ["ptxas", "--version"],
                "returncode": 127,
                "stdout": "",
                "stderr": "ptxas is not on PATH; Triton resolved its packaged CUDA toolchain internally",
            }
        ),
    }


def _cuda_samples(
    torch: Any,
    operation: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> tuple[Any, list[float]]:
    output = None
    for _ in range(warmups):
        output = operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)
    assert output is not None
    return output, samples


def _selected_attention_reference(
    torch: Any,
    query: Any,
    key: Any,
    value: Any,
    relation: RelationPlan,
    *,
    block_size: int,
    scale: float,
    checked_query_blocks: np.ndarray,
) -> tuple[Any, Any]:
    """Evaluate checked query blocks with an independent source-ordered FP32 softmax."""
    head_map = torch.arange(query.shape[1], device=query.device) // (query.shape[1] // key.shape[1])
    outputs = []
    token_indices = []
    for query_block in checked_query_blocks.tolist():
        query_start = query_block * block_size
        query_stop = min(query_start + block_size, query.shape[0])
        query_positions = torch.arange(query_start, query_stop, device=query.device)
        route_start = query_block * relation.route_slots
        route_stop = route_start + relation.route_slots
        routes = np.arange(route_start, route_stop, dtype=np.int32)
        routes = routes[relation.edge_valid[query_block]]
        chosen = relation.destination_item[routes]
        selected_tokens = torch.cat(
            [
                torch.arange(
                    int(block) * block_size,
                    min((int(block) + 1) * block_size, key.shape[0]),
                    device=query.device,
                )
                for block in chosen
            ]
        )
        expanded_key = key[selected_tokens][:, head_map].float()
        scores = torch.einsum("qhd,khd->qhk", query[query_start:query_stop].float(), expanded_key) * scale
        scores.masked_fill_(query_positions[:, None, None] < selected_tokens[None, None, :], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        expanded_value = value[selected_tokens][:, head_map].float()
        outputs.append(torch.einsum("qhk,khv->qhv", probabilities, expanded_value))
        token_indices.append(query_positions)
    return torch.cat(token_indices), torch.cat(outputs)


def _executable_result(
    args: argparse.Namespace,
    relation: RelationPlan,
    fsa_indices: np.ndarray,
) -> dict[str, Any]:
    if args.fsa_root is None:
        raise ValueError("--fsa-root is required unless --planning-only is set")
    torch, fsa_attention = _import_torch_and_fsa(args.fsa_root)
    if not torch.cuda.is_available():
        raise RuntimeError("the executable FSA benchmark requires a CUDA GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] not in {8, 9}:
        raise RuntimeError(
            f"pinned FSA is tested on Ampere and Hopper, not compute capability {capability[0]}.{capability[1]}"
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
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
    cumulative_sequence = torch.tensor([0, args.sequence_length], dtype=torch.int32, device=device)

    transfer_start = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    transfer_start.record()
    device_indices = torch.from_numpy(fsa_indices).to(device=device)
    transfer_end.record()
    transfer_end.synchronize()
    index_transfer_ms = float(transfer_start.elapsed_time(transfer_end))

    scale = 1.0 / math.sqrt(args.head_dimension)

    def operation() -> Any:
        with torch.inference_mode():
            return fsa_attention(
                query,
                key,
                value,
                device_indices,
                args.block_size,
                cumulative_sequence,
                scale,
            )

    first_started = time.perf_counter()
    first_output = operation()
    torch.cuda.synchronize()
    compile_and_first_run_seconds = time.perf_counter() - first_started

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = int(torch.cuda.memory_allocated(device))
    peak_output = operation()
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    measured_peak_increment_bytes = max(0, peak_allocated - allocated_before)
    del peak_output

    output, kernel_samples = _cuda_samples(
        torch,
        operation,
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    selected_work_flops = _selected_qk_pv_flops(
        relation,
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        query_heads=args.query_heads,
        head_dimension=args.head_dimension,
    )
    kernel_summary = _sample_summary(kernel_samples)
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
    difference = output[checked_tokens].float() - reference
    absolute = difference.abs()
    first_output_sha256 = _torch_tensor_sha256(torch, first_output)
    timed_output_sha256 = _torch_tensor_sha256(torch, output)
    return {
        "status": "ok",
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "index_host_to_device_ms": index_transfer_ms,
        "kernel": {
            "scope": (
                "FSA public selected-attention call, including FSA-private inverse-relation planning, allocations, "
                "QK/PV partial computation, and reduction"
            ),
            **kernel_summary,
            "selected_qk_pv_flops": selected_work_flops,
            "selected_qk_pv_tflops_at_median": selected_work_flops / kernel_summary["median_ms"] / 1.0e9,
        },
        "measured_peak_allocator_increment_bytes": measured_peak_increment_bytes,
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
        "bitwise_deterministic_repeated_output": bool(
            first_output_sha256 == timed_output_sha256 and torch.equal(first_output, output)
        ),
        "hashes": {
            "query_sha256": _torch_tensor_sha256(torch, query),
            "key_sha256": _torch_tensor_sha256(torch, key),
            "value_sha256": _torch_tensor_sha256(torch, value),
            "first_output_sha256": first_output_sha256,
            "timed_output_sha256": timed_output_sha256,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": importlib.import_module("triton").__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(capability),
            "toolchain": _toolchain_snapshot(),
            "gpu_telemetry": {"final": nvidia_smi_snapshot()},
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--selected-blocks", type=int, default=16)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--key-value-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--planning-repeats", type=int, default=20)
    parser.add_argument("--correctness-blocks", type=int, default=8)
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--fsa-root", type=Path)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--clock-policy", default="cluster_default_unpinned")
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.block_size not in FSA_SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"FSA block size must be one of {sorted(FSA_SUPPORTED_BLOCK_SIZES)}")
    if args.query_heads % args.key_value_heads:
        raise ValueError("query heads must map evenly onto KV heads")
    if args.head_dimension > 256:
        raise ValueError("pinned FSA requires equal Q/K/V head dimensions no greater than 256")
    if args.planning_repeats <= 0 or args.repeats <= 0 or args.iterations <= 0:
        raise ValueError("planning repeats, kernel repeats, and iterations must be positive")

    selected, edge_valid = make_causal_block_relation(
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        selected_blocks=args.selected_blocks,
    )

    def build_relation() -> RelationPlan:
        return build_routed_attention_relation(selected, edge_valid=edge_valid)

    relation, relation_samples = _timed_cpu_samples(build_relation, args.planning_repeats)

    def build_fsa_indices() -> np.ndarray:
        return fsa_indices_from_relation(
            relation,
            sequence_length=args.sequence_length,
            query_block_size=args.block_size,
            key_value_heads=args.key_value_heads,
        )

    fsa_indices, adapter_samples = _timed_cpu_samples(build_fsa_indices, args.planning_repeats)
    selected_work_flops = _selected_qk_pv_flops(
        relation,
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        query_heads=args.query_heads,
        head_dimension=args.head_dimension,
    )
    _, kv_major = compile_routed_attention_candidates(
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

    result = {
        "schema_version": 1,
        "benchmark": "shuttle_routed_sparse_attention_fsa_kv_major_h100",
        "status": "planning_only" if args.planning_only else "ok",
        "source": {
            "shuttle_revision": args.shuttle_revision,
            "fsa": _source_record(args.fsa_root),
        },
        "shape": {
            "sequence_length": args.sequence_length,
            "query_block_size": args.block_size,
            "key_value_block_size": args.block_size,
            "selected_blocks": args.selected_blocks,
            "query_heads": args.query_heads,
            "key_value_heads": args.key_value_heads,
            "head_dimension": args.head_dimension,
            "value_dimension": args.head_dimension,
            "causal": True,
            "dtype": "bfloat16",
        },
        "relation": {
            "orientation": "block-shared generic relation adapted to FSA token input; FSA executes KV-major",
            "valid_block_edges": relation.route_count,
            "token_edges_after_fsa_expansion": int(np.count_nonzero(fsa_indices >= 0)),
            "source_degrees": np.count_nonzero(relation.edge_valid, axis=1).tolist(),
            "destination_degrees": relation.group_count.tolist(),
            "relation_plan_timing": _sample_summary(relation_samples),
            "fsa_adapter_timing": _sample_summary(adapter_samples),
            "relation_plan_sha256": _relation_sha256(relation),
            "semantic_boolean_mask_raw_sha256": _semantic_relation_mask_sha256(relation),
            "fsa_indices_sha256": _array_sha256(fsa_indices),
            "merge_order": relation.merge_order,
        },
        "buffers": {
            "shuttle_declared_edge_partial_state_bytes": kv_major.partial_state_materialization_bytes,
            "shuttle_plan_scope": (
                "One FP32 (max, denominator, weighted-value) state per block edge, query token, and query head"
            ),
            "pinned_fsa_source_estimate": _fsa_buffer_bytes(
                fsa_indices,
                sequence_length=args.sequence_length,
                block_size=args.block_size,
                head_dimension=args.head_dimension,
            ),
        },
        "selected_work": {
            "qk_plus_pv_flops": selected_work_flops,
            "formula": "tail-aware 4 * sum_edges(query_block_tokens * kv_block_tokens) * query_heads * head_dimension",
        },
        "candidate": {
            "shuttle_kv_major_plan": kv_major.dump(),
            "executable_seam": (
                "RelationPlan -> repeated token-level topk_idx -> pinned FSA public selected-attention call"
            ),
            "semantic_mismatches": _semantic_mismatches(),
        },
        "protocol": {
            "planning_repeats": args.planning_repeats,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations": args.iterations,
            "kernel_selection_metric": "median CUDA-event milliseconds",
            "clock_policy": args.clock_policy,
            "command": command_record(),
        },
    }
    if not args.planning_only:
        result["execution"] = _executable_result(args, relation, fsa_indices)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
