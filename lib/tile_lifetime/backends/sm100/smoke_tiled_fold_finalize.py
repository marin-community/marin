# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual GB200 combine-only benchmark for generic tiled Fold schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from clean_routed_streaming_emitter import (  # noqa: E402
    real_col_to_stg128_half_col,
    render_partial_merge_cuda,
    tiled_fold_merge_program,
)
from clean_routed_streaming_runtime import compile_tiled_fold_finalize  # noqa: E402

from tile_lifetime import DType  # noqa: E402
from tile_lifetime.tensor_program import TensorAxis, serialize_scalar_expression  # noqa: E402
from tile_lifetime.tiled_fold_finalize import (  # noqa: E402
    FoldFeatureLayout,
    FoldPartialAddressing,
    FoldPhysicalAxis,
    TiledFoldAxes,
    TiledFoldFinalizeSchedule,
    TiledFoldInputLayout,
    deterministic_weighted_sum_fold_program,
    normalized_exponential_fold_program,
)


@dataclass(frozen=True)
class Candidate:
    """One bounded workload-neutral physical Fold schedule."""

    name: str
    feature_tile: int
    shared_buffers: int


@dataclass(frozen=True)
class PreparedCandidate:
    """Compiled candidate and its preallocated combine-only call boundary."""

    candidate: Candidate
    source: str
    semantic_program: dict[str, Any]
    module: Any
    call_arguments: tuple[Any, ...]
    output: torch.Tensor
    expected: torch.Tensor


CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        Candidate("feature64_buffer1_no_overlap", feature_tile=64, shared_buffers=1),
        Candidate("feature64_buffer2_ping_pong", feature_tile=64, shared_buffers=2),
        Candidate("feature128_buffer2_ping_pong", feature_tile=128, shared_buffers=2),
    )
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--source-rows", type=int, default=8192)
    parser.add_argument("--partials", type=int, default=16)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--semantics", choices=("normalized_exp", "weighted_sum"), default="normalized_exp")
    parser.add_argument("--candidate", action="append", choices=tuple(CANDIDATES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _schedule(arguments: argparse.Namespace, candidate: Candidate, addressing: FoldPartialAddressing):
    axes = TiledFoldAxes(
        partial=TensorAxis(0, arguments.partials, "partial"),
        row=TensorAxis(1, arguments.rows, "row"),
        feature=TensorAxis(2, arguments.features, "feature"),
    )
    if arguments.features % candidate.feature_tile:
        raise ValueError(
            f"feature width {arguments.features} must be divisible by candidate tile {candidate.feature_tile}"
        )
    if addressing is FoldPartialAddressing.DENSE:
        layout = TiledFoldInputLayout(
            addressing=addressing,
            value_axis_order=(FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW, FoldPhysicalAxis.FEATURE),
            scalar_axis_order=(FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW),
            feature_layout=FoldFeatureLayout.STG128_LANE_PERMUTED,
        )
        partial_lanes = 32
    else:
        layout = TiledFoldInputLayout(
            addressing=addressing,
            value_axis_order=(FoldPhysicalAxis.SOURCE, FoldPhysicalAxis.FEATURE),
            scalar_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
            index_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
            feature_layout=FoldFeatureLayout.CONTIGUOUS,
        )
        partial_lanes = 1
    return TiledFoldFinalizeSchedule(
        axes=axes,
        partial_addressing=addressing,
        row_tile=8,
        feature_tile=candidate.feature_tile,
        vector_bytes=16,
        shared_stages=4,
        threads=256,
        partial_lanes=partial_lanes,
        shared_buffers=candidate.shared_buffers,
        input_layout=layout,
    )


def _program(arguments: argparse.Namespace, candidate: Candidate):
    if arguments.semantics == "normalized_exp":
        generic = normalized_exponential_fold_program(
            _schedule(arguments, candidate, FoldPartialAddressing.DENSE),
            partial_value_dtype=DType.BF16,
            output_dtype=DType.BF16,
        )
    else:
        generic = deterministic_weighted_sum_fold_program(
            _schedule(arguments, candidate, FoldPartialAddressing.INDEXED),
            partial_value_dtype=DType.BF16,
            output_dtype=DType.BF16,
        )
    return tiled_fold_merge_program(generic)


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _stg128_storage(value: torch.Tensor) -> torch.Tensor:
    stored = torch.empty_like(value)
    for real_feature in range(value.shape[-1]):
        stored[..., real_col_to_stg128_half_col(real_feature)] = value[..., real_feature]
    return stored


def _normalized_exp_inputs(arguments: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor | int]:
    canonical_value = torch.randn(
        arguments.partials,
        arguments.rows,
        1,
        arguments.features,
        dtype=torch.bfloat16,
        device=device,
    )
    partial_scalar = torch.randn(
        arguments.partials,
        arguments.rows,
        1,
        dtype=torch.float32,
        device=device,
    )
    valid_counts = torch.full(
        (arguments.rows, 1),
        arguments.partials,
        dtype=torch.int32,
        device=device,
    )
    if arguments.partials > 2:
        valid_counts[::3, 0] = arguments.partials - 2
        partial_scalar[-2:, ::3, 0] = 1.0e30
    partial = torch.arange(arguments.partials, device=device)[:, None]
    valid = partial < valid_counts[:, 0][None, :]
    scalar = partial_scalar[:, :, 0]
    masked_scalar = torch.where(valid, scalar, -torch.inf)
    common = masked_scalar.amax(dim=0)
    weight = torch.where(valid, torch.exp(masked_scalar - common[None, :]), 0.0)
    numerator = torch.sum(weight[..., None] * canonical_value[:, :, 0].float(), dim=0, dtype=torch.float32)
    denominator = torch.sum(weight, dim=0, dtype=torch.float32)
    expected = (numerator / denominator[:, None]).to(torch.bfloat16).unsqueeze(1)
    return {
        "partial_scalar": partial_scalar,
        "partial_value": _stg128_storage(canonical_value),
        "partial_metadata": valid_counts,
        "query_heads_per_key_value_head": 1,
        "expected": expected,
    }


def _weighted_sum_inputs(arguments: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor | int]:
    values = torch.randn(
        arguments.source_rows,
        arguments.features,
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.randint(
        arguments.source_rows,
        (arguments.rows, arguments.partials),
        dtype=torch.int32,
        device=device,
    )
    if arguments.partials > 1:
        indices[::3, 1] = -1
    weights = torch.randn(
        arguments.rows,
        arguments.partials,
        dtype=torch.float32,
        device=device,
    )
    result = torch.zeros(arguments.rows, arguments.features, dtype=torch.float32, device=device)
    for partial in range(arguments.partials):
        source = indices[:, partial]
        valid = source >= 0
        contribution = torch.zeros_like(result)
        contribution[valid] = values[source[valid]].float() * weights[valid, partial, None]
        result = result + contribution
    return {
        "partial_scalar": weights,
        "partial_value": values,
        "partial_metadata": indices,
        "query_heads_per_key_value_head": 1,
        "expected": result.to(torch.bfloat16),
    }


def _semantic_program(program: Any) -> dict[str, Any]:
    generic = program.generic_program
    assert generic is not None
    semantics = generic.semantics
    return {
        "accumulation_dtype": semantics.accumulation_dtype.value,
        "contribution_expression": json.loads(serialize_scalar_expression(semantics.contribution_expression)),
        "denominator": semantics.denominator.value,
        "finalize_expression": json.loads(serialize_scalar_expression(semantics.finalize_expression)),
        "output_dtype": semantics.output_dtype.value,
        "partial_scalar_dtype": semantics.partial_scalar_dtype.value,
        "partial_value_dtype": semantics.partial_value_dtype.value,
        "reassociation": semantics.reassociation.value,
        "scalar_reduction": semantics.scalar_reduction.value,
        "update_expression": json.loads(serialize_scalar_expression(semantics.update_expression)),
        "weight_expression": json.loads(serialize_scalar_expression(semantics.weight_expression)),
    }


def _canonical_sha256(value: dict[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _prepare_candidate(
    arguments: argparse.Namespace,
    candidate: Candidate,
    inputs: dict[str, torch.Tensor | int],
) -> PreparedCandidate:
    program = _program(arguments, candidate)
    source = render_partial_merge_cuda(program)
    build_directory = None
    if arguments.build_directory is not None:
        build_directory = arguments.build_directory / candidate.name
    module = compile_tiled_fold_finalize(program, build_directory=build_directory)
    expected = inputs["expected"]
    assert isinstance(expected, torch.Tensor)
    output = torch.empty_like(expected)
    call_arguments = (
        inputs["partial_scalar"],
        inputs["partial_value"],
        inputs["partial_metadata"],
        inputs["query_heads_per_key_value_head"],
        output,
    )

    return PreparedCandidate(
        candidate=candidate,
        source=source,
        semantic_program=_semantic_program(program),
        module=module,
        call_arguments=call_arguments,
        output=output,
        expected=expected,
    )


def _validate_candidate(prepared: PreparedCandidate) -> dict[str, Any]:
    observed = prepared.module.merge_out(*prepared.call_arguments)
    torch.cuda.synchronize()
    difference = (observed.float() - prepared.expected.float()).abs()
    first_hash = _tensor_sha256(observed)
    prepared.module.merge_out(*prepared.call_arguments)
    torch.cuda.synchronize()
    deterministic = first_hash == _tensor_sha256(prepared.output)
    return {
        "deterministic": deterministic,
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "output_sha256": first_hash,
    }


def _rotated(candidates: list[PreparedCandidate], iteration: int) -> list[PreparedCandidate]:
    offset = iteration % len(candidates)
    return candidates[offset:] + candidates[:offset]


def _warm_candidates(candidates: list[PreparedCandidate], warmups: int) -> None:
    for warmup in range(warmups):
        for prepared in _rotated(candidates, warmup):
            prepared.module.merge_out(*prepared.call_arguments)
    torch.cuda.synchronize()


def _time_candidates(
    candidates: list[PreparedCandidate],
    repeats: int,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    samples = {prepared.candidate.name: [] for prepared in candidates}
    repeat_orders = []
    for repeat in range(repeats):
        execution_order = _rotated(candidates, repeat)
        repeat_orders.append([prepared.candidate.name for prepared in execution_order])
        for prepared in execution_order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            prepared.module.merge_out(*prepared.call_arguments)
            end.record()
            end.synchronize()
            samples[prepared.candidate.name].append(float(start.elapsed_time(end)))
    return samples, repeat_orders


def _candidate_record(
    prepared: PreparedCandidate,
    validation: dict[str, Any],
    samples: list[float],
) -> dict[str, Any]:
    return {
        "candidate": prepared.candidate.name,
        "combine_only": True,
        "deterministic": validation["deterministic"],
        "feature_tile": prepared.candidate.feature_tile,
        "maximum_absolute_error": validation["maximum_absolute_error"],
        "mean_absolute_error": validation["mean_absolute_error"],
        "median_ms": float(np.median(samples)),
        "output_sha256": validation["output_sha256"],
        "samples_ms": samples,
        "semantic_sha256": _canonical_sha256(prepared.semantic_program),
        "shared_buffers": prepared.candidate.shared_buffers,
        "source_sha256": hashlib.sha256(prepared.source.encode()).hexdigest(),
    }


def _device_record() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda_runtime": torch.version.cuda,
        "device_name": properties.name,
        "torch_version": torch.__version__,
    }


def main() -> None:
    arguments = _arguments()
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda")
    if arguments.semantics == "normalized_exp":
        inputs = _normalized_exp_inputs(arguments, device)
    else:
        inputs = _weighted_sum_inputs(arguments, device)
    selected = arguments.candidate or list(CANDIDATES)
    # Compile every candidate before any warmup or timed execution. This keeps
    # extension compilation and first-use initialization outside the measured
    # order comparison.
    prepared = [_prepare_candidate(arguments, CANDIDATES[name], inputs) for name in selected]
    validations = {candidate.candidate.name: _validate_candidate(candidate) for candidate in prepared}
    _warm_candidates(prepared, arguments.warmups)
    samples, repeat_orders = _time_candidates(prepared, arguments.repeats)
    records = [
        _candidate_record(candidate, validations[candidate.candidate.name], samples[candidate.candidate.name])
        for candidate in prepared
    ]
    semantic_hashes = {record["semantic_sha256"] for record in records}
    if len(semantic_hashes) != 1:
        raise AssertionError("physical candidates changed the generic Fold semantics")
    semantic_programs = {json.dumps(candidate.semantic_program, sort_keys=True) for candidate in prepared}
    if len(semantic_programs) != 1:
        raise AssertionError("physical candidates changed the canonical generic Fold program")
    record = {
        "candidates": records,
        "command": sys.argv,
        "device": _device_record(),
        "features": arguments.features,
        "partials": arguments.partials,
        "reference_sha256": _tensor_sha256(inputs["expected"]),
        "repeat_orders": repeat_orders,
        "rows": arguments.rows,
        "seed": arguments.seed,
        "semantic_program": prepared[0].semantic_program,
        "semantics": arguments.semantics,
        "warmups_per_candidate": arguments.warmups,
    }
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
