# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual GB200 smoke for generic dense and indexed Fold finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from clean_routed_streaming_emitter import (  # noqa: E402
    render_partial_merge_cuda,
    tiled_fold_merge_program,
)
from clean_routed_streaming_runtime import compile_tiled_fold_finalize  # noqa: E402

from tile_lifetime import DType  # noqa: E402
from tile_lifetime.tensor_program import TensorAxis  # noqa: E402
from tile_lifetime.tiled_fold_finalize import (  # noqa: E402
    FoldFeatureLayout,
    FoldPartialAddressing,
    FoldPhysicalAxis,
    TiledFoldAxes,
    TiledFoldFinalizeSchedule,
    TiledFoldInputLayout,
    deterministic_weighted_sum_fold_program,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--source-rows", type=int, default=8192)
    parser.add_argument("--partials", type=int, default=6)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _program(arguments: argparse.Namespace):
    axes = TiledFoldAxes(
        partial=TensorAxis(0, arguments.partials, "route_slot"),
        row=TensorAxis(1, arguments.rows, "destination_row"),
        feature=TensorAxis(2, arguments.features, "feature"),
    )
    schedule = TiledFoldFinalizeSchedule(
        axes=axes,
        partial_addressing=FoldPartialAddressing.INDEXED,
        row_tile=8,
        feature_tile=128,
        vector_bytes=16,
        shared_stages=4,
        threads=256,
        partial_lanes=1,
        shared_buffers=2,
        input_layout=TiledFoldInputLayout(
            addressing=FoldPartialAddressing.INDEXED,
            value_axis_order=(FoldPhysicalAxis.SOURCE, FoldPhysicalAxis.FEATURE),
            scalar_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
            index_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
            feature_layout=FoldFeatureLayout.CONTIGUOUS,
        ),
    )
    return tiled_fold_merge_program(
        deterministic_weighted_sum_fold_program(
            schedule,
            partial_value_dtype=DType.BF16,
            output_dtype=DType.BF16,
        )
    )


def _reference(values: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(
        indices.shape[0],
        values.shape[1],
        dtype=torch.float32,
        device=values.device,
    )
    for partial in range(indices.shape[1]):
        source = indices[:, partial]
        valid = source >= 0
        contribution = torch.zeros_like(result)
        contribution[valid] = values[source[valid]].float() * weights[valid, partial, None]
        result = result + contribution
    return result.to(torch.bfloat16)


def main() -> None:
    arguments = _arguments()
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda")
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
    # Exercise arbitrary non-prefix validity rather than an attention-style
    # count. Every third row has an invalid interior route slot.
    indices[::3, 1] = -1
    weights = torch.randn(
        arguments.rows,
        arguments.partials,
        dtype=torch.float32,
        device=device,
    )
    program = _program(arguments)
    module = compile_tiled_fold_finalize(program, build_directory=arguments.build_directory)
    source = render_partial_merge_cuda(program)

    observed = module.merge(weights, values, indices, 1)
    expected = _reference(values, indices, weights)
    torch.cuda.synchronize()
    difference = (observed.float() - expected.float()).abs()
    repeated = module.merge(weights, values, indices, 1)
    deterministic = bool(torch.equal(observed, repeated))

    for _ in range(arguments.warmups):
        module.merge(weights, values, indices, 1)
    torch.cuda.synchronize()
    samples = []
    for _ in range(arguments.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.merge(weights, values, indices, 1)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    record = {
        "deterministic": deterministic,
        "features": arguments.features,
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "median_ms": float(np.median(samples)),
        "non_prefix_validity": True,
        "partials": arguments.partials,
        "rows": arguments.rows,
        "samples_ms": samples,
        "source_rows": arguments.source_rows,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    rendered = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
