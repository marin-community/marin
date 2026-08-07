# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CODA delayed RMS scaling and source-ordered pre-scaling."""

import argparse
import statistics
from collections.abc import Callable

import torch

try:
    from coda.core.gemm.functional import gemm_rmsnorm as coda_gemm_rmsnorm
except ImportError:
    coda_gemm_rmsnorm = None


TensorFunction = Callable[[], torch.Tensor]


def _benchmark(
    function: TensorFunction,
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> tuple[float, float]:
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
    return statistics.median(samples), min(samples)


def _parse_shape(value: str) -> tuple[int, int, int]:
    dimensions = tuple(int(dimension) for dimension in value.split("x"))
    if len(dimensions) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    return dimensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=_parse_shape,
        nargs="+",
        default=((128, 512, 2816), (2048, 4096, 6144)),
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__} coda_available={coda_gemm_rmsnorm is not None}")

    for m, k, n in args.shapes:
        activation = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        weight = torch.randn(k, n, dtype=torch.bfloat16, device=device)
        inverse_rms = torch.rand(m, dtype=torch.float32, device=device) * 1.5 + 0.25
        strict_input = (activation.float() * inverse_rms[:, None]).bfloat16()
        strict_reference = torch.mm(strict_input, weight)
        ideal_fp32_reference = torch.mm(activation.float() * inverse_rms[:, None], weight.float())

        functions: list[tuple[str, TensorFunction]] = [
            ("raw_torch_gemm", lambda a=activation, b=weight: torch.mm(a, b)),
            (
                "materialized_bf16_prescale_then_gemm",
                lambda a=activation, b=weight, r=inverse_rms: torch.mm((a.float() * r[:, None]).bfloat16(), b),
            ),
            (
                "torch_postscale",
                lambda a=activation, b=weight, r=inverse_rms: (torch.mm(a, b).float() * r[:, None]).bfloat16(),
            ),
        ]
        if coda_gemm_rmsnorm is not None:
            functions.insert(
                1,
                (
                    "coda_consumer_epilogue",
                    lambda a=activation, b=weight, r=inverse_rms: coda_gemm_rmsnorm(a, b, r),
                ),
            )

        print(f"shape={m}x{k}x{n}")
        for name, function in functions:
            actual = function()
            difference = (actual.float() - strict_reference.float()).abs()
            ideal_difference = (actual.float() - ideal_fp32_reference).abs()
            median_ms, minimum_ms = _benchmark(
                function,
                warmups=args.warmups,
                repeats=args.repeats,
                iterations=args.iterations,
            )
            tflops = 2 * m * n * k / (median_ms * 1e9)
            print(
                f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f} "
                f"tflops={tflops:.1f} max_abs_vs_strict={difference.max().item():.6f} "
                f"mean_abs_vs_strict={difference.mean().item():.6f} "
                f"max_abs_vs_ideal_fp32={ideal_difference.max().item():.6f} "
                f"mean_abs_vs_ideal_fp32={ideal_difference.mean().item():.6f}"
            )


if __name__ == "__main__":
    main()
