# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CODA and QuACK pairwise SwiGLU epilogues on H100."""

import argparse
import statistics
from collections.abc import Callable

import torch
from coda.core.gemm.functional import (
    gemm_rmsnorm_swiglu as coda_gemm_rmsnorm_swiglu,
)
from coda.core.gemm.functional import (
    gemm_swiglu as coda_gemm_swiglu,
)
from quack.gemm_interface import gemm_act as quack_gemm_act

TensorFunction = Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]


def _output(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return value[-1] if isinstance(value, tuple) else value


def _benchmark(
    function: TensorFunction,
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmups):
        _output(function())
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            _output(function())
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples), min(samples)


def _parse_shape(value: str) -> tuple[int, int, int]:
    dimensions = tuple(int(dimension) for dimension in value.split("x"))
    if len(dimensions) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    return dimensions


def _torch_swiglu(activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    preactivation = torch.mm(activation, weight)
    pairs = preactivation.reshape(*preactivation.shape[:-1], preactivation.shape[-1] // 2, 2)
    return torch.nn.functional.silu(pairs[..., 0]) * pairs[..., 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=_parse_shape, nargs="+", default=((2048, 4096, 28_672),))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__}")

    for m, k, n in args.shapes:
        if n % 2:
            raise ValueError("SwiGLU projection width must be even")
        activation = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        weight = torch.randn(k, n, dtype=torch.bfloat16, device=device) / k**0.5
        reference = _torch_swiglu(activation, weight)
        inverse_rms = torch.rand(m, dtype=torch.float32, device=device) * 1.5 + 0.25
        source_ordered_rms_reference = _torch_swiglu(
            (activation.float() * inverse_rms[:, None]).bfloat16(),
            weight,
        )
        ideal_preactivation = activation.float() @ weight.float()
        ideal_pairs = ideal_preactivation.reshape(m, n // 2, 2)
        ideal_reference = torch.nn.functional.silu(ideal_pairs[..., 0]) * ideal_pairs[..., 1]

        functions: list[tuple[str, TensorFunction, torch.Tensor]] = [
            ("torch_materialized_swiglu", lambda a=activation, b=weight: _torch_swiglu(a, b), reference),
            ("coda_swiglu_store_preact", lambda a=activation, b=weight: coda_gemm_swiglu(a, b), reference),
            (
                "quack_swiglu_dead_preact",
                lambda a=activation, b=weight: quack_gemm_act(
                    a,
                    b,
                    activation="swiglu",
                    store_preact=False,
                    tuned=True,
                ),
                reference,
            ),
            (
                "coda_delayed_rms_swiglu_store_preact",
                lambda a=activation, b=weight, r=inverse_rms: coda_gemm_rmsnorm_swiglu(a, b, r),
                source_ordered_rms_reference,
            ),
        ]

        print(f"shape={m}x{k}x{n}")
        for name, function, function_reference in functions:
            actual = _output(function())
            difference = (actual.float() - function_reference.float()).abs()
            ideal_difference = (
                (actual.float() - ideal_reference).abs()
                if function_reference is reference
                else torch.full((), float("nan"), device=device)
            )
            median_ms, minimum_ms = _benchmark(
                function,
                warmups=args.warmups,
                repeats=args.repeats,
                iterations=args.iterations,
            )
            gemm_tflops = 2 * m * n * k / (median_ms * 1e9)
            print(
                f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f} "
                f"gemm_tflops={gemm_tflops:.1f} max_abs={difference.max().item():.6f} "
                f"mean_abs={difference.mean().item():.6f} "
                f"max_abs_vs_ideal_fp32={ideal_difference.max().item():.6f} "
                f"mean_abs_vs_ideal_fp32={ideal_difference.mean().item():.6f}"
            )


if __name__ == "__main__":
    main()
