# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune the CODA residual/gamma/RMS-partial projection boundary."""

import argparse
import math
import statistics
from collections.abc import Callable

import torch
from quack.epilogue.library import rms_partial_epi
from quack.rms_final_reduce import _rms_final_reduce_out

TensorFunction = Callable[[], object]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--pingpong", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    if args.n % args.tile_n:
        raise ValueError("N must be divisible by tile N for the RMS-partial buffer")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    activation = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device)
    weight_nk = torch.randn(args.n, args.k, dtype=torch.bfloat16, device=device) / math.sqrt(args.k)
    residual = torch.randn(args.m, args.n, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(args.n, dtype=torch.float32, device=device)
    scaled = torch.empty(args.m, args.n, dtype=torch.bfloat16, device=device)
    residual_out = torch.empty_like(scaled)
    partials = torch.empty(args.m, args.n // args.tile_n, dtype=torch.float32, device=device)
    inverse_rms = torch.empty(args.m, dtype=torch.float32, device=device)

    def gemm_partials() -> None:
        rms_partial_epi.gemm(
            activation,
            weight_nk,
            scaled,
            residual,
            epi_args={"weight": gamma, "resid_out": residual_out, "sqsum": partials},
            tile_M=args.tile_m,
            tile_N=args.tile_n,
            cluster_M=1,
            cluster_N=args.cluster_n,
            pingpong=args.pingpong,
        )

    def reduce_partials() -> None:
        _rms_final_reduce_out(partials, inverse_rms, 1.0 / args.n, 1e-6)

    def combined() -> None:
        gemm_partials()
        reduce_partials()

    combined()
    expected_residual = torch.mm(activation.float(), weight_nk.float().mT) + residual.float()
    expected_scaled = (expected_residual * gamma).bfloat16()
    scaled_difference = (scaled.float() - expected_scaled.float()).abs()
    expected_inverse_rms = torch.rsqrt(expected_residual.square().mean(-1) + 1e-6)
    inverse_rms_difference = (inverse_rms - expected_inverse_rms).abs()
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__}")
    print(
        f"shape={args.m}x{args.k}x{args.n} tile={args.tile_m}x{args.tile_n}x64 "
        f"cluster=1x{args.cluster_n} pingpong={args.pingpong}"
    )
    print(
        f"scaled_max_abs={scaled_difference.max().item():.6f} "
        f"scaled_mean_abs={scaled_difference.mean().item():.6f} "
        f"inverse_rms_max_abs={inverse_rms_difference.max().item():.6f} "
        f"inverse_rms_mean_abs={inverse_rms_difference.mean().item():.6f}"
    )
    for name, function in (
        ("gemm_partials", gemm_partials),
        ("reduce_partials", reduce_partials),
        ("combined", combined),
    ):
        median_ms, minimum_ms = _benchmark(
            function,
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        print(f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f}")


if __name__ == "__main__":
    main()
