# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark an experimental QuACK consumer-GEMM A-operand RMS scale."""

import argparse
import statistics
from collections.abc import Callable

import torch
from quack.epilogue import gemm_epilogue
from quack.operand_transform import a_transform, transform_a_operand

TensorFunction = Callable[[], torch.Tensor]


@a_transform(vec_size=8, args={"inverse_rms": "colvec_ktile"})
def _scale_a_by_bf16_inverse_rms(activation, inverse_rms):
    return activation * inverse_rms


@a_transform(vec_size=8, args={"inverse_rms": "colvec_ktile_fp32"})
def _scale_a_by_fp32_inverse_rms(activation, inverse_rms):
    return activation * inverse_rms


@gemm_epilogue()
def _identity_epilogue(acc):
    return {"D": acc}


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
    parser.add_argument("--shapes", type=_parse_shape, nargs="+", default=((2048, 4096, 6144),))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--pingpong", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__}")

    for m, k, n in args.shapes:
        if k % args.tile_k != 0:
            raise ValueError(f"K={k} must be divisible by the experimental tile K={args.tile_k}")
        activation = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        weight_nk = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        inverse_rms = torch.rand(m, dtype=torch.float32, device=device) * 1.5 + 0.25
        strict_input = (activation.float() * inverse_rms[:, None]).bfloat16()
        strict_reference = torch.mm(strict_input, weight_nk.mT)
        ideal_fp32_reference = torch.mm(activation.float() * inverse_rms[:, None], weight_nk.float().mT)

        # QuACK's existing strip operand is BF16 and carries one copy per K tile.
        # This isolates the cost of applying the scale in the RS WGMMA A-fragment
        # path. A follow-up backend change will carry the row vector in FP32.
        scale_strip = inverse_rms.bfloat16().repeat(k // args.tile_k, 1)
        bf16_activation_bundle = transform_a_operand(
            _scale_a_by_bf16_inverse_rms,
            activation,
            {"inverse_rms": scale_strip},
            args.tile_m,
            args.tile_k,
        )
        fp32_scale_strip = inverse_rms.repeat(k // args.tile_k, 1)
        fp32_activation_bundle = transform_a_operand(
            _scale_a_by_fp32_inverse_rms,
            activation,
            {"inverse_rms": fp32_scale_strip},
            args.tile_m,
            args.tile_k,
        )
        bf16_prologue_output = torch.empty(m, n, dtype=torch.bfloat16, device=device)
        fp32_prologue_output = torch.empty(m, n, dtype=torch.bfloat16, device=device)

        def fused_bf16_prologue(
            activation_arg=bf16_activation_bundle,
            weight_arg=weight_nk,
            output_arg=bf16_prologue_output,
        ) -> torch.Tensor:
            _identity_epilogue.gemm(
                activation_arg,
                weight_arg,
                output_arg,
                epi_args={},
                transform_a=_scale_a_by_bf16_inverse_rms,
                tile_M=args.tile_m,
                tile_N=args.tile_n,
                cluster_M=args.cluster_m,
                cluster_N=args.cluster_n,
                pingpong=args.pingpong,
            )
            return output_arg

        def fused_fp32_prologue(
            activation_arg=fp32_activation_bundle,
            weight_arg=weight_nk,
            output_arg=fp32_prologue_output,
        ) -> torch.Tensor:
            _identity_epilogue.gemm(
                activation_arg,
                weight_arg,
                output_arg,
                epi_args={},
                transform_a=_scale_a_by_fp32_inverse_rms,
                tile_M=args.tile_m,
                tile_N=args.tile_n,
                cluster_M=args.cluster_m,
                cluster_N=args.cluster_n,
                pingpong=args.pingpong,
            )
            return output_arg

        functions: list[tuple[str, TensorFunction]] = [
            ("raw_torch_gemm", lambda a=activation, b=weight_nk: torch.mm(a, b.mT)),
            ("quack_consumer_prologue_bf16_scale", fused_bf16_prologue),
            ("quack_consumer_prologue_fp32_scale", fused_fp32_prologue),
            (
                "materialized_bf16_prescale_then_gemm",
                lambda a=activation, b=weight_nk, r=inverse_rms: torch.mm((a.float() * r[:, None]).bfloat16(), b.mT),
            ),
        ]

        print(
            f"shape={m}x{k}x{n} tile={args.tile_m}x{args.tile_n}x{args.tile_k} "
            f"cluster={args.cluster_m}x{args.cluster_n} pingpong={args.pingpong}"
        )
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
