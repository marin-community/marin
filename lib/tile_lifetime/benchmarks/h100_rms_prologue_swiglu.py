# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the composed FP32 RMS prologue and dead-output SwiGLU epilogue."""

import argparse
import statistics
from collections.abc import Callable

import torch
from quack.epilogue.library import rstd_swiglu_epi, swiglu_mod
from quack.operand_transform import a_transform, transform_a_operand

TensorFunction = Callable[[], torch.Tensor]
TILE_M = 128
TILE_N = 256
TILE_K = 64


@a_transform(vec_size=8, args={"inverse_rms": "colvec_ktile_fp32"})
def _scale_a_by_fp32_inverse_rms(activation, inverse_rms):
    return activation * inverse_rms


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


def _torch_swiglu(activation: torch.Tensor, weight_nk: torch.Tensor) -> torch.Tensor:
    preactivation = torch.mm(activation, weight_nk.mT)
    pairs = preactivation.reshape(*preactivation.shape[:-1], preactivation.shape[-1] // 2, 2)
    return torch.nn.functional.silu(pairs[..., 0]) * pairs[..., 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=28_672)
    parser.add_argument("--cluster-n", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    if args.k % TILE_K:
        raise ValueError(f"K={args.k} must be divisible by {TILE_K}")
    if args.n % 2:
        raise ValueError("SwiGLU projection width must be even")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__}")

    activation = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device)
    weight_nk = torch.randn(args.n, args.k, dtype=torch.bfloat16, device=device) / args.k**0.5
    inverse_rms = torch.rand(args.m, dtype=torch.float32, device=device) * 1.5 + 0.25
    strict_input = (activation.float() * inverse_rms[:, None]).bfloat16()
    strict_reference = _torch_swiglu(strict_input, weight_nk)

    scale_strip = inverse_rms.repeat(args.k // TILE_K, 1)
    activation_bundle = transform_a_operand(
        _scale_a_by_fp32_inverse_rms,
        activation,
        {"inverse_rms": scale_strip},
        TILE_M,
        TILE_K,
    )
    fused_output = torch.empty(args.m, args.n // 2, dtype=torch.bfloat16, device=device)
    delayed_output = torch.empty_like(fused_output)

    def fused_prologue_swiglu(
        activation_arg=activation_bundle,
        weight_arg=weight_nk,
        output_arg=fused_output,
    ) -> torch.Tensor:
        swiglu_mod.gemm(
            activation_arg,
            weight_arg,
            None,
            epi_args={"postact": output_arg},
            transform_a=_scale_a_by_fp32_inverse_rms,
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=args.cluster_n,
            pingpong=False,
        )
        return output_arg

    def delayed_rms_swiglu(
        activation_arg=activation,
        weight_arg=weight_nk,
        output_arg=delayed_output,
    ) -> torch.Tensor:
        rstd_swiglu_epi.gemm(
            activation_arg,
            weight_arg,
            None,
            epi_args={"rstd": inverse_rms, "postact": output_arg},
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=args.cluster_n,
            pingpong=False,
        )
        return output_arg

    functions: list[tuple[str, TensorFunction]] = [
        ("quack_fp32_prologue_dead_preact_swiglu", fused_prologue_swiglu),
        ("coda_delayed_rms_dead_preact_swiglu", delayed_rms_swiglu),
        (
            "materialized_bf16_prescale_gemm_swiglu",
            lambda a=activation, b=weight_nk, r=inverse_rms: _torch_swiglu((a.float() * r[:, None]).bfloat16(), b),
        ),
    ]

    print(f"shape={args.m}x{args.k}x{args.n} tile={TILE_M}x{TILE_N}x{TILE_K} " f"cluster=1x{args.cluster_n}")
    for name, function in functions:
        actual = function()
        difference = (actual.float() - strict_reference.float()).abs()
        median_ms, minimum_ms = _benchmark(
            function,
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        gemm_tflops = 2 * args.m * args.n * args.k / (median_ms * 1e9)
        print(
            f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f} "
            f"gemm_tflops={gemm_tflops:.1f} max_abs={difference.max().item():.6f} "
            f"mean_abs={difference.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
