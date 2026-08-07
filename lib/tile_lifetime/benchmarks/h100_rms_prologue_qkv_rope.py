# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark source-ordered RMS scaling composed with a packed QKV/RoPE GEMM."""

import argparse
import math
import statistics
from collections.abc import Callable

import torch
from quack.epilogue.rotary import make_interleaved_inv_freq, rope_posfreq_epi, rstd_rope_posfreq_epi
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


def _rope_reference(
    projected: torch.Tensor,
    *,
    positions: torch.Tensor,
    inverse_frequency: torch.Tensor,
    query_key_width: int,
    head_dimension: int,
) -> torch.Tensor:
    query_key = (
        projected[:, :query_key_width].float().unflatten(-1, (query_key_width // head_dimension, head_dimension // 2, 2))
    )
    angle = positions.double()[:, None] * inverse_frequency[None, :]
    cosine = angle.cos().float()[:, None, :]
    sine = angle.sin().float()[:, None, :]
    rotated = torch.empty_like(query_key)
    rotated[..., 0] = query_key[..., 0] * cosine - query_key[..., 1] * sine
    rotated[..., 1] = query_key[..., 0] * sine + query_key[..., 1] * cosine
    return torch.cat((rotated.flatten(-3), projected[:, query_key_width:].float()), dim=-1).bfloat16()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    if args.k % TILE_K:
        raise ValueError(f"K={args.k} must be divisible by {TILE_K}")
    if args.head_dimension % 2:
        raise ValueError("RoPE head dimension must be even")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__}")

    query_key_width = (args.query_heads + args.kv_heads) * args.head_dimension
    value_width = args.kv_heads * args.head_dimension
    output_width = query_key_width + value_width
    activation = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device)
    weight_nk = torch.randn(output_width, args.k, dtype=torch.bfloat16, device=device) / math.sqrt(args.k)
    inverse_rms = torch.rand(args.m, dtype=torch.float32, device=device) * 1.5 + 0.25
    positions = torch.arange(args.m, dtype=torch.float32, device=device)
    base_frequency = 10_000.0 ** (
        -torch.arange(args.head_dimension // 2, dtype=torch.float64, device=device) / (args.head_dimension // 2)
    )
    frequency = make_interleaved_inv_freq(base_frequency, query_key_width, value_width)
    epilogue_args = {"pos": positions, "freq": frequency}

    strict_input = (activation.float() * inverse_rms[:, None]).bfloat16()
    strict_projection = torch.mm(strict_input, weight_nk.mT)
    strict_reference = _rope_reference(
        strict_projection,
        positions=positions,
        inverse_frequency=base_frequency,
        query_key_width=query_key_width,
        head_dimension=args.head_dimension,
    )

    scale_strip = inverse_rms.repeat(args.k // TILE_K, 1)
    activation_bundle = transform_a_operand(
        _scale_a_by_fp32_inverse_rms,
        activation,
        {"inverse_rms": scale_strip},
        TILE_M,
        TILE_K,
    )
    prologue_output = torch.empty(args.m, output_width, dtype=torch.bfloat16, device=device)
    delayed_output = torch.empty_like(prologue_output)
    materialized_output = torch.empty_like(prologue_output)

    def prologue_qkv_rope(
        activation_arg=activation_bundle,
        weight_arg=weight_nk,
        output_arg=prologue_output,
    ) -> torch.Tensor:
        rope_posfreq_epi.gemm(
            activation_arg,
            weight_arg,
            output_arg,
            epi_args=epilogue_args,
            transform_a=_scale_a_by_fp32_inverse_rms,
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=args.cluster_n,
            pingpong=False,
        )
        return output_arg

    def delayed_qkv_rope(
        activation_arg=activation,
        weight_arg=weight_nk,
        output_arg=delayed_output,
    ) -> torch.Tensor:
        rstd_rope_posfreq_epi.gemm(
            activation_arg,
            weight_arg,
            output_arg,
            epi_args={**epilogue_args, "rstd": inverse_rms},
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=args.cluster_n,
            pingpong=False,
        )
        return output_arg

    def materialized_qkv_rope(
        input_arg=activation,
        weight_arg=weight_nk,
        output_arg=materialized_output,
    ) -> torch.Tensor:
        scaled_input = (input_arg.float() * inverse_rms[:, None]).bfloat16()
        rope_posfreq_epi.gemm(
            scaled_input,
            weight_arg,
            output_arg,
            epi_args=epilogue_args,
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=args.cluster_n,
            pingpong=False,
        )
        return output_arg

    functions: list[tuple[str, TensorFunction]] = [
        ("quack_fp32_prologue_qkv_rope", prologue_qkv_rope),
        ("coda_delayed_rms_qkv_rope", delayed_qkv_rope),
        ("materialized_bf16_prescale_qkv_rope", materialized_qkv_rope),
    ]

    print(
        f"shape={args.m}x{args.k}x{output_width} head_dim={args.head_dimension} "
        f"tile={TILE_M}x{TILE_N}x{TILE_K} cluster=1x{args.cluster_n}"
    )
    boundary_reference = materialized_qkv_rope().clone()
    for name, function in functions:
        actual = function()
        source_difference = (actual.float() - strict_reference.float()).abs()
        boundary_difference = (actual.float() - boundary_reference.float()).abs()
        median_ms, minimum_ms = _benchmark(
            function,
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        gemm_tflops = 2 * args.m * output_width * args.k / (median_ms * 1e9)
        print(
            f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f} "
            f"gemm_tflops={gemm_tflops:.1f} "
            f"max_abs_vs_materialized={boundary_difference.max().item():.6f} "
            f"mean_abs_vs_materialized={boundary_difference.mean().item():.6f} "
            f"max_abs_vs_source={source_difference.max().item():.6f} "
            f"mean_abs_vs_source={source_difference.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
