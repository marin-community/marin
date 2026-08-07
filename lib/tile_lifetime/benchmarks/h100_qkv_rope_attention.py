# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the packed QKV/RoPE output contract into official FA3."""

import argparse
import math
import statistics
from collections.abc import Callable

import torch
from flash_attn_interface import flash_attn_func
from quack.epilogue.rotary import make_interleaved_inv_freq, rope_posfreq_epi

TensorFunction = Callable[[], torch.Tensor]


def _attention_output(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    tokens = args.batch * args.sequence
    query_width = args.query_heads * args.head_dimension
    key_value_width = args.kv_heads * args.head_dimension
    query_key_width = query_width + key_value_width
    output_width = query_width + 2 * key_value_width

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__}")

    activation = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device=device)
    weight_nk = torch.randn(output_width, args.hidden, dtype=torch.bfloat16, device=device) / math.sqrt(args.hidden)
    positions = torch.arange(args.sequence, dtype=torch.float32, device=device).repeat(args.batch)
    base_frequency = 10_000.0 ** (
        -torch.arange(args.head_dimension // 2, dtype=torch.float64, device=device) / (args.head_dimension // 2)
    )
    frequency = make_interleaved_inv_freq(base_frequency, query_key_width, key_value_width)
    packed_qkv = torch.empty(tokens, output_width, dtype=torch.bfloat16, device=device)

    def project_qkv_rope() -> torch.Tensor:
        rope_posfreq_epi.gemm(
            activation,
            weight_nk,
            packed_qkv,
            epi_args={"pos": positions, "freq": frequency},
            tile_M=128,
            tile_N=256,
            cluster_M=1,
            cluster_N=2,
            pingpong=False,
        )
        return packed_qkv

    def qkv_views() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = packed_qkv[:, :query_width].view(args.batch, args.sequence, args.query_heads, args.head_dimension)
        key = packed_qkv[:, query_width:query_key_width].view(
            args.batch, args.sequence, args.kv_heads, args.head_dimension
        )
        value = packed_qkv[:, query_key_width:].view(args.batch, args.sequence, args.kv_heads, args.head_dimension)
        return query, key, value

    def packed_boundary() -> torch.Tensor:
        project_qkv_rope()
        query, key, value = qkv_views()
        return _attention_output(flash_attn_func(query, key, value, causal=True))

    def repacked_boundary() -> torch.Tensor:
        project_qkv_rope()
        query, key, value = qkv_views()
        return _attention_output(flash_attn_func(query.contiguous(), key.contiguous(), value.contiguous(), causal=True))

    packed = packed_boundary()
    repacked = repacked_boundary()
    difference = (packed.float() - repacked.float()).abs()
    query, key, value = qkv_views()
    print(f"packed_strides=q{query.stride()} k{key.stride()} v{value.stride()}")
    print(f"packed_vs_repacked max_abs={difference.max().item():.6f} " f"mean_abs={difference.mean().item():.6f}")
    for name, function in (("packed_segment_views", packed_boundary), ("explicit_contiguous_repack", repacked_boundary)):
        median_ms, minimum_ms = _benchmark(
            function,
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        print(f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f}")


if __name__ == "__main__":
    main()
