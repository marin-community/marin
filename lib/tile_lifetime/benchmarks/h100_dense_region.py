# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a hand-composed QuACK/CODA plus official-FA3 dense region."""

import argparse
import math
import statistics
from collections.abc import Callable

import torch
from flash_attn_interface import flash_attn_func
from quack.epilogue.library import rms_partial_epi, rstd_swiglu_epi, swiglu_mod
from quack.epilogue.rotary import make_interleaved_inv_freq, rope_posfreq_epi, rstd_rope_posfreq_epi
from quack.operand_transform import a_transform, transform_a_operand
from quack.rms_final_reduce import _rms_final_reduce_out

TensorFunction = Callable[[], object]
TILE_M = 128
TILE_N = 256
TILE_K = 64


@a_transform(vec_size=8, args={"inverse_rms": "colvec_ktile_fp32"})
def _scale_a_by_fp32_inverse_rms(activation, inverse_rms):
    return activation * inverse_rms


def _attention_output(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _benchmark_variants(
    variants: tuple[tuple[str, TensorFunction], ...],
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, tuple[float, float]]:
    for _ in range(warmups):
        for _, function in variants:
            function()
    torch.cuda.synchronize()

    samples = {name: [] for name, _ in variants}
    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        for name, function in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                function()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    return {name: (statistics.median(values), min(values)) for name, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14_336)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--profile-phases", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    if args.hidden % TILE_K:
        raise ValueError(f"hidden size must be divisible by {TILE_K}")
    tokens = args.batch * args.sequence
    query_width = args.query_heads * args.head_dimension
    key_value_width = args.kv_heads * args.head_dimension
    query_key_width = query_width + key_value_width
    qkv_width = query_width + 2 * key_value_width

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    print(f"torch={torch.__version__}")
    print(
        f"shape=B{args.batch} S{args.sequence} H{args.hidden} I{args.intermediate} "
        f"Hq{args.query_heads} Hkv{args.kv_heads} D{args.head_dimension}"
    )

    x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device=device)
    qkv_weight = torch.randn(qkv_width, args.hidden, dtype=torch.bfloat16, device=device) / math.sqrt(args.hidden)
    output_weight = torch.randn(args.hidden, args.hidden, dtype=torch.bfloat16, device=device) / math.sqrt(args.hidden)
    mlp_gamma = torch.randn(args.hidden, dtype=torch.bfloat16, device=device)
    gate_up_weight = torch.randn(2 * args.intermediate, args.hidden, dtype=torch.bfloat16, device=device) / math.sqrt(
        args.hidden
    )
    down_weight = torch.randn(args.hidden, args.intermediate, dtype=torch.bfloat16, device=device) / math.sqrt(
        args.intermediate
    )
    next_gamma = torch.randn(args.hidden, dtype=torch.bfloat16, device=device)
    next_qkv_weight = torch.randn(qkv_width, args.hidden, dtype=torch.bfloat16, device=device) / math.sqrt(args.hidden)

    positions = torch.arange(args.sequence, dtype=torch.float32, device=device).repeat(args.batch)
    base_frequency = 10_000.0 ** (
        -torch.arange(args.head_dimension // 2, dtype=torch.float64, device=device) / (args.head_dimension // 2)
    )
    frequency = make_interleaved_inv_freq(base_frequency, query_key_width, key_value_width)
    rope_args = {"pos": positions, "freq": frequency}

    packed_qkv = torch.empty(tokens, qkv_width, dtype=torch.bfloat16, device=device)
    gamma_scaled_mlp = torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device=device)
    x1 = torch.empty_like(gamma_scaled_mlp)
    mlp_partials = torch.empty(tokens, args.hidden // TILE_N, dtype=torch.float32, device=device)
    mlp_inverse_rms = torch.empty(tokens, dtype=torch.float32, device=device)
    mlp_scale_strip = torch.empty(args.hidden // TILE_K, tokens, dtype=torch.float32, device=device)
    activated = torch.empty(tokens, args.intermediate, dtype=torch.bfloat16, device=device)
    gamma_scaled_next = torch.empty_like(gamma_scaled_mlp)
    x2 = torch.empty_like(gamma_scaled_mlp)
    next_partials = torch.empty_like(mlp_partials)
    next_inverse_rms = torch.empty_like(mlp_inverse_rms)
    next_scale_strip = torch.empty_like(mlp_scale_strip)
    next_packed_qkv = torch.empty_like(packed_qkv)
    source_next_packed_qkv = torch.empty_like(packed_qkv)

    mlp_bundle = transform_a_operand(
        _scale_a_by_fp32_inverse_rms,
        gamma_scaled_mlp,
        {"inverse_rms": mlp_scale_strip},
        TILE_M,
        TILE_K,
    )
    next_bundle = transform_a_operand(
        _scale_a_by_fp32_inverse_rms,
        gamma_scaled_next,
        {"inverse_rms": next_scale_strip},
        TILE_M,
        TILE_K,
    )

    def qkv_attention() -> torch.Tensor:
        rope_posfreq_epi.gemm(
            x,
            qkv_weight,
            packed_qkv,
            epi_args=rope_args,
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=2,
            pingpong=False,
        )
        query = packed_qkv[:, :query_width].view(args.batch, args.sequence, args.query_heads, args.head_dimension)
        key = packed_qkv[:, query_width:query_key_width].view(
            args.batch, args.sequence, args.kv_heads, args.head_dimension
        )
        value = packed_qkv[:, query_key_width:].view(args.batch, args.sequence, args.kv_heads, args.head_dimension)
        return _attention_output(flash_attn_func(query, key, value, causal=True)).view(tokens, args.hidden)

    def output_projection_and_rms(attention: torch.Tensor) -> None:
        rms_partial_epi.gemm(
            attention,
            output_weight,
            gamma_scaled_mlp,
            x,
            epi_args={"weight": mlp_gamma, "resid_out": x1, "sqsum": mlp_partials},
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=1,
            pingpong=False,
        )
        _rms_final_reduce_out(mlp_partials, mlp_inverse_rms, 1.0 / args.hidden, args.epsilon)

    def down_projection_and_rms() -> None:
        rms_partial_epi.gemm(
            activated,
            down_weight,
            gamma_scaled_next,
            x1,
            epi_args={"weight": next_gamma, "resid_out": x2, "sqsum": next_partials},
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=1,
            pingpong=False,
        )
        _rms_final_reduce_out(next_partials, next_inverse_rms, 1.0 / args.hidden, args.epsilon)

    def copy_mlp_scale_strip() -> None:
        mlp_scale_strip.copy_(mlp_inverse_rms[None, :])

    def prologue_gate_up_kernel() -> None:
        swiglu_mod.gemm(
            mlp_bundle,
            gate_up_weight,
            None,
            epi_args={"postact": activated},
            transform_a=_scale_a_by_fp32_inverse_rms,
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=2,
            pingpong=False,
        )

    def prologue_gate_up() -> None:
        copy_mlp_scale_strip()
        prologue_gate_up_kernel()

    def delayed_gate_up() -> None:
        rstd_swiglu_epi.gemm(
            gamma_scaled_mlp,
            gate_up_weight,
            None,
            epi_args={"rstd": mlp_inverse_rms, "postact": activated},
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=2,
            pingpong=False,
        )

    def copy_next_scale_strip() -> None:
        next_scale_strip.copy_(next_inverse_rms[None, :])

    def prologue_next_qkv_kernel() -> None:
        rope_posfreq_epi.gemm(
            next_bundle,
            next_qkv_weight,
            next_packed_qkv,
            epi_args=rope_args,
            transform_a=_scale_a_by_fp32_inverse_rms,
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=2,
            pingpong=False,
        )

    def prologue_next_qkv() -> None:
        copy_next_scale_strip()
        prologue_next_qkv_kernel()

    def delayed_next_qkv() -> None:
        rstd_rope_posfreq_epi.gemm(
            gamma_scaled_next,
            next_qkv_weight,
            next_packed_qkv,
            epi_args={**rope_args, "rstd": next_inverse_rms},
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=2,
            pingpong=False,
        )

    def prologue_region() -> tuple[torch.Tensor, torch.Tensor]:
        attention = qkv_attention()
        output_projection_and_rms(attention)
        prologue_gate_up()
        down_projection_and_rms()
        prologue_next_qkv()
        return x2, next_packed_qkv

    def delayed_region() -> tuple[torch.Tensor, torch.Tensor]:
        attention = qkv_attention()
        output_projection_and_rms(attention)
        delayed_gate_up()
        down_projection_and_rms()
        delayed_next_qkv()
        return x2, next_packed_qkv

    def materialized_region() -> tuple[torch.Tensor, torch.Tensor]:
        attention = qkv_attention()
        projected = torch.mm(attention, output_weight.mT)
        source_x1 = projected + x
        source_x1_fp32 = source_x1.float()
        source_mlp_inverse_rms = torch.rsqrt(source_x1_fp32.square().mean(-1, keepdim=True) + args.epsilon)
        source_mlp_input = (source_x1_fp32 * mlp_gamma * source_mlp_inverse_rms).bfloat16()
        source_gate_up = torch.mm(source_mlp_input, gate_up_weight.mT).unflatten(-1, (args.intermediate, 2))
        source_activated = torch.nn.functional.silu(source_gate_up[..., 0]) * source_gate_up[..., 1]
        source_down = torch.mm(source_activated, down_weight.mT)
        source_x2 = source_down + source_x1
        source_x2_fp32 = source_x2.float()
        source_next_inverse_rms = torch.rsqrt(source_x2_fp32.square().mean(-1, keepdim=True) + args.epsilon)
        source_next_input = (source_x2_fp32 * next_gamma * source_next_inverse_rms).bfloat16()
        rope_posfreq_epi.gemm(
            source_next_input,
            next_qkv_weight,
            source_next_packed_qkv,
            epi_args=rope_args,
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=1,
            cluster_N=2,
            pingpong=False,
        )
        return source_x2, source_next_packed_qkv

    prologue_x2, prologue_qkv = (value.clone() for value in prologue_region())
    delayed_x2, delayed_qkv = (value.clone() for value in delayed_region())
    source_x2, source_qkv = materialized_region()
    x2_difference = (prologue_x2.float() - delayed_x2.float()).abs()
    qkv_difference = (prologue_qkv.float() - delayed_qkv.float()).abs()
    print(
        f"prologue_vs_delayed x2_max_abs={x2_difference.max().item():.6f} "
        f"x2_mean_abs={x2_difference.mean().item():.6f} "
        f"next_qkv_max_abs={qkv_difference.max().item():.6f} "
        f"next_qkv_mean_abs={qkv_difference.mean().item():.6f}"
    )
    for name, x2_output, qkv_output in (
        ("consumer_prologue", prologue_x2, prologue_qkv),
        ("delayed_epilogue", delayed_x2, delayed_qkv),
    ):
        x2_source_difference = (x2_output.float() - source_x2.float()).abs()
        qkv_source_difference = (qkv_output.float() - source_qkv.float()).abs()
        print(
            f"{name}_vs_materialized x2_max_abs={x2_source_difference.max().item():.6f} "
            f"x2_mean_abs={x2_source_difference.mean().item():.6f} "
            f"next_qkv_max_abs={qkv_source_difference.max().item():.6f} "
            f"next_qkv_mean_abs={qkv_source_difference.mean().item():.6f}"
        )
    variants = (
        ("consumer_prologue", prologue_region),
        ("delayed_epilogue", delayed_region),
        ("materialized_torch", materialized_region),
    )
    measurements = _benchmark_variants(
        variants,
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    for name, _ in variants:
        median_ms, minimum_ms = measurements[name]
        print(f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f}")

    if args.profile_phases:
        attention_for_profile = qkv_attention()
        output_projection_and_rms(attention_for_profile)
        prologue_gate_up()
        down_projection_and_rms()
        prologue_next_qkv()
        phase_variants = (
            ("qkv_rope_attention", qkv_attention),
            ("output_projection_rms", lambda: output_projection_and_rms(attention_for_profile)),
            ("mlp_scale_strip_copy", copy_mlp_scale_strip),
            ("gate_up_prologue_kernel", prologue_gate_up_kernel),
            ("gate_up_prologue", prologue_gate_up),
            ("gate_up_delayed", delayed_gate_up),
            ("down_projection_rms", down_projection_and_rms),
            ("next_scale_strip_copy", copy_next_scale_strip),
            ("next_qkv_prologue_kernel", prologue_next_qkv_kernel),
            ("next_qkv_prologue", prologue_next_qkv),
            ("next_qkv_delayed", delayed_next_qkv),
        )
        phase_measurements = _benchmark_variants(
            phase_variants,
            warmups=args.warmups,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        for name, _ in phase_variants:
            median_ms, minimum_ms = phase_measurements[name]
            print(f"  phase_{name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f}")


if __name__ == "__main__":
    main()
