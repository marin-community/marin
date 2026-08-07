# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark official FA3, FA4 CuTe, and Torch SDPA on one H100."""

import argparse
import importlib.metadata
import statistics
from collections.abc import Callable

import torch

try:
    from flash_attn.cute import flash_attn_func as fa4_attention
except ImportError:
    fa4_attention = None

try:
    from flash_attn_interface import flash_attn_func as fa3_attention
except ImportError:
    fa3_attention = None


TensorFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]]


def _output(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _benchmark(
    function: TensorFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> tuple[float, float]:
    for _ in range(warmups):
        _output(function(q, k, v))
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            _output(function(q, k, v))
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples), min(samples)


def _torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
        enable_gqa=True,
    ).transpose(1, 2)


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if fa4_attention is not None:
        return _output(fa4_attention(q, k, v, causal=True))
    return _torch_sdpa(q, k, v)


def _backends() -> list[tuple[str, TensorFunction]]:
    backends: list[tuple[str, TensorFunction]] = [("torch_sdpa", _torch_sdpa)]
    if fa4_attention is not None:
        backends.insert(0, ("fa4_cute_sm90", lambda q, k, v: fa4_attention(q, k, v, causal=True)))
    if fa3_attention is not None:
        backends.insert(
            0,
            ("official_fa3", lambda q, k, v: fa3_attention(q, k, v, causal=True, pack_gqa=True)),
        )
    return backends


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, nargs="+", default=(2048, 4096))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"gpu={properties.name} capability={properties.major}.{properties.minor}")
    for package in ("torch", "flash-attn-4", "flash-attn-3", "nvidia-cutlass-dsl"):
        try:
            print(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}=not-installed")

    batch = 1
    query_heads = 32
    key_value_heads = 8
    head_dimension = 128
    for sequence in args.sequences:
        q = torch.randn(
            batch,
            sequence,
            query_heads,
            head_dimension,
            dtype=torch.bfloat16,
            device=device,
        )
        k = torch.randn(
            batch,
            sequence,
            key_value_heads,
            head_dimension,
            dtype=torch.bfloat16,
            device=device,
        )
        v = torch.randn_like(k)
        reference = _reference(q, k, v)

        print(f"sequence={sequence}")
        for name, function in _backends():
            actual = _output(function(q, k, v))
            difference = (actual.float() - reference.float()).abs()
            median_ms, minimum_ms = _benchmark(
                function,
                q,
                k,
                v,
                warmups=args.warmups,
                repeats=args.repeats,
                iterations=args.iterations,
            )
            causal_flops = 2 * batch * query_heads * sequence * sequence * head_dimension
            tflops = causal_flops / (median_ms * 1e9)
            print(
                f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f} "
                f"causal_tflops={tflops:.1f} max_abs={difference.max().item():.6f} "
                f"mean_abs={difference.mean().item():.6f}"
            )


if __name__ == "__main__":
    main()
