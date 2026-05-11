# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for SonicMoE side-by-side token gather/sum comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

DTypeName = Literal["bf16", "fp32"]


@dataclass(frozen=True, slots=True)
class TokenGatherSumConfig:
    """Common fixed-top-k token gather/sum benchmark shape."""

    tokens: int = 8192
    hidden: int = 2048
    experts: int = 8
    topk: int = 2
    dtype: DTypeName = "bf16"
    weighted: bool = False
    kernel_repeat: int = 1
    replicate_input: bool = False
    warmup: int = 5
    steps: int = 20
    seed: int = 0

    @property
    def assignments(self) -> int:
        return self.tokens * self.topk

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--kernel-repeat", type=int, default=1)
    parser.add_argument("--replicate-input", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)


def config_from_args(args: argparse.Namespace) -> TokenGatherSumConfig:
    return TokenGatherSumConfig(
        tokens=args.tokens,
        hidden=args.hidden,
        experts=args.experts,
        topk=args.topk,
        dtype=args.dtype,
        weighted=args.weighted,
        kernel_repeat=args.kernel_repeat,
        replicate_input=args.replicate_input,
        warmup=args.warmup,
        steps=args.steps,
        seed=args.seed,
    )


def emit_record(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def make_selected_experts(config: TokenGatherSumConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    return rng.integers(0, config.experts, size=(config.tokens, config.topk), dtype=np.int32)


def reverse_scatter_from_selected_experts(selected_experts: np.ndarray) -> np.ndarray:
    flat = selected_experts.reshape(-1)
    order = np.argsort(flat, kind="stable").astype(np.int32)
    reverse = np.empty_like(order, dtype=np.int32)
    reverse[order] = np.arange(order.size, dtype=np.int32)
    return reverse


def timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty timing list.")
    return {
        "steady_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def time_blocking_call(call: Callable[[], Any], *, warmup: int, steps: int) -> tuple[float, dict[str, float], Any]:
    start = time.perf_counter()
    result = call()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    compile_inclusive = time.perf_counter() - start

    for _ in range(warmup):
        result = call()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        result = call()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        timings.append(time.perf_counter() - start)
    return compile_inclusive, timing_stats(timings), result
