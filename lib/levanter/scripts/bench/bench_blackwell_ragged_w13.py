# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark local Blackwell ragged W13/GMM for staged source-push MoE.

This measures the local compute side of the staged Blackwell plan after
Warpgroup peer refs have been ruled out for a true fused source-push kernel.
Rows are emitted as JSONL so Hopper, copy-only, local-GMM, and staged inbox rows
can stay separate in downstream analysis.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu, blackwell_ragged_dot_mgpu
from levanter.grug._moe.source_push_inbox_blackwell import BLACKWELL_TARGET_W13_TUNING_CONFIG


TFLOPS = 1.0e12
TARGET_M = 65_536
TARGET_HIDDEN_DIM = 3_072
TARGET_INTERMEDIATE_DIM = 3_072
TARGET_EP_SIZE = 256
TARGET_TOPK = 4
DEFAULT_TUNING_CONFIG = BLACKWELL_TARGET_W13_TUNING_CONFIG
DEFAULT_GRID_MINOR_DIM = blackwell_matmul_mgpu.MatmulDimension[DEFAULT_TUNING_CONFIG.grid_minor_dim.value]


@dataclass(frozen=True)
class RaggedW13Shape:
    m: int = TARGET_M
    hidden_dim: int = TARGET_HIDDEN_DIM
    intermediate_dim: int = TARGET_INTERMEDIATE_DIM
    num_groups: int = 1
    dtype: str = "bfloat16"

    @property
    def n(self) -> int:
        return 2 * self.intermediate_dim

    def validate(self) -> None:
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {self.intermediate_dim}")
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {self.num_groups}")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError(f"dtype must be 'bfloat16' or 'float16', got {self.dtype!r}")


@dataclass(frozen=True)
class RaggedW13RunSettings:
    warmup: int = 1
    steps: int = 5
    check: bool = False

    def validate(self) -> None:
        if self.warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {self.warmup}")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def _parse_bool_csv(value: str) -> tuple[bool, ...]:
    parsed = []
    for part in value.split(","):
        if not part:
            continue
        normalized = part.lower()
        if normalized in ("1", "true", "yes"):
            parsed.append(True)
        elif normalized in ("0", "false", "no"):
            parsed.append(False)
        else:
            raise argparse.ArgumentTypeError(f"expected booleans, got {part!r}")
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated list of booleans")
    return tuple(parsed)


def _parse_minor_dim_csv(value: str) -> tuple[blackwell_matmul_mgpu.MatmulDimension, ...]:
    dims = []
    for part in value.split(","):
        if not part:
            continue
        try:
            dims.append(blackwell_matmul_mgpu.MatmulDimension[part.upper()])
        except KeyError as exc:
            raise argparse.ArgumentTypeError("grid minor dimensions must be M or N") from exc
    if not dims:
        raise argparse.ArgumentTypeError("expected a comma-separated list of M/N values")
    return tuple(dims)


def _require_blackwell_gpu() -> str:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Blackwell W13 benchmark requires a GPU backend, got {jax.default_backend()!r}")
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("Blackwell W13 benchmark requires visible GPU devices")
    device = devices[0]
    device_kind = getattr(device, "device_kind", "")
    compute_capability = getattr(device, "compute_capability", None)
    if compute_capability is not None:
        try:
            if float(compute_capability) >= 10.0:
                return device_kind
        except (TypeError, ValueError):
            pass
    if any(name in device_kind for name in ("B200", "B300", "GB200", "GB300")):
        return device_kind
    raise RuntimeError(f"Blackwell W13 benchmark requires Blackwell GPUs, got {device_kind!r}")


def _dtype(name: str) -> Any:
    return {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }[name]


def _balanced_group_sizes(num_groups: int, m: int) -> jax.Array:
    base = m // num_groups
    remainder = m % num_groups
    sizes = np.full((num_groups,), base, dtype=np.int32)
    sizes[:remainder] += 1
    return jnp.asarray(sizes)


def _make_inputs(shape: RaggedW13Shape) -> tuple[jax.Array, jax.Array, jax.Array]:
    dtype = _dtype(shape.dtype)
    lhs_key, rhs_key = jax.random.split(jax.random.key(0))
    lhs = jax.random.normal(lhs_key, (shape.m, shape.hidden_dim), dtype=dtype)
    rhs = jax.random.normal(rhs_key, (shape.num_groups, shape.hidden_dim, shape.n), dtype=dtype)
    return lhs, rhs, _balanced_group_sizes(shape.num_groups, shape.m)


def _useful_tflops(shape: RaggedW13Shape, seconds: float) -> float:
    flops = 2 * shape.m * shape.hidden_dim * shape.n
    return flops / seconds / TFLOPS


def _candidate_configs(args: argparse.Namespace) -> Iterable[blackwell_ragged_dot_mgpu.TuningConfig]:
    if args.preset == "target":
        tile_m_values = (64, 128)
        tile_n_values = (128, 256)
        tile_k_values = (64, 128)
        max_concurrent_steps_values = (3, 4, 6, 8)
        collective_values = (False, True)
        grid_tile_width_values = (1, 4, 8, 12, 16)
        grid_minor_dim_values = tuple(blackwell_matmul_mgpu.MatmulDimension)
        epilogue_tile_n_values = (64, 128)
    else:
        tile_m_values = args.tile_m
        tile_n_values = args.tile_n
        tile_k_values = args.tile_k
        max_concurrent_steps_values = args.max_concurrent_steps
        collective_values = args.collective
        grid_tile_width_values = args.grid_tile_width
        grid_minor_dim_values = args.grid_minor_dim
        epilogue_tile_n_values = args.epilogue_tile_n

    for (
        tile_m,
        tile_n,
        tile_k,
        max_concurrent_steps,
        collective,
        grid_tile_width,
        grid_minor_dim,
        epilogue_tile_n,
    ) in itertools.product(
        tile_m_values,
        tile_n_values,
        tile_k_values,
        max_concurrent_steps_values,
        collective_values,
        grid_tile_width_values,
        grid_minor_dim_values,
        epilogue_tile_n_values,
    ):
        yield blackwell_ragged_dot_mgpu.TuningConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            max_concurrent_steps=max_concurrent_steps,
            collective=collective,
            grid_tile_width=grid_tile_width,
            grid_minor_dim=grid_minor_dim,
            epilogue_tile_n=epilogue_tile_n,
        )


def _config_row(config: blackwell_ragged_dot_mgpu.TuningConfig) -> dict[str, Any]:
    row = asdict(config)
    row["grid_minor_dim"] = config.grid_minor_dim.name
    return row


def _benchmark_config(
    shape: RaggedW13Shape,
    settings: RaggedW13RunSettings,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: blackwell_ragged_dot_mgpu.TuningConfig,
) -> dict[str, Any]:
    fn = jax.jit(functools.partial(blackwell_ragged_dot_mgpu.ragged_dot_kernel, config=config))
    row: dict[str, Any] = {
        "kernel": "blackwell_ragged_w13",
        "implementation": "jax_blackwell_ragged_dot_mgpu",
        "target_ep_size": TARGET_EP_SIZE,
        "target_topk": TARGET_TOPK,
        **asdict(shape),
        **_config_row(config),
    }
    try:
        compile_start = time.perf_counter()
        out = fn(lhs, rhs, group_sizes)
        jax.block_until_ready(out)
        row["compile_time"] = time.perf_counter() - compile_start

        if settings.check:
            expected = jax.lax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type=jnp.float32)
            max_abs_diff = jnp.max(jnp.abs(out.astype(jnp.float32) - expected.astype(jnp.float32)))
            row["max_abs_diff"] = float(max_abs_diff)

        for _ in range(settings.warmup):
            out = fn(lhs, rhs, group_sizes)
            jax.block_until_ready(out)

        step_times = []
        for _ in range(settings.steps):
            start = time.perf_counter()
            out = fn(lhs, rhs, group_sizes)
            jax.block_until_ready(out)
            step_times.append(time.perf_counter() - start)

        median = float(np.median(step_times))
        row.update(
            {
                "ok": True,
                "steady_state_median": median,
                "steady_state_min": float(np.min(step_times)),
                "steady_state_max": float(np.max(step_times)),
                "useful_tflops_per_rank": _useful_tflops(shape, median),
            }
        )
    except Exception as exc:
        row.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "target"), default="quick")
    parser.add_argument("--m", type=int, default=TARGET_M)
    parser.add_argument("--hidden-dim", type=int, default=TARGET_HIDDEN_DIM)
    parser.add_argument("--intermediate-dim", type=int, default=TARGET_INTERMEDIATE_DIM)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--tile-m", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.tile_m,))
    parser.add_argument("--tile-n", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.tile_n,))
    parser.add_argument("--tile-k", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.tile_k,))
    parser.add_argument(
        "--max-concurrent-steps", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.max_concurrent_steps,)
    )
    parser.add_argument("--collective", type=_parse_bool_csv, default=(DEFAULT_TUNING_CONFIG.collective,))
    parser.add_argument("--grid-tile-width", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.grid_tile_width,))
    parser.add_argument("--grid-minor-dim", type=_parse_minor_dim_csv, default=(DEFAULT_GRID_MINOR_DIM,))
    parser.add_argument("--epilogue-tile-n", type=_parse_int_csv, default=(DEFAULT_TUNING_CONFIG.epilogue_tile_n,))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device_kind = _require_blackwell_gpu()
    shape = RaggedW13Shape(
        m=args.m,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_groups=args.num_groups,
        dtype=args.dtype,
    )
    settings = RaggedW13RunSettings(warmup=args.warmup, steps=args.steps, check=args.check)
    shape.validate()
    settings.validate()
    lhs, rhs, group_sizes = _make_inputs(shape)

    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    for config in _candidate_configs(args):
        row = _benchmark_config(shape, settings, lhs, rhs, group_sizes, config)
        row["device_kind"] = device_kind
        row["jax_version"] = jax.__version__
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                print(line, file=f, flush=True)


if __name__ == "__main__":
    main()
