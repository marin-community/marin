# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the Pallas-Triton grouped GEMM behind ``haliax.nn.ragged_dot`` on one GPU.

A routed-expert MLP runs three grouped-GEMM layouts per weight. With ``M`` routed rows, ``G``
local experts and an expert weight ``[G, K, N]``:

    fwd:   lhs [M, K] x rhs [G, K, N]   -> [M, N]
    dlhs:  dout [M, N] x rhs [G, K, N]^T -> [M, K]
    drhs:  lhs [M, K]^T x dout [M, N]    -> [G, K, N]

Each case is timed for both Triton kernel families (``tile_map``, used on AMD Instinct GPUs, and
``group_grid``, used on every other GPU), for XLA's ``ragged_dot_general`` (in ``--xla-dtype``,
since hipBLASLt's grouped GEMM rejects bf16 on gfx950), and for a dense ``jnp.matmul`` with the
same FLOPs. Tile-map blocks follow this device's entry in ``_TILE_MAP_CONFIGS``.
Each timing warms up for ``--warmup-seconds``, then reports the median of ``--repeats`` windows of
back-to-back calls.

With ``--sweep``, the script instead times the tile-map kernel for every block configuration in a
bounded grid, optionally sharded across processes with ``--shard i/n`` so one node's GPUs can
split the grid.

Rows are JSON lines with the fields of ``BenchRow``. Example::

    HIP_VISIBLE_DEVICES=0 python lib/levanter/scripts/bench/bench_ragged_dot_triton.py \
        --rows 16384 131072 --output results.jsonl
"""

import argparse
import dataclasses
import importlib
import itertools
import json
import logging
import math
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)

# ``haliax.nn.ragged_dot`` is also the name of the function the package re-exports.
ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")

DTYPES = {"bfloat16": jnp.bfloat16, "float16": jnp.float16}
LAYOUTS = [layout.value for layout in ragged_dot_module.RaggedLayout]
KERNEL_FAMILIES = [family.value for family in ragged_dot_module.TritonKernelFamily]
# June expert MLP per GPU: hidden 2560, expert intermediate 1280 with gate and up fused.
DEFAULT_WEIGHTS = ("w13:2560x2560", "w2:1280x2560")

# Default sweep grid; each axis can be narrowed from the command line.
SWEEP_GRID = {
    "block_m": (64, 128, 256),
    "block_n": (64, 128, 256),
    "block_k": (32, 64, 128),
    "num_warps": (4, 8),
    "num_stages": (1, 2),
    "num_xcds": (1, 8),
    "group_m": (1,),
}


@dataclasses.dataclass(frozen=True)
class Case:
    weight: str
    layout: str
    rows: int
    groups: int
    k: int
    n: int

    @property
    def flops(self) -> int:
        return 2 * self.rows * self.k * self.n


@dataclasses.dataclass(frozen=True)
class Timed:
    """One implementation of one case, ready to time."""

    name: str
    dtype: str
    fn: Callable
    inputs: tuple
    block_sizes: str


@dataclasses.dataclass(frozen=True)
class BenchRow:
    kernel: str
    implementation: str
    weight: str
    layout: str
    shape: str
    dtype: str
    backend: str
    device_type: str
    device_count: int
    block_sizes: str
    compile_time: float | None
    steady_state_time: float | None
    tflops: float | None
    error: str | None
    git_sha: str
    xla_flags: str
    backend_env: str


def parse_weight(text: str) -> tuple[str, int, int]:
    name, dims = text.split(":")
    k, n = (int(d) for d in dims.split("x"))
    return name, k, n


def balanced_group_sizes(rows: int, groups: int) -> jax.Array:
    base = jnp.full((groups,), rows // groups, dtype=jnp.int32)
    return base.at[: rows % groups].add(1)


def case_inputs(case: Case, dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    key_a, key_b = jax.random.split(jax.random.key(0))
    group_sizes = balanced_group_sizes(case.rows, case.groups)
    if case.layout == "fwd":
        lhs = jax.random.normal(key_a, (case.rows, case.k), dtype)
        rhs = jax.random.normal(key_b, (case.groups, case.k, case.n), dtype)
    elif case.layout == "dlhs":
        lhs = jax.random.normal(key_a, (case.rows, case.n), dtype)
        rhs = jax.random.normal(key_b, (case.groups, case.k, case.n), dtype)
    else:
        lhs = jax.random.normal(key_a, (case.rows, case.k), dtype)
        rhs = jax.random.normal(key_b, (case.rows, case.n), dtype)
    return lhs, rhs, group_sizes


def time_call(fn: Callable, args, *, warmup_seconds: float, window_seconds: float, repeats: int):
    jax.block_until_ready(args)
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    out = fn(*args)
    calls = 1
    while time.perf_counter() - start < warmup_seconds:
        out = fn(*args)
        calls += 1
    jax.block_until_ready(out)
    seconds_per_call = (time.perf_counter() - start) / calls
    iterations = max(10, math.ceil(window_seconds / seconds_per_call))

    windows = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            out = fn(*args)
        jax.block_until_ready(out)
        windows.append((time.perf_counter() - start) / iterations)
    return compile_time, sorted(windows)[len(windows) // 2]


def implementations(case: Case, args) -> list[Timed]:
    """Both Triton kernel families, plus XLA and a dense matmul unless skipped."""
    layout = ragged_dot_module.RaggedLayout(case.layout)
    bf16_inputs = case_inputs(case, jnp.bfloat16)
    timed = []
    for family in args.kernel_families:
        kernels = ragged_dot_module._TRITON_KERNELS[ragged_dot_module.TritonKernelFamily(family)]
        fn = jax.jit(lambda lhs, rhs, gs, kernels=kernels: kernels(lhs, rhs, gs, layout))
        timed.append(Timed(f"triton_{family}", "bfloat16", fn, bf16_inputs, describe_triton_config(case, family)))
    if not args.skip_xla:
        dim_nums = ragged_dot_module._LAYOUT_DIM_NUMS[layout]
        xla = jax.jit(
            lambda lhs, rhs, gs: jax.lax.ragged_dot_general(lhs, rhs, gs, ragged_dot_dimension_numbers=dim_nums)
        )
        timed.append(Timed("xla", args.xla_dtype, xla, case_inputs(case, DTYPES[args.xla_dtype]), ""))
    if not args.skip_dense:
        key_a, key_b = jax.random.split(jax.random.key(1))
        dense_inputs = (
            jax.random.normal(key_a, (case.rows, case.k), jnp.bfloat16),
            jax.random.normal(key_b, (case.k, case.n), jnp.bfloat16),
        )
        timed.append(Timed("dense_matmul", "bfloat16", jax.jit(jnp.matmul), dense_inputs, ""))
    return timed


def sweep_implementations(case: Case, configs: list) -> list[Timed]:
    """The tile-map kernel once per distinct block config after fitting the configs to this case."""
    layout = ragged_dot_module.RaggedLayout(case.layout)
    inputs = case_inputs(case, jnp.bfloat16)
    fitted = dict.fromkeys(config.fit(*triton_problem_dims(case)) for config in configs)
    timed = []
    for config in fitted:
        fn = jax.jit(
            lambda lhs, rhs, gs, config=config: ragged_dot_module._tile_map_pallas_call(lhs, rhs, gs, layout, config)
        )
        timed.append(Timed("triton_tile_map", "bfloat16", fn, inputs, str(dataclasses.asdict(config))))
    return timed


def describe_triton_config(case: Case, family: str) -> str:
    if family == ragged_dot_module.TritonKernelFamily.GROUP_GRID:
        return "group_grid defaults"
    layout = ragged_dot_module.RaggedLayout(case.layout)
    m, k, n = triton_problem_dims(case)
    return str(dataclasses.asdict(ragged_dot_module._tile_map_block_config(layout, m, k, n)))


def triton_problem_dims(case: Case) -> tuple[int, int, int]:
    """(rows, contraction, columns) as the Triton kernel sees them for each layout."""
    if case.layout == "fwd":
        return case.rows, case.k, case.n
    if case.layout == "dlhs":
        return case.rows, case.n, case.k
    return case.k, case.rows, case.n


def sweep_configs(grid: dict[str, tuple[int, ...]]) -> list:
    config_cls = ragged_dot_module.TritonBlockConfig
    configs = [config_cls(**dict(zip(grid, values))) for values in itertools.product(*grid.values())]
    # Very large tiles spill registers on 64-wide CDNA wavefronts; skip the hopeless corner.
    return [
        c
        for c in configs
        if c.block_m * c.block_n <= 256 * 256 and c.block_m * c.block_n * c.block_k <= 256 * 256 * 64
    ]


def git_sha() -> str:
    env_sha = os.environ.get("MARIN_COMMIT")
    if env_sha:
        return env_sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("MARIN_COMMIT is unset and git rev-parse failed (%s); recording git_sha=unknown", exc)
        return "unknown"


def run_metadata() -> dict:
    """The ``BenchRow`` fields that describe the run rather than the case."""
    device = jax.devices()[0]
    return dict(
        kernel="ragged_dot",
        backend=device.platform,
        device_type=device.device_kind,
        device_count=1,
        git_sha=git_sha(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        backend_env=f"jax={jax.__version__} RAGGED_DOT_IMPL={os.environ.get('RAGGED_DOT_IMPL', '')}",
    )


def bench_row(case: Case, timed: Timed, metadata: dict, timing: dict) -> BenchRow:
    compile_time = steady = tflops = None
    error = None
    try:
        compile_time, steady = time_call(timed.fn, timed.inputs, **timing)
        tflops = case.flops / steady / 1e12
    except Exception as exc:  # A failing config is a result to record, not a reason to stop the sweep.
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
    return BenchRow(
        implementation=timed.name,
        weight=case.weight,
        layout=case.layout,
        shape=f"M={case.rows} G={case.groups} K={case.k} N={case.n}",
        dtype=timed.dtype,
        block_sizes=timed.block_sizes,
        compile_time=compile_time,
        steady_state_time=steady,
        tflops=tflops,
        error=error,
        **metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--weights", nargs="+", default=list(DEFAULT_WEIGHTS), help="name:KxN")
    parser.add_argument("--layouts", nargs="+", default=LAYOUTS, choices=LAYOUTS)
    parser.add_argument("--kernel-families", nargs="+", default=KERNEL_FAMILIES, choices=KERNEL_FAMILIES)
    parser.add_argument("--xla-dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--skip-xla", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="time the tile-map kernel over the block-config grid")
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th sweep config starting at i")
    for field, values in SWEEP_GRID.items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=int, nargs="+", default=list(values))
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    metadata = run_metadata()
    logger.info("%s", metadata)
    cases = [
        Case(name, layout, rows, args.groups, k, n)
        for rows in args.rows
        for name, k, n in map(parse_weight, args.weights)
        for layout in args.layouts
    ]
    timing = dict(warmup_seconds=args.warmup_seconds, window_seconds=args.window_seconds, repeats=args.repeats)
    index, count = (int(x) for x in args.shard.split("/"))
    configs = sweep_configs({field: tuple(getattr(args, field)) for field in SWEEP_GRID})[index::count]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as out:
        for case in cases:
            for timed in sweep_implementations(case, configs) if args.sweep else implementations(case, args):
                row = bench_row(case, timed, metadata, timing)
                out.write(json.dumps(dataclasses.asdict(row)) + "\n")
                out.flush()
                result = row.error or f"{row.steady_state_time * 1e3:.3f} ms {row.tflops:.1f} TFLOP/s"
                logger.info(
                    "%s %s %s M=%d %s: %s", case.weight, case.layout, timed.name, case.rows, timed.block_sizes, result
                )


if __name__ == "__main__":
    main()
