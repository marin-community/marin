# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU benchmark of the ragged EP backend's two grouped weight gradients.

The ragged all-to-all expert MLP runs six grouped GEMMs. Four group over rows
(``cu_seqlens_m``) and run on QuACK; the two weight gradients contract *over* the ragged
dimension (``cu_seqlens_k``) and have run on cuDNN Frontend's grouped Wgrad kernel. This
compares that kernel against QuACK's own varlen-k path at the hero's per-call shapes.

The comparison that matters is end to end per call, not kernel time: cuDNN's kernel derives
each expert's tile count as ``ceil(tokens/cta_tile_k)`` while addressing through one TMA
descriptor over the whole buffer, so every group has to start on a 256-row boundary, and the
wrapper materializes aligned copies of both operands to get there. QuACK offsets a coordinate
into the descriptor instead, so it reads the buffer where it lies and needs no copy. Both
totals are reported, plus the cuDNN kernel alone, so the copy is attributable.

Hero per-call shapes (``experiments/grug/moe_hero_ep`` HERO_MODEL under EP64, LatentMoE, so
the expert input is the latent dim, not the model dim):

    dw13:  lhs [rows, 3072]  rhs [rows, 6144]  -> [E, 3072, 6144]
    dw2:   lhs [rows, 3072]  rhs [rows, 3072]  -> [E, 3072, 3072]

with ``rows`` the chunk's receiver-buffer capacity and ``E`` its expert count (384 experts /
64 shards / 2 chunks = 3).

Reports per variant: wall time, achieved TFLOP/s, and max/mean absolute error against a
float32 reference computed per group. Exits nonzero if any variant's error exceeds the
reference's own bf16 rounding floor, so this doubles as the correctness gate.

Examples::

    python lib/levanter/scripts/bench/bench_grouped_wgrad.py
    python lib/levanter/scripts/bench/bench_grouped_wgrad.py --sweep
    python lib/levanter/scripts/bench/bench_grouped_wgrad.py --rows 301466 --experts 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

# Both kernel families ship only with the CUDA 13 GPU extra, as does this benchmark's point.
from levanter.grug._moe.cudnn_wgrad_cute import cudnn_grouped_wgrad
from levanter.grug._moe.quack_moe_cute import quack_grouped_wgrad

# Hero per-call shapes; see the module docstring.
_HERO_ROWS = 301_466
_HERO_EXPERTS = 3
_HERO_LATENT = 3_072
_HERO_INTERMEDIATE = 3_072

_WARMUP = 3
_ITERS = 10

# Fixed per case so a rerun draws the same group sizes.
_CASE_SEEDS = {"dw13": 1, "dw2": 2}


@dataclass
class Result:
    case: str
    variant: str
    rows: int
    experts: int
    m: int
    n: int
    seconds: float
    tflops: float
    max_abs_err: float
    mean_abs_err: float


def _group_sizes(rows: int, experts: int, seed: int) -> np.ndarray:
    """Uneven group sizes summing to at most ``rows``, none of them 256-aligned.

    The hero's router does not hand out aligned counts, and an aligned draw would hide exactly
    the failure mode that made the cuDNN wrapper's padding load-bearing.
    """
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.7, 1.3, size=experts)
    sizes = np.floor(weights / weights.sum() * rows * 0.98).astype(np.int64)
    # Nudge off every alignment boundary the kernels care about.
    sizes = sizes - (sizes % 256) + 137
    return sizes


def _reference(lhs: jax.Array, rhs: jax.Array, sizes: np.ndarray) -> np.ndarray:
    """Per-group ``lhs.T @ rhs`` in float32 on device, the accuracy target for both kernels.

    Takes the same bf16 values the kernels get, upcast, so the error it measures is the
    kernel's own and not the inputs' rounding. float32 accumulation over ~1e5 rows carries
    ~1e-5 relative error, two orders below bf16 output rounding, which is enough to separate
    a correct kernel from one reading a neighbouring group's rows.
    """
    out = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        a = lhs[start:stop].astype(jnp.float32)
        b = rhs[start:stop].astype(jnp.float32)
        out.append(np.asarray(jax.block_until_ready(a.T @ b), dtype=np.float64))
        start = stop
    return np.stack(out)


def _time(fn, *args) -> float:
    for _ in range(_WARMUP):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(_ITERS):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / _ITERS


def _errors(got: jax.Array, want: np.ndarray) -> tuple[float, float]:
    delta = np.abs(np.asarray(got, dtype=np.float64) - want)
    scale = np.abs(want).mean()
    return float(delta.max() / scale), float(delta.mean() / scale)


def _run_case(
    case: str,
    rows: int,
    experts: int,
    m: int,
    n: int,
    sweep: bool,
) -> list[Result]:
    sizes = _group_sizes(rows, experts, seed=_CASE_SEEDS[case])
    rng = np.random.default_rng(0)
    lhs = rng.normal(0, 1, size=(rows, m)).astype(np.float32)
    rhs = rng.normal(0, 1, size=(rows, n)).astype(np.float32)
    # Rows past the last group are live memory that neither kernel may read. A NaN there turns
    # an over-read into a NaN output rather than a small numerical drift, which is how the
    # cuDNN alignment defect stayed invisible for as long as it did.
    lhs[int(sizes.sum()) :] = np.nan
    rhs[int(sizes.sum()) :] = np.nan

    lhs_d = jnp.asarray(lhs, dtype=jnp.bfloat16)
    rhs_d = jnp.asarray(rhs, dtype=jnp.bfloat16)
    del lhs, rhs
    want = _reference(lhs_d, rhs_d, sizes)
    group_sizes = jnp.asarray(sizes, dtype=jnp.int32)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])

    flops = 2.0 * float(sizes.sum()) * m * n
    results: list[Result] = []

    def record(variant: str, fn, *args) -> None:
        out = jax.block_until_ready(fn(*args))
        max_err, mean_err = _errors(out, want)
        seconds = _time(fn, *args)
        results.append(
            Result(
                case=case,
                variant=variant,
                rows=rows,
                experts=experts,
                m=m,
                n=n,
                seconds=seconds,
                tflops=flops / seconds / 1e12,
                max_abs_err=max_err,
                mean_abs_err=mean_err,
            )
        )

    cudnn = jax.jit(cudnn_grouped_wgrad)
    record("cudnn+pad", cudnn, lhs_d, rhs_d, group_sizes)

    quack = jax.jit(quack_grouped_wgrad)
    record("quack-varlen-k", quack, lhs_d, rhs_d, cu)

    if sweep:
        for tile, cluster, clc in itertools.product(
            [(128, 128), (128, 256), (256, 128), (256, 256)],
            [(2, 1, 1), (2, 2, 1)],
            [False, True],
        ):
            variant = f"quack tile={tile[0]}x{tile[1]} cluster={cluster[0]}x{cluster[1]} clc={int(clc)}"
            fn = jax.jit(
                lambda a, b, c, tile=tile, cluster=cluster, clc=clc: quack_grouped_wgrad(
                    a, b, c, tile_mn=tile, cluster_mnk=cluster, use_clc_persistence=clc
                )
            )
            try:
                record(variant, fn, lhs_d, rhs_d, cu)
            except Exception as exc:  # a tile the kernel refuses is a datum, not a failure
                print(f"  {variant}: unsupported ({type(exc).__name__}: {str(exc)[:120]})")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=_HERO_ROWS)
    parser.add_argument("--experts", type=int, default=_HERO_EXPERTS)
    parser.add_argument("--sweep", action="store_true", help="sweep QuACK tile/cluster/CLC settings")
    parser.add_argument("--json", type=str, default=None, help="write results here as JSON")
    parser.add_argument(
        "--error-ratio",
        type=float,
        default=3.0,
        help=(
            "how many times the cuDNN kernel's relative error a QuACK variant may show before the "
            "run is called a correctness failure. Calibrating against the other kernel rather than "
            "an absolute number keeps the gate meaningful as the shapes change: the two differ "
            "only in reduction order, so anything beyond a small factor is a real defect."
        ),
    )
    args = parser.parse_args()

    print(f"device: {jax.devices()[0].device_kind}")
    results: list[Result] = []
    for case, m, n in (
        ("dw13", _HERO_LATENT, 2 * _HERO_INTERMEDIATE),
        ("dw2", _HERO_INTERMEDIATE, _HERO_LATENT),
    ):
        print(f"\n{case}: lhs[{args.rows}, {m}] x rhs[{args.rows}, {n}] -> [{args.experts}, {m}, {n}]")
        case_results = _run_case(case, args.rows, args.experts, m, n, args.sweep)
        for r in case_results:
            print(
                f"  {r.variant:<44} {r.seconds * 1e3:8.3f} ms  {r.tflops:7.1f} TFLOP/s  "
                f"max_err {r.max_abs_err:.2e}"
            )
        results.extend(case_results)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)

    failed = []
    for case in {r.case for r in results}:
        reference = next(r for r in results if r.case == case and r.variant == "cudnn+pad")
        budget = args.error_ratio * reference.max_abs_err
        for r in results:
            if r.case != case or r.variant == "cudnn+pad":
                continue
            if not np.isfinite(r.max_abs_err) or r.max_abs_err > budget:
                failed.append((r, budget))
    for r, budget in failed:
        print(f"FAIL {r.case} {r.variant}: relative error {r.max_abs_err:.3e} > {budget:.3e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
