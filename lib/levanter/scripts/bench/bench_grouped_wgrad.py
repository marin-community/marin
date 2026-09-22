# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune and check QuACK's grouped weight-gradient kernel for a grouped-expert MLP.

A grouped-expert MLP runs six grouped GEMMs. Four group over rows (``cu_seqlens_m``). The two
weight gradients contract *over* the ragged dimension, so they group with ``cu_seqlens_k``.
This script covers that pair.

It times `quack_grouped_wgrad` against XLA's `ragged_dot`, which is the portable fallback, and
with ``--sweep`` it walks the tile, cluster, and CLC grid. Use it to pick the values a backend
passes to the kernel, and to re-pick them when the shapes change.

It is also the correctness gate. Every variant is checked against a per-group float32
reference, and the run exits nonzero when a variant's relative error exceeds ``--error-ratio``
times XLA's on the same case. Rows past the last group hold NaN, so a kernel that reads past
its own group returns NaN rather than a small drift.

Shapes come from the command line. The defaults are one EP64 chunk of a d6144 LatentMoE model,
where the expert input is the latent dim rather than the model dim::

    dw13:  lhs [rows, hidden] rhs [rows, 2 * intermediate] -> [experts, hidden, 2 * intermediate]
    dw2:   lhs [rows, intermediate] rhs [rows, hidden]     -> [experts, intermediate, hidden]

``rows`` is the chunk's receiver-buffer capacity and ``experts`` is its expert count.

Examples::

    python lib/levanter/scripts/bench/bench_grouped_wgrad.py --sweep
    python lib/levanter/scripts/bench/bench_grouped_wgrad.py --rows 65536 --experts 8 \
        --hidden 2048 --intermediate 1024
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from haliax.nn.ragged_dot import ragged_dot

# QuACK ships only with the CUDA 13 GPU extra, as does this benchmark's point.
from levanter.grug._moe.quack_moe_cute import quack_grouped_wgrad
from levanter.grug._moe.sonic_cute import _QUACK_WGRAD_KW

# One EP64 chunk of the d6144 LatentMoE model; see the module docstring.
_DEFAULT_ROWS = 301_466
_DEFAULT_EXPERTS = 3
_DEFAULT_HIDDEN = 3_072
_DEFAULT_INTERMEDIATE = 3_072

_WARMUP = 3
_ITERS = 10

# Fixed so a rerun draws the same group sizes. Both cases share it because the backward drives
# both weight gradients from one `cu`, so scoring them at different raggedness would compare a
# pairing production never runs.
_GROUP_SEED = 1

# The row every other variant is judged against.
_REFERENCE_VARIANT = "xla ragged_dot"


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
    max_rel_err: float
    mean_rel_err: float


def _group_sizes(rows: int, experts: int, seed: int) -> np.ndarray:
    """Uneven group sizes summing to at most ``rows``, none of them 256-aligned.

    A real router does not hand out aligned counts, and an aligned draw would hide a kernel that
    reads past its own group.
    """
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.7, 1.3, size=experts)
    sizes = np.floor(weights / weights.sum() * rows * 0.98).astype(np.int64)
    # Nudge off every alignment boundary the kernels care about, without letting the total run
    # past the buffer: an overshoot would disable the NaN sentinel below, truncate the reference,
    # and hand the kernel offsets past the end of the allocation.
    sizes = np.maximum(sizes - (sizes % 256) + 137, 1)
    overshoot = int(sizes.sum()) - rows
    if overshoot > 0:
        sizes[-1] -= overshoot
    if sizes.min() < 1 or sizes.sum() > rows:
        raise ValueError(f"cannot fit {experts} uneven groups into {rows} rows")
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
    sizes = _group_sizes(rows, experts, seed=_GROUP_SEED)
    rng = np.random.default_rng(0)
    # `normal` has no dtype and would materialise float64 temporaries -- ~26 GB at the defaults.
    lhs = rng.standard_normal(size=(rows, m), dtype=np.float32)
    rhs = rng.standard_normal(size=(rows, n), dtype=np.float32)
    # Rows past the last group are live memory that neither kernel may read. A NaN there turns
    # an over-read into a NaN output rather than a small numerical drift.
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
                max_rel_err=max_err,
                mean_rel_err=mean_err,
            )
        )

    def xla_wgrad(lhs, rhs, sizes):
        """The portable weight gradient: `ragged_dot`'s own transpose."""
        weights = jnp.zeros((len(sizes), lhs.shape[1], rhs.shape[1]), dtype=lhs.dtype)
        return jax.vjp(lambda w: ragged_dot(lhs, w, sizes), weights)[1](rhs)[0]

    record(_REFERENCE_VARIANT, jax.jit(xla_wgrad), lhs_d, rhs_d, group_sizes)

    # The shipped configuration, imported rather than restated, so this gate cannot drift away
    # from what training runs.
    shipped = jax.jit(functools.partial(quack_grouped_wgrad, **_QUACK_WGRAD_KW))
    record("quack varlen-k (shipped)", shipped, lhs_d, rhs_d, cu)

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
            except ValueError as exc:
                # A shape or tile the wrapper itself refuses is a datum about the grid. Anything
                # else -- a signature change, or a CUDA fault that leaves the context unusable for
                # every later variant -- must not be reported as "unsupported" and exit zero.
                print(f"  {variant}: unsupported ({type(exc).__name__}: {str(exc)[:120]})")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS, help="rows in the receiver buffer")
    parser.add_argument("--experts", type=int, default=_DEFAULT_EXPERTS, help="expert groups in the buffer")
    parser.add_argument("--hidden", type=int, default=_DEFAULT_HIDDEN, help="expert input width")
    parser.add_argument("--intermediate", type=int, default=_DEFAULT_INTERMEDIATE, help="expert hidden width")
    parser.add_argument("--sweep", action="store_true", help="sweep QuACK tile/cluster/CLC settings")
    parser.add_argument("--json", type=str, default=None, help="write results here as JSON")
    parser.add_argument(
        "--error-ratio",
        type=float,
        default=3.0,
        help=(
            "how many times XLA's relative error a QuACK variant may show before the "
            "run is called a correctness failure. Calibrating against the other kernel rather than "
            "an absolute number keeps the gate meaningful as the shapes change: the two differ "
            "only in reduction order, so anything beyond a small factor is a real defect."
        ),
    )
    args = parser.parse_args()

    print(f"device: {jax.devices()[0].device_kind}")
    results: list[Result] = []
    for case, m, n in (
        ("dw13", args.hidden, 2 * args.intermediate),
        ("dw2", args.intermediate, args.hidden),
    ):
        print(f"\n{case}: lhs[{args.rows}, {m}] x rhs[{args.rows}, {n}] -> [{args.experts}, {m}, {n}]")
        case_results = _run_case(case, args.rows, args.experts, m, n, args.sweep)
        for r in case_results:
            print(
                f"  {r.variant:<44} {r.seconds * 1e3:8.3f} ms  {r.tflops:7.1f} TFLOP/s  "
                f"max_err {r.max_rel_err:.2e}"
            )
        results.extend(case_results)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)

    failed = []
    for case in sorted({r.case for r in results}):
        reference = next(r for r in results if r.case == case and r.variant == _REFERENCE_VARIANT)
        # Everything below is judged against this row, so a NaN or zero reference would silently
        # make every comparison vacuous (`x > nan` is False) rather than failing anything.
        if not np.isfinite(reference.max_rel_err) or reference.max_rel_err <= 0:
            failed.append((reference, float("nan")))
            continue
        budget = args.error_ratio * reference.max_rel_err
        for r in results:
            if r.case != case:
                continue
            if not np.isfinite(r.max_rel_err) or (r.variant != _REFERENCE_VARIANT and r.max_rel_err > budget):
                failed.append((r, budget))
    for r, budget in failed:
        print(f"FAIL {r.case} {r.variant}: relative error {r.max_rel_err:.3e} > {budget:.3e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
