#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerics harness for the ragged-EP expert MLP: QuACK versus ragged_dot versus an fp32 reference.

Exercises exactly the two `_ExpertMlp` implementations the ragged all-to-all backend selects
between (`levanter.grug._moe.ep_ragged_all_to_all._cute_expert_mlp` and
`_ragged_dot_expert_mlp`), on a receiver buffer laid out the way the transport lays it out:
expert-major groups, trailing padding past the last active row. For each case it reports, per
tensor (forward output, dx, dw13, dw2), the max and mean absolute error and the max relative
error against an fp32 per-group reference computed from the same bf16-rounded inputs.

Cases cover the edge conditions the transport produces: tile-unaligned group boundaries, empty
groups (capacity-clipped experts), leading empty groups, a single expert holding every row,
one-row groups, and an all-empty chunk.

Runs on CPU for the ragged_dot and reference rows (the QuACK rows skip without an SM100 GPU),
which is how the harness itself was validated. On a GB200::

    uv run python .agents/projects/moe-grouped-gemm-derisking/scripts/numerics_expert_mlp.py --hero
    uv run python .agents/projects/moe-grouped-gemm-derisking/scripts/numerics_expert_mlp.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

# Force XLA's ragged_dot_general for the portable row so the comparison is kernel-vs-XLA rather
# than kernel-vs-Pallas-Triton. Must be set before haliax.nn.ragged_dot is imported.
os.environ.setdefault("RAGGED_DOT_IMPL", "xla")

from levanter.grug._moe.ep_ragged_all_to_all import (
    _quack_grouped_gemm_available,
    _ragged_dot_expert_mlp,
)

# One EP64 chunk of the d6144 hero: capacity 1.15 x 65,536 tokens x top-8 / 2 chunks, 3 experts,
# latent 3072 in, intermediate 3072.
HERO_ROWS = 301_466
HERO_EXPERTS = 3
HERO_HIDDEN = 3_072
HERO_INTERMEDIATE = 3_072

SMALL_ROWS = 1_024
SMALL_EXPERTS = 3
SMALL_HIDDEN = 64
SMALL_INTERMEDIATE = 96


@dataclass
class TensorError:
    max_abs: float
    mean_abs: float
    max_rel: float
    # max |delta| / max |reference|: the error in units of the tensor's own scale.
    max_abs_over_scale: float


@dataclass
class Row:
    case: str
    impl: str
    sizes: list[int]
    forward: TensorError
    dx: TensorError
    dw13: TensorError
    dw2: TensorError
    padding_rows_zero: bool | None


def _cases(rows: int, experts: int) -> dict[str, list[int]]:
    third = rows // 3
    return {
        "balanced": [third] * experts,
        "uneven": [int(rows * f) for f in (0.5, 0.3, 0.15)][:experts],
        # Off every 256-row boundary, scaled to the buffer so the case also fits the CPU smoke size.
        "tile-unaligned": [37 + 256 * (rows // 1024), 91 + 256 * (rows // 512), 5 + 256 * (rows // 2048)][:experts],
        "empty-middle": [third, 0, third][:experts],
        "leading-empty": [0, 0, min(rows, 3 * third)][:experts],
        "one-expert-all-rows": [rows, 0, 0][:experts],
        "one-row-groups": [1] * experts,
        "extreme-imbalance": [rows - 2 * (experts - 1), 1, 1][:experts],
        "all-empty": [0] * experts,
    }


def _errors(got: np.ndarray, want: np.ndarray) -> TensorError:
    got = got.astype(np.float64)
    want = want.astype(np.float64)
    delta = np.abs(got - want)
    scale = float(np.abs(want).max()) if want.size else 0.0
    rel = delta / (np.abs(want) + 1e-6)
    return TensorError(
        max_abs=float(delta.max()) if delta.size else 0.0,
        mean_abs=float(delta.mean()) if delta.size else 0.0,
        max_rel=float(rel.max()) if rel.size else 0.0,
        max_abs_over_scale=float(delta.max() / scale) if scale > 0 else 0.0,
    )


def _reference_fn(sizes: np.ndarray):
    """fp32 per-group dense expert MLP over exactly the active rows, differentiable."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    def fn(x, w13, w2):
        x = x.astype(jnp.float32)
        w13 = w13.astype(jnp.float32)
        w2 = w2.astype(jnp.float32)
        moe_dim = w2.shape[1]
        parts = []
        for e, (start, size) in enumerate(zip(starts, sizes, strict=True)):
            if size == 0:
                continue
            xs = jax.lax.dynamic_slice_in_dim(x, int(start), int(size), axis=0)
            h = xs @ w13[e]
            gate, up = jnp.split(h, [moe_dim], axis=-1)
            parts.append((jax.nn.silu(gate) * up) @ w2[e])
        active = int(sizes.sum())
        out = jnp.concatenate(parts, axis=0) if parts else jnp.zeros((0, x.shape[1]), jnp.float32)
        return jnp.pad(out, ((0, x.shape[0] - active), (0, 0)))

    return fn


def _impl_fn(impl, sizes: np.ndarray, rows: int):
    """Wrap an `_ExpertMlp` in the buffer bookkeeping the ragged backend does around it."""
    sizes_d = jnp.asarray(sizes, dtype=jnp.int32)
    total = int(sizes.sum())
    physical = sizes_d.at[-1].add(rows - total)

    def fn(x, w13, w2):
        return impl(x, w13, w2, physical, sizes_d, jax.nn.silu)

    return fn


def _run_case(name: str, sizes: list[int], rows: int, hidden: int, intermediate: int, impls) -> list[Row]:
    sizes_np = np.asarray(sizes, dtype=np.int64)
    if sizes_np.sum() > rows:
        raise ValueError(f"{name}: sizes {sizes} exceed rows {rows}")
    active = int(sizes_np.sum())
    experts = len(sizes)
    key = jax.random.key(hash(name) % (2**31))
    k_x, k_13, k_2, k_ct = jax.random.split(key, 4)
    x = jax.random.normal(k_x, (rows, hidden), dtype=jnp.bfloat16)
    # Padding rows are live memory the grouped kernels must not read past the last group; the
    # ragged_dot path charges them to the last expert and computes on them, so they stay finite.
    w13 = (jax.random.normal(k_13, (experts, hidden, 2 * intermediate)) / np.sqrt(hidden)).astype(jnp.bfloat16)
    w2 = (jax.random.normal(k_2, (experts, intermediate, hidden)) / np.sqrt(intermediate)).astype(jnp.bfloat16)
    cotangent = jax.random.normal(k_ct, (rows, hidden), dtype=jnp.bfloat16)
    # Only active rows carry a cotangent: the transport's return direction never reads padding.
    cotangent = jnp.where(jnp.arange(rows)[:, None] < active, cotangent, jnp.zeros((), cotangent.dtype))

    ref = _reference_fn(sizes_np)
    ref_out, ref_vjp = jax.vjp(ref, x, w13, w2)
    ref_dx, ref_dw13, ref_dw2 = ref_vjp(cotangent.astype(jnp.float32))
    ref_out, ref_dx, ref_dw13, ref_dw2 = jax.block_until_ready((ref_out, ref_dx, ref_dw13, ref_dw2))

    rows_out = []
    for impl_name, impl in impls.items():
        fn = jax.jit(_impl_fn(impl, sizes_np, rows))
        out, vjp = jax.vjp(fn, x, w13, w2)
        dx, dw13, dw2 = vjp(cotangent)
        out, dx, dw13, dw2 = jax.block_until_ready((out, dx, dw13, dw2))
        out_np = np.asarray(out, dtype=np.float32)
        padding_zero = bool(np.all(out_np[active:] == 0)) if impl_name == "quack" else None
        rows_out.append(
            Row(
                case=name,
                impl=impl_name,
                sizes=sizes,
                forward=_errors(out_np[:active], np.asarray(ref_out)[:active]),
                dx=_errors(np.asarray(dx, dtype=np.float32)[:active], np.asarray(ref_dx)[:active]),
                dw13=_errors(np.asarray(dw13, dtype=np.float32), np.asarray(ref_dw13)),
                dw2=_errors(np.asarray(dw2, dtype=np.float32), np.asarray(ref_dw2)),
                padding_rows_zero=padding_zero,
            )
        )
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hero", action="store_true", help="use the d6144 EP64 chunk shape")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--cases", type=str, default=None, help="comma-separated subset of case names")
    args = parser.parse_args()

    rows = args.rows or (HERO_ROWS if args.hero else SMALL_ROWS)
    hidden = args.hidden or (HERO_HIDDEN if args.hero else SMALL_HIDDEN)
    intermediate = args.intermediate or (HERO_INTERMEDIATE if args.hero else SMALL_INTERMEDIATE)
    experts = HERO_EXPERTS if args.hero else SMALL_EXPERTS

    impls = {"ragged_dot[xla]": _ragged_dot_expert_mlp}
    if _quack_grouped_gemm_available():
        from levanter.grug._moe.ep_ragged_all_to_all import _cute_expert_mlp  # noqa: PLC0415

        impls["quack"] = _cute_expert_mlp
    else:
        print("QuACK grouped GEMM unavailable on this device: reporting the ragged_dot row only")
    print(f"device: {jax.devices()[0].device_kind}  rows={rows} experts={experts} hidden={hidden} inter={intermediate}")

    cases = _cases(rows, experts)
    if args.cases:
        cases = {k: cases[k] for k in args.cases.split(",")}

    results: list[Row] = []
    for name, sizes in cases.items():
        for row in _run_case(name, sizes, rows, hidden, intermediate, impls):
            results.append(row)
            print(
                f"{row.case:<22} {row.impl:<16} fwd max_abs {row.forward.max_abs:.3e} rel-scale "
                f"{row.forward.max_abs_over_scale:.2e} | dx {row.dx.max_abs_over_scale:.2e} | "
                f"dw13 {row.dw13.max_abs_over_scale:.2e} | dw2 {row.dw2.max_abs_over_scale:.2e}"
                + ("" if row.padding_rows_zero is None else f" | padding zero {row.padding_rows_zero}")
            )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)

    bad = [r for r in results if not np.isfinite([r.forward.max_abs, r.dx.max_abs, r.dw13.max_abs, r.dw2.max_abs]).all()]
    bad += [r for r in results if r.padding_rows_zero is False]
    for r in bad:
        print(f"FAIL {r.case} {r.impl}: non-finite error or unmasked padding")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
