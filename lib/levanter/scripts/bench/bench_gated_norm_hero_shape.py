# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU benchmark of the Grug MoE EP64 hero's RMSNorm -> GatedNorm at the per-rank shape.

Every hero block normalizes twice with ``GatedNorm(RMSNorm(x))`` (``Block.__call__`` in
``experiments/grug/moe_hero_ep/model.py``), and the embedding and final norm add one call each:
98 calls per step. Per rank a call sees ``x [65_536, 6_144]`` bf16 and a rank-128 bottleneck.

Each variant is timed forward, and forward plus backward as one VJP against a fixed cotangent. The
per-step cost is ``98 * (fwd_bwd + fwd)``, because the hero's ``offload_carry`` remat reruns the
forward in the backward. ``passes`` converts a time to ``[T, D]`` HBM passes at the bandwidth an
elementwise ``x -> 2x`` reaches; the floor is 2 forward and 5 forward plus backward.

Example::

    python lib/levanter/scripts/bench/bench_gated_norm_hero_shape.py
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.triton.gated_norm import gated_rms_norm, gated_rms_norm_reference

RESULT_PREFIX = "RESULT_JSON "

# experiments/grug/moe_hero_ep/heuristic.py HERO_MODEL, per rank.
HERO_T = 65_536
HERO_D = 6_144
# experiments/grug/moe_hero_ep/model.py `_GATED_NORM_RANK`.
HERO_RANK = 128
HERO_EPS = 1e-5
# 48 layers x (attention, MLP) + the embedding norm + the final norm.
HERO_CALLS_PER_STEP = 98

Weights = tuple[jax.Array, jax.Array, jax.Array]  # (rms weight [D], w_down [D, R], w_up [R, D])


def _fused(implementation: str) -> Callable[[jax.Array, Weights], jax.Array]:
    def fn(x, weights):
        return gated_rms_norm(x, *weights, eps=HERO_EPS, implementation=implementation)

    return fn


VARIANTS: dict[str, Callable[[jax.Array, Weights], jax.Array]] = {
    "reference": lambda x, weights: gated_rms_norm_reference(x, *weights, HERO_EPS),
    "xla": _fused("xla"),
    "triton_gpu": _fused("triton_gpu"),
}


@dataclass
class Result:
    variant: str
    device_kind: str
    fwd_ms: float
    fwd_bwd_ms: float
    per_step_ms: float
    fwd_passes: float
    fwd_bwd_passes: float
    rel_rms_vs_fp32: dict[str, float]


def _make_inputs(args):
    kx, kdy, kw, kd, ku = jax.random.split(jax.random.PRNGKey(args.seed), 5)
    x = jax.random.normal(kx, (args.tokens, args.hidden)).astype(jnp.bfloat16)
    dy = jax.random.normal(kdy, (args.tokens, args.hidden)).astype(jnp.bfloat16)
    # A non-unit RMS weight and fan-in-scaled bottleneck weights, so the gate spans the sigmoid.
    rms_weight = (1.0 + 0.1 * jax.random.normal(kw, (args.hidden,))).astype(jnp.bfloat16)
    w_down = (jax.random.normal(kd, (args.hidden, args.rank)) / np.sqrt(args.hidden)).astype(jnp.bfloat16)
    w_up = (jax.random.normal(ku, (args.rank, args.hidden)) / np.sqrt(args.rank)).astype(jnp.bfloat16)
    return jax.block_until_ready((x, dy, (rms_weight, w_down, w_up)))


def _time(fn, call_args, steps: int, warmup: int) -> tuple[float, object]:
    """Mean seconds per call; calls are dispatched back to back and synchronized once."""
    out = jax.block_until_ready(fn(*call_args))
    for _ in range(warmup):
        out = fn(*call_args)
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(steps):
        out = fn(*call_args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / steps, out


def _out_and_grads(fn):
    @jax.jit
    def run(x, weights, dy):
        y, vjp = jax.vjp(fn, x, weights)
        return y, vjp(dy)

    return run


def _flatten(y, grads):
    dx, (d_rms_weight, d_w_down, d_w_up) = grads
    return {"y": y, "dx": dx, "d_rms_weight": d_rms_weight, "d_w_down": d_w_down, "d_w_up": d_w_up}


def _fp32_reference(x, dy, weights):
    up = lambda a: a.astype(jnp.float32)  # noqa: E731
    with jax.default_matmul_precision("highest"):
        y, grads = _out_and_grads(VARIANTS["reference"])(up(x), jax.tree.map(up, weights), up(dy))
    return {k: np.asarray(v, np.float64) for k, v in _flatten(y, grads).items()}


def _rel_rms(actual, expected: np.ndarray) -> float:
    a = np.asarray(actual, np.float64)
    return float(np.sqrt(np.mean((a - expected) ** 2)) / np.sqrt(np.mean(expected**2)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=HERO_T)
    p.add_argument("--hidden", type=int, default=HERO_D)
    p.add_argument("--rank", type=int, default=HERO_RANK)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    args = p.parse_args()

    x, dy, weights = _make_inputs(args)
    copy_s, _ = _time(jax.jit(lambda a: a * jnp.asarray(2, a.dtype)), (x,), args.steps, args.warmup)
    pass_s = copy_s / 2
    print(f"one [T, D] pass: {pass_s * 1e3:.3f} ms ({x.size * x.dtype.itemsize / pass_s / 1e12:.2f} TB/s)")
    reference = _fp32_reference(x, dy, weights)

    results = []
    for name in args.variants.split(","):
        fn = VARIANTS[name]
        fwd_s, _ = _time(jax.jit(fn), (x, weights), args.steps, args.warmup)
        fwd_bwd_s, (y, grads) = _time(_out_and_grads(fn), (x, weights, dy), args.steps, args.warmup)
        res = Result(
            variant=name,
            device_kind=jax.local_devices()[0].device_kind,
            fwd_ms=fwd_s * 1e3,
            fwd_bwd_ms=fwd_bwd_s * 1e3,
            per_step_ms=HERO_CALLS_PER_STEP * (fwd_bwd_s + fwd_s) * 1e3,
            fwd_passes=fwd_s / pass_s,
            fwd_bwd_passes=fwd_bwd_s / pass_s,
            rel_rms_vs_fp32={k: _rel_rms(v, reference[k]) for k, v in _flatten(y, grads).items()},
        )
        print(RESULT_PREFIX + json.dumps(asdict(res)), flush=True)
        results.append(res)

    base = next((r for r in results if r.variant == "reference"), None)
    print(
        f"\n{'variant':<12}{'fwd ms':>9}{'passes':>8}{'fwd+bwd ms':>12}{'passes':>8}{'per step ms':>13}{'vs ref':>8}"
    )
    for r in results:
        delta = f"{r.per_step_ms - base.per_step_ms:+.0f}" if base else ""
        print(
            f"{r.variant:<12}{r.fwd_ms:>9.3f}{r.fwd_passes:>8.1f}{r.fwd_bwd_ms:>12.3f}{r.fwd_bwd_passes:>8.1f}"
            f"{r.per_step_ms:>13.0f}{delta:>8}"
        )
    print(f"\n{'variant':<12}" + "".join(f"{k:>14}" for k in reference) + "   (rel RMS vs fp32)")
    for r in results:
        print(f"{r.variant:<12}" + "".join(f"{v:>14.2e}" for v in r.rel_rms_vs_fp32.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
