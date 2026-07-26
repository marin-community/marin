# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP8W-006: sm100 parity and timing for the MXFP8 forward-dispatch wire (#7665).

Compares two arms of the ring EP backend that differ only in how the dispatch
buffer reaches the MXFP8 grouped kernels:

  control    bf16 dispatch collective, op quantizes on arrival (today)
  treatment  quantize before the collective, payload feeds the kernels directly

The forward operand is bit-identical between the arms by construction; the wgrad
operand is rebuilt from the arrived payload, which CPU study FP8W-001 measured at
2.7e-6 to 1.1e-3 of the error the control already carries. This script checks
that on real kernels, and reports the byte and step-time deltas.

Needs a Blackwell (sm100) node with at least 2 visible GPUs for the expert axis.

Usage:
  python experiments/grug/moe/standalone/test_mxfp8_dispatch_gpu.py --tokens 512 --hidden 512
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

_REPO = Path(__file__).resolve().parents[4]
for _p in (str(_REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from levanter.grug.grug_moe import moe_mlp  # noqa: E402

from experiments.grug.moe.mxfp8 import MxFp8MoeMlpOp  # noqa: E402


def relfrob(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def build_mesh():
    devices = jax.devices()
    if len(devices) < 2:
        raise SystemExit(f"needs >= 2 GPUs for the expert axis; saw {len(devices)}")
    return Mesh(np.array(devices), axis_names=("expert",), axis_types=(AxisType.Explicit,))


def run_arm(mesh, tensors, *, mxfp8_dispatch, producer):
    x, sel, cw, w13, w2, cot = tensors
    op = MxFp8MoeMlpOp(producer=producer)

    def forward(x_, w13_, w2_):
        return moe_mlp(
            x_,
            sel,
            cw,
            w13_,
            w2_,
            implementation="ring",
            mesh=mesh,
            expert_mlp_op=op,
            mxfp8_dispatch=mxfp8_dispatch,
        )

    def loss(x_, w13_, w2_):
        return jnp.sum(forward(x_, w13_, w2_) * cot)

    fwd = jax.jit(forward)
    grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    out = fwd(x, w13, w2)
    grads = grad(x, w13, w2)
    jax.block_until_ready((out, grads))
    return out, grads, fwd, grad


def time_fn(fn, args, *, iters=20, warmup=5):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - start) / iters * 1e3


def check_operands(args):
    """Compare the two paths' forward operands directly, outside the EP backend.

    FP8W-001 predicts the row-orientation operand is bit-identical whether the
    quantization happens before or after the dispatch. The end-to-end arms
    differ by ~1e-3, so this isolates where that comes from: it feeds one
    dispatch buffer through both forward pipelines and diffs the operands the
    grouped kernels actually receive.
    """
    from levanter.grug._moe.mxfp8_wire import quantize_mxfp8_rows  # noqa: PLC0415

    from experiments.grug.moe.mxfp8 import _forward_pipeline, _forward_pipeline_quantized  # noqa: PLC0415

    op = MxFp8MoeMlpOp(producer=args.producer)
    d, i, e = args.hidden, args.intermediate, 4
    capacity = 512
    keys = jax.random.split(jax.random.key(7), 4)
    x = (jax.random.normal(keys[0], (capacity, d), jnp.float32) * 0.5).astype(jnp.bfloat16)
    # Leave a tail of exactly-zero rows: dropped slots and pad rows are routine,
    # and they are where the two quantizers are known to differ.
    x = x.at[-96:].set(0)
    group_sizes = jnp.array([160, 128, 128, capacity - 416], jnp.int32)
    w13 = (jax.random.normal(keys[1], (e, d, 2 * i), jnp.float32) * 0.02).astype(jnp.bfloat16)
    w2 = (jax.random.normal(keys[2], (e, i, d), jnp.float32) * 0.02).astype(jnp.bfloat16)

    payload, scales = quantize_mxfp8_rows(x.astype(jnp.float32))
    ctl = jax.jit(lambda a, b, c: _forward_pipeline(op, a, b, c, group_sizes))(x, w13, w2)
    trt = jax.jit(lambda p, s, b, c: _forward_pipeline_quantized(op, p, s, b, c, group_sizes))(payload, scales, w13, w2)

    same_q = bool(jnp.all(ctl["x_q"].view(jnp.uint8) == trt["x_q"].view(jnp.uint8)))
    same_sf = bool(jnp.all(ctl["x_sf"].view(jnp.uint8) == trt["x_sf"].view(jnp.uint8)))
    same_col = bool(jnp.all(ctl["x_col"].view(jnp.uint8) == trt["x_col"].view(jnp.uint8)))
    print(
        json.dumps(
            {
                "row_operand_bit_identical": same_q,
                "row_scales_bit_identical": same_sf,
                "col_operand_bit_identical": same_col,
                "y_relfrob": relfrob(trt["y"], ctl["y"]),
                "control_x_q_has_nan": bool(jnp.any(jnp.isnan(ctl["x_q"].astype(jnp.float32)))),
                "treatment_x_q_has_nan": bool(jnp.any(jnp.isnan(trt["x_q"].astype(jnp.float32)))),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--intermediate", type=int, default=256)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--producer", default="xla", choices=("auto", "cute", "xla"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--check-operands", action="store_true", help="diff the two forward pipelines' operands")
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(
        f"device: {dev.device_kind} (cc {getattr(dev, 'compute_capability', '?')}), "
        f"jax {jax.__version__}, {len(jax.devices())} devices",
        flush=True,
    )

    if args.check_operands:
        check_operands(args)
        return

    mesh = build_mesh()
    keys = jax.random.split(jax.random.key(0), 6)
    t, d, i, e, k = args.tokens, args.hidden, args.intermediate, args.experts, args.topk

    with jax.set_mesh(mesh):
        shard = NamedSharding(mesh, P("expert"))
        x = jax.device_put(jax.random.normal(keys[0], (t, d), jnp.bfloat16), shard)
        cot = jax.device_put(jax.random.normal(keys[1], (t, d), jnp.bfloat16), shard)
        sel = jax.device_put(jax.random.randint(keys[2], (t, k), 0, e), shard)
        cw = jax.device_put(jax.random.uniform(keys[3], (t, k), dtype=jnp.bfloat16), shard)
        w13 = jax.random.normal(keys[4], (e, d, 2 * i), jnp.bfloat16) * 0.02
        w2 = jax.random.normal(keys[5], (e, i, d), jnp.bfloat16) * 0.02
        tensors = (x, sel, cw, w13, w2, cot)

        out_ctl, grad_ctl, fwd_ctl, gfn_ctl = run_arm(mesh, tensors, mxfp8_dispatch=False, producer=args.producer)
        out_wire, grad_wire, fwd_wire, gfn_wire = run_arm(mesh, tensors, mxfp8_dispatch=True, producer=args.producer)

        results = {
            "config": vars(args) | {"devices": len(jax.devices())},
            "forward_relfrob": relfrob(out_wire, out_ctl),
            "dx_relfrob": relfrob(grad_wire[0], grad_ctl[0]),
            "dw13_relfrob": relfrob(grad_wire[1], grad_ctl[1]),
            "dw2_relfrob": relfrob(grad_wire[2], grad_ctl[2]),
            "dx_nonzero": bool(np.any(np.asarray(grad_wire[0], np.float32) != 0)),
            "dispatch_bytes_ratio": 33.0 / 64.0,
        }

        results["fwd_ms_control"] = time_fn(fwd_ctl, (x, w13, w2), iters=args.iters)
        results["fwd_ms_wire"] = time_fn(fwd_wire, (x, w13, w2), iters=args.iters)
        results["grad_ms_control"] = time_fn(gfn_ctl, (x, w13, w2), iters=args.iters)
        results["grad_ms_wire"] = time_fn(gfn_wire, (x, w13, w2), iters=args.iters)

    results["fwd_speedup"] = results["fwd_ms_control"] / results["fwd_ms_wire"]
    results["grad_speedup"] = results["grad_ms_control"] / results["grad_ms_wire"]

    print("\nFP8W-006 RESULTS")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)

    if not results["dx_nonzero"]:
        raise SystemExit("FAIL: dispatch gradient is identically zero (the FP8W-005 failure mode)")
    print("\nOK: dispatch gradient is nonzero")


if __name__ == "__main__":
    main()
