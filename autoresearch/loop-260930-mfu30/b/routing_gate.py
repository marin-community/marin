# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exactness + component-timing gate for ragged all-to-all MoE routing changes.

Runs the ragged EP `moe_mlp` forward and backward once per variant: the branch's
`ep_ragged_all_to_all` module (candidate) and frozen module copies in this directory
(control = main's module). Reports bitwise equality of the output and of the
gradients with respect to x, combine weights and both expert weight banks, plus same-code
repeatability of the control and a component timing of each variant.

Usage (on a 4-GPU GB200 node): python autoresearch/loop-260930-mfu30/b/routing_gate.py [--quick]
"""

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import levanter.grug._moe.ep_ragged_all_to_all as candidate_module
import levanter.grug.grug_moe as grug_moe
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

HERE = pathlib.Path(__file__).resolve().parent


def _load_frozen(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# control: main f38da1173d. sonic: inverse permutation + chained cotangents + unfilled transport
# buffers + expert-side routing-weight gradient (ce112504f1). candidate: the branch's live module.
VARIANTS = {
    "control": _load_frozen("control_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local,
    "sonic": _load_frozen("sonic_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local,
    "candidate": candidate_module._moe_mlp_ep_ragged_a2a_local,
}
# Outputs whose values must match main exactly. The routing-weight gradient is compared to rounding.
EXACT_OUTPUTS = ("out", "dropped", "d_x", "d_w13", "d_w2")
ROUNDED_OUTPUT = "d_weights"
# bf16 carries 8 significand bits; allow a few ulps of the largest gradient.
ROUNDED_MAX_REL_TOL = 2.0**-6


def _mesh() -> Mesh:
    devices = np.array(jax.devices()).reshape(1, len(jax.devices()), 1)
    return Mesh(devices, ("data", "expert", "model"), axis_types=(AxisType.Explicit,) * 3)


def _inputs(case, mesh):
    rng = np.random.default_rng(case["seed"])
    shards = mesh.shape["expert"]
    tokens = case["tokens_per_shard"] * shards
    hidden, inter, experts, topk = case["hidden"], case["inter"], case["experts"], case["topk"]
    scores = rng.standard_normal((tokens, experts), dtype=np.float32)
    if case["routing"] == "skewed":
        scores += np.linspace(2.0, 0.0, experts, dtype=np.float32)[None, :]
    elif case["routing"] == "one_hot":
        scores[:, 0] += 100.0
    selected = np.argsort(-scores, axis=1)[:, :topk].astype(np.int32)
    # Weights from their own logits: the routing boost above would drive most weights to exact
    # zero in bf16, a case the expert-side weight gradient maps to zero by design.
    logits = rng.standard_normal((tokens, topk), dtype=np.float32)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    valid = np.ones((tokens,), dtype=bool)
    if case["padded"]:
        valid[rng.random(tokens) < 0.125] = False
    x = rng.standard_normal((tokens, hidden), dtype=np.float32)
    w13 = rng.standard_normal((experts, hidden, 2 * inter), dtype=np.float32) / np.sqrt(hidden)
    w2 = rng.standard_normal((experts, inter, hidden), dtype=np.float32) / np.sqrt(inter)
    ct = rng.standard_normal((tokens, hidden), dtype=np.float32)

    token = NamedSharding(mesh, P(("data", "expert"), None))
    token1 = NamedSharding(mesh, P(("data", "expert")))
    expert = NamedSharding(mesh, P("expert", None, None))
    bf16 = jnp.bfloat16
    return dict(
        x=jax.device_put(jnp.asarray(x, bf16), token),
        selected=jax.device_put(jnp.asarray(selected), token),
        weights=jax.device_put(jnp.asarray(weights, bf16), token),
        valid=jax.device_put(jnp.asarray(valid), token1),
        w13=jax.device_put(jnp.asarray(w13, bf16), expert),
        w2=jax.device_put(jnp.asarray(w2, bf16), expert),
        ct=jax.device_put(jnp.asarray(ct, bf16), token),
    )


def _build(variant, case, mesh, inp):
    local_fn = VARIANTS[variant]

    def forward(x, weights, w13, w2):
        grug_moe._moe_mlp_ep_ragged_a2a_local = local_fn
        out, counts = grug_moe.moe_mlp(
            x,
            inp["selected"],
            weights,
            w13,
            w2,
            token_valid=inp["valid"],
            implementation="ragged_all_to_all",
            mesh=mesh,
            capacity_factor=case["capacity_factor"],
            report_capacity_overflow=True,
        )
        return out, counts.dropped

    def loss_and_grads(x, weights, w13, w2, ct):
        def loss(x, weights, w13, w2):
            out, dropped = forward(x, weights, w13, w2)
            return jnp.sum(out.astype(jnp.float32) * ct.astype(jnp.float32)), (out, dropped)

        (_value, (out, dropped)), grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(
            x, weights, w13, w2
        )
        return out, dropped, grads

    with jax.set_mesh(mesh):
        return jax.jit(loss_and_grads).lower(inp["x"], inp["weights"], inp["w13"], inp["w2"], inp["ct"]).compile()


def _bits(a):
    a = np.asarray(jax.device_get(a))
    return a.view(np.uint16) if a.dtype == jnp.bfloat16 else a


def _compare(a, b):
    names = ["out", "dropped", "d_x", "d_weights", "d_w13", "d_w2"]
    flat_a = [a[0], a[1], *a[2]]
    flat_b = [b[0], b[1], *b[2]]
    result = {}
    for name, u, v in zip(names, flat_a, flat_b, strict=True):
        bu, bv = _bits(u), _bits(v)
        fu = np.asarray(jax.device_get(u), dtype=np.float32)
        fv = np.asarray(jax.device_get(v), dtype=np.float32)
        diff = np.abs(fu - fv)
        scale = float(np.max(np.abs(fu))) if fu.size else 0.0
        nonzero = np.abs(fu) > 0
        result[name] = dict(
            bitwise_equal=bool(np.array_equal(bu, bv)),
            # Equal values, counting +0 and -0 as equal; NaN never compares equal.
            value_equal=bool(np.array_equal(fu, fv)),
            max_abs_diff=float(np.max(diff)) if fu.size else 0.0,
            # Largest difference relative to the largest reference magnitude.
            max_rel_diff=float(np.max(diff)) / scale if scale else 0.0,
            # Median elementwise relative difference over nonzero reference entries.
            median_rel_diff=float(np.median(diff[nonzero] / np.abs(fu[nonzero]))) if nonzero.any() else 0.0,
            fraction_equal=float(np.mean(fu == fv)) if fu.size else 1.0,
            finite=bool(np.isfinite(fu).all() and np.isfinite(fv).all()),
        )
    return result


def _acceptable(comparison):
    exact = all(comparison[k]["value_equal"] and comparison[k]["finite"] for k in EXACT_OUTPUTS)
    rounded = comparison[ROUNDED_OUTPUT]
    return exact and rounded["finite"] and rounded["max_rel_diff"] <= ROUNDED_MAX_REL_TOL


def _time(compiled, inp, iters):
    args = (inp["x"], inp["weights"], inp["w13"], inp["w2"], inp["ct"])
    for _ in range(2):
        jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _temp_bytes(compiled):
    stats = compiled.memory_analysis()
    return None if stats is None else int(stats.temp_size_in_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small shapes only")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    mesh = _mesh()
    shards = mesh.shape["expert"]
    base = dict(hidden=3072, inter=3072, experts=6 * shards, topk=8, capacity_factor=1.15, padded=False, seed=0)
    cases = [
        dict(base, name="small-uniform", tokens_per_shard=2048, hidden=512, inter=512, routing="uniform"),
        dict(base, name="small-skewed-drops", tokens_per_shard=2048, hidden=512, inter=512, routing="skewed"),
        dict(base, name="small-padded", tokens_per_shard=2048, hidden=512, inter=512, routing="uniform", padded=True),
        dict(base, name="small-one-hot", tokens_per_shard=2048, hidden=512, inter=512, routing="one_hot"),
    ]
    if not args.quick:
        cases += [
            dict(base, name="hero-uniform", tokens_per_shard=65536, routing="uniform"),
            dict(base, name="hero-skewed-drops", tokens_per_shard=65536, routing="skewed", padded=True),
        ]
    print(json.dumps(dict(devices=[str(d) for d in jax.devices()], jax=jax.__version__)), flush=True)
    failures = 0
    for case in cases:
        inp = _inputs(case, mesh)
        compiled = {v: _build(v, case, mesh, inp) for v in VARIANTS}
        args_ = (inp["x"], inp["weights"], inp["w13"], inp["w2"], inp["ct"])
        control_a = compiled["control"](*args_)
        control_b = compiled["control"](*args_)
        treated = {v: compiled[v](*args_) for v in VARIANTS if v != "control"}
        exact = {v: _compare(control_a, out) for v, out in treated.items()}
        repeat = _compare(control_a, control_b)
        ok = all(_acceptable(cmp) for cmp in exact.values())
        failures += not ok
        record = dict(
            case=case["name"],
            exact=ok,
            dropped=int(jax.device_get(control_a[1])),
            vs_control=exact,
            control_repeatable=all(r["bitwise_equal"] for r in repeat.values()),
            control_repeat_detail={k: v["max_abs_diff"] for k, v in repeat.items()},
            temp_bytes={v: _temp_bytes(c) for v, c in compiled.items()},
        )
        if case["tokens_per_shard"] >= 65536:
            # Rotate the order so no variant always runs first.
            names = list(VARIANTS)
            times = {v: [] for v in names}
            for rotation in range(len(names)):
                for v in names[rotation:] + names[:rotation]:
                    times[v].append(_time(compiled[v], inp, args.iters))
            medians = {v: statistics.median(t) for v, t in times.items()}
            record["median_step_seconds"] = medians
            record["speedup_vs_control"] = {v: medians["control"] / medians[v] for v in names if v != "control"}
        print(json.dumps(record), flush=True)
        del compiled, inp
    print(json.dumps(dict(failures=failures)), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
