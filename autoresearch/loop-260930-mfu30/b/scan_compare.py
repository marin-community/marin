# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ragged a2a module variants inside a rematted layer scan.

Builds a scan over rematted ragged-EP MoE layers with per-layer routing and differentiated
routing weights, forward and backward, for each variant. Per variant it reports, per HLO
computation, the kernels that touch transport-sized buffers (fills, copies, Triton kernels,
QuACK GEMMs, ragged all-to-alls), compiler temp bytes, the loss, and the median step time with
the variant order rotated. It also checks every gradient against main: x, w13 and w2 exactly,
the routing weights to rounding.

Variants:
  control:   main's module, full remat.
  unfilled:  inverse permutation + chained cotangents + unfilled buffers, full remat.
  stack:     unfilled + PR #9481's pipelined chunks and mirror parameters, full remat.
  sonic:     unfilled + expert-side routing-weight gradient, remat saving the MoE output.
  candidate: the branch module, remat saving the MoE output (the hero's new policy).

Usage (GB200x4): python autoresearch/loop-260930-mfu30/b/scan_compare.py
"""

import importlib.util
import json
import pathlib
import re
import statistics
import time
from collections import Counter

import jax
import jax.numpy as jnp
import levanter.grug._moe.ep_ragged_all_to_all as candidate_module
import levanter.grug._moe.sonic_cute as sonic_cute
import levanter.grug.grug_moe as grug_moe
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

HERE = pathlib.Path(__file__).resolve().parent
LAYERS = 3
TOKENS_PER_SHARD = 65536
HIDDEN = 3072
INTER = 3072
TOPK = 8
CAPACITY_FACTOR = 1.15
MOE_OUTPUT = "moe_output"


def _load_frozen(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BACKWARD = sonic_cute._expert_mlp_quack_wgrad_backward


# name -> (module-local function, remat policy, QuACK backward)
_SAVE_OUTPUT = jax.checkpoint_policies.save_only_these_names(MOE_OUTPUT)
VARIANTS = {
    "control": (_load_frozen("control_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local, None, _BACKWARD),
    "unfilled": (_load_frozen("unfilled_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local, None, _BACKWARD),
    # unfilled + PR #9481's pipelined chunks and mirror transpose parameters (mfu30-stack 423e8c50e4).
    "stack": (_load_frozen("stack_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local, None, _BACKWARD),
    # unfilled + expert-side routing-weight gradient (ce112504f1).
    "sonic": (_load_frozen("sonic_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local, _SAVE_OUTPUT, _BACKWARD),
    "candidate": (candidate_module._moe_mlp_ep_ragged_a2a_local, _SAVE_OUTPUT, _BACKWARD),
}


def _mesh():
    devices = np.array(jax.devices()).reshape(1, len(jax.devices()), 1)
    return Mesh(devices, ("data", "expert", "model"), axis_types=(AxisType.Explicit,) * 3)


def _inputs(mesh):
    shards = mesh.shape["expert"]
    experts = 6 * shards
    tokens = TOKENS_PER_SHARD * shards
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((tokens, experts), dtype=np.float32)
    selected = np.argsort(-scores, axis=1)[:, :TOPK].astype(np.int32)
    weights = rng.random((LAYERS, tokens, TOPK), dtype=np.float32) + 0.05
    weights /= weights.sum(axis=-1, keepdims=True)
    token = NamedSharding(mesh, P(("data", "expert"), None))
    token1 = NamedSharding(mesh, P(("data", "expert")))
    stacked = NamedSharding(mesh, P(None, "expert", None, None))
    stacked_token = NamedSharding(mesh, P(None, ("data", "expert"), None))
    bf16 = jnp.bfloat16
    return dict(
        experts=experts,
        selected=jax.device_put(jnp.asarray(selected), token),
        valid=jax.device_put(jnp.ones((tokens,), bool), token1),
        x=jax.device_put(jnp.asarray(rng.standard_normal((tokens, HIDDEN), dtype=np.float32), bf16), token),
        weights=jax.device_put(jnp.asarray(weights, bf16), stacked_token),
        w13=jax.device_put(
            jnp.asarray(rng.standard_normal((LAYERS, experts, HIDDEN, 2 * INTER), dtype=np.float32) * 0.02, bf16),
            stacked,
        ),
        w2=jax.device_put(
            jnp.asarray(rng.standard_normal((LAYERS, experts, INTER, HIDDEN), dtype=np.float32) * 0.02, bf16),
            stacked,
        ),
    )


def _build(mesh, inp, local_fn, policy, backward):
    def layer(x, ws):
        w13_l, w2_l, weights_l, shift = ws
        grug_moe._moe_mlp_ep_ragged_a2a_local = local_fn
        sonic_cute._expert_mlp_quack_wgrad_backward = backward
        # Routing varies per layer, as in the model, so nothing routing-derived is loop invariant.
        out = grug_moe.moe_mlp(
            x,
            (inp["selected"] + shift) % inp["experts"],
            weights_l,
            w13_l,
            w2_l,
            token_valid=inp["valid"],
            implementation="ragged_all_to_all",
            mesh=mesh,
            capacity_factor=CAPACITY_FACTOR,
        )
        out = checkpoint_name(out, MOE_OUTPUT)
        # A nonlinear consumer, like the model's norm and short conv, so the backward needs the value.
        return x + jnp.tanh(out.astype(jnp.float32)).astype(x.dtype), None

    def loss(x, w13, w2, weights):
        shifts = jnp.arange(LAYERS, dtype=jnp.int32)
        body = layer if policy == "none" else jax.checkpoint(layer, policy=policy)
        y, _ = jax.lax.scan(body, x, (w13, w2, weights, shifts))
        return jnp.sum(y.astype(jnp.float32) ** 2)

    args = (inp["x"], inp["w13"], inp["w2"], inp["weights"])
    with jax.set_mesh(mesh):
        compiled = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3))).lower(*args).compile()
    sonic_cute._expert_mlp_quack_wgrad_backward = _BACKWARD
    return compiled, args


def _census(hlo_text, big_shapes, small_shapes):
    """Count kernels touching transport-sized buffers per computation."""
    counts = Counter()
    computation = "?"
    for line in hlo_text.splitlines():
        header = re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
        if header:
            computation = "entry" if header.group(1) else header.group(2)
            continue
        if "=" not in line:
            continue
        result = line.split("=", 1)[1]
        result_shape = result.split("(")[0]
        kind = None
        if "CutlassCall" in line:
            kind = "quack"
        elif re.search(r"ragged-all-to-all(-start)?\(", result):
            kind = "a2a_small" if any(s in result_shape for s in small_shapes) else "a2a"
        elif any(shape in result_shape for shape in big_shapes):
            if re.search(r" copy\(", result):
                kind = "copy"
            elif "calls=%fused_broadcast" in line or "calls=fused_broadcast" in line:
                kind = "fill"
            elif "triton_kernel_call" in line:
                kind = "triton_buffer"
        elif "triton_kernel_call" in line and f"[{TOKENS_PER_SHARD},{HIDDEN}]" in result_shape:
            kind = "gather_sum"
        if kind:
            counts[f"{computation}:{kind}"] += 1
    return dict(counts)


def _row_dot_fusions(hlo_text):
    """Name the fusions that pack the SwiGLU backward or reduce [C, I] rows, and what each holds."""
    fusions = {}
    name = None
    body = []
    for line in hlo_text.splitlines():
        header = re.match(r"^%?([\w.\-]+) .*\{\s*$", line)
        if header:
            name, body = header.group(1), []
            continue
        if line.startswith("}") and name:
            text = "\n".join(body)
            has_or = " or(" in text
            has_reduce = " reduce(" in text
            if has_or or (has_reduce and f",{INTER}]" in text):
                fusions[name] = dict(packs_swiglu=has_or, reduces=has_reduce)
            name = None
            continue
        if name:
            body.append(line)
    return fusions


def _compare(reference, other):
    names = ["loss", "d_x", "d_w13", "d_w2", "d_weights"]
    ref = [reference[0], *reference[1]]
    oth = [other[0], *other[1]]
    result = {}
    for n, a, b in zip(names, ref, oth, strict=True):
        fa = np.asarray(jax.device_get(a), np.float32)
        fb = np.asarray(jax.device_get(b), np.float32)
        diff = np.abs(fa - fb)
        scale = float(np.max(np.abs(fa))) or 1.0
        nonzero = np.abs(fa) > 0
        result[n] = dict(
            equal=bool(np.array_equal(fa, fb)),
            finite=bool(np.isfinite(fb).all()),
            max_rel=float(np.max(diff)) / scale,
            median_rel=float(np.median(diff[nonzero] / np.abs(fa[nonzero]))) if nonzero.any() else 0.0,
        )
    return result


def main():
    mesh = _mesh()
    chunk_capacity = int(np.ceil(np.ceil(CAPACITY_FACTOR * TOKENS_PER_SHARD * TOPK) / 2))
    big_shapes = (f"[{TOKENS_PER_SHARD * TOPK},{HIDDEN}]", f"[{chunk_capacity},{HIDDEN}]")
    small_shapes = (f"[{TOKENS_PER_SHARD * TOPK},1]", f"[{chunk_capacity},1]")
    inp = _inputs(mesh)
    compiled = {name: _build(mesh, inp, *spec) for name, spec in VARIANTS.items()}
    results = {}
    for name, (exe, args) in compiled.items():
        stats = exe.memory_analysis()
        results[name] = exe(*args)
        text = exe.as_text()
        print(
            json.dumps(
                dict(
                    variant=name,
                    census=_census(text, big_shapes, small_shapes),
                    swiglu_backward_fusions=_row_dot_fusions(text) if name.startswith("candidate") else None,
                    temp_bytes=None if stats is None else int(stats.temp_size_in_bytes),
                    loss=float(results[name][0]),
                )
            ),
            flush=True,
        )
    print(
        json.dumps(dict(gradients_vs_control={n: _compare(results["control"], r) for n, r in results.items()})),
        flush=True,
    )
    names = list(VARIANTS)
    times = {name: [] for name in names}
    for rotation in range(len(names)):
        for name in names[rotation:] + names[:rotation]:
            exe, args = compiled[name]
            for _ in range(2):
                jax.block_until_ready(exe(*args))
            samples = []
            for _ in range(10):
                start = time.perf_counter()
                jax.block_until_ready(exe(*args))
                samples.append(time.perf_counter() - start)
            times[name].append(statistics.median(samples))
    medians = {name: statistics.median(t) for name, t in times.items()}
    print(
        json.dumps(dict(median_seconds=medians, speedup_vs_control={n: medians["control"] / medians[n] for n in names})),
        flush=True,
    )


if __name__ == "__main__":
    main()
