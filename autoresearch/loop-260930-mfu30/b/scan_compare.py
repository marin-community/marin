# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ragged a2a module variants inside a rematted layer scan.

Builds a scan over rematted ragged-EP MoE layers with per-layer routing, forward and backward,
for each module variant (frozen copies in this directory plus the branch's live module). For
each it reports the compiled HLO's fills, copies and no-op kernels on transport-sized buffers per
computation, compiler temp bytes, and the median step time with the variant order rotated.

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
import levanter.grug.grug_moe as grug_moe
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

HERE = pathlib.Path(__file__).resolve().parent
LAYERS = 3
TOKENS_PER_SHARD = 32768
HIDDEN = 2048
INTER = 2048
TOPK = 8
CAPACITY_FACTOR = 1.15


def _load_frozen(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VARIANTS = {
    "control": _load_frozen("control_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local,
    "chain": _load_frozen("chain_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local,
    "candidate": candidate_module._moe_mlp_ep_ragged_a2a_local,
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
    token = NamedSharding(mesh, P(("data", "expert"), None))
    token1 = NamedSharding(mesh, P(("data", "expert")))
    stacked = NamedSharding(mesh, P(None, "expert", None, None))
    bf16 = jnp.bfloat16
    return dict(
        experts=experts,
        selected=jax.device_put(jnp.asarray(selected), token),
        weights=jax.device_put(jnp.full((tokens, TOPK), 1.0 / TOPK, bf16), token),
        valid=jax.device_put(jnp.ones((tokens,), bool), token1),
        x=jax.device_put(jnp.asarray(rng.standard_normal((tokens, HIDDEN), dtype=np.float32), bf16), token),
        w13=jax.device_put(
            jnp.asarray(rng.standard_normal((LAYERS, experts, HIDDEN, 2 * INTER), dtype=np.float32) * 0.02, bf16),
            stacked,
        ),
        w2=jax.device_put(
            jnp.asarray(rng.standard_normal((LAYERS, experts, INTER, HIDDEN), dtype=np.float32) * 0.02, bf16),
            stacked,
        ),
    )


def _build(mesh, inp, local_fn):
    def layer(x, ws):
        w13_l, w2_l, shift = ws
        grug_moe._moe_mlp_ep_ragged_a2a_local = local_fn
        # Routing varies per layer, as in the model, so nothing routing-derived is loop invariant.
        out = grug_moe.moe_mlp(
            x,
            (inp["selected"] + shift) % inp["experts"],
            inp["weights"],
            w13_l,
            w2_l,
            token_valid=inp["valid"],
            implementation="ragged_all_to_all",
            mesh=mesh,
            capacity_factor=CAPACITY_FACTOR,
        )
        return x + out.astype(x.dtype), None

    def loss(x, w13, w2):
        shifts = jnp.arange(LAYERS, dtype=jnp.int32)
        y, _ = jax.lax.scan(jax.checkpoint(layer), x, (w13, w2, shifts))
        return jnp.sum(y.astype(jnp.float32))

    args = (inp["x"], inp["w13"], inp["w2"])
    with jax.set_mesh(mesh):
        compiled = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2))).lower(*args).compile()
    return compiled, args


def _census(hlo_text, big_shapes):
    """Count fills, copies and no-op kernels of transport-sized buffers per computation."""
    counts = Counter()
    computation = "?"
    for line in hlo_text.splitlines():
        header = re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
        if header:
            computation = "entry" if header.group(1) else header.group(2)
            continue
        if "=" not in line or not any(shape in line.split("=")[1] for shape in big_shapes):
            continue
        if re.search(r"= \S+ copy\(", line):
            kind = "copy"
        elif "calls=%fused_broadcast" in line or "calls=fused_broadcast" in line:
            kind = "fill"
        elif "triton_kernel_call" in line and "custom-call(" in line:
            kind = "triton"
        else:
            continue
        counts[f"{computation}:{kind}"] += 1
    return dict(counts)


def main():
    mesh = _mesh()
    chunk_capacity = int(np.ceil(np.ceil(CAPACITY_FACTOR * TOKENS_PER_SHARD * TOPK) / 2))
    big_shapes = (f"[{TOKENS_PER_SHARD * TOPK},{HIDDEN}]", f"[{chunk_capacity},{HIDDEN}]")
    inp = _inputs(mesh)
    compiled = {name: _build(mesh, inp, fn) for name, fn in VARIANTS.items()}
    losses = {}
    for name, (exe, args) in compiled.items():
        stats = exe.memory_analysis()
        value, _grads = exe(*args)
        losses[name] = float(value)
        print(
            json.dumps(
                dict(
                    variant=name,
                    census=_census(exe.as_text(), big_shapes),
                    temp_bytes=None if stats is None else int(stats.temp_size_in_bytes),
                    loss=losses[name],
                )
            ),
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
        json.dumps(
            dict(
                median_seconds=medians,
                speedup_vs_control={n: medians["control"] / medians[n] for n in names},
                losses_equal=len(set(losses.values())) == 1,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
