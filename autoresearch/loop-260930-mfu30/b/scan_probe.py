# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where do the ragged a2a buffer inits land in a rematted layer scan?

Builds a scan over rematted ragged-EP MoE layers (forward and backward), compiles it on the
available GPUs, and counts, per HLO computation, the instructions that fill or copy transport-sized
buffers: broadcast fusions (zero fills), `copy` instructions, and `AllocateBuffer` custom calls.
Variants:
  zeros: the branch module as is (loop-local zero inits).
  empty: every loop-local zero init replaced by `jax.lax.empty` (an uninitialized buffer). Values
         are wrong in unwritten rows, so only the HLO and timing are meaningful.
  triton_empty: as empty, from a Triton kernel that writes nothing and reads the loop-variant tie.

Usage (GB200x4): python autoresearch/loop-260930-mfu30/b/scan_probe.py
"""

import json
import re
import statistics
import time
from collections import Counter

import jax
import jax.numpy as jnp
import jax_triton as jt
import levanter.grug._moe.ep_ragged_all_to_all as ragged
import levanter.grug.grug_moe as grug_moe
import numpy as np
import triton
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

LAYERS = 3
TOKENS_PER_SHARD = 16384
HIDDEN = 1024
INTER = 1024
TOPK = 8
CAPACITY_FACTOR = 1.15

_zeros = ragged._loop_local_zeros


def _empty(rows, hidden_dim, dtype, tie, site):
    del tie, site
    return jax.lax.empty((rows, hidden_dim), dtype)


@triton.jit
def _leave_output_unwritten(tie_ptr, out_ptr):
    pass


def _triton_empty(rows, hidden_dim, dtype, tie, site):
    """An uninitialized buffer from a kernel that writes nothing, tied to a loop-variant input."""
    del site
    return jt.triton_call(
        tie,
        kernel=_leave_output_unwritten,
        out_shape=jax.ShapeDtypeStruct((rows, hidden_dim), dtype),
        grid=(1,),
    )


def _mesh():
    devices = np.array(jax.devices()).reshape(1, len(jax.devices()), 1)
    return Mesh(devices, ("data", "expert", "model"), axis_types=(AxisType.Explicit,) * 3)


def _build(mesh, init_fn):
    shards = mesh.shape["expert"]
    experts = 6 * shards
    tokens = TOKENS_PER_SHARD * shards
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((tokens, experts), dtype=np.float32)
    selected = np.argsort(-scores, axis=1)[:, :TOPK].astype(np.int32)
    weights = np.full((tokens, TOPK), 1.0 / TOPK, dtype=np.float32)
    token = NamedSharding(mesh, P(("data", "expert"), None))
    token1 = NamedSharding(mesh, P(("data", "expert")))
    stacked = NamedSharding(mesh, P(None, "expert", None, None))
    bf16 = jnp.bfloat16
    selected = jax.device_put(jnp.asarray(selected), token)
    weights = jax.device_put(jnp.asarray(weights, bf16), token)
    valid = jax.device_put(jnp.ones((tokens,), bool), token1)
    x = jax.device_put(jnp.asarray(rng.standard_normal((tokens, HIDDEN), dtype=np.float32), bf16), token)
    w13 = jax.device_put(
        jnp.asarray(rng.standard_normal((LAYERS, experts, HIDDEN, 2 * INTER), dtype=np.float32) * 0.03, bf16), stacked
    )
    w2 = jax.device_put(
        jnp.asarray(rng.standard_normal((LAYERS, experts, INTER, HIDDEN), dtype=np.float32) * 0.03, bf16), stacked
    )

    def layer(x, ws):
        w13_l, w2_l, shift = ws
        ragged._loop_local_zeros = init_fn
        # Routing varies per layer, as in the model, so nothing routing-derived is loop invariant.
        layer_selected = (selected + shift) % experts
        out = grug_moe.moe_mlp(
            x,
            layer_selected,
            weights,
            w13_l,
            w2_l,
            token_valid=valid,
            implementation="ragged_all_to_all",
            mesh=mesh,
            capacity_factor=CAPACITY_FACTOR,
        )
        return x + out.astype(x.dtype), None

    def loss(x, w13, w2):
        shifts = jnp.arange(LAYERS, dtype=jnp.int32)
        y, _ = jax.lax.scan(jax.checkpoint(layer), x, (w13, w2, shifts))
        return jnp.sum(y.astype(jnp.float32))

    with jax.set_mesh(mesh):
        compiled = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(x, w13, w2).compile()
    return compiled, (x, w13, w2)


def _census(hlo_text, big_shapes, verbose=False):
    """Count fills/copies/allocations of transport-sized buffers per computation kind."""
    counts = Counter()
    computation = "?"
    for line in hlo_text.splitlines():
        header = re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
        if header:
            computation = "entry" if header.group(1) else header.group(2)
            continue
        if not any(shape in line.split("=")[1] if "=" in line else False for shape in big_shapes):
            continue
        kind = None
        if "AllocateBuffer" in line:
            kind = "allocate"
        elif re.search(r"= \S+ copy\(", line):
            kind = "copy"
        elif re.search(r"fusion\(.*calls=%?fused_broadcast|kind=kLoop, calls=%?fused_broadcast", line):
            kind = "broadcast_fusion"
        if kind:
            counts[f"{computation}:{kind}"] += 1
            if verbose:
                print(f"  [{computation}] {line.strip()[:260]}", flush=True)
    return dict(counts)


def main():
    mesh = _mesh()
    shards = mesh.shape["expert"]
    chunk_capacity = int(np.ceil(np.ceil(CAPACITY_FACTOR * TOKENS_PER_SHARD * TOPK) / 2))
    big_shapes = (f"[{TOKENS_PER_SHARD * TOPK},{HIDDEN}]", f"[{chunk_capacity},{HIDDEN}]")
    print(json.dumps(dict(devices=len(jax.devices()), shards=shards, big_shapes=big_shapes)), flush=True)
    for name, init_fn in (("zeros", _zeros), ("empty", _empty), ("triton_empty", _triton_empty)):
        compiled, args = _build(mesh, init_fn)
        text = compiled.as_text()
        for _ in range(2):
            jax.block_until_ready(compiled(*args))
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            jax.block_until_ready(compiled(*args))
            samples.append(time.perf_counter() - start)
        stats = compiled.memory_analysis()
        print(
            json.dumps(
                dict(
                    variant=name,
                    census=_census(text, big_shapes, verbose=True),
                    median_seconds=statistics.median(samples),
                    temp_bytes=None if stats is None else int(stats.temp_size_in_bytes),
                )
            ),
            flush=True,
        )
    ragged._loop_local_zeros = _zeros


if __name__ == "__main__":
    main()
