# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate for the shared-expert holdback (`held_back_shared_experts`), run from a checkout that has it.

1. Model: C's model smoke (reference attention) with 0 and 1 held-back shared experts in one process:
   same parameter paths and shapes, and bitwise-equal loss, router metrics and gradients.
2. Module: `moe_mlp` vs `moe_mlp_with_holdback(holdback=x)` on the routing gate's cases: bitwise-equal
   output, drop counts and gradients, and the returned holdback equal to its input.
3. Schedule (informative; EP4 does not predict EP64): the scheduled forward scan body of a hero-shaped EP4
   model with 0 and 1 held-back shared experts.

Usage (GB200x4, hero env): python autoresearch/loop-260930-mfu30/b/holdback_gate.py
"""

import dataclasses
import hashlib
import json
import sys

sys.path.insert(0, "autoresearch/loop-260930-mfu30/stack")
sys.path.insert(0, "autoresearch/loop-260930-mfu30/b")
import forward_schedule
import jax
import jax.numpy as jnp
import levanter.grug.grug_moe as grug_moe
import model_smoke as ms
import numpy as np
import routing_gate
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.hero_recipe import HERO_MODEL_CONFIG, with_transport_remat_mode


def _digest(tree):
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        array = np.asarray(jax.device_get(leaf))
        out[jax.tree_util.keystr(path)] = hashlib.sha256(array.tobytes()).hexdigest()[:16]
    return out


def _model_part():
    devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
    results = {}
    shapes = {}
    for held in (0, 1):
        cfg = dataclasses.replace(ms.CFG, attention_implementation="reference", held_back_shared_experts=held)
        with jax.set_mesh(mesh):
            m = model.Transformer.init(cfg, key=jax.random.key(0))
            m = jax.tree.map(
                lambda a: (
                    a.astype(jnp.bfloat16) if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jnp.floating) else a
                ),
                m,
            )
            shapes[held] = [(jax.tree_util.keystr(p), leaf.shape) for p, leaf in jax.tree_util.tree_leaves_with_path(m)]
            spec = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
            tokens = jax.device_put(jax.random.randint(jax.random.key(1), (ms.BATCH, ms.SEQ), 0, cfg.vocab_size), spec)
            weight = jax.device_put(jnp.ones((ms.BATCH, ms.SEQ), jnp.float32), spec)
            step = jax.jit(
                jax.value_and_grad(
                    lambda mm, tokens=tokens, weight=weight: mm.next_token_loss(
                        tokens, weight, return_router_metrics=True
                    ),
                    has_aux=True,
                )
            )
            (loss, metrics), grads = step(m)
            results[held] = dict(loss=_digest(loss), metrics=_digest(metrics), grads=_digest(grads))
    differ = {
        part: [k for k in results[0][part] if results[0][part][k] != results[1][part].get(k)]
        for part in ("loss", "metrics", "grads")
    }
    ok = shapes[0] == shapes[1] and not any(differ.values())
    print(json.dumps(dict(part="model", ok=ok, same_params=shapes[0] == shapes[1], differ=differ)), flush=True)
    return ok


def _module_part():
    mesh = routing_gate._mesh()
    shards = mesh.shape["expert"]
    base = dict(hidden=3072, inter=3072, experts=6 * shards, topk=8, capacity_factor=1.15, padded=False, seed=0)
    cases = [
        dict(base, name="small-skewed-drops", tokens_per_shard=2048, hidden=512, inter=512, routing="skewed"),
        dict(base, name="small-padded", tokens_per_shard=2048, hidden=512, inter=512, routing="uniform", padded=True),
        dict(base, name="hero-skewed-drops", tokens_per_shard=65536, routing="skewed", padded=True),
    ]
    ok = True
    for case in cases:
        inp = routing_gate._inputs(case, mesh)

        def run(with_holdback, inp=inp, case=case):
            def loss(x, weights, w13, w2):
                kwargs = dict(token_valid=inp["valid"], mesh=mesh, capacity_factor=case["capacity_factor"])
                if with_holdback:
                    out, counts, held = grug_moe.moe_mlp_with_holdback(x, inp["selected"], weights, w13, w2, x, **kwargs)
                else:
                    out, counts = grug_moe.moe_mlp(
                        x,
                        inp["selected"],
                        weights,
                        w13,
                        w2,
                        implementation="ragged_all_to_all",
                        report_capacity_overflow=True,
                        **kwargs,
                    )
                    held = x
                value = jnp.sum(out.astype(jnp.float32) * inp["ct"].astype(jnp.float32))
                return value, (out, counts.sender_dropped, held)

            with jax.set_mesh(mesh):
                fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                (_, aux), grads = fn(inp["x"], inp["weights"], inp["w13"], inp["w2"])
            return jax.device_get((aux, grads))

        (aux0, g0), (aux1, g1) = run(False), run(True)
        record = dict(
            case=case["name"],
            out=bool(np.array_equal(np.asarray(aux0[0]), np.asarray(aux1[0]))),
            dropped=int(aux0[1]) == int(aux1[1]),
            holdback_is_input=bool(np.array_equal(np.asarray(aux1[2]), np.asarray(jax.device_get(inp["x"])))),
            grads={
                n: bool(np.array_equal(np.asarray(a), np.asarray(b)))
                for n, a, b in zip(("d_x", "d_weights", "d_w13", "d_w2"), g0, g1, strict=True)
            },
        )
        record["ok"] = (
            record["out"] and record["dropped"] and record["holdback_is_input"] and all(record["grads"].values())
        )
        ok = ok and record["ok"]
        print(json.dumps(dict(part="module", **record)), flush=True)
    return ok


def _schedule_part():
    shards = len(jax.devices())
    for held in (0, 1):
        cfg = with_transport_remat_mode(
            dataclasses.replace(
                HERO_MODEL_CONFIG, vocab_size=2048, num_layers=4, num_experts=6 * shards, held_back_shared_experts=held
            )
        )
        text = forward_schedule._compile(cfg, 16 * shards, 4096)
        print(f"== hero4 held_back_shared_experts={held}\n{forward_schedule._forward_schedule(text)}\n", flush=True)


def main():
    model_ok = _model_part()
    module_ok = _module_part()
    _schedule_part()
    print(json.dumps(dict(model_ok=model_ok, module_ok=module_ok)), flush=True)
    sys.exit(0 if model_ok and module_ok else 1)


if __name__ == "__main__":
    main()
