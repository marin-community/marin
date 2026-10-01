# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What each ragged all-to-all of the hero backward loop waits on, from the grad jaxpr.

Traces the stack's `stack/model_smoke.py` model (reference attention) on CPU, takes the backward scan body
(the scan body with the most ragged all-to-alls), inlines every call-like sub-jaxpr (pjit, shard_map,
checkpoint, custom_jvp/vjp calls) into one dataflow graph, and for every ragged all-to-all lists the earlier
ragged all-to-alls and the expert-MLP matmuls (`ragged_dot_general`) among its transitive inputs. Data
dependencies bind the GPU scheduler; everything else is its choice. A recomputed chunk-1 dispatch with the
chunk-0 recompute matmuls among its inputs cannot run under them.

Run from a stack checkout (optional integer config overrides as key=value):
  XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu python <this file> [key=value ...]
"""

import collections
import dataclasses
import re
import sys

sys.path.insert(0, "autoresearch/loop-260930-mfu30/stack")
import jax
import jax.numpy as jnp
import model_smoke as ms
import numpy as np
from jax.extend import core as jax_core
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import model

CHUNK = re.compile(r"moe_chunk_(\d+)")
CALL_PARAMS = ("jaxpr", "call_jaxpr")
OPAQUE = {"scan", "while", "cond"}


def _sub_jaxpr(eqn):
    if eqn.primitive.name in OPAQUE:
        return None
    for key in CALL_PARAMS:
        sub = eqn.params.get(key)
        if isinstance(sub, jax_core.ClosedJaxpr):
            sub = sub.jaxpr
        if (
            isinstance(sub, jax_core.Jaxpr)
            and len(sub.invars) == len(eqn.invars)
            and len(sub.outvars) == len(eqn.outvars)
        ):
            return sub
    return None


class Graph:
    """Flattened dataflow: node id -> (primitive, name stack, input node ids)."""

    def __init__(self):
        self.nodes = []

    def add(self, jaxpr, env, stack=""):
        for eqn in jaxpr.eqns:
            inputs = [env.get(v) for v in eqn.invars if isinstance(v, jax_core.Var)]
            inputs = [i for i in inputs if i is not None]
            name_stack = f"{stack}/{eqn.source_info.name_stack}"
            sub = _sub_jaxpr(eqn)
            if sub is not None:
                inner = {
                    iv: env.get(ov)
                    for iv, ov in zip(sub.invars, eqn.invars, strict=True)
                    if isinstance(ov, jax_core.Var)
                }
                self.add(sub, inner, name_stack)
                for outer, inner_var in zip(eqn.outvars, sub.outvars, strict=True):
                    if isinstance(inner_var, jax_core.Var) and inner_var in inner:
                        env[outer] = inner[inner_var]
                continue
            node = len(self.nodes)
            self.nodes.append((eqn.primitive.name, name_stack, inputs))
            for v in eqn.outvars:
                env[v] = node


def _bodies(jaxpr, out):
    for eqn in jaxpr.eqns:
        for param in eqn.params.values():
            subs = param if isinstance(param, (tuple, list)) else [param]
            for sub in subs:
                sub = sub.jaxpr if isinstance(sub, jax_core.ClosedJaxpr) else sub
                if isinstance(sub, jax_core.Jaxpr):
                    if eqn.primitive.name == "scan":
                        out.append(sub)
                    _bodies(sub, out)
    return out


def _count(jaxpr, prim):
    graph = Graph()
    graph.add(jaxpr, {})
    return sum(p == prim for p, _, _ in graph.nodes)


def _label(name_stack):
    """Recompute ops sit under the checkpoint's rematted computation; in the backward body the rest is backward."""
    chunk = CHUNK.findall(name_stack)
    phase = "remat" if "rematted_computation" in name_stack else "bwd"
    return f"{phase}:chunk{chunk[-1] if chunk else '?'}"


def main():
    # Optional integer config overrides as key=value arguments, e.g. held_back_shared_experts=1.
    overrides = {k: int(v) for k, v in (arg.split("=", 1) for arg in sys.argv[1:])}
    cfg = dataclasses.replace(ms.CFG, attention_implementation="reference", **overrides)
    devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
    with jax.set_mesh(mesh):
        m = model.Transformer.init(cfg, key=jax.random.key(0))
        spec = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        tokens = jax.device_put(jax.random.randint(jax.random.key(1), (ms.BATCH, ms.SEQ), 0, cfg.vocab_size), spec)
        weight = jax.device_put(jnp.ones((ms.BATCH, ms.SEQ), jnp.float32), spec)
        closed = jax.make_jaxpr(jax.value_and_grad(lambda mm: mm.next_token_loss(tokens, weight)))(m)
    bodies = _bodies(closed.jaxpr, [])
    body = max(bodies, key=lambda b: _count(b, "ragged_all_to_all"))
    graph = Graph()
    graph.add(body, {})
    nodes = graph.nodes

    ancestors = {}
    for i, (_, _, inputs) in enumerate(nodes):
        acc = set(inputs)
        for j in inputs:
            acc |= ancestors[j]
        ancestors[i] = acc

    a2a = [i for i, (p, _, _) in enumerate(nodes) if p == "ragged_all_to_all"]
    mlp = [i for i, (p, _, _) in enumerate(nodes) if p == "ragged_dot_general"]
    barriers = [i for i, (p, _, _) in enumerate(nodes) if p == "optimization_barrier"]
    print(f"backward body: {len(a2a)} ragged all-to-alls, {len(mlp)} ragged_dot_general, {len(barriers)} barriers")
    for k, i in enumerate(a2a):
        before = [f"#{a2a.index(j)}({_label(nodes[j][1])})" for j in a2a if j in ancestors[i]]
        matmuls = collections.Counter(_label(nodes[j][1]) for j in mlp if j in ancestors[i])
        print(f"#{k} {_label(nodes[i][1]):16s} after a2a {before}; after matmuls {dict(matmuls)}")
        print(f"     {nodes[i][1][-150:]}")


if __name__ == "__main__":
    main()
