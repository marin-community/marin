# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count transports and expert GEMMs in each scan body of the hero model's grad jaxpr.

Traces the stack's `stack/model_smoke.py` config on CPU (no compile), so it checks JAX-level
dead-code elimination only. Run from a stack checkout:
  XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu python <this file> reference
"""
import collections
import dataclasses
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


def body_counts(jaxpr, prims, out, path="top"):
    c = collections.Counter()
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in prims:
            c[eqn.primitive.name] += 1
        for param in eqn.params.values():
            subs = []
            if isinstance(param, jax_core.ClosedJaxpr):
                subs = [param.jaxpr]
            elif isinstance(param, jax_core.Jaxpr):
                subs = [param]
            elif isinstance(param, (tuple, list)):
                subs = [
                    p.jaxpr if isinstance(p, jax_core.ClosedJaxpr) else p
                    for p in param
                    if isinstance(p, (jax_core.ClosedJaxpr, jax_core.Jaxpr))
                ]
            for i, sub in enumerate(subs):
                sub_path = f"{path}/{eqn.primitive.name}"
                if eqn.primitive.name == "scan":
                    out.append((sub_path + f"#{len(out)}", body_counts(sub, prims, out, sub_path)))
                else:
                    c += body_counts(sub, prims, out, sub_path)
    return c


cfg = dataclasses.replace(ms.CFG, attention_implementation=sys.argv[1] if len(sys.argv) > 1 else "reference")
devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
with jax.set_mesh(mesh):
    m = model.Transformer.init(cfg, key=jax.random.key(0))
    tokens = jax.device_put(
        jax.random.randint(jax.random.key(1), (ms.BATCH, ms.SEQ), 0, cfg.vocab_size),
        NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None)),
    )
    weight = jax.device_put(
        jnp.ones((ms.BATCH, ms.SEQ), jnp.float32), NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    )
    jaxpr = jax.make_jaxpr(jax.value_and_grad(lambda mm: mm.next_token_loss(tokens, weight)))(m)
prims = {"ragged_all_to_all", "ragged_dot_general", "optimization_barrier", "all_gather", "custom_vjp_call"}
out = []
top = body_counts(jaxpr.jaxpr, prims, out)
for path, c in out:
    print(path, dict(c))
