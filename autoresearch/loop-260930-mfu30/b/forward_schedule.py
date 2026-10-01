# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the scheduler puts the shared-expert forward GEMMs relative to the ragged transports.

Compiles (does not run) the hero train step on GB200x4 for two configs: C's model smoke (`stack/model_smoke.py`)
and a hero-shaped EP4 model (hero per-GPU shapes: 16 x 4096 tokens, d6144, latent 3072, 6 local experts, top-8,
2 shared experts; 4 layers). Prints the scheduled order of the forward scan body: ragged all-to-all starts/dones,
expert GEMMs (EX), shared-expert GEMMs (SH), other GEMMs (G), collectives with their scope, and barriers.

Usage (GB200x4, hero env, from a stack checkout): python autoresearch/loop-260930-mfu30/b/forward_schedule.py
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
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.hero_recipe import HERO_MODEL_CONFIG, with_transport_remat_mode

INST = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
OP_NAME = re.compile(r'op_name="([^"]*)"')
COLLECTIVE = re.compile(
    r"\s(all-reduce-start|all-gather-start|reduce-scatter|all-to-all|collective-permute-start|all-reduce|all-gather)\("
)


def _bodies(text):
    bodies, name, lines = {}, None, []
    for line in text.splitlines():
        header = re.match(r"^%?([\w.\-]+) .*\{\s*$", line)
        if header:
            name, lines = header.group(1), []
            continue
        if line.startswith("}") and name:
            bodies[name] = lines
            name = None
            continue
        if name:
            lines.append(line)
    return bodies


def _token(rest):
    op = OP_NAME.search(rest)
    op = op.group(1) if op else ""
    if "ragged-all-to-all-start(" in rest:
        shapes = re.findall(r"\[(\d+),(\d+)\]", rest.split("ragged-all-to-all-start(")[0])
        return f"S:a2a[{shapes[0][0]}->{shapes[1][0]}]" if len(shapes) >= 2 else "S:a2a"
    if "ragged-all-to-all-done(" in rest:
        return "D:a2a"
    if "CutlassCall" in rest:
        return "EX"
    is_gemm = "__cublas$gemm" in rest or "__cublas$lt" in rest or ("kind=kCustom" in rest and "gemm" in rest)
    if is_gemm:
        return "SH" if "DenseMLP" in op else "G"
    m = COLLECTIVE.search(" " + rest)
    if m:
        tail = "/".join(op.split("/")[-2:])[-40:]
        return f"C:{m.group(1)}({tail})"
    if " opt-barrier(" in rest:
        return "B"
    return None


def _forward_schedule(text):
    bodies = _bodies(text)
    starts = {n: sum("ragged-all-to-all-start(" in line for line in b) for n, b in bodies.items()}
    forward = [n for n, c in starts.items() if c == 4]
    if not forward:
        return f"no 4-transport body; counts {sorted(set(starts.values()))}"
    body = bodies[forward[0]]
    events = []
    for line in body:
        m = INST.match(line)
        if not m:
            continue
        tok = _token(m.group(2))
        if tok:
            events.append(tok)
    out = []
    for e in events:
        if out and out[-1][0] == e:
            out[-1][1] += 1
        else:
            out.append([e, 1])
    return " ".join(f"{e}x{n}" if n > 1 else e for e, n in out)


def _compile(cfg, batch, seq):
    devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
    with jax.set_mesh(mesh):
        m = model.Transformer.init(cfg, key=jax.random.key(0))
        m = jax.tree.map(
            lambda a: (
                a.astype(jnp.bfloat16) if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jnp.floating) else a
            ),
            m,
        )
        spec = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        tokens = jax.device_put(jax.random.randint(jax.random.key(1), (batch, seq), 0, cfg.vocab_size), spec)
        weight = jax.device_put(jnp.ones((batch, seq), jnp.float32), spec)
        step = jax.jit(jax.value_and_grad(lambda mm: mm.next_token_loss(tokens, weight)))
        return step.lower(m).compile().as_text()


def main():
    shards = len(jax.devices())
    hero4 = with_transport_remat_mode(
        dataclasses.replace(HERO_MODEL_CONFIG, vocab_size=2048, num_layers=4, num_experts=6 * shards)
    )
    for name, cfg, batch, seq in (("smoke", ms.CFG, ms.BATCH, ms.SEQ), ("hero4", hero4, 16 * shards, 4096)):
        text = _compile(cfg, batch, seq)
        sched = _forward_schedule(text)
        counts = collections.Counter(t.split("x")[0] for t in sched.split())
        print(f"== {name}: {dict(counts)}\n{sched}\n", flush=True)


if __name__ == "__main__":
    main()
