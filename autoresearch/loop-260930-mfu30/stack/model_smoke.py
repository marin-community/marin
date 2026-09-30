"""Model-level GB200x4 smoke of the stacked hero code path at a reduced shape.

Runs `Transformer.next_token_loss` forward and backward on an EP4 mesh with the ragged
all-to-all backend, the carry-offload remat policy (which, with B's D, saves the routed MoE
output), #9481's model commits, and the fused gated norm on and off. Checks finite loss and
gradients, compares the two norm settings, and reports compiled temp memory. Needs the hero env
(cuda_async allocator, ragged XLA flags, overlap limit 1).

Usage (GB200x4): python autoresearch/loop-260930-mfu30/stack/model_smoke.py
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.model import OFFLOAD_CARRY_REMAT_MODE, QbEstimator

BATCH, SEQ = 8, 1024

CFG = model.GrugModelConfig(
    vocab_size=2048,
    hidden_dim=1024,
    intermediate_dim=512,
    shared_expert_intermediate_dim=512,
    num_shared_experts=2,
    num_experts=16,
    num_experts_per_token=4,
    latent_dim=512,
    num_layers=4,
    num_heads=8,
    num_kv_heads=2,
    local_kv_heads=2,
    global_kv_heads=1,
    head_dim=128,
    max_seq_len=SEQ,
    sliding_window=512,
    global_every=4,
    capacity_factor=1.15,
    initializer_std=0.5 / math.sqrt(1024),
    qk_mult=1.3,
    sconv=True,
    attention_implementation="gpu_fa4_cute_sm100",
    moe_implementation="ragged_all_to_all",
    report_capacity_overflow=True,
    rope_fused=True,
    remat_mode=OFFLOAD_CARRY_REMAT_MODE,
    qb_estimator=QbEstimator.HIST,
    qb_hist_bins=1000,
)


def run(mesh, cfg):
    with jax.set_mesh(mesh):
        m = model.Transformer.init(cfg, key=jax.random.key(0))
        m = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jnp.floating) else a, m)
        tokens = jax.device_put(
            jax.random.randint(jax.random.key(1), (BATCH, SEQ), 0, cfg.vocab_size),
            NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None)),
        )
        weight = jax.device_put(jnp.ones((BATCH, SEQ), jnp.float32), NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None)))
        step = jax.jit(jax.value_and_grad(lambda mm: mm.next_token_loss(tokens, weight)))
        compiled = step.lower(m).compile()
        loss, grads = step(m)
        loss = float(loss)
        leaves = [np.asarray(g, np.float32) for g in jax.tree.leaves(grads)]
    return loss, leaves, compiled.memory_analysis().temp_size_in_bytes / 2**30


def main():
    devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
    results = {}
    for impl in (None, "pallas_gpu"):
        loss, leaves, temp = run(mesh, dataclasses.replace(CFG, gated_norm_implementation=impl))
        finite = all(np.all(np.isfinite(g)) for g in leaves)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in leaves))
        results[impl] = leaves
        print(f"gated_norm={impl}: loss {loss:.6f} grad-norm {norm:.4e} finite {finite} temp {temp:.2f} GiB", flush=True)
        assert finite, impl
    worst = max(
        float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        for a, b in zip(results[None], results["pallas_gpu"], strict=True)
    )
    print(f"worst per-leaf relative gradient difference, fused vs modules: {worst:.3e}", flush=True)
    print("MODEL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
