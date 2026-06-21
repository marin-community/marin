# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the grug-MoE pipeline model's gradients against a non-pipelined reference.

Drives the reshard-free grug MoE transformer (``model.py``) through both pipeline
backends and checks the loss and per-group gradients against an autodiff oracle
that runs the same model sequentially with no mesh.

Run on a forced 8-device CPU mesh:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        uv run python -m experiments.grug.moe_zb.check_grug_parity
"""

from __future__ import annotations

import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_zb.model import GrugMoEConfig, build_pipeline_params, reference_loss
from experiments.grug.moe_zb.pipeline import (
    PipelineParams,
    pipeline_value_and_grad,
    zero_bubble_value_and_grad,
)

# M*MICROBATCH must be divisible by NUM_STAGES so the hoisted head's batch axis
# shards evenly over the stage axis (8 * 2 = 16, divisible by 8).
NUM_STAGES = 8
NUM_MICROBATCHES = 8
MICROBATCH = 2
SEQ_LEN = 16

LOSS_TOL = 1e-4
GRAD_REL_TOL = 1e-3

CONFIG = GrugMoEConfig(
    vocab_size=256,
    hidden_dim=32,
    intermediate_dim=64,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=8,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=SEQ_LEN,
    num_stages=NUM_STAGES,
)


def main() -> int:
    devices = np.array(jax.devices())
    assert devices.size == NUM_STAGES, f"need {NUM_STAGES} devices, got {devices.size}"
    mesh = Mesh(devices.reshape(NUM_STAGES), ("stage",), axis_types=(AxisType.Auto,))

    key = jax.random.PRNGKey(0)
    k_params, k_tokens = jax.random.split(key, 2)
    params, model = build_pipeline_params(CONFIG, key=k_params)
    tokens = jax.random.randint(k_tokens, (NUM_MICROBATCHES, MICROBATCH, SEQ_LEN), 0, CONFIG.vocab_size, dtype=jnp.int32)

    def batched_reference_loss(p: PipelineParams) -> jax.Array:
        losses = jnp.stack([reference_loss(p, model, tokens[m], CONFIG) for m in range(NUM_MICROBATCHES)])
        return jnp.mean(losses)

    ref_loss, ref_grads = jax.value_and_grad(batched_reference_loss)(params)

    hidden_shape = (MICROBATCH, SEQ_LEN, CONFIG.hidden_dim)
    with jax.set_mesh(mesh):
        params_sharded = PipelineParams(
            embed=jax.device_put(params.embed, NamedSharding(mesh, P())),
            stage=jax.device_put(params.stage, NamedSharding(mesh, P("stage"))),
            head=jax.device_put(params.head, NamedSharding(mesh, P())),
        )
        tokens_r = jax.device_put(tokens, NamedSharding(mesh, P()))
        gpipe_loss, gpipe_grads = pipeline_value_and_grad(
            params_sharded,
            tokens_r,
            tokens_r,
            model=model,
            mesh=mesh,
            num_microbatches=NUM_MICROBATCHES,
            hidden_shape=hidden_shape,
        )
        zb_loss, zb_grads = zero_bubble_value_and_grad(
            params_sharded,
            tokens_r,
            tokens_r,
            model=model,
            mesh=mesh,
            num_microbatches=NUM_MICROBATCHES,
            hidden_shape=hidden_shape,
        )

    ref_loss = np.asarray(ref_loss)

    def report(label: str, loss: jax.Array, grads: PipelineParams) -> bool:
        loss = np.asarray(loss)
        loss_diff = abs(ref_loss - loss)
        print(f"\n[{label}] loss={loss:.8f} diff_vs_ref={loss_diff:.2e}")
        rels = []
        for name, ref_group, group in (
            ("embed", ref_grads.embed, grads.embed),
            ("stage", ref_grads.stage, grads.stage),
            ("head", ref_grads.head, grads.head),
        ):
            ref_leaves = jax.tree_util.tree_leaves(ref_group)
            leaves = jax.tree_util.tree_leaves(group)
            max_abs = 0.0
            rel = 0.0
            for a, b in zip(ref_leaves, leaves, strict=True):
                a, b = np.asarray(a), np.asarray(b)
                leaf_max_abs = float(np.max(np.abs(a - b)))
                max_abs = max(max_abs, leaf_max_abs)
                rel = max(rel, leaf_max_abs / (float(np.max(np.abs(a))) + 1e-12))
            print(f"  grad {name:6s} max_abs_diff={max_abs:.2e}  rel={rel:.2e}")
            rels.append(rel)
        return loss_diff < LOSS_TOL and max(rels) < GRAD_REL_TOL

    ok_gpipe = report("GPipe autodiff oracle", gpipe_loss, gpipe_grads)
    ok_zb = report("Zero-bubble B/W split", zb_loss, zb_grads)
    ok = ok_gpipe and ok_zb
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
