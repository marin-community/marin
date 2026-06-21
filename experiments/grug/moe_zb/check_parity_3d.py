# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate PP x FSDP x EP gradients against a non-pipelined reference.

Drives the reshard-free grug MoE through the pipeline on a 3-D
``(stage, data, expert)`` mesh -- pipeline parallelism on ``stage``, FSDP on
``data``, expert parallelism on ``expert`` -- and checks the loss and per-group
gradients against the same single-device autodiff oracle the 1-D parity check
uses. Sharding must not change the math, so the gradients must match to
tolerance even with GSPMD inserting the FSDP all-gathers and EP reduces.

Run on a forced 32-device CPU mesh (4 x 4 x 2):

    XLA_FLAGS=--xla_force_host_platform_device_count=32 \
        uv run python -m experiments.grug.moe_zb.check_parity_3d
"""

from __future__ import annotations

import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_zb.model import GrugMoEConfig, build_pipeline_params, reference_loss
from experiments.grug.moe_zb.parallelism import DATA_AXIS, make_pipeline_mesh, shard_pipeline_params
from experiments.grug.moe_zb.pipeline import (
    PipelineParams,
    pipeline_value_and_grad,
    zero_bubble_value_and_grad,
)

NUM_STAGES = 4
NUM_DATA = 4
NUM_EXPERT = 2

NUM_MICROBATCHES = 6
MICROBATCH = 4
SEQ_LEN = 16

LOSS_TOL = 1e-4
GRAD_REL_TOL = 1e-3

CONFIG = GrugMoEConfig(
    vocab_size=64,
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
    mesh = make_pipeline_mesh(NUM_STAGES, NUM_DATA, NUM_EXPERT)

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
        params_sharded = shard_pipeline_params(params, mesh)
        # Replicate over stage (in_specs P()), data-parallel over `data` on the
        # microbatch axis so the activation flow exercises FSDP, not just params.
        tokens_r = jax.device_put(tokens, NamedSharding(mesh, P(None, DATA_AXIS, None)))
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

    ok_gpipe = report("GPipe (PPxFSDPxEP)", gpipe_loss, gpipe_grads)
    ok_zb = report("Zero-bubble (PPxFSDPxEP)", zb_loss, zb_grads)
    ok = ok_gpipe and ok_zb
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
