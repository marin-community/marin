# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient parity for the device-group pipeline (pipeline_zb) vs the oracle.

Validates :func:`experiments.grug.moe_pp.pipeline_zb.zb_value_and_grad` -- stages
on disjoint device slices, microbatched, activations transported by ``device_put``
-- against the non-pipelined oracle on a forced 8-CPU mesh. The loss equals the
oracle's over the same global batch by construction (averaged over microbatches);
the grads must match the whole-program autodiff to float reassociation tolerance.
Unlike the single-mesh manual driver this is NOT bit-exact: microbatch averaging
and per-stage sub-meshes reassociate the sums.

Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        uv run python -m experiments.grug.moe_pp.check_zb
"""

from __future__ import annotations

import dataclasses
import logging
import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe_pp.oracle import oracle_loss
from experiments.grug.moe_pp.pipeline_manual import _EMBED_HEAD_FIELDS
from experiments.grug.moe_pp.pipeline_zb import zb_value_and_grad

logger = logging.getLogger(__name__)

BATCH = 16
SEQ_LEN = 64
LOSS_TOL = 1e-3
GRAD_REL_TOL = 5e-3

CONFIG = GrugModelConfig(
    vocab_size=512,
    hidden_dim=128,
    intermediate_dim=256,
    shared_expert_intermediate_dim=256,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=8,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=SEQ_LEN,
    sliding_window=SEQ_LEN,
    moe_implementation="ring",
    attention_implementation="reference",
)


def _set_capacity_free(model: Transformer, expert_axis_size: int) -> Transformer:
    """Set every expert MLP's ``capacity_factor`` to the expert axis size so no path drops tokens."""
    capacity_factor = float(expert_axis_size)

    def _fix(block):
        expert_mlp = dataclasses.replace(block.mlp.expert_mlp, capacity_factor=capacity_factor)
        return dataclasses.replace(block, mlp=dataclasses.replace(block.mlp, expert_mlp=expert_mlp))

    return dataclasses.replace(model, blocks=tuple(_fix(b) for b in model.blocks))


def _rel_err(ref, got) -> float:
    a, b = np.asarray(ref, dtype=np.float64), np.asarray(got, dtype=np.float64)
    return float(np.max(np.abs(a - b))) / (float(np.max(np.abs(a))) + 1e-12)


def _group_rel_err(ref_leaves, got_leaves) -> float:
    return max((_rel_err(a, b) for a, b in zip(ref_leaves, got_leaves, strict=True)), default=0.0)


def _expert_router_leaves(block_grads) -> list:
    leaves = []
    for b in block_grads:
        leaves += [b.mlp.expert_mlp.w_gate_up, b.mlp.expert_mlp.w_down, b.mlp.router]
    return leaves


def _run(label: str, *, num_stages: int, expert_per_stage: int, data_per_stage: int, num_microbatches: int) -> bool:
    model_key = jax.random.PRNGKey(0)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (BATCH, SEQ_LEN), 0, CONFIG.vocab_size, dtype=jnp.int32)
    weight = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.float32)

    # Oracle: full-mesh non-pipelined value_and_grad at the same expert axis as the stage
    # sub-meshes, so MoE capacity (no-drop) matches; values are otherwise mesh-independent.
    oracle_mesh = compact_grug_mesh(
        expert_axis_size=expert_per_stage, replica_axis_size=1, model_axis_size=1, stage_axis_size=1
    )
    with set_mesh(oracle_mesh):
        model = _set_capacity_free(Transformer.init(CONFIG, key=model_key), expert_per_stage)
        arrays, static = eqx.partition(model, eqx.is_array)
        ref_loss, ref_grads = jax.jit(jax.value_and_grad(lambda a: oracle_loss(eqx.combine(a, static), tokens, weight)))(
            arrays
        )

    pipe_loss, g_eh, g_blocks = zb_value_and_grad(
        model,
        tokens,
        weight,
        num_stages=num_stages,
        num_microbatches=num_microbatches,
        expert_per_stage=expert_per_stage,
        data_per_stage=data_per_stage,
    )

    ref_loss, pipe_loss = float(np.asarray(ref_loss)), float(np.asarray(pipe_loss))
    loss_diff = abs(ref_loss - pipe_loss)

    ref_eh = tuple(getattr(ref_grads, f) for f in _EMBED_HEAD_FIELDS)
    embed_rel = _group_rel_err(jax.tree_util.tree_leaves(ref_eh), jax.tree_util.tree_leaves(g_eh))
    block_rel = _group_rel_err(jax.tree_util.tree_leaves(list(ref_grads.blocks)), jax.tree_util.tree_leaves(g_blocks))
    er_rel = _group_rel_err(_expert_router_leaves(ref_grads.blocks), _expert_router_leaves(g_blocks))

    ok = np.isfinite(pipe_loss) and loss_diff < LOSS_TOL and max(embed_rel, block_rel, er_rel) < GRAD_REL_TOL
    logger.info(
        "[%-12s] stages=%d eps=%d dps=%d M=%d | loss diff=%.2e | grad rel embed=%.2e block=%.2e er=%.2e -> %s",
        label,
        num_stages,
        expert_per_stage,
        data_per_stage,
        num_microbatches,
        loss_diff,
        embed_rel,
        block_rel,
        er_rel,
        "PASS" if ok else "FAIL",
    )
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = jax.device_count()
    logger.info("device-group pipeline gradient parity on %d %s device(s)", n, jax.devices()[0].platform)
    if n < 8:
        logger.error("want 8 devices; got %d (set XLA_FLAGS=--xla_force_host_platform_device_count=8)", n)
        return 1

    ok = True
    ok = _run("pure-PP", num_stages=8, expert_per_stage=1, data_per_stage=1, num_microbatches=4) and ok
    ok = _run("PP-single-mb", num_stages=8, expert_per_stage=1, data_per_stage=1, num_microbatches=1) and ok
    ok = _run("PPxEP", num_stages=4, expert_per_stage=2, data_per_stage=1, num_microbatches=4) and ok
    ok = _run("PPxFSDP", num_stages=4, expert_per_stage=1, data_per_stage=2, num_microbatches=4) and ok
    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
