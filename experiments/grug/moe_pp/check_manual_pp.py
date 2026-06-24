# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient parity for the manually threaded pipeline vs the oracle.

Compares :func:`experiments.grug.moe_pp.pipeline_manual.manual_pp_value_and_grad`
-- the Python-driven per-stage ``jax.vjp`` backward, NO ``stage`` mesh axis --
against the unmodified non-pipelined oracle ``Transformer.next_token_loss``, on the
SAME mesh and the same params/tokens. The only difference is that the manual driver
decomposes the backward per stage; the loss is identical by construction, so the
grads must match the whole-program autodiff to float reassociation tolerance.

This is the keystone correctness check for the manual-threading direction: if the
per-stage vjp composition reproduces the oracle grads, PP needs no ``stage`` mesh
axis (and so the stage-stacked weight-grad that OOMs the GPU partitioner never
forms). EP and FSDP run as in production (real reshards, ring-EP nested shard_map);
each composition exercises a different (expert, data) layout under the logical
2-stage pipeline.

Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        uv run python -m experiments.grug.moe_pp.check_manual_pp
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
from experiments.grug.moe_pp.pipeline_manual import _EMBED_HEAD_FIELDS, manual_pp_value_and_grad

logger = logging.getLogger(__name__)

STAGE = 2
BATCH = 16
SEQ_LEN = 64
LOSS_TOL = 1e-4
GRAD_REL_TOL = 1e-3

CONFIG = GrugModelConfig(
    vocab_size=512,
    hidden_dim=128,
    intermediate_dim=256,
    shared_expert_intermediate_dim=256,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=4,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=SEQ_LEN,
    sliding_window=SEQ_LEN,
    moe_implementation="ring",
    attention_implementation="reference",
)


def _set_capacity_free(model: Transformer, expert_axis_size: int) -> Transformer:
    """Set every expert MLP's ``capacity_factor`` to ep_size so neither side drops tokens."""
    capacity_factor = float(expert_axis_size)

    def _fix(block):
        expert_mlp = dataclasses.replace(block.mlp.expert_mlp, capacity_factor=capacity_factor)
        return dataclasses.replace(block, mlp=dataclasses.replace(block.mlp, expert_mlp=expert_mlp))

    return dataclasses.replace(model, blocks=tuple(_fix(b) for b in model.blocks))


def _rel_err(ref: jax.Array, got: jax.Array) -> float:
    a, b = np.asarray(ref), np.asarray(got)
    return float(np.max(np.abs(a - b))) / (float(np.max(np.abs(a))) + 1e-12)


def _group_rel_err(ref_leaves, got_leaves) -> float:
    return max((_rel_err(a, b) for a, b in zip(ref_leaves, got_leaves, strict=True)), default=0.0)


def _expert_router_leaves(block_grads) -> list[jax.Array]:
    """The EP-sensitive leaves (a mis-scaled EP transpose corrupts these first)."""
    leaves = []
    for b in block_grads:
        leaves += [b.mlp.expert_mlp.w_gate_up, b.mlp.expert_mlp.w_down, b.mlp.router]
    return leaves


def _run_config(label: str, expert: int, replica: int, model_key, tokens, weight) -> bool:
    mesh = compact_grug_mesh(expert_axis_size=expert, replica_axis_size=replica, model_axis_size=1, stage_axis_size=1)
    data = mesh.shape["data"]

    with set_mesh(mesh):
        model = _set_capacity_free(Transformer.init(CONFIG, key=model_key), expert)
        arrays, static = eqx.partition(model, eqx.is_array)

        def loss_fn(arrays):
            return oracle_loss(eqx.combine(arrays, static), tokens, weight)

        ref_loss, ref_grads = jax.jit(jax.value_and_grad(loss_fn))(arrays)
        pipe_loss, eh_grads, block_grads = jax.jit(
            lambda m: manual_pp_value_and_grad(m, tokens, weight, num_stages=STAGE)
        )(model)

    ref_loss, pipe_loss = float(np.asarray(ref_loss)), float(np.asarray(pipe_loss))
    loss_diff = abs(ref_loss - pipe_loss)

    ref_eh = tuple(getattr(ref_grads, f) for f in _EMBED_HEAD_FIELDS)
    embed_rel = _group_rel_err(jax.tree_util.tree_leaves(ref_eh), jax.tree_util.tree_leaves(eh_grads))
    block_rel = _group_rel_err(jax.tree_util.tree_leaves(list(ref_grads.blocks)), jax.tree_util.tree_leaves(block_grads))
    er_rel = _group_rel_err(_expert_router_leaves(ref_grads.blocks), _expert_router_leaves(block_grads))

    ok = np.isfinite(pipe_loss) and loss_diff < LOSS_TOL and max(embed_rel, block_rel, er_rel) < GRAD_REL_TOL
    logger.info(
        "[%-11s] s=%d e=%d d=%d | loss diff=%.2e | grad rel: embed=%.2e block=%.2e expert/router=%.2e -> %s",
        label,
        STAGE,
        expert,
        data,
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
    logger.info("manual-PP gradient parity on %d %s device(s)", n, jax.devices()[0].platform)
    if n < 8:
        logger.error("want 8 devices; got %d (set XLA_FLAGS=--xla_force_host_platform_device_count=8)", n)
        return 1

    key = jax.random.PRNGKey(0)
    k_model, k_tokens = jax.random.split(key)
    tokens = jax.random.randint(k_tokens, (BATCH, SEQ_LEN), 0, CONFIG.vocab_size, dtype=jnp.int32)
    weight = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.float32)

    ok = True
    for label, expert, replica in (("EP", 2, 1), ("FSDP", 1, 1), ("EP+FSDP", 4, 1)):
        ok = _run_config(label, expert, replica, k_model, tokens, weight) and ok
    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
