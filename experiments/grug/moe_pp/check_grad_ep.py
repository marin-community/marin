# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-vs-real gradient parity for the inline ring-EP pipeline.

Compares :func:`experiments.grug.moe_pp.pipeline.pipeline_value_and_grad_ep` -- the
single ``{stage, expert}``-manual ``shard_map`` running the PRODUCTION ring EP
(``_moe_mlp_ep_ring_local``: all_gather dispatch + ragged_dot GMM + psum_scatter
collect) INLINE -- against the unmodified non-pipelined oracle
``Transformer.next_token_loss`` (which uses the same ring ``moe_mlp``). Same params,
same tokens. Both sides run the real sparse ring EP, so this is real-vs-real parity,
not the dense-vs-dense check in ``check_grad_autodiff.py``.

Three compositions on the forced 8-CPU mesh:

- ``PP x EP``        : ``(stage=2, data=1, expert=2)``
- ``PP x FSDP``      : ``(stage=2, data=4, expert=1)``
- ``PP x FSDP x EP`` : ``(stage=2, data=2, expert=2)``

The pipeline uses an :func:`ep_pipeline_mesh` (``stage``/``expert`` Explicit, the rest
Auto) so ``ragged_dot_general`` lowers and GSPMD inserts the FSDP weight-grad reduce.
The expert/router grads -- the ones a mis-scaled EP transpose corrupts -- are reported
as a dedicated group.

Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        uv run python -m experiments.grug.moe_pp.check_grad_ep
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

from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.check_forward import CONFIG, SEQ_LEN, STAGE
from experiments.grug.moe_pp.oracle import oracle_loss
from experiments.grug.moe_pp.pipeline import (
    ep_pipeline_mesh,
    pipeline_value_and_grad_ep,
    stack_blocks_for_stages,
)

logger = logging.getLogger(__name__)

BATCH = 16
LOSS_TOL = 1e-4
GRAD_REL_TOL = 1e-3


def _set_capacity_free(model: Transformer, expert_axis_size: int) -> Transformer:
    """Set every expert MLP's ``capacity_factor`` to ``expert_axis_size`` (= ep_size).

    With ``local_capacity = ceil(capacity_factor * assignments / ep_size)``, choosing
    ``capacity_factor = ep_size`` makes ``local_capacity == assignments`` -- the
    maximum a shard could receive -- so the ring drops NOTHING. Both the oracle and
    the pipeline then route every assignment, making the EP forward (and grads)
    exactly comparable instead of differing by the few layout-dependent drops a tight
    capacity would discard.
    """
    capacity_factor = float(expert_axis_size)

    def _fix(block):
        expert_mlp = dataclasses.replace(block.mlp.expert_mlp, capacity_factor=capacity_factor)
        return dataclasses.replace(block, mlp=dataclasses.replace(block.mlp, expert_mlp=expert_mlp))

    return dataclasses.replace(model, blocks=tuple(_fix(b) for b in model.blocks))


def _rel_err(ref: jax.Array, got: jax.Array) -> float:
    a, b = np.asarray(ref), np.asarray(got)
    max_abs = float(np.max(np.abs(a - b)))
    return max_abs / (float(np.max(np.abs(a))) + 1e-12)


def _group_rel_err(ref_leaves, got_leaves) -> float:
    return max((_rel_err(a, b) for a, b in zip(ref_leaves, got_leaves, strict=True)), default=0.0)


def _expert_router_leaves(stage_block_arrays) -> list[jax.Array]:
    """Router + expert-MLP grad leaves inside a ``[stage, layers_per_stage, ...]`` tree."""
    mlp = stage_block_arrays.mlp
    return [
        mlp.router,
        mlp.router_bias,
        *jax.tree_util.tree_leaves(mlp.expert_mlp),
    ]


def _run_config(label: str, expert: int, replica: int, data: int, model_key, tokens, weight) -> bool:
    oracle_mesh = compact_grug_mesh(
        expert_axis_size=expert, replica_axis_size=replica, model_axis_size=1, stage_axis_size=1
    )
    pipe_mesh = ep_pipeline_mesh(stage=STAGE, expert=expert, replica=replica, data=data)

    # Oracle: unmodified production ring-EP loss, non-pipelined, on stage=1. Run
    # capacity-free (capacity_factor = ep_size) so neither side drops assignments.
    with set_mesh(oracle_mesh):
        oracle_model = _set_capacity_free(Transformer.init(CONFIG, key=model_key), expert)
        oracle_arrays, oracle_static = eqx.partition(oracle_model, eqx.is_array)

        def loss_fn(arrays):
            m = eqx.combine(arrays, oracle_static)
            return oracle_loss(m, tokens, weight)

        ref_loss, ref_grads = jax.jit(jax.value_and_grad(loss_fn))(oracle_arrays)
    ref_loss = float(np.asarray(ref_loss))

    # Pipeline: init under the same compact Explicit mesh (deterministic in model_key),
    # then run the value_and_grad under the EP mesh.
    pipe_init_mesh = compact_grug_mesh(
        expert_axis_size=expert, replica_axis_size=replica, model_axis_size=1, stage_axis_size=STAGE
    )
    with set_mesh(pipe_init_mesh):
        pipe_model = _set_capacity_free(Transformer.init(CONFIG, key=model_key), expert)
        stage_arrays, block_static = stack_blocks_for_stages(pipe_model, STAGE)

    with set_mesh(pipe_mesh):

        @jax.jit
        def run(model, stage_arrays):
            return pipeline_value_and_grad_ep(
                model,
                stage_arrays,
                block_static,
                tokens,
                weight,
                mesh=pipe_mesh,
                num_stages=STAGE,
            )

        pipe_loss, g_embed, g_stage = run(pipe_model, stage_arrays)
    pipe_loss = float(np.asarray(pipe_loss))

    g_te, g_en, g_eg, g_fn, g_fgn, g_op = g_embed
    embed_ref = [
        ref_grads.token_embed,
        *jax.tree_util.tree_leaves(ref_grads.embed_norm),
        *jax.tree_util.tree_leaves(ref_grads.embed_gated_norm),
        *jax.tree_util.tree_leaves(ref_grads.final_norm),
        *jax.tree_util.tree_leaves(ref_grads.final_gated_norm),
        ref_grads.output_proj,
    ]
    embed_got = [
        g_te,
        *jax.tree_util.tree_leaves(g_en),
        *jax.tree_util.tree_leaves(g_eg),
        *jax.tree_util.tree_leaves(g_fn),
        *jax.tree_util.tree_leaves(g_fgn),
        g_op,
    ]

    block_ref_arrays = [eqx.partition(b, eqx.is_array)[0] for b in ref_grads.blocks]
    layers_per_stage = CONFIG.num_layers // STAGE
    stacked_ref = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *block_ref_arrays)
    stacked_ref = jax.tree_util.tree_map(lambda x: x.reshape((STAGE, layers_per_stage, *x.shape[1:])), stacked_ref)

    stage_ref_leaves = jax.tree_util.tree_leaves(stacked_ref)
    stage_got_leaves = jax.tree_util.tree_leaves(g_stage)
    er_ref_leaves = _expert_router_leaves(stacked_ref)
    er_got_leaves = _expert_router_leaves(g_stage)

    loss_diff = abs(ref_loss - pipe_loss)
    embed_rel = _group_rel_err(embed_ref, embed_got)
    stage_rel = _group_rel_err(stage_ref_leaves, stage_got_leaves)
    er_rel = _group_rel_err(er_ref_leaves, er_got_leaves)
    ok = np.isfinite(pipe_loss) and loss_diff < LOSS_TOL and max(embed_rel, stage_rel, er_rel) < GRAD_REL_TOL

    logger.info(
        "[%s] mesh=%s loss=%.6f oracle=%.6f diff=%.2e | grad rel: embed=%.2e stage=%.2e expert/router=%.2e -> %s",
        label,
        dict(pipe_mesh.shape),
        pipe_loss,
        ref_loss,
        loss_diff,
        embed_rel,
        stage_rel,
        er_rel,
        "PASS" if ok else "FAIL",
    )
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    key = jax.random.PRNGKey(0)
    k_model, k_tokens = jax.random.split(key, 2)
    tokens = jax.random.randint(k_tokens, (BATCH, SEQ_LEN), 0, CONFIG.vocab_size, dtype=jnp.int32)
    weight = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.float32)

    ok_ep = _run_config("PP x EP        (stage=2,data=1,expert=2)", 2, 2, 1, k_model, tokens, weight)
    ok_fsdp = _run_config("PP x FSDP      (stage=2,data=4,expert=1)", 1, 1, 4, k_model, tokens, weight)
    ok_both = _run_config("PP x FSDP x EP (stage=2,data=2,expert=2)", 2, 1, 2, k_model, tokens, weight)

    ok = ok_ep and ok_fsdp and ok_both
    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
