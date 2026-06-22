# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step-2 gradient parity: the manual-backward pipelined production Transformer
vs the autodiff oracle.

``jax.value_and_grad`` of the pipelined loss is structurally blocked (see
``check_grad_blocked.py``), so :func:`experiments.grug.moe_pp.pipeline.pipeline_value_and_grad`
runs the backward BY HAND inside the stage-manual ``shard_map``: a forward sweep
captures per-(stage, microbatch) ``jax.vjp`` closures, then a backward sweep seeds
the loss cotangent on the last stage and ``ppermute``s activation-cotangents
upstream. This compares its ``(loss, grads)`` against
``jax.value_and_grad(oracle_loss)`` where ``oracle_loss`` is the UNMODIFIED,
non-pipelined ``Transformer.next_token_loss`` run at ``stage=1``.

Three compositions are checked on the forced 8-CPU mesh, exactly as the forward
de-risk (:mod:`experiments.grug.moe_pp.check_forward`):

- ``PP x EP``        : ``(stage=2, data=1, expert=2)``
- ``PP x FSDP``      : ``(stage=2, data=4, expert=1)``
- ``PP x FSDP x EP`` : ``(stage=2, data=2, expert=2)``

Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        uv run python -m experiments.grug.moe_pp.check_grad
"""

from __future__ import annotations

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
from experiments.grug.moe_pp.pipeline import pipeline_value_and_grad, stack_blocks_for_stages

logger = logging.getLogger(__name__)

NUM_MICROBATCHES = 2
MICROBATCH = 8
BATCH = NUM_MICROBATCHES * MICROBATCH
LOSS_TOL = 1e-4
GRAD_REL_TOL = 1e-3


def _rel_err(ref: jax.Array, got: jax.Array) -> tuple[float, float]:
    a, b = np.asarray(ref), np.asarray(got)
    max_abs = float(np.max(np.abs(a - b)))
    rel = max_abs / (float(np.max(np.abs(a))) + 1e-12)
    return max_abs, rel


def _group_rel_err(ref_leaves, got_leaves) -> tuple[float, float]:
    max_abs = 0.0
    rel = 0.0
    for a, b in zip(ref_leaves, got_leaves, strict=True):
        ma, r = _rel_err(a, b)
        max_abs = max(max_abs, ma)
        rel = max(rel, r)
    return max_abs, rel


def _run_config(label: str, expert: int, replica: int, model_key, tokens, weight, mb_tokens, mb_weight) -> bool:
    pipe_mesh = compact_grug_mesh(
        expert_axis_size=expert, replica_axis_size=replica, model_axis_size=1, stage_axis_size=STAGE
    )
    oracle_mesh = compact_grug_mesh(
        expert_axis_size=expert, replica_axis_size=replica, model_axis_size=1, stage_axis_size=1
    )

    # Oracle: value_and_grad of the UNMODIFIED non-pipelined production loss at stage=1.
    with set_mesh(oracle_mesh):
        oracle_model = Transformer.init(CONFIG, key=model_key)
        oracle_arrays, _ = eqx.partition(oracle_model, eqx.is_array)

        def loss_fn(arrays):
            m = eqx.combine(arrays, eqx.partition(oracle_model, eqx.is_array)[1])
            return oracle_loss(m, tokens, weight)

        ref_loss, ref_grads = jax.jit(jax.value_and_grad(loss_fn))(oracle_arrays)
    ref_loss = float(np.asarray(ref_loss))

    with set_mesh(pipe_mesh):
        pipe_model = Transformer.init(CONFIG, key=model_key)
        stage_arrays, block_static = stack_blocks_for_stages(pipe_model, STAGE)

        @jax.jit
        def run(model, stage_arrays):
            return pipeline_value_and_grad(
                model,
                stage_arrays,
                block_static,
                mb_tokens,
                mb_weight,
                mesh=pipe_mesh,
                num_stages=STAGE,
                num_microbatches=NUM_MICROBATCHES,
            )

        pipe_loss, g_embed, g_stage = run(pipe_model, stage_arrays)
    pipe_loss = float(np.asarray(pipe_loss))

    # Map the oracle's full-Transformer grads onto the pipeline's two grad groups.
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

    # The oracle's per-block grads are a tuple of Block grads; stack them into the
    # pipeline's [stage, layers_per_stage, ...] layout to compare leaf-for-leaf.
    block_ref_arrays = [eqx.partition(b, eqx.is_array)[0] for b in ref_grads.blocks]
    layers_per_stage = CONFIG.num_layers // STAGE
    stacked_ref = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *block_ref_arrays)
    stacked_ref = jax.tree_util.tree_map(lambda x: x.reshape((STAGE, layers_per_stage, *x.shape[1:])), stacked_ref)
    stage_ref_leaves = jax.tree_util.tree_leaves(stacked_ref)
    stage_got_leaves = jax.tree_util.tree_leaves(g_stage)

    loss_diff = abs(ref_loss - pipe_loss)
    _, embed_rel = _group_rel_err(embed_ref, embed_got)
    _, stage_rel = _group_rel_err(stage_ref_leaves, stage_got_leaves)
    ok = np.isfinite(pipe_loss) and loss_diff < LOSS_TOL and max(embed_rel, stage_rel) < GRAD_REL_TOL

    logger.info(
        "[%s] mesh=%s loss=%.6f diff=%.2e | grad rel: embed=%.2e stage=%.2e -> %s",
        label,
        dict(pipe_mesh.shape),
        pipe_loss,
        loss_diff,
        embed_rel,
        stage_rel,
        "PASS" if ok else "FAIL",
    )
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    key = jax.random.PRNGKey(0)
    k_model, k_tokens = jax.random.split(key, 2)
    tokens = jax.random.randint(k_tokens, (BATCH, SEQ_LEN), 0, CONFIG.vocab_size, dtype=jnp.int32)
    weight = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.float32)
    # The pipeline consumes [num_microbatches, microbatch, seq]; the oracle the flat
    # [batch, seq]. Reshape preserves token order so both score the same global batch.
    mb_tokens = tokens.reshape(NUM_MICROBATCHES, MICROBATCH, SEQ_LEN)
    mb_weight = weight.reshape(NUM_MICROBATCHES, MICROBATCH, SEQ_LEN)

    ok_ep = _run_config("PP x EP        (stage=2,data=1,expert=2)", 2, 2, k_model, tokens, weight, mb_tokens, mb_weight)
    ok_fsdp = _run_config(
        "PP x FSDP      (stage=2,data=4,expert=1)", 1, 1, k_model, tokens, weight, mb_tokens, mb_weight
    )
    ok_both = _run_config(
        "PP x FSDP x EP (stage=2,data=2,expert=2)", 2, 1, k_model, tokens, weight, mb_tokens, mb_weight
    )

    ok = ok_ep and ok_fsdp and ok_both
    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
