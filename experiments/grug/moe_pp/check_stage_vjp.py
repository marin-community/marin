# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step-1 make-or-break: does ``jax.vjp`` of ONE production stage's forward work
*inside* the stage-manual ``shard_map``?

The manual-backward pipeline plan (see ``check_grad_blocked.py`` for why autodiff
of the whole outer shard_map is impossible) hinges on differentiating each stage
locally, inside the ``stage``-manual region, with ``jax.vjp``. A stage forward is
(optionally embed) + a ``lax.scan`` over ``layers_per_stage`` production ``Block``s,
each carrying the inner EP ``moe_mlp`` shard_map over ``expert`` and the in-body
Explicit-axis ``_batch_reshard`` / ``out_sharding=`` reshards over ``data``/``expert``.

This script puts ``out, vjp_fn = jax.vjp(stage_fwd, hidden, stage_params)`` then
``dh, dparams = vjp_fn(cotangent)`` inside a ``shard_map(axis_names={"stage"},
check_vma=False)`` and runs it on the 8-CPU mesh (stage=2, data=2, expert=2). The
inner EP shard_map is a normal manual shard_map so it must transpose; the risk is
the per-stage-body Explicit reshards. If ``jax.vjp`` rejects them, the exact error
is logged and the approach must change.

Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        uv run python -m experiments.grug.moe_pp.check_stage_vjp
"""

from __future__ import annotations

import logging
import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import Transformer, _batch_spec
from experiments.grug.moe_pp.check_forward import CONFIG, SEQ_LEN, STAGE
from experiments.grug.moe_pp.pipeline import _run_stage_blocks, build_layer_masks, stack_blocks_for_stages

logger = logging.getLogger(__name__)

STAGE_AXIS = "stage"
BATCH = 16


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mesh = compact_grug_mesh(expert_axis_size=2, replica_axis_size=1, model_axis_size=1, stage_axis_size=STAGE)
    key = jax.random.PRNGKey(0)
    k_model, k_hidden = jax.random.split(key, 2)

    with set_mesh(mesh):
        model = Transformer.init(CONFIG, key=k_model)
        stage_arrays, block_static = stack_blocks_for_stages(model, STAGE)
        layer_masks = build_layer_masks(model, STAGE, SEQ_LEN)

        hidden_dim = CONFIG.hidden_dim
        # A per-stage activation [B, S, D]; the stage_fwd runs this stage's block scan.
        hidden = jax.random.normal(k_hidden, (BATCH, SEQ_LEN, hidden_dim), dtype=jnp.float32)
        cotangent = jnp.ones((BATCH, SEQ_LEN, hidden_dim), dtype=jnp.float32)

        stage_spec = P(STAGE_AXIS)
        repl = P()
        stage_in_specs = jax.tree_util.tree_map(lambda _: stage_spec, stage_arrays)

        def body(stage_arrays_in, masks, hidden_in, cot):
            # Squeeze the size-1 stage shard: [1, layers_per_stage, ...] -> [layers_per_stage, ...].
            stage_blocks = jax.tree_util.tree_map(lambda x: x[0], stage_arrays_in)
            stage_masks = masks[0]
            # Carry the production inter-layer token sharding so the scanned blocks
            # see exactly what they reshard to internally.
            hidden_local = reshard(hidden_in, _batch_spec())

            def stage_fwd(h, blocks):
                # Returns just the activation; the z-loss is a separate scalar output
                # of _run_stage_blocks but we differentiate the activation path.
                out, _z = _run_stage_blocks(blocks, block_static, h, stage_masks)
                return out

            out, vjp_fn = jax.vjp(stage_fwd, hidden_local, stage_blocks)
            # The stage output carries the Explicit (replica_dcn,data,expert) batch
            # sharding from its final _batch_reshard; the cotangent vjp_fn expects must
            # match that aval sharding, so reshard the incoming cotangent to it.
            cot = reshard(cot, _batch_spec())
            dh, dparams = vjp_fn(cot)
            # Return the activation, its input-cotangent, and the summed param-grad
            # magnitude so XLA can't DCE the vjp.
            dparam_norm = sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(dparams))
            return out, dh, dparam_norm

        def _place(x, spec):
            return reshard(x, NamedSharding(mesh, spec))

        stage_arrays = jax.tree_util.tree_map(_place, stage_arrays, stage_in_specs)
        layer_masks = _place(layer_masks, stage_spec)
        hidden = _place(hidden, repl)
        cotangent = _place(cotangent, repl)

        run = jax.jit(
            lambda sa, lm, h, c: jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(stage_in_specs, stage_spec, repl, repl),
                out_specs=(repl, repl, repl),
                axis_names=frozenset({STAGE_AXIS}),
                check_vma=False,
            )(sa, lm, h, c)
        )

        try:
            out, dh, dparam_norm = run(stage_arrays, layer_masks, hidden, cotangent)
            out = np.asarray(out)
            dh = np.asarray(dh)
            dparam_norm = float(np.asarray(dparam_norm))
        except Exception as ex:
            logger.info("STEP 1 FAIL: jax.vjp of a stage forward was REJECTED inside the stage-manual shard_map")
            logger.info("Exception type: %s", type(ex).__name__)
            logger.info("Message:\n%s", str(ex))
            return 1

        ok = (
            np.all(np.isfinite(out))
            and np.all(np.isfinite(dh))
            and np.isfinite(dparam_norm)
            and dparam_norm > 0.0
            and float(np.max(np.abs(dh))) > 0.0
        )
        logger.info(
            "STEP 1 result: out_finite=%s dh_max=%.4e dparam_grad_sq=%.4e -> %s",
            bool(np.all(np.isfinite(out))),
            float(np.max(np.abs(dh))),
            dparam_norm,
            "PASS" if ok else "FAIL",
        )
        logger.info(
            "jax.vjp of a stage forward (embed-free block scan w/ inner EP shard_map "
            "+ _batch_reshard) LOWERS and RUNS inside the stage-manual shard_map (check_vma=False)."
        )
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
