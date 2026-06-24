# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU make-or-break: manual-threaded PP (ring EP) on H100 -- does it lower + fit?

Runs :func:`experiments.grug.moe_pp.pipeline_manual.manual_pp_value_and_grad`
(forward + backward) for the production grug-MoE -- real ring EP, FSDP over
``data`` -- on the GPU mesh and reports peak HBM. Answers the two open questions
for the GPU port:

1. **Does the production ``ring`` EP lower + execute on GPU?** If this runs, yes --
   so the Lever-2 GPU Pallas EP is a perf option, not a correctness requirement.
2. **Does the manual per-stage backward avoid the OOM at scale?** There is NO
   ``stage`` mesh axis, so the ``[num_stages, ...]`` weight-grad buffer that
   whole-program autodiff through a stage-manual ``shard_map`` materializes
   (the ~48 GiB H100 OOM) never forms.

Model size and layout come from the same ``MOE_PP_*`` env vars as ``benchmark.py``.
``MOE_PP_STAGE`` is the LOGICAL pipeline depth (how many vjp chunks the backward is
split into); there is no stage mesh axis, so all stages share the GPU mesh here.

    iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources \\
      -e MOE_PP_HIDDEN 2048 -e MOE_PP_LAYERS 16 -e MOE_PP_EXPERTS 16 \\
      -- python -m experiments.grug.moe_pp.gpu_smoke
"""

from __future__ import annotations

import logging
import os

import jax
import jax.numpy as jnp
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.benchmark import _config, _param_count, init_distributed
from experiments.grug.moe_pp.pipeline_manual import manual_pp_value_and_grad

logger = logging.getLogger(__name__)


def _peak_hbm_gib() -> float:
    """Max ``peak_bytes_in_use`` across local devices, in GiB (0 where unavailable, e.g. CPU)."""
    peaks = [(d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in jax.local_devices()]
    return max(peaks, default=0) / 2**30


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_distributed()
    n = jax.device_count()
    platform = jax.devices()[0].platform
    logger.info("gpu-smoke on %d %s device(s), %d host(s)", n, platform, jax.process_count())

    stage = int(os.environ.get("MOE_PP_STAGE", "4"))  # logical pipeline depth (vjp chunks)
    expert = int(os.environ.get("MOE_PP_EP", "2"))
    hidden_dim = int(os.environ.get("MOE_PP_HIDDEN", "2048"))
    num_layers = int(os.environ.get("MOE_PP_LAYERS", "16"))
    num_experts = int(os.environ.get("MOE_PP_EXPERTS", "16"))
    num_experts_per_token = int(os.environ.get("MOE_PP_EPT", "2"))
    seq_len = int(os.environ.get("MOE_PP_SEQ", "1024"))
    vocab_size = int(os.environ.get("MOE_PP_VOCAB", "32768"))
    batch = int(os.environ.get("MOE_PP_BATCH", "16"))
    iters = int(os.environ.get("MOE_PP_ITERS", "3"))

    if num_layers % stage != 0:
        raise ValueError(f"num_layers={num_layers} must be divisible by logical stage={stage}")
    if n % expert != 0:
        raise ValueError(f"device_count={n} must be divisible by expert={expert}")

    mesh = compact_grug_mesh(expert_axis_size=expert, replica_axis_size=1, model_axis_size=1, stage_axis_size=1)
    data = mesh.shape["data"]
    cfg = _config(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        seq_len=seq_len,
        attention_implementation="reference",
    )
    total_b, active_b = _param_count(cfg)
    logger.info(
        "mesh: data=%d expert=%d | ~%.1fB total / ~%.1fB active | hidden=%d L=%d E=%d ept=%d seq=%d vocab=%d "
        "| logical stages=%d batch=%d",
        data,
        expert,
        total_b,
        active_b,
        hidden_dim,
        num_layers,
        num_experts,
        num_experts_per_token,
        seq_len,
        vocab_size,
        stage,
        batch,
    )

    with set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = jax.random.randint(jax.random.PRNGKey(1), (batch, seq_len), 0, vocab_size, dtype=jnp.int32)
        weight = jnp.ones((batch, seq_len), jnp.float32)

        @jax.jit
        def step(m):
            loss, g_embed_head, g_blocks = manual_pp_value_and_grad(m, tokens, weight, num_stages=stage)
            # Reduce the grads to a scalar so the backward is not dead-code eliminated.
            grad_sq = sum(
                jnp.sum(leaf.astype(jnp.float32) ** 2) for leaf in jax.tree_util.tree_leaves((g_embed_head, g_blocks))
            )
            return loss, grad_sq

        logger.info("compiling + running manual PP fwd+bwd (ring EP) ...")
        for i in range(iters):
            loss, grad_sq = step(model)
            loss.block_until_ready()
            logger.info(
                "iter %d: loss=%.4f grad_sq=%.3e peak_hbm=%.1f GiB", i, float(loss), float(grad_sq), _peak_hbm_gib()
            )

    logger.info(
        "RESULT: manual PP fwd+bwd ran on %s WITHOUT OOM (ring EP lowered). ~%.1fB model, peak_hbm=%.1f GiB",
        platform,
        total_b,
        _peak_hbm_gib(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
