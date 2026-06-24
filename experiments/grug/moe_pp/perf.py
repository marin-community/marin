# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput: manual-threaded PP vs the non-pipelined FSDP/EP baseline on GPU.

Times one forward+backward step of each path on the SAME model / mesh / global
batch and reports seconds/step, tokens/s, and peak HBM:

* **FSDP/EP baseline** -- ``jax.value_and_grad(Transformer.next_token_loss)`` at
  ``stage=1`` (pure FSDP over ``data``, EP over ``expert``); the production path.
* **Manual PP (as-is)** -- ``manual_pp_value_and_grad`` with ``num_stages`` logical
  stages and per-block remat, on the same single mesh (no ``stage`` mesh axis).

The optimizer update is identical for both and cheap relative to fwd+bwd, so it is
excluded to isolate what PP actually changes. The as-is manual driver runs all
stages on one mesh (no cross-device overlap), so this is the reference the
microbatched zero-bubble schedule must beat -- not a pipeline speedup yet.

Size via the ``MOE_PP_*`` env vars (see ``benchmark.py``).

    iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --extra gpu \\
      -e MOE_PP_HIDDEN 2048 -e MOE_PP_LAYERS 16 -e MOE_PP_EXPERTS 16 \\
      -- python -m experiments.grug.moe_pp.perf
"""

from __future__ import annotations

import logging
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.benchmark import _config, _param_count, init_distributed, peak_hbm_gib
from experiments.grug.moe_pp.oracle import oracle_loss
from experiments.grug.moe_pp.pipeline_manual import manual_pp_value_and_grad

logger = logging.getLogger(__name__)


def _time_fwd_bwd(grad_fn, arg, *, warmup: int, iters: int) -> tuple[float, float]:
    """Mean seconds/step for a jitted fwd+bwd ``grad_fn(arg)``; returns ``(sec, loss)``."""
    last = None
    for _ in range(warmup):
        last = grad_fn(arg)
        jax.block_until_ready(last)
    start = time.perf_counter()
    for _ in range(iters):
        last = grad_fn(arg)
    jax.block_until_ready(last)
    seconds = (time.perf_counter() - start) / iters
    return seconds, float(np.asarray(last[0]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_distributed()
    n = jax.device_count()
    platform = jax.devices()[0].platform
    logger.info("perf on %d %s device(s), %d host(s)", n, platform, jax.process_count())

    stage = int(os.environ.get("MOE_PP_STAGE", "4"))
    expert = int(os.environ.get("MOE_PP_EP", "2"))
    hidden_dim = int(os.environ.get("MOE_PP_HIDDEN", "2048"))
    num_layers = int(os.environ.get("MOE_PP_LAYERS", "16"))
    num_experts = int(os.environ.get("MOE_PP_EXPERTS", "16"))
    num_experts_per_token = int(os.environ.get("MOE_PP_EPT", "2"))
    seq_len = int(os.environ.get("MOE_PP_SEQ", "1024"))
    vocab_size = int(os.environ.get("MOE_PP_VOCAB", "32768"))
    global_batch = int(os.environ.get("MOE_PP_BATCH", "16"))
    warmup = int(os.environ.get("MOE_PP_WARMUP", "2"))
    iters = int(os.environ.get("MOE_PP_ITERS", "5"))

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
    mesh = compact_grug_mesh(expert_axis_size=expert, replica_axis_size=1, model_axis_size=1, stage_axis_size=1)
    data = mesh.shape["data"]
    tokens_per_step = global_batch * seq_len
    logger.info(
        "model ~%.1fB total / ~%.1fB active | hidden=%d L=%d E=%d ept=%d seq=%d vocab=%d | "
        "mesh data=%d expert=%d | global_batch=%d logical_stages=%d | tokens/step=%d",
        total_b,
        active_b,
        hidden_dim,
        num_layers,
        num_experts,
        num_experts_per_token,
        seq_len,
        vocab_size,
        data,
        expert,
        global_batch,
        stage,
        tokens_per_step,
    )

    with set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = jax.random.randint(jax.random.PRNGKey(1), (global_batch, seq_len), 0, vocab_size, dtype=jnp.int32)
        weight = jnp.ones((global_batch, seq_len), jnp.float32)
        arrays, static = eqx.partition(model, eqx.is_array)

        @jax.jit
        def fsdp_grad(arrays):
            return jax.value_and_grad(lambda p: oracle_loss(eqx.combine(p, static), tokens, weight))(arrays)

        @jax.jit
        def mpp_grad(model):
            loss, g_eh, g_blocks = manual_pp_value_and_grad(model, tokens, weight, num_stages=stage)
            return loss, (g_eh, g_blocks)

        fsdp_sec, fsdp_loss = _time_fwd_bwd(fsdp_grad, arrays, warmup=warmup, iters=iters)
        fsdp_hbm = peak_hbm_gib()
        mpp_sec, mpp_loss = _time_fwd_bwd(mpp_grad, model, warmup=warmup, iters=iters)
        mpp_hbm = peak_hbm_gib()

    logger.info(
        "PERF fsdp_baseline : %.4f s/step  %9.0f tok/s  loss=%.4f  peak_hbm=%.1f GiB",
        fsdp_sec,
        tokens_per_step / fsdp_sec,
        fsdp_loss,
        fsdp_hbm,
    )
    logger.info(
        "PERF manual_pp_asis: %.4f s/step  %9.0f tok/s  loss=%.4f  peak_hbm=%.1f GiB",
        mpp_sec,
        tokens_per_step / mpp_sec,
        mpp_loss,
        mpp_hbm,
    )
    logger.info(
        "PERF manual_pp / fsdp = %.2fx step time (>1 = PP slower, expected for a single-mesh non-overlapped baseline)",
        mpp_sec / fsdp_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
