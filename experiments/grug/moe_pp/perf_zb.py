# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput: device-group pipeline (pipeline_zb) vs the FSDP/EP baseline on GPU.

Times one full forward+backward step of each on the SAME model and global batch:

* **FSDP/EP baseline** -- ``jax.value_and_grad(Transformer.next_token_loss)`` over
  all devices (FSDP over ``data``, EP over ``expert``); one jitted program.
* **Device-group pipeline** -- :func:`pipeline_zb.zb_build` with ``num_stages``
  stages on disjoint device slices, microbatched, ordered by ``MOE_PP_SCHED``
  (:class:`pipeline_zb.Schedule`: ``gpipe`` / ``1f1b`` / ``zb``). Compiled stage calls
  on disjoint slices dispatch asynchronously so the runtime overlaps them.

Defaults to 8-way PP (1 device/stage) at ~1B -- the original ``v6e-8`` goal scale.
On a single NVLink node FSDP may still win on throughput (its all-gather is cheap
and the bubble is ``(P-1)/(M+P-1)``); PP's win is multi-node comm and depth/memory
scaling. Size via the ``MOE_PP_*`` env vars.

    iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --extra gpu \\
      -e MOE_PP_SCHED zb -e MOE_PP_HIDDEN 1536 -e MOE_PP_LAYERS 24 -e MOE_PP_EXPERTS 8 \\
      -e MOE_PP_STAGE 8 -e MOE_PP_NMICRO 8 -e MOE_PP_BATCH 32 \\
      -- python -m experiments.grug.moe_pp.perf_zb
"""

from __future__ import annotations

import logging
import os
import time

import equinox as eqx
import jax
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.benchmark import _config, _param_count, init_distributed, peak_hbm_gib
from experiments.grug.moe_pp.oracle import oracle_loss
from experiments.grug.moe_pp.pipeline_zb import Schedule, _stage_submesh, orthogonalize_tree, zb_build

logger = logging.getLogger(__name__)


def _replicate(x, mesh):
    """Replicate a host-local array onto every device of ``mesh`` (multi-host aware)."""
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if jax.process_count() == 1:
        return jax.device_put(x, sharding)
    host = np.asarray(x)
    shards = [jax.device_put(host, d) for d in mesh.devices.flat if d.process_index == jax.process_index()]
    return jax.make_array_from_single_device_arrays(host.shape, sharding, shards)


def _time(fn, *, warmup: int, iters: int):
    last = None
    for _ in range(warmup):
        last = fn()
        jax.block_until_ready(last)
    start = time.perf_counter()
    for _ in range(iters):
        last = fn()
    jax.block_until_ready(last)
    return (time.perf_counter() - start) / iters, last


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_distributed()
    n = jax.device_count()
    platform = jax.devices()[0].platform

    num_stages = int(os.environ.get("MOE_PP_STAGE", "8"))
    expert_per_stage = int(os.environ.get("MOE_PP_EPS", "1"))
    data_per_stage = int(os.environ.get("MOE_PP_DPS", "1"))
    num_microbatches = int(os.environ.get("MOE_PP_NMICRO", "8"))
    schedule = Schedule(os.environ.get("MOE_PP_SCHED", "zb"))
    remat = os.environ.get("MOE_PP_REMAT", "1") == "1"
    muon = os.environ.get("MOE_PP_MUON", "0") == "1"
    async_transport = os.environ.get("MOE_PP_ASYNC_XPORT", "0") == "1"
    p2p_transport = os.environ.get("MOE_PP_P2P", "0") == "1"
    hidden_dim = int(os.environ.get("MOE_PP_HIDDEN", "1536"))
    num_layers = int(os.environ.get("MOE_PP_LAYERS", "24"))
    num_experts = int(os.environ.get("MOE_PP_EXPERTS", "8"))
    num_experts_per_token = int(os.environ.get("MOE_PP_EPT", "2"))
    seq_len = int(os.environ.get("MOE_PP_SEQ", "1024"))
    vocab_size = int(os.environ.get("MOE_PP_VOCAB", "32768"))
    global_batch = int(os.environ.get("MOE_PP_BATCH", "32"))
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
    tokens_per_step = global_batch * seq_len
    logger.info("perf_zb on %d %s device(s), %d host(s)", n, platform, jax.process_count())
    logger.info(
        "model ~%.1fB total / ~%.1fB active | hidden=%d L=%d E=%d seq=%d | global_batch=%d | "
        "PP stages=%d eps=%d dps=%d microbatches=%d | tokens/step=%d",
        total_b,
        active_b,
        hidden_dim,
        num_layers,
        num_experts,
        seq_len,
        global_batch,
        num_stages,
        expert_per_stage,
        data_per_stage,
        num_microbatches,
        tokens_per_step,
    )

    # Inputs are host-local numpy (identical on every process via the shared seed): the
    # pipeline places them onto the stage sub-meshes, and the FSDP baseline replicates
    # them. Numpy (not a jax array) keeps them fully addressable on every host -- a global
    # device array would reshard on the first per-microbatch slice.
    multihost = jax.process_count() > 1
    tokens = np.random.default_rng(1).integers(0, vocab_size, (global_batch, seq_len), dtype=np.int32)
    weight = np.ones((global_batch, seq_len), np.float32)
    skip_fsdp = os.environ.get("MOE_PP_SKIP_FSDP", "0") == "1"

    # --- FSDP/EP baseline: full mesh, one jitted value_and_grad ---
    fsdp_expert = expert_per_stage * data_per_stage if expert_per_stage > 1 else 1
    fsdp_mesh = compact_grug_mesh(
        expert_axis_size=fsdp_expert, replica_axis_size=1, model_axis_size=1, stage_axis_size=1
    )
    fsdp_sec = fsdp_loss = fsdp_hbm = float("nan")
    with set_mesh(fsdp_mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        if not skip_fsdp:
            arrays, static = eqx.partition(model, eqx.is_array)
            fsdp_tokens = _replicate(tokens, fsdp_mesh)
            fsdp_weight = _replicate(weight, fsdp_mesh)

            @jax.jit
            def fsdp_grad(arrays):
                loss, grads = jax.value_and_grad(
                    lambda p: oracle_loss(eqx.combine(p, static), fsdp_tokens, fsdp_weight)
                )(arrays)
                # Muon orthogonalizes each block weight-grad; under the FSDP mesh the
                # grads are sharded over the data axis, so Newton-Schulz all-gathers them.
                block_grads = orthogonalize_tree(grads.blocks) if muon else grads.blocks
                return loss, block_grads

            fsdp_sec, fsdp_out = _time(lambda: fsdp_grad(arrays), warmup=warmup, iters=iters)
            fsdp_loss = float(np.asarray(fsdp_out[0]))
            fsdp_hbm = peak_hbm_gib()

    # --- device-group pipeline (zb_build re-places per stage). Multi-host: init under a
    # mesh of THIS process's local devices (identical params on every process via the
    # shared PRNG, fully addressable on each host) so zb_build builds each stage's shards
    # without a cross-host reshard from a 16-device array. ---
    if multihost:
        host_mesh = _stage_submesh(jax.local_devices(), expert=1, data=jax.local_device_count())
    else:
        host_mesh = fsdp_mesh
    with set_mesh(host_mesh):
        pp_model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
    step = zb_build(
        pp_model,
        num_stages=num_stages,
        num_microbatches=num_microbatches,
        expert_per_stage=expert_per_stage,
        data_per_stage=data_per_stage,
        schedule=schedule,
        remat=remat,
        muon=muon,
        async_transport=async_transport,
        p2p_transport=p2p_transport,
    )
    pp_sec, pp_out = _time(lambda: step(tokens, weight), warmup=warmup, iters=iters)
    pp_loss = float(np.asarray(pp_out[0]))
    pp_hbm = peak_hbm_gib()

    logger.info(
        "PERF fsdp_baseline   : %.4f s/step  %9.0f tok/s  loss=%.4f  peak_hbm=%.1f GiB",
        fsdp_sec,
        tokens_per_step / fsdp_sec,
        fsdp_loss,
        fsdp_hbm,
    )
    logger.info(
        "PERF zb_pipeline mode: schedule=%s remat=%s muon=%s async_transport=%s p2p=%s",
        schedule.value,
        remat,
        muon,
        async_transport,
        p2p_transport,
    )
    logger.info(
        "PERF zb_pipeline     : %.4f s/step  %9.0f tok/s  loss=%.4f  peak_hbm=%.1f GiB  microbatch=%d (%d tok/call)",
        pp_sec,
        tokens_per_step / pp_sec,
        pp_loss,
        pp_hbm,
        global_batch // num_microbatches,
        (global_batch // num_microbatches) * seq_len,
    )
    logger.info(
        "PERF zb / fsdp = %.2fx step time | GPipe bubble ~%.0f%% ((P-1)/(M+P-1), P=%d M=%d)",
        pp_sec / fsdp_sec,
        100.0 * (num_stages - 1) / (num_microbatches + num_stages - 1),
        num_stages,
        num_microbatches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
