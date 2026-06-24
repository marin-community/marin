# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Do the REAL grug stage kernels overlap across devices, or do they serialize?

The device-group pipeline's forward overlaps but its backward runs ~serially, even
the W-pass whose calls are fully independent. The bare-matmul probes (overlap_probe)
pipeline at 100%, so the suspect is the grug stage kernel itself -- specifically the
MoE ``shard_map`` ring collectives (all_gather / psum_scatter over the expert axis),
which XLA may serialize globally even when every mesh axis is size 1.

This builds ONE real stage (``lps`` grug blocks) and places a replicated copy on each
of ``P`` single-device sub-meshes, then times a single forward/backward against ``P``
independent ones launched together. If wall(P) ~= wall(1) the kernels overlap; if
wall(P) ~= P*wall(1) they serialize.

    iris --cluster=cw-us-east-02a job run --gpu H100x8 --enable-extra-resources --extra gpu \\
      -e MOE_PP_HIDDEN 1536 -e MOE_PP_LAYERS 24 -e MOE_PP_STAGE 8 -e MOE_PP_EXPERTS 8 \\
      -- python -m experiments.grug.moe_pp.kernel_overlap_probe
"""

from __future__ import annotations

import logging
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.partitioning import set_mesh

from experiments.grug.moe import model as grug_model
from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.benchmark import _config, init_distributed
from experiments.grug.moe_pp.pipeline_manual import _stage_forward
from experiments.grug.moe_pp.pipeline_zb import _put_act, _put_params, _stage_submesh

logger = logging.getLogger(__name__)


def _time(fn, *, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn())
    t = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t) / iters


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_distributed()
    devs = jax.devices()
    p = min(int(os.environ.get("MOE_PP_STAGE", "8")), len(devs))

    hidden_dim = int(os.environ.get("MOE_PP_HIDDEN", "1536"))
    num_layers = int(os.environ.get("MOE_PP_LAYERS", "24"))
    num_experts = int(os.environ.get("MOE_PP_EXPERTS", "8"))
    seq_len = int(os.environ.get("MOE_PP_SEQ", "1024"))
    microbatch = int(os.environ.get("MOE_PP_MB", "8"))
    lps = num_layers // p

    cfg = _config(
        vocab_size=32768,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_token=2,
        seq_len=seq_len,
        attention_implementation="reference",
    )
    logger.info(
        "kernel_overlap_probe on %d %s | stage=%d blocks hidden=%d E=%d seq=%d microbatch=%d",
        len(devs),
        devs[0].platform,
        lps,
        hidden_dim,
        num_experts,
        seq_len,
        microbatch,
    )

    submeshes = [_stage_submesh([devs[s]], expert=1, data=1) for s in range(p)]
    with set_mesh(submeshes[0]):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
    block_static = eqx.partition(model.blocks[0], eqx.is_array)[1]
    block_arrays = [eqx.partition(b, eqx.is_array)[0] for b in model.blocks[:lps]]
    base_mask = grug_model.AttentionMask.causal()
    short_mask, long_mask = grug_model._layer_attention_masks(base_mask, sliding_window=cfg.sliding_window)
    masks = tuple(long_mask if (i % 4 == 3) else short_mask for i in range(lps))

    params = [_put_params(block_arrays, submeshes[s]) for s in range(p)]
    x0 = jnp.zeros((microbatch, seq_len, hidden_dim), jnp.float32)
    xs = [_put_act(x0, submeshes[s]) for s in range(p)]

    fwd = jax.jit(lambda pr, x: _stage_forward(pr, block_static, x, masks))

    def bwd_fn(pr, x):
        _, vjp = jax.vjp(lambda pp, hh: _stage_forward(pp, block_static, hh, masks), pr, x)
        y, _z = _stage_forward(pr, block_static, x, masks)
        return vjp((jnp.ones_like(y), jnp.ones((), jnp.float32)))

    bwd = jax.jit(bwd_fn)

    def call_fwd(s):
        with set_mesh(submeshes[s]):
            return fwd(params[s], xs[s])

    def call_bwd(s):
        with set_mesh(submeshes[s]):
            return bwd(params[s], xs[s])

    one_f = _time(lambda: call_fwd(0))
    fan_f = _time(lambda: [call_fwd(s) for s in range(p)])
    logger.info(
        "FORWARD: 1 call=%.1fms | %d independent=%.1fms | fanout/1=%.2fx (%.0f%% overlap; %d=serial 1=perfect)",
        one_f * 1e3,
        p,
        fan_f * 1e3,
        fan_f / one_f,
        100.0 * (p - fan_f / one_f) / (p - 1),
        p,
    )

    one_b = _time(lambda: call_bwd(0))
    fan_b = _time(lambda: [call_bwd(s) for s in range(p)])
    logger.info(
        "BACKWARD: 1 call=%.1fms | %d independent=%.1fms | fanout/1=%.2fx (%.0f%% overlap; %d=serial 1=perfect)",
        one_b * 1e3,
        p,
        fan_b * 1e3,
        fan_b / one_b,
        100.0 * (p - fan_b / one_b) / (p - 1),
        p,
    )

    # --- W-pass mimics: M calls per device (M*P total) under different dispatch order and grad
    # accumulation, to isolate why the real W-pass serializes. Ideal (full cross-device overlap)
    # is ~M*one_b (each device runs its M serially, all devices concurrent). ---
    big_m = int(os.environ.get("MOE_PP_MWP", "8"))
    ideal = big_m * one_b

    def stage_major_noaccum():
        return [call_bwd(s) for s in range(p) for _m in range(big_m)]

    def mb_major_noaccum():
        return [call_bwd(s) for _m in range(big_m) for s in range(p)]

    def stage_major_accum():
        acc: list = [None] * p
        for s in range(p):
            for _m in range(big_m):
                g = call_bwd(s)
                acc[s] = g if acc[s] is None else jax.tree_util.tree_map(jnp.add, acc[s], g)
        return acc

    for name, fn in [
        ("stage-major no-accum", stage_major_noaccum),
        ("mb-major   no-accum", mb_major_noaccum),
        ("stage-major +accum ", stage_major_accum),
    ]:
        wall = _time(fn, warmup=1, iters=3)
        logger.info(
            "WPASS %s: %dx%d calls=%.0fms | ideal(M*1call)=%.0fms | wall/ideal=%.2fx %s",
            name,
            p,
            big_m,
            wall * 1e3,
            ideal * 1e3,
            wall / ideal,
            "OVERLAPS" if wall < 2 * ideal else "SERIALIZES",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
