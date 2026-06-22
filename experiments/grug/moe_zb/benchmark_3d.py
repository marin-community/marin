# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput benchmark for composed parallelism: PP x FSDP x EP vs pure FSDP.

Times the zero-bubble pipeline on a ``(stage, data, expert)`` mesh (4 x 4 x 2 by
default -- 4-way pipeline, 4-way FSDP, 2-way expert parallelism) against a pure
FSDP baseline that spreads the same model over all the same chips on a single
``data`` axis. Same model, same global batch, same device count -- only the
parallelism layout differs.

Run on a forced 32-device CPU mesh for a smoke, or on a real v6e-32 (the same
``python -m`` command) for representative numbers.
"""

from __future__ import annotations

import logging
import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"

import jax
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_zb.benchmark import _config, _fsdp_sharding, _timed_steps, init_distributed
from experiments.grug.moe_zb.model import build_pipeline_params, reference_loss
from experiments.grug.moe_zb.parallelism import DATA_AXIS, make_pipeline_mesh, shard_pipeline_params
from experiments.grug.moe_zb.train import Schedule, _arithmetic_batch, make_train_step

logger = logging.getLogger(__name__)


def _param_count(num_layers, hidden_dim, num_experts, vocab_size) -> tuple[float, float]:
    """(total, active) parameter counts for the grug-MoE config, in billions.

    Per layer: attention ~4*D^2 (square q/k/v/o at this head config), dense-eval MoE
    3 expert matrices of [E,D,2D] plus a [D,E] router. Embed + untied head are 2*V*D.
    "Active" replaces all-experts with top-2 in the FFN term.
    """
    d = hidden_dim
    attn = 4 * d * d
    moe_total = 6 * num_experts * d * d + d * num_experts
    moe_active = 6 * 2 * d * d + d * num_experts
    embed_head = 2 * vocab_size * d
    total = embed_head + num_layers * (attn + moe_total)
    active = embed_head + num_layers * (attn + moe_active)
    return total / 1e9, active / 1e9


def bench_pipeline_3d(
    cfg, num_stages, num_data, num_expert, *, num_microbatches, microbatch, seq_len, lr, warmup, iters, seed
) -> float:
    mesh = make_pipeline_mesh(num_stages, num_data, num_expert)
    params, model = build_pipeline_params(cfg, key=jax.random.PRNGKey(seed))
    optimizer = optax.adam(lr)
    hidden_shape = (microbatch, seq_len, cfg.hidden_dim)
    with jax.set_mesh(mesh):
        params = shard_pipeline_params(params, mesh)
        opt_state = optimizer.init(params)
        step = make_train_step(
            model,
            optimizer,
            mesh,
            num_microbatches=num_microbatches,
            hidden_shape=hidden_shape,
            schedule=Schedule.ZERO_BUBBLE,
        )

        def data_fn(i):
            tokens = _arithmetic_batch(
                jax.random.PRNGKey(1000 + i),
                num_microbatches=num_microbatches,
                microbatch=microbatch,
                seq_len=seq_len,
                vocab_size=cfg.vocab_size,
            )
            return jax.device_put(tokens, NamedSharding(mesh, P(None, DATA_AXIS, None)))

        def step_fn(state, tokens):
            params, opt_state = state
            params, opt_state, _ = step(params, opt_state, tokens)
            return params, opt_state

        return _timed_steps(step_fn, (params, opt_state), data_fn, warmup=warmup, iters=iters)


def bench_fsdp(cfg, num_devices, *, global_batch, seq_len, lr, warmup, iters, seed) -> float:
    devices = jax.devices()[:num_devices]
    mesh = Mesh(np.array(devices), (DATA_AXIS,))
    params, model = build_pipeline_params(cfg, key=jax.random.PRNGKey(seed))
    param_shardings = jax.tree_util.tree_map(lambda x: _fsdp_sharding(x, mesh, num_devices), params)
    params = jax.device_put(params, param_shardings)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    tok_sharding = NamedSharding(mesh, P(DATA_AXIS, None))

    @jax.jit
    def step(params, opt_state, tokens):
        loss, grads = jax.value_and_grad(lambda p: reference_loss(p, model, tokens, cfg))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def data_fn(i):
        tokens = _arithmetic_batch(
            jax.random.PRNGKey(1000 + i),
            num_microbatches=1,
            microbatch=global_batch,
            seq_len=seq_len,
            vocab_size=cfg.vocab_size,
        )[0]
        return jax.device_put(tokens, tok_sharding)

    def step_fn(state, tokens):
        params, opt_state = state
        params, opt_state, _ = step(params, opt_state, tokens)
        return params, opt_state

    return _timed_steps(step_fn, (params, opt_state), data_fn, warmup=warmup, iters=iters)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_distributed()
    on_tpu = jax.devices()[0].platform == "tpu"
    logger.info(
        "benchmarking on %d %s device(s) across %d host(s)",
        jax.device_count(),
        jax.devices()[0].platform,
        jax.process_count(),
    )

    num_stages = int(os.environ.get("MOE_ZB_PP", "2"))
    if on_tpu:
        num_layers, hidden_dim, num_experts = 8, 1024, 8
        seq_len, vocab_size = 512, 8192
        num_microbatches, microbatch = 16, 4
        warmup, iters = 5, 20
    else:
        num_layers, hidden_dim, num_experts = 8, 64, 4
        seq_len, vocab_size = 32, 256
        num_microbatches, microbatch = 8, 4
        warmup, iters = 2, 5
    # Model-dimension overrides for a largest-that-fits sweep on hardware.
    num_layers = int(os.environ.get("MOE_ZB_LAYERS", num_layers))
    hidden_dim = int(os.environ.get("MOE_ZB_HIDDEN", hidden_dim))
    num_experts = int(os.environ.get("MOE_ZB_EXPERTS", num_experts))
    seq_len = int(os.environ.get("MOE_ZB_SEQ", seq_len))
    vocab_size = int(os.environ.get("MOE_ZB_VOCAB", vocab_size))
    microbatch = int(os.environ.get("MOE_ZB_MICROBATCH", microbatch))
    # The schedule keeps every in-flight microbatch's activations live (no
    # cross-microbatch remat), so fewer microbatches lowers the activation peak.
    num_microbatches = int(os.environ.get("MOE_ZB_NMICRO", num_microbatches))

    num_devices = jax.device_count()
    fill = num_devices // num_stages
    global_batch = num_microbatches * microbatch
    tokens_per_step = global_batch * seq_len
    cfg = _config(num_stages, num_layers, hidden_dim, num_experts, seq_len, vocab_size)

    # The pipeline shard_map manualizes `stage` while `data`/`expert` stay GSPMD.
    # Composing PP with ONE GSPMD axis lowers on TPU; composing both data AND expert
    # at once trips XLA's SPMD partitioner (see parallelism.make_pipeline_mesh), so we
    # sweep the two pairwise compositions. MOE_ZB_EP forces a single layout instead.
    if "MOE_ZB_EP" in os.environ:
        ep = int(os.environ["MOE_ZB_EP"])
        layouts = [(num_stages, max(1, fill // ep), ep)]
    else:
        layouts = [(num_stages, fill, 1), (num_stages, 1, fill)]

    total_b, active_b = _param_count(num_layers, hidden_dim, num_experts, vocab_size)
    logger.info(
        "model: layers=%d hidden=%d experts=%d seq=%d vocab=%d global_batch=%d tok/step=%d"
        " | ~%.1fB total / ~%.1fB active",
        num_layers,
        hidden_dim,
        num_experts,
        seq_len,
        vocab_size,
        global_batch,
        tokens_per_step,
        total_b,
        active_b,
    )

    # Run the pipeline layouts first: each microbatches the global batch, so it fits
    # where the one-shot FSDP baseline below may not. Then time the FSDP baseline,
    # tolerating an OOM and reporting it (that gap IS the memory case for PP).
    pp_results = []
    for ns, nd, ne in layouts:
        kind = "PPxEP " if nd == 1 else "PPxFSDP" if ne == 1 else "PPxFSDPxEP"
        pp_s = bench_pipeline_3d(
            cfg,
            ns,
            nd,
            ne,
            num_microbatches=num_microbatches,
            microbatch=microbatch,
            seq_len=seq_len,
            lr=3e-3,
            warmup=warmup,
            iters=iters,
            seed=0,
        )
        pp_results.append((kind, ns, nd, ne, pp_s))
        logger.info("%s (%dx%dx%d): %.1f ms/step  %.0f tokens/sec", kind, ns, nd, ne, pp_s * 1e3, tokens_per_step / pp_s)

    try:
        fsdp_s = bench_fsdp(
            cfg, num_devices, global_batch=global_batch, seq_len=seq_len, lr=3e-3, warmup=warmup, iters=iters, seed=0
        )
    except jax.errors.JaxRuntimeError as e:
        if "RESOURCE_EXHAUSTED" not in str(e):
            raise
        logger.info(
            "FSDP (%d-way) baseline: OOM at this size (one-shot global batch); "
            "pipeline microbatching fits where pure FSDP does not",
            num_devices,
        )
        return 0

    fsdp_tps = tokens_per_step / fsdp_s
    logger.info("FSDP (%d-way) baseline: %.1f ms/step  %.0f tokens/sec", num_devices, fsdp_s * 1e3, fsdp_tps)
    for kind, ns, nd, ne, pp_s in pp_results:
        logger.info("%s (%dx%dx%d): %.2fx FSDP", kind, ns, nd, ne, (tokens_per_step / pp_s) / fsdp_tps)

    logger.info(
        "note: PPxFSDPxEP with data>1 AND expert>1 does not lower on TPU "
        "(XLA SPMD partitioner; see parallelism.make_pipeline_mesh)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
