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

from experiments.grug.moe_zb.benchmark import _config, _fsdp_sharding, _timed_steps
from experiments.grug.moe_zb.model import build_pipeline_params, reference_loss
from experiments.grug.moe_zb.parallelism import DATA_AXIS, make_pipeline_mesh, shard_pipeline_params
from experiments.grug.moe_zb.train import Schedule, _arithmetic_batch, make_train_step

logger = logging.getLogger(__name__)


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
    on_tpu = jax.devices()[0].platform == "tpu"
    logger.info("benchmarking on %d %s device(s)", jax.device_count(), jax.devices()[0].platform)

    # Keep 2-way PP and 2-way EP; spend the rest on FSDP so all three axes stay >1
    # on every slice (v6e-8 -> 2x2x2, v6e-16 -> 2x4x2, v6e-32 -> 2x8x2). 2x2x2 fits
    # one v6e-8 host, so the full PP x FSDP x EP composition runs without any DCN.
    num_stages, num_expert = 2, 2
    num_data = max(1, jax.device_count() // (num_stages * num_expert))
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

    num_devices = num_stages * num_data * num_expert
    global_batch = num_microbatches * microbatch
    tokens_per_step = global_batch * seq_len
    cfg = _config(num_stages, num_layers, hidden_dim, num_experts, seq_len, vocab_size)
    logger.info(
        "mesh=%dx%dx%d (stage,data,expert)=%d chips | layers=%d hidden=%d experts=%d seq=%d global_batch=%d tok/step=%d",
        num_stages,
        num_data,
        num_expert,
        num_devices,
        num_layers,
        hidden_dim,
        num_experts,
        seq_len,
        global_batch,
        tokens_per_step,
    )

    pp_s = bench_pipeline_3d(
        cfg,
        num_stages,
        num_data,
        num_expert,
        num_microbatches=num_microbatches,
        microbatch=microbatch,
        seq_len=seq_len,
        lr=3e-3,
        warmup=warmup,
        iters=iters,
        seed=0,
    )
    fsdp_s = bench_fsdp(
        cfg, num_devices, global_batch=global_batch, seq_len=seq_len, lr=3e-3, warmup=warmup, iters=iters, seed=0
    )

    pp_tps = tokens_per_step / pp_s
    fsdp_tps = tokens_per_step / fsdp_s
    logger.info(
        "PPxFSDPxEP (%dx%dx%d): %.1f ms/step  %.0f tokens/sec", num_stages, num_data, num_expert, pp_s * 1e3, pp_tps
    )
    logger.info("FSDP (%d-way)        : %.1f ms/step  %.0f tokens/sec", num_devices, fsdp_s * 1e3, fsdp_tps)
    logger.info("PPxFSDPxEP / FSDP throughput ratio: %.2fx", pp_tps / fsdp_tps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
