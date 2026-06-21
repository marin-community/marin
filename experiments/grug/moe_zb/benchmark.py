# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput benchmark: zero-bubble pipeline vs. FSDP for the same grug-MoE.

Both paths train the identical reshard-free grug-MoE model on the same devices
with the same global batch, so the only difference is the parallelism strategy:

- **Zero-bubble PP** (``zero_bubble_value_and_grad``): one stage per chip, the
  global batch split into microbatches that pipeline across stages; cross-stage
  comms are nearest-neighbour ``ppermute`` of activations.
- **FSDP** (GSPMD): every layer's weights sharded over a ``data`` axis spanning
  all chips, the global batch data-parallel; comms are all-gather of weights and
  reduce-scatter of grads. This is the standard grug-style baseline.

Reports steady-state ms/step and tokens/sec for each over a timed window after
warmup. Run on a forced multi-device CPU mesh for a smoke, or on a real TPU
slice (the same ``python -m`` command) for representative numbers.
"""

from __future__ import annotations

import logging
import os
import time

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_zb.model import GrugMoEConfig, build_pipeline_params, reference_loss
from experiments.grug.moe_zb.train import (
    Schedule,
    _arithmetic_batch,
    make_stage_mesh,
    make_train_step,
    shard_params,
)

logger = logging.getLogger(__name__)

DATA_AXIS = "data"


def _fsdp_sharding(leaf: jax.Array, mesh: Mesh, num_data: int) -> NamedSharding:
    """Shard a parameter's largest data-divisible axis over ``data`` (else replicate)."""
    best_axis = None
    for axis, dim in enumerate(leaf.shape):
        if dim % num_data == 0 and dim >= num_data and (best_axis is None or dim > leaf.shape[best_axis]):
            best_axis = axis
    if best_axis is None:
        return NamedSharding(mesh, P())
    spec = [None] * leaf.ndim
    spec[best_axis] = DATA_AXIS
    return NamedSharding(mesh, P(*spec))


def _timed_steps(step_fn, state, data_fn, *, warmup: int, iters: int) -> float:
    """Return mean seconds/step over ``iters`` steps after ``warmup`` warmup steps."""
    for i in range(warmup):
        state = step_fn(state, data_fn(i))
    jax.block_until_ready(state)
    start = time.perf_counter()
    for i in range(iters):
        state = step_fn(state, data_fn(warmup + i))
    jax.block_until_ready(state)
    return (time.perf_counter() - start) / iters


def _config(num_stages, num_layers, hidden_dim, num_experts, seq_len, vocab_size) -> GrugMoEConfig:
    return GrugMoEConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=2 * hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=2,
        num_layers=num_layers,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=seq_len,
        num_stages=num_stages,
    )


def bench_pipeline(cfg, schedule, *, num_microbatches, microbatch, seq_len, lr, warmup, iters, seed) -> float:
    mesh = make_stage_mesh(cfg.num_stages)
    params, model = build_pipeline_params(cfg, key=jax.random.PRNGKey(seed))
    optimizer = optax.adam(lr)
    hidden_shape = (microbatch, seq_len, cfg.hidden_dim)
    with jax.set_mesh(mesh):
        params = shard_params(params, mesh)
        opt_state = optimizer.init(params)
        step = make_train_step(
            model,
            optimizer,
            mesh,
            num_microbatches=num_microbatches,
            hidden_shape=hidden_shape,
            schedule=schedule,
        )

        def data_fn(i):
            tokens = _arithmetic_batch(
                jax.random.PRNGKey(1000 + i),
                num_microbatches=num_microbatches,
                microbatch=microbatch,
                seq_len=seq_len,
                vocab_size=cfg.vocab_size,
            )
            return jax.device_put(tokens, NamedSharding(mesh, P()))

        def step_fn(state, tokens):
            params, opt_state = state
            params, opt_state, _ = step(params, opt_state, tokens)
            return params, opt_state

        return _timed_steps(step_fn, (params, opt_state), data_fn, warmup=warmup, iters=iters)


def bench_fsdp(cfg, *, global_batch, seq_len, lr, warmup, iters, seed) -> float:
    devices = jax.devices()[: cfg.num_stages]
    mesh = Mesh(np.array(devices), (DATA_AXIS,))
    params, model = build_pipeline_params(cfg, key=jax.random.PRNGKey(seed))
    param_shardings = jax.tree_util.tree_map(lambda x: _fsdp_sharding(x, mesh, len(devices)), params)
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
        starts = jax.random.randint(jax.random.PRNGKey(1000 + i), (global_batch, 1), 0, cfg.vocab_size)
        tokens = ((starts + jnp.arange(seq_len)[None, :]) % cfg.vocab_size).astype(jnp.int32)
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

    if on_tpu:
        num_stages, num_layers, hidden_dim, num_experts = 8, 8, 1024, 8
        seq_len = 512
        # Vocab is overridable to probe how the head ([D, V]) cost scales the
        # zero-bubble-vs-FSDP gap: small vocab hides the head, a large (LLM-class)
        # vocab makes it the dominant per-stage matmul.
        vocab_size = int(os.environ.get("MOE_ZB_VOCAB", "8192"))
        num_microbatches, microbatch = 16, 4
        warmup, iters = 5, 30
    else:
        num_stages, num_layers, hidden_dim, num_experts = 8, 8, 64, 4
        seq_len, vocab_size = 32, 256
        num_microbatches, microbatch = 8, 2
        warmup, iters = 2, 5

    global_batch = num_microbatches * microbatch
    tokens_per_step = global_batch * seq_len
    cfg = _config(num_stages, num_layers, hidden_dim, num_experts, seq_len, vocab_size)
    logger.info(
        "config: stages=%d layers=%d hidden=%d experts=%d seq=%d global_batch=%d tokens/step=%d",
        num_stages,
        num_layers,
        hidden_dim,
        num_experts,
        seq_len,
        global_batch,
        tokens_per_step,
    )

    bench_kwargs = dict(
        num_microbatches=num_microbatches,
        microbatch=microbatch,
        seq_len=seq_len,
        lr=3e-3,
        warmup=warmup,
        iters=iters,
        seed=0,
    )
    # GPIPE is the head-hoisted backend (head scored once, distributed over the
    # stage axis); ZERO_BUBBLE keeps the head inside the per-stage program.
    gpipe_s = bench_pipeline(cfg, Schedule.GPIPE, **bench_kwargs)
    zb_s = bench_pipeline(cfg, Schedule.ZERO_BUBBLE, **bench_kwargs)
    fsdp_s = bench_fsdp(cfg, global_batch=global_batch, seq_len=seq_len, lr=3e-3, warmup=warmup, iters=iters, seed=0)

    gpipe_tps = tokens_per_step / gpipe_s
    zb_tps = tokens_per_step / zb_s
    fsdp_tps = tokens_per_step / fsdp_s
    logger.info("GPipe (head-hoisted): %.1f ms/step  %.0f tokens/sec", gpipe_s * 1e3, gpipe_tps)
    logger.info("ZeroBubble PP       : %.1f ms/step  %.0f tokens/sec", zb_s * 1e3, zb_tps)
    logger.info("FSDP                : %.1f ms/step  %.0f tokens/sec", fsdp_s * 1e3, fsdp_tps)
    logger.info("GPipe/FSDP throughput ratio:      %.2fx", gpipe_tps / fsdp_tps)
    logger.info("ZeroBubble/FSDP throughput ratio: %.2fx", zb_tps / fsdp_tps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
