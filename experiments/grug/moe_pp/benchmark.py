# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput benchmark: PP x FSDP x EP pipeline vs pure FSDP for the PRODUCTION grug-MoE.

Both paths train the IDENTICAL production ``Transformer`` (real ring-EP ``moe_mlp``,
FSDP over ``data``, vocab-TP over ``model``, fused-CE head) on the same chips with
the same global batch -- only the parallelism layout differs:

- **PP x FSDP x EP** (the pipeline): ``compact_grug_mesh(stage_axis_size=S,
  expert_axis_size=E)`` with ``data`` filling the rest. The global batch is split
  into ``num_microbatches`` microbatches that pipeline across the ``stage`` axis;
  the gradient-exact manual GPipe backward (:func:`pipeline_value_and_grad`) feeds
  an ``optax.adamw`` update.
- **FSDP baseline**: the same production ``Transformer`` run NON-pipelined
  (``stage=1``, optionally EP over ``expert``), the global batch consumed one-shot,
  ``jax.value_and_grad(Transformer.next_token_loss)`` (via the oracle) feeding the
  same ``optax.adamw`` update.

Reports steady-state ms/step and tokens/sec for each over a timed window after
warmup, plus the PP/FSDP throughput ratio. The one-shot FSDP baseline may OOM at
large sizes where pipeline microbatching still fits; that is caught and reported
(the memory gap IS the PP value proposition), and the PP numbers are returned
regardless (PP runs first).

Run on a forced multi-device CPU mesh for a smoke, or on a real TPU slice (the
same ``python -m`` command) for representative numbers.
"""

from __future__ import annotations

import logging
import os
import time

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from iris.runtime.jax_init import initialize_jax
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe_pp.oracle import oracle_loss
from experiments.grug.moe_pp.pipeline import pipeline_value_and_grad, stack_blocks_for_stages

logger = logging.getLogger(__name__)

LR = 3e-3


def init_distributed() -> None:
    """Bring up JAX distributed for a multi-host TPU slice (no-op on one host).

    A v6e-32 is 8 hosts; without this each host only sees its 4 local chips.
    ``iris.runtime.jax_init.initialize_jax`` calls ``jax.distributed.initialize()``
    via TPU runtime autodiscovery inside an Iris job and skips cleanly off-cluster
    (e.g. the forced-CPU smoke), so it is safe to call unconditionally.
    """
    initialize_jax()


def _param_count(cfg: GrugModelConfig) -> tuple[float, float]:
    """(total, active) parameter counts for the production grug-MoE config, in billions.

    Per layer: attention ~4*D^2 (square q/k/v/o), the MoE block has 3 expert matrices
    of [E, D, I] (gate/up/down) plus a [D, E] router, and a shared expert of 3
    [D, I_shared] matrices. "Active" replaces all-experts with the top-k routed
    experts (6 * ept * D * I), keeping the shared expert and router. Embed + untied
    head are 2 * V * D.
    """
    d = cfg.hidden_dim
    e = cfg.num_experts
    i = cfg.intermediate_dim
    ept = cfg.num_experts_per_token
    i_shared = cfg.shared_expert_intermediate_dim
    attn = 4 * d * d
    router = d * e
    shared = 3 * d * i_shared
    moe_total = 6 * e * d * i + router + shared
    moe_active = 6 * ept * d * i + router + shared
    embed_head = 2 * cfg.vocab_size * d
    total = embed_head + cfg.num_layers * (attn + moe_total)
    active = embed_head + cfg.num_layers * (attn + moe_active)
    return total / 1e9, active / 1e9


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


def _config(*, vocab_size, hidden_dim, num_layers, num_experts, num_experts_per_token, seq_len) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=2 * hidden_dim,
        shared_expert_intermediate_dim=2 * hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_layers=num_layers,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=seq_len,
        sliding_window=seq_len,
        moe_implementation="ring",
        # Reference (plain-JAX einsum) attention so the manual pipeline backward can
        # differentiate it: the TPU splash kernel emits a custom-VJP/ShapeDtypeStruct
        # that does not transpose through the stage-manual shard_map. Both the PP and
        # FSDP paths use the same impl, so the throughput ratio stays apples-to-apples.
        attention_implementation="reference",
    )


def _microbatch_tokens(rng_seed: int, *, num_microbatches, microbatch, seq_len, vocab_size) -> jax.Array:
    """Random ``[num_microbatches, microbatch, seq]`` token ids for one pipeline step."""
    key = jax.random.PRNGKey(1000 + rng_seed)
    return jax.random.randint(key, (num_microbatches, microbatch, seq_len), 0, vocab_size, dtype=jnp.int32)


def bench_pipeline(
    cfg: GrugModelConfig,
    *,
    stage: int,
    expert: int,
    num_microbatches: int,
    microbatch: int,
    seq_len: int,
    warmup: int,
    iters: int,
    seed: int,
) -> tuple[float, float]:
    """Time a full PP x FSDP x EP training step. Returns ``(seconds_per_step, loss)``.

    ``data`` fills whatever ``stage`` and ``expert`` leave; the global batch is
    ``num_microbatches * microbatch`` and the microbatch dim shards over ``data``.
    """
    mesh = compact_grug_mesh(expert_axis_size=expert, replica_axis_size=1, model_axis_size=1, stage_axis_size=stage)
    weight_microbatches = jnp.ones((num_microbatches, microbatch, seq_len), dtype=jnp.float32)
    optimizer = optax.adamw(LR)

    with set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(seed))
        stage_arrays, block_static = stack_blocks_for_stages(model, stage)
        embed_arrays, _ = eqx.partition(
            (
                model.token_embed,
                model.embed_norm,
                model.embed_gated_norm,
                model.final_norm,
                model.final_gated_norm,
                model.output_proj,
            ),
            eqx.is_array,
        )
        # The trainable leaves are the replicated embed/norm/head tuple and the
        # stage-sharded stacked blocks -- the exact two grad groups
        # pipeline_value_and_grad returns.
        params = (embed_arrays, stage_arrays)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, tokens):
            embed_arrays, stage_arrays = params
            host = eqx.combine(
                embed_arrays,
                eqx.partition(
                    (
                        model.token_embed,
                        model.embed_norm,
                        model.embed_gated_norm,
                        model.final_norm,
                        model.final_gated_norm,
                        model.output_proj,
                    ),
                    eqx.is_array,
                )[1],
            )
            # Rebuild a Transformer carrying the (possibly updated) embed/norm/head
            # leaves so pipeline_value_and_grad reads the current head params.
            updated_model = eqx.tree_at(
                lambda t: (
                    t.token_embed,
                    t.embed_norm,
                    t.embed_gated_norm,
                    t.final_norm,
                    t.final_gated_norm,
                    t.output_proj,
                ),
                model,
                host,
            )
            loss, g_embed, g_stage = pipeline_value_and_grad(
                updated_model,
                stage_arrays,
                block_static,
                tokens,
                weight_microbatches,
                mesh=mesh,
                num_stages=stage,
                num_microbatches=num_microbatches,
            )
            grads = (g_embed, g_stage)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        def data_fn(i):
            return _microbatch_tokens(
                i,
                num_microbatches=num_microbatches,
                microbatch=microbatch,
                seq_len=seq_len,
                vocab_size=cfg.vocab_size,
            )

        last_loss = jnp.zeros(())

        def step_fn(state, tokens):
            nonlocal last_loss
            params, opt_state = state
            params, opt_state, last_loss = step(params, opt_state, tokens)
            return params, opt_state

        seconds = _timed_steps(step_fn, (params, opt_state), data_fn, warmup=warmup, iters=iters)
    return seconds, float(np.asarray(last_loss))


def bench_fsdp(
    cfg: GrugModelConfig,
    *,
    expert: int,
    global_batch: int,
    seq_len: int,
    warmup: int,
    iters: int,
    seed: int,
) -> tuple[float, float]:
    """Time a full non-pipelined FSDP (+ EP) training step. Returns ``(seconds_per_step, loss)``.

    Same production ``Transformer`` at ``stage=1``: pure FSDP over ``data`` (with EP
    over ``expert`` if ``expert > 1``), the global batch consumed one-shot via
    ``jax.value_and_grad(Transformer.next_token_loss)``.
    """
    mesh = compact_grug_mesh(expert_axis_size=expert, replica_axis_size=1, model_axis_size=1, stage_axis_size=1)
    weight = jnp.ones((global_batch, seq_len), dtype=jnp.float32)
    optimizer = optax.adamw(LR)

    with set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(seed))
        params, static = eqx.partition(model, eqx.is_array)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, tokens):
            def loss_fn(p):
                return oracle_loss(eqx.combine(p, static), tokens, weight)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        def data_fn(i):
            key = jax.random.PRNGKey(1000 + i)
            return jax.random.randint(key, (global_batch, seq_len), 0, cfg.vocab_size, dtype=jnp.int32)

        last_loss = jnp.zeros(())

        def step_fn(state, tokens):
            nonlocal last_loss
            params, opt_state = state
            params, opt_state, last_loss = step(params, opt_state, tokens)
            return params, opt_state

        seconds = _timed_steps(step_fn, (params, opt_state), data_fn, warmup=warmup, iters=iters)
    return seconds, float(np.asarray(last_loss))


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

    stage = int(os.environ.get("MOE_PP_STAGE", "4" if on_tpu else "2"))
    expert = int(os.environ.get("MOE_PP_EP", "2"))
    if on_tpu:
        hidden_dim, num_layers, num_experts = 2048, 24, 32
        seq_len, vocab_size = 1024, 32768
        microbatch, num_microbatches = 4, 8
        warmup, iters = 5, 20
    else:
        hidden_dim, num_layers, num_experts = 128, 4, 4
        seq_len, vocab_size = 64, 512
        microbatch, num_microbatches = 4, 2
        warmup, iters = 2, 5
    num_experts_per_token = int(os.environ.get("MOE_PP_EPT", "2"))
    # Model-dimension overrides for a largest-that-fits sweep on hardware.
    hidden_dim = int(os.environ.get("MOE_PP_HIDDEN", hidden_dim))
    num_layers = int(os.environ.get("MOE_PP_LAYERS", num_layers))
    num_experts = int(os.environ.get("MOE_PP_EXPERTS", num_experts))
    seq_len = int(os.environ.get("MOE_PP_SEQ", seq_len))
    vocab_size = int(os.environ.get("MOE_PP_VOCAB", vocab_size))
    microbatch = int(os.environ.get("MOE_PP_MICROBATCH", microbatch))
    num_microbatches = int(os.environ.get("MOE_PP_NMICRO", num_microbatches))

    num_devices = jax.device_count()
    data = num_devices // (stage * expert)
    if num_devices % (stage * expert) != 0:
        raise ValueError(f"device_count={num_devices} must be divisible by stage*expert={stage * expert}")
    if num_layers % stage != 0:
        raise ValueError(f"num_layers={num_layers} must be divisible by stage={stage}")
    # The production model shards a microbatch's batch dim over (replica_dcn, data,
    # expert); replica_dcn is 1 here, so the per-microbatch batch must divide data*expert.
    batch_shards = data * expert
    if microbatch % batch_shards != 0:
        raise ValueError(
            f"microbatch={microbatch} must be divisible by data*expert={batch_shards} "
            "(the microbatch batch dim shards over data and expert)"
        )
    if num_experts % expert != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by expert={expert}")

    cfg = _config(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        seq_len=seq_len,
    )
    global_batch = num_microbatches * microbatch
    tokens_per_step = global_batch * seq_len
    total_b, active_b = _param_count(cfg)
    logger.info("mesh: stage=%d data=%d expert=%d (devices=%d)", stage, data, expert, num_devices)
    logger.info(
        "model: hidden=%d layers=%d experts=%d ept=%d seq=%d vocab=%d | microbatch=%d nmicro=%d"
        " global_batch=%d tok/step=%d | ~%.2fB total / ~%.2fB active",
        hidden_dim,
        num_layers,
        num_experts,
        num_experts_per_token,
        seq_len,
        vocab_size,
        microbatch,
        num_microbatches,
        global_batch,
        tokens_per_step,
        total_b,
        active_b,
    )

    bench_kwargs = dict(warmup=warmup, iters=iters, seed=0)
    # Run the pipeline first: it microbatches the global batch, so it fits where the
    # one-shot FSDP baseline below may not. An FSDP OOM then does not block the PP result.
    pp_s, pp_loss = bench_pipeline(
        cfg,
        stage=stage,
        expert=expert,
        num_microbatches=num_microbatches,
        microbatch=microbatch,
        seq_len=seq_len,
        **bench_kwargs,
    )
    pp_tps = tokens_per_step / pp_s
    logger.info(
        "PPxFSDPxEP (stage=%d,data=%d,expert=%d): %.1f ms/step  %.0f tokens/sec  (loss=%.4f)",
        stage,
        data,
        expert,
        pp_s * 1e3,
        pp_tps,
        pp_loss,
    )

    try:
        fsdp_s, fsdp_loss = bench_fsdp(cfg, expert=expert, global_batch=global_batch, seq_len=seq_len, **bench_kwargs)
    except jax.errors.JaxRuntimeError as e:
        if "RESOURCE_EXHAUSTED" not in str(e):
            raise
        logger.info("FSDP OOM at this size -- pipeline microbatching fits where pure FSDP does not")
        return 0

    fsdp_tps = tokens_per_step / fsdp_s
    logger.info(
        "FSDP       (stage=1,data=%d,expert=%d): %.1f ms/step  %.0f tokens/sec  (loss=%.4f)",
        num_devices // expert,
        expert,
        fsdp_s * 1e3,
        fsdp_tps,
        fsdp_loss,
    )
    logger.info("PP/FSDP throughput ratio: %.2fx", pp_tps / fsdp_tps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
