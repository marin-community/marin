# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Throughput benchmark: PP x FSDP x EP pipeline vs pure FSDP for the PRODUCTION grug-MoE.

Both paths train the IDENTICAL production ``Transformer`` (real ring-EP ``moe_mlp``,
FSDP over ``data``, vocab-TP over ``model``, fused-CE head) on the same chips with
the same global batch -- only the parallelism layout differs:

- **PP x FSDP x EP** (the pipeline): an :func:`ep_pipeline_mesh` with ``stage`` /
  ``expert`` / ``data`` all manual (``data`` fills whatever ``stage`` and ``expert``
  leave). The global batch is split into ``num_microbatches`` microbatches that
  pipeline across the ``stage`` axis with the real ring EP inline; the gradient-exact
  whole-program backward (:func:`pipeline_value_and_grad_ep_microbatched`) feeds an
  ``optax.adamw`` update.
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
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe_pp.oracle import oracle_loss
from experiments.grug.moe_pp.pipeline import (
    _stage_in_specs,
    ep_pipeline_mesh,
    pipeline_value_and_grad_ep_microbatched,
    stack_blocks_for_stages,
)

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


def _config(
    *, vocab_size, hidden_dim, num_layers, num_experts, num_experts_per_token, seq_len, attention_implementation
) -> GrugModelConfig:
    # The pipeline path requires ``reference`` (plain-JAX einsum) attention: it
    # differentiates the forward with whole-program autodiff inside the
    # {stage, expert, data}-manual check_vma=False shard_map, and the TPU splash kernel
    # emits a custom-VJP/ShapeDtypeStruct that does not transpose through it. The FSDP
    # baseline (ordinary top-level autodiff) can run either impl, so it is timed under
    # both to measure the splash-vs-reference delta.
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
        attention_implementation=attention_implementation,
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
    data: int,
    num_microbatches: int,
    microbatch: int,
    seq_len: int,
    warmup: int,
    iters: int,
    seed: int,
) -> tuple[float, float]:
    """Time a full PP x FSDP x EP training step. Returns ``(seconds_per_step, loss)``.

    Runs the inline ring-EP pipeline (:func:`pipeline_value_and_grad_ep_microbatched`)
    under an :func:`ep_pipeline_mesh` -- ``stage`` / ``expert`` / ``data`` all manual so
    the megablox GMM lowers on TPU and the FSDP weight-grad reduce over ``data`` is
    inserted by the single shard_map's transpose. The model is initialized under a
    ``compact_grug_mesh(stage_axis_size=stage)`` (the init-time weight sharding) and the
    value_and_grad runs under the EP mesh. The global batch is ``num_microbatches *
    microbatch``; each microbatch's batch dim shards over both ``expert`` and ``data``.
    """
    init_mesh = compact_grug_mesh(expert_axis_size=expert, replica_axis_size=1, model_axis_size=1, stage_axis_size=stage)
    run_mesh = ep_pipeline_mesh(stage=stage, expert=expert, replica=1, data=data)
    weight_microbatches = jnp.ones((num_microbatches, microbatch, seq_len), dtype=jnp.float32)
    optimizer = optax.adamw(LR)

    with set_mesh(init_mesh):
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

    with set_mesh(run_mesh):
        # The params, optimizer state, and grads must all live on the run mesh: the EP
        # value_and_grad reshards its inputs to the run mesh internally and returns grads
        # there, so reshard the init-mesh params to the run mesh's in-specs (embed/head
        # replicated, blocks stage-sharded with expert-sharded MLP weights) up front so
        # optax never mixes the all-Explicit init aval mesh with the run mesh.
        stage_in_specs = _stage_in_specs(stage_arrays)
        embed_in_specs = jax.tree_util.tree_map(lambda _: P(), embed_arrays)
        embed_arrays = jax.tree_util.tree_map(
            lambda x, spec: reshard(x, NamedSharding(run_mesh, spec)), embed_arrays, embed_in_specs
        )
        stage_arrays = jax.tree_util.tree_map(
            lambda x, spec: reshard(x, NamedSharding(run_mesh, spec)), stage_arrays, stage_in_specs
        )
        # The trainable leaves are the replicated embed/norm/head tuple and the
        # stage-sharded stacked blocks -- the exact two grad groups the EP pipeline
        # value_and_grad returns.
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
            # leaves so the pipeline reads the current head params.
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
            loss, g_embed, g_stage = pipeline_value_and_grad_ep_microbatched(
                updated_model,
                stage_arrays,
                block_static,
                tokens,
                weight_microbatches,
                mesh=run_mesh,
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

    cfg_kwargs = dict(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        seq_len=seq_len,
    )
    cfg = _config(**cfg_kwargs, attention_implementation="reference")
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
        data=data,
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

    def run_fsdp(cfg_variant: GrugModelConfig, label: str, *, optional: bool) -> float | None:
        """Time an FSDP baseline; return tokens/sec, or None if it OOMs (or, for an
        ``optional`` variant, if the attention impl is incompatible with this mesh)."""
        tolerated: tuple[type[Exception], ...] = (jax.errors.JaxRuntimeError,)
        if optional:
            # A splash-attention variant may be incompatible with this mesh; that is a
            # diagnostic, not a failure, so do not let it abort the PP/FSDP comparison.
            tolerated = (jax.errors.JaxRuntimeError, ValueError, RuntimeError)
        try:
            seconds, loss = bench_fsdp(
                cfg_variant, expert=expert, global_batch=global_batch, seq_len=seq_len, **bench_kwargs
            )
        except tolerated as e:
            if not optional and "RESOURCE_EXHAUSTED" not in str(e):
                raise
            reason = "OOM" if "RESOURCE_EXHAUSTED" in str(e) else "unsupported on this mesh"
            logger.info("FSDP[%s] skipped (%s)", label, reason)
            return None
        tps = tokens_per_step / seconds
        logger.info(
            "FSDP[%s]  (stage=1,data=%d,expert=%d): %.1f ms/step  %.0f tokens/sec  (loss=%.4f)",
            label,
            num_devices // expert,
            expert,
            seconds * 1e3,
            tps,
            loss,
        )
        return tps

    # Reference-attention FSDP is the apples-to-apples baseline (same attention as PP).
    # Splash-attention FSDP is production-representative and quantifies the attention-impl
    # cost; it runs only on TPU (no splash kernel on CPU) and is best-effort.
    fsdp_ref_tps = run_fsdp(cfg, "reference", optional=False)
    fsdp_splash_tps = None
    if on_tpu:
        cfg_splash = _config(**cfg_kwargs, attention_implementation="tpu_splash")
        fsdp_splash_tps = run_fsdp(cfg_splash, "splash", optional=True)

    if fsdp_ref_tps is not None:
        logger.info("PP / FSDP(reference) ratio: %.2fx  [apples-to-apples, same attention]", pp_tps / fsdp_ref_tps)
    else:
        logger.info("FSDP(reference) OOM -- pipeline microbatching fits where one-shot FSDP does not")
    if fsdp_ref_tps is not None and fsdp_splash_tps is not None:
        logger.info(
            "FSDP splash / reference ratio: %.2fx  [attention-impl speedup; how much reference understates production]",
            fsdp_splash_tps / fsdp_ref_tps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
