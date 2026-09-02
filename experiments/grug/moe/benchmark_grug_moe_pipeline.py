# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configurable multi-host H100 benchmark for the canonical Grug MoE pipeline."""

import dataclasses
import json
import os
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import cast


class PipelineSchedule(StrEnum):
    ZERO_BUBBLE = "automatic_zero_bubble"
    DUALPIPE_V = "automatic_dualpipe_v"

    @property
    def automatic_schedule(self):
        from experiments.grug.moe.grug_moe_pipeline import AutomaticPipelineSchedule  # noqa: PLC0415

        if self == self.DUALPIPE_V:
            return AutomaticPipelineSchedule.DUALPIPE_V
        if self == self.ZERO_BUBBLE:
            return AutomaticPipelineSchedule.ZERO_BUBBLE
        raise ValueError(f"unknown pipeline schedule: {self.value}")


@dataclass(frozen=True)
class BenchmarkConfig:
    stages: int
    physical_stages: int
    microbatches: int
    batch_size: int
    seq_len: int
    hidden_dim: int
    intermediate_dim: int
    shared_expert_intermediate_dim: int
    num_layers: int
    num_experts: int
    top_k: int
    expert_axis_size: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    sliding_window: int
    qk_mult: float
    steps: int
    warmup_steps: int
    profile_start_step: int
    profile_steps: int
    profile_run_id: str | None
    layer_counts: tuple[int, ...] | None
    memory_threshold: int | None
    mp_policy_string: str
    remat_mode: str
    attention_implementation: str
    moe_implementation: str
    schedule: PipelineSchedule


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    return default if not value else int(value)


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name)
    return default if not value else float(value)


def _resolve_benchmark_config(environ: Mapping[str, str]) -> BenchmarkConfig:
    from experiments.grug.moe.heuristic import MoeHeuristic  # noqa: PLC0415

    stages = _env_int(environ, "PIPELINE_STAGES", 4)
    physical_stages = _env_int(environ, "PIPELINE_PHYSICAL_STAGES", stages)
    seq_len = _env_int(environ, "PIPELINE_SEQ_LEN", 4096)
    hidden_dim = _env_int(environ, "PIPELINE_HIDDEN_DIM", 2560)
    warmup_steps = _env_int(environ, "PIPELINE_WARMUP_STEPS", 1)
    profile_start_step = _env_int(environ, "PIPELINE_PROFILE_START_STEP", warmup_steps)
    profile_steps = _env_int(environ, "PIPELINE_PROFILE_STEPS", 0)
    profile_run_id = environ.get("PIPELINE_PROFILE_RUN_ID")
    steps = _env_int(environ, "PIPELINE_STEPS", 4)
    if profile_steps > 0:
        if not profile_run_id:
            raise ValueError("PIPELINE_PROFILE_RUN_ID is required when profiling")
        if profile_start_step + profile_steps > steps:
            raise ValueError("pipeline profile window must fit within PIPELINE_STEPS")

    layer_counts_value = environ.get("PIPELINE_LAYERS_PER_STAGE")
    layer_counts = None if not layer_counts_value else tuple(int(value) for value in layer_counts_value.split(","))
    memory_threshold_value = environ.get("PIPELINE_RESHARD_THRESHOLD_BYTES")
    memory_threshold = None if not memory_threshold_value else int(memory_threshold_value)
    base_model_config = MoeHeuristic().build_model_config(hidden_dim, seq_len=seq_len)
    return BenchmarkConfig(
        stages=stages,
        physical_stages=physical_stages,
        microbatches=_env_int(environ, "PIPELINE_MICROBATCHES", 8),
        batch_size=_env_int(environ, "PIPELINE_BATCH", 256),
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        intermediate_dim=_env_int(environ, "PIPELINE_INTERMEDIATE_DIM", base_model_config.intermediate_dim),
        shared_expert_intermediate_dim=_env_int(
            environ,
            "PIPELINE_SHARED_EXPERT_INTERMEDIATE_DIM",
            base_model_config.shared_expert_intermediate_dim,
        ),
        num_layers=_env_int(environ, "PIPELINE_LAYERS", 24),
        num_experts=_env_int(environ, "PIPELINE_EXPERTS", 256),
        top_k=_env_int(environ, "PIPELINE_TOP_K", 4),
        expert_axis_size=_env_int(environ, "PIPELINE_EXPERT_AXIS", 8),
        vocab_size=_env_int(environ, "PIPELINE_VOCAB_SIZE", base_model_config.vocab_size),
        num_heads=_env_int(environ, "PIPELINE_HEADS", base_model_config.num_heads),
        num_kv_heads=_env_int(environ, "PIPELINE_KV_HEADS", base_model_config.num_kv_heads),
        sliding_window=_env_int(environ, "PIPELINE_SLIDING_WINDOW", base_model_config.sliding_window),
        qk_mult=_env_float(environ, "PIPELINE_QK_MULT", base_model_config.qk_mult),
        steps=steps,
        warmup_steps=warmup_steps,
        profile_start_step=profile_start_step,
        profile_steps=profile_steps,
        profile_run_id=profile_run_id,
        layer_counts=layer_counts,
        memory_threshold=memory_threshold,
        mp_policy_string=environ.get("PIPELINE_MP", "params=bfloat16,compute=bfloat16,output=bfloat16"),
        remat_mode=environ.get("PIPELINE_REMAT", "recompute_all"),
        attention_implementation=environ.get("PIPELINE_ATTENTION", "gpu_fa4_cute"),
        moe_implementation=environ.get("PIPELINE_MOE", "ring"),
        schedule=PipelineSchedule(environ.get("PIPELINE_SCHEDULE", PipelineSchedule.ZERO_BUBBLE)),
    )


def _log(event: str, **values) -> None:
    print(f"{event} {json.dumps(values, sort_keys=True, default=str)}", flush=True)


def _validate_local_mesh(
    *,
    local_device_count: int,
    expert_axis_size: int,
    batch_size: int,
    microbatches: int,
) -> None:
    if local_device_count % expert_axis_size != 0:
        raise ValueError(f"local device count {local_device_count} must be divisible by expert axis {expert_axis_size}")
    if batch_size % microbatches != 0:
        raise ValueError(f"batch size {batch_size} must be divisible by {microbatches} microbatches")
    microbatch_size = batch_size // microbatches
    if microbatch_size % local_device_count != 0:
        raise ValueError(
            f"microbatch size {microbatch_size} must be divisible by the {local_device_count} devices in each stage"
        )


def _run_benchmark(config: BenchmarkConfig) -> None:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import jmp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import optax  # noqa: PLC0415
    from fray.device_flops import device_flops  # noqa: PLC0415
    from jax.sharding import NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from jaxpp.array import MpmdArray  # noqa: PLC0415
    from levanter.data.text.examples import GrugLmExample  # noqa: PLC0415
    from levanter.utils.flop_utils import lm_flops_per_token  # noqa: PLC0415

    from experiments.grug.moe.grug_moe_pipeline import (  # noqa: PLC0415
        TRAIN_LOSS_KEY,
        GrugMoePipelineConfig,
        automatic_stage_to_mpmd_indices,
        make_automatic_pipeline_step,
        make_mpmd_automatic_pipeline_state,
        make_pipeline_mesh,
        prepare_automatic_mpmd_step,
        stacked_microbatches,
    )
    from experiments.grug.moe.heuristic import MoeHeuristic  # noqa: PLC0415
    from experiments.grug.moe.model import BATCH_AXES, Transformer  # noqa: PLC0415

    mp_policy = jmp.get_policy(config.mp_policy_string)
    pipeline_config = GrugMoePipelineConfig(
        stages=config.stages,
        microbatches=config.microbatches,
        physical_stages=None if config.physical_stages == config.stages else config.physical_stages,
    )

    if jax.process_count() != config.physical_stages:
        raise ValueError(
            f"expected one process per physical stage ({config.physical_stages}), got {jax.process_count()}"
        )
    _validate_local_mesh(
        local_device_count=jax.local_device_count(),
        expert_axis_size=config.expert_axis_size,
        batch_size=config.batch_size,
        microbatches=config.microbatches,
    )

    base_model_config = MoeHeuristic().build_model_config(config.hidden_dim, seq_len=config.seq_len)
    model_config = dataclasses.replace(
        base_model_config,
        vocab_size=config.vocab_size,
        intermediate_dim=config.intermediate_dim,
        shared_expert_intermediate_dim=config.shared_expert_intermediate_dim,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        num_experts_per_token=config.top_k,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        sliding_window=config.sliding_window,
        qk_mult=config.qk_mult,
        router_z_loss_coef=0.0,
        attention_implementation=cast(str, config.attention_implementation),
        moe_implementation=cast(str, config.moe_implementation),
        remat_mode=cast(str, config.remat_mode),
    )
    mesh, mpmd_mesh = make_pipeline_mesh(
        pipeline_config,
        expert_axis_size=config.expert_axis_size,
        replica_axis_size=1,
    )
    optimizer = optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.95, weight_decay=0.1)
    peak_flops_per_device = device_flops("h100")
    if peak_flops_per_device is None:
        raise ValueError("Fray does not define H100 BF16 peak FLOP/s")

    _log(
        "PIPELINE_CONFIG",
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        global_devices=jax.device_count(),
        local_devices=jax.local_device_count(),
        mesh_shape=dict(mesh.shape),
        stages=config.stages,
        physical_stages=config.physical_stages,
        microbatches=config.microbatches,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        hidden_dim=config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_expert_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=config.num_layers,
        layers_per_stage=config.layer_counts,
        num_experts=config.num_experts,
        top_k=config.top_k,
        vocab_size=model_config.vocab_size,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        sliding_window=model_config.sliding_window,
        qk_mult=model_config.qk_mult,
        steps=config.steps,
        profile_start_step=config.profile_start_step,
        profile_steps=config.profile_steps,
        mp_policy=config.mp_policy_string,
        remat_mode=config.remat_mode,
        schedule=config.schedule,
        jax_version=jax.__version__,
    )

    init_started = time.monotonic()
    with jax.set_mesh(mesh):
        model = mp_policy.cast_to_param(Transformer.init(model_config, key=jax.random.PRNGKey(0)))
        batch_sharding = NamedSharding(mesh, P(BATCH_AXES, None))
        token_row = np.arange(config.seq_len, dtype=np.int32) % model_config.vocab_size
        host_tokens = np.broadcast_to(token_row, (config.batch_size, config.seq_len)).copy()
        host_loss_weight = np.ones((config.batch_size, config.seq_len), dtype=np.float32)
        host_loss_weight[:, -1] = 0
        batch = GrugLmExample(
            tokens=jax.device_put(host_tokens, batch_sharding),
            loss_weight=jax.device_put(host_loss_weight, batch_sharding),
        )
        loss_denominator = jnp.sum(batch.loss_weight.astype(jnp.float32))
        batches = stacked_microbatches(batch, config.microbatches)
        if config.schedule == PipelineSchedule.DUALPIPE_V:
            stage_to_mpmd_index = automatic_stage_to_mpmd_indices(pipeline_config, config.schedule.automatic_schedule)
        else:
            stage_to_mpmd_index = None
        state, static_stages = make_mpmd_automatic_pipeline_state(
            model,
            optimizer,
            mpmd_mesh,
            num_stages=config.stages,
            layer_counts=config.layer_counts,
            stage_to_mpmd_index=stage_to_mpmd_index,
            memory_threshold=config.memory_threshold,
        )
    jax.block_until_ready((state, batches, loss_denominator))
    _log(
        "PIPELINE_INIT",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - init_started,
    )

    build_started = time.monotonic()
    step = make_automatic_pipeline_step(
        optimizer,
        mp_policy,
        static_stages,
        state,
        batches,
        config=pipeline_config,
        mpmd_mesh=mpmd_mesh,
        schedule_name=config.schedule.automatic_schedule,
    )
    _log(
        "PIPELINE_BUILD",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - build_started,
    )
    lower_started = time.monotonic()
    prepared = prepare_automatic_mpmd_step(
        step,
        state,
        batches,
        loss_denominator,
        mpmd_mesh,
        memory_threshold=config.memory_threshold,
    )
    step = prepared.step
    state = prepared.state
    batches = prepared.batches
    loss_denominator = prepared.loss_denominator
    _log(
        "PIPELINE_LOWER",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - lower_started,
    )

    profiler_callback = None
    if config.profile_steps > 0:
        from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig  # noqa: PLC0415

        profiler_callback = ProfilerConfig(
            enabled=True,
            start_step=config.profile_start_step,
            num_steps=config.profile_steps,
            perfetto_link=False,
            profile_options=ProfileOptionsConfig(
                host_tracer_level=1,
                python_tracer_level=0,
                enable_hlo_proto=True,
            ),
        ).build(
            "/tmp/grug-moe-pipeline-profiler",
            run_id=config.profile_run_id,
        )
        profiler_callback(SimpleNamespace(step=-1))

    step_times = []
    loss = None
    for step_index in range(config.steps):
        started = time.monotonic()
        state, metrics = step(state, batches, loss_denominator)
        jax.block_until_ready((state, metrics))
        elapsed = time.monotonic() - started
        step_times.append(elapsed)
        metric_loss = metrics[TRAIN_LOSS_KEY]
        if profiler_callback is not None:
            profiler_callback(SimpleNamespace(step=step_index))
        if isinstance(metric_loss, MpmdArray):
            if not metric_loss.is_partially_addressable:
                continue
            local_loss = metric_loss.to_mpmd_local_array
            if not isinstance(local_loss, jax.Array):
                raise TypeError(f"expected one local loss array, got {type(local_loss).__name__}")
        else:
            local_loss = metric_loss
        loss = float(local_loss)
        _log(
            "PIPELINE_STEP",
            step=step_index,
            elapsed_seconds=elapsed,
            tokens_per_second=config.batch_size * config.seq_len / elapsed,
            loss=loss,
        )

    measured = step_times[config.warmup_steps :]
    if not measured:
        measured = step_times
    mean_step_seconds = sum(measured) / len(measured)
    median_step_seconds = statistics.median(measured)
    tokens_per_second = config.batch_size * config.seq_len / mean_step_seconds
    median_tokens_per_second = config.batch_size * config.seq_len / median_step_seconds
    forward_flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=1 if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
    )
    analytic_mfu = 3 * forward_flops_per_token * tokens_per_second / (jax.device_count() * peak_flops_per_device)
    median_analytic_mfu = (
        3 * forward_flops_per_token * median_tokens_per_second / (jax.device_count() * peak_flops_per_device)
    )
    if loss is not None:
        _log(
            "PIPELINE_SUMMARY",
            mean_step_seconds=mean_step_seconds,
            median_step_seconds=median_step_seconds,
            tokens_per_second=tokens_per_second,
            median_tokens_per_second=median_tokens_per_second,
            analytic_mfu=analytic_mfu,
            median_analytic_mfu=median_analytic_mfu,
            measured_step_seconds=measured,
            loss=loss,
            measured_steps=len(measured),
        )

    for device in jax.local_devices():
        stats = device.memory_stats()
        if stats is not None:
            _log(
                "PIPELINE_MEMORY",
                process_index=jax.process_index(),
                device_id=device.id,
                peak_bytes_in_use=stats.get("peak_bytes_in_use"),
                bytes_limit=stats.get("bytes_limit"),
            )


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415

    initialize_jax()
    config = _resolve_benchmark_config(os.environ)
    _run_benchmark(config)


if __name__ == "__main__":
    main()
