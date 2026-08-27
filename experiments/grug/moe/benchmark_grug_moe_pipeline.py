# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configurable multi-host H100 benchmark for the canonical Grug MoE pipeline."""

import dataclasses
import json
import os
import statistics
import time
from types import SimpleNamespace
from typing import cast

H100_BF16_PEAK_FLOPS = 989e12


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if not value else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if not value else float(value)


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


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415

    initialize_jax()

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import jmp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import optax  # noqa: PLC0415
    from jax.sharding import NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from levanter.data.text.examples import GrugLmExample  # noqa: PLC0415
    from levanter.utils.flop_utils import lm_flops_per_token  # noqa: PLC0415

    from experiments.grug.moe.grug_moe_pipeline import (  # noqa: PLC0415
        GrugMoePipelineConfig,
        automatic_stage_to_mpmd_indices,
        batches_for_pipeline,
        make_automatic_zero_bubble_step,
        make_explicit_1f1b_step,
        make_explicit_zero_bubble_step,
        make_mpmd_automatic_pipeline_state,
        make_mpmd_pipeline_state,
        make_pipeline_mesh,
        place_pipeline_batches,
        prepare_automatic_mpmd_step,
        prepare_explicit_step,
        stacked_microbatches,
    )
    from experiments.grug.moe.heuristic import MoeHeuristic  # noqa: PLC0415
    from experiments.grug.moe.model import Transformer  # noqa: PLC0415

    stages = _env_int("PIPELINE_STAGES", 4)
    physical_stages = _env_int("PIPELINE_PHYSICAL_STAGES", stages)
    microbatches = _env_int("PIPELINE_MICROBATCHES", 8)
    batch_size = _env_int("PIPELINE_BATCH", 256)
    seq_len = _env_int("PIPELINE_SEQ_LEN", 4096)
    hidden_dim = _env_int("PIPELINE_HIDDEN_DIM", 2560)
    num_layers = _env_int("PIPELINE_LAYERS", 24)
    num_experts = _env_int("PIPELINE_EXPERTS", 256)
    top_k = _env_int("PIPELINE_TOP_K", 4)
    expert_axis_size = _env_int("PIPELINE_EXPERT_AXIS", 8)
    steps = _env_int("PIPELINE_STEPS", 4)
    warmup_steps = _env_int("PIPELINE_WARMUP_STEPS", 1)
    profile_start_step = _env_int("PIPELINE_PROFILE_START_STEP", warmup_steps)
    profile_steps = _env_int("PIPELINE_PROFILE_STEPS", 0)
    profile_run_id = os.environ.get("PIPELINE_PROFILE_RUN_ID")
    if profile_steps > 0:
        if not profile_run_id:
            raise ValueError("PIPELINE_PROFILE_RUN_ID is required when profiling")
        if profile_start_step + profile_steps > steps:
            raise ValueError("pipeline profile window must fit within PIPELINE_STEPS")
    layer_counts_value = os.environ.get("PIPELINE_LAYERS_PER_STAGE")
    layer_counts = None if not layer_counts_value else tuple(int(value) for value in layer_counts_value.split(","))
    memory_threshold_value = os.environ.get("PIPELINE_RESHARD_THRESHOLD_BYTES")
    memory_threshold = None if not memory_threshold_value else int(memory_threshold_value)
    mp_policy_string = os.environ.get("PIPELINE_MP", "params=bfloat16,compute=bfloat16,output=bfloat16")
    mp_policy = jmp.get_policy(mp_policy_string)
    remat_mode = os.environ.get("PIPELINE_REMAT", "recompute_all")
    schedule = os.environ.get("PIPELINE_SCHEDULE", "1f1b")
    pipeline_config = GrugMoePipelineConfig(
        stages=stages,
        microbatches=microbatches,
        physical_stages=None if physical_stages == stages else physical_stages,
    )

    if jax.process_count() != physical_stages:
        raise ValueError(f"expected one process per physical stage ({physical_stages}), got {jax.process_count()}")
    _validate_local_mesh(
        local_device_count=jax.local_device_count(),
        expert_axis_size=expert_axis_size,
        batch_size=batch_size,
        microbatches=microbatches,
    )

    base_model_config = MoeHeuristic().build_model_config(hidden_dim, seq_len=seq_len)
    model_config = dataclasses.replace(
        base_model_config,
        vocab_size=_env_int("PIPELINE_VOCAB_SIZE", base_model_config.vocab_size),
        intermediate_dim=_env_int("PIPELINE_INTERMEDIATE_DIM", base_model_config.intermediate_dim),
        shared_expert_intermediate_dim=_env_int(
            "PIPELINE_SHARED_EXPERT_INTERMEDIATE_DIM",
            base_model_config.shared_expert_intermediate_dim,
        ),
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        num_heads=_env_int("PIPELINE_HEADS", base_model_config.num_heads),
        num_kv_heads=_env_int("PIPELINE_KV_HEADS", base_model_config.num_kv_heads),
        sliding_window=_env_int("PIPELINE_SLIDING_WINDOW", base_model_config.sliding_window),
        qk_mult=_env_float("PIPELINE_QK_MULT", base_model_config.qk_mult),
        router_z_loss_coef=0.0,
        attention_implementation=cast(str, os.environ.get("PIPELINE_ATTENTION", "gpu_fa4_cute")),
        moe_implementation=cast(str, os.environ.get("PIPELINE_MOE", "ring")),
        remat_mode=cast(str, remat_mode),
    )
    mesh, mpmd_mesh = make_pipeline_mesh(
        pipeline_config,
        expert_axis_size=expert_axis_size,
        replica_axis_size=1,
    )
    optimizer = optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.95, weight_decay=0.1)

    _log(
        "PIPELINE_CONFIG",
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        global_devices=jax.device_count(),
        local_devices=jax.local_device_count(),
        mesh_shape=dict(mesh.shape),
        stages=stages,
        physical_stages=physical_stages,
        microbatches=microbatches,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_expert_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=num_layers,
        layers_per_stage=layer_counts,
        num_experts=num_experts,
        top_k=top_k,
        vocab_size=model_config.vocab_size,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        sliding_window=model_config.sliding_window,
        qk_mult=model_config.qk_mult,
        steps=steps,
        profile_start_step=profile_start_step,
        profile_steps=profile_steps,
        mp_policy=mp_policy_string,
        remat_mode=remat_mode,
        schedule=schedule,
        jax_version=jax.__version__,
    )

    init_started = time.monotonic()
    with jax.set_mesh(mesh):
        model = mp_policy.cast_to_param(Transformer.init(model_config, key=jax.random.PRNGKey(0)))
        batch_sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        token_row = np.arange(seq_len, dtype=np.int32) % model_config.vocab_size
        host_tokens = np.broadcast_to(token_row, (batch_size, seq_len)).copy()
        host_loss_weight = np.ones((batch_size, seq_len), dtype=np.float32)
        host_loss_weight[:, -1] = 0
        batch = GrugLmExample(
            tokens=jax.device_put(host_tokens, batch_sharding),
            loss_weight=jax.device_put(host_loss_weight, batch_sharding),
        )
        loss_denominator = jnp.sum(batch.loss_weight.astype(jnp.float32))
        if schedule in {"automatic_zero_bubble", "automatic_dualpipe_v"}:
            batches = stacked_microbatches(batch, microbatches)
            if schedule == "automatic_dualpipe_v":
                stage_to_mpmd_index = automatic_stage_to_mpmd_indices(pipeline_config, "dualpipe_v")
            else:
                stage_to_mpmd_index = None
            state, static_stages = make_mpmd_automatic_pipeline_state(
                model,
                optimizer,
                mpmd_mesh,
                num_stages=stages,
                layer_counts=layer_counts,
                stage_to_mpmd_index=stage_to_mpmd_index,
                memory_threshold=memory_threshold,
            )
        else:
            batches = batches_for_pipeline(batch, pipeline_config)
            state = make_mpmd_pipeline_state(
                model,
                optimizer,
                mpmd_mesh,
                num_stages=stages,
                layer_counts=layer_counts,
                memory_threshold=memory_threshold,
            )
            batches = place_pipeline_batches(
                mpmd_mesh,
                batches,
                memory_threshold=memory_threshold,
            )
    jax.block_until_ready((state, batches, loss_denominator))
    _log(
        "PIPELINE_INIT",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - init_started,
    )

    build_started = time.monotonic()
    if schedule in {"automatic_zero_bubble", "automatic_dualpipe_v"}:
        step = make_automatic_zero_bubble_step(
            optimizer,
            mp_policy,
            static_stages,
            state,
            batches,
            config=pipeline_config,
            mpmd_mesh=mpmd_mesh,
            schedule_name="dualpipe_v" if schedule == "automatic_dualpipe_v" else "zero_bubble",
        )
    else:
        if schedule == "1f1b":
            make_step = make_explicit_1f1b_step
        elif schedule == "zero_bubble":
            make_step = make_explicit_zero_bubble_step
        else:
            raise ValueError(f"unknown pipeline schedule: {schedule}")
        step = make_step(
            optimizer,
            mp_policy,
            config=pipeline_config,
            mpmd_mesh=mpmd_mesh,
            sample_state=state,
            sample_batches=batches,
        )
    _log(
        "PIPELINE_BUILD",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - build_started,
    )
    lower_started = time.monotonic()
    if schedule in {"automatic_zero_bubble", "automatic_dualpipe_v"}:
        step, state, batches, loss_denominator = prepare_automatic_mpmd_step(
            step,
            state,
            batches,
            loss_denominator,
            mpmd_mesh,
            memory_threshold=memory_threshold,
        )
    else:
        step = prepare_explicit_step(step, state, batches, mpmd_mesh)
    _log(
        "PIPELINE_LOWER",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - lower_started,
    )

    profiler_callback = None
    if profile_steps > 0:
        from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig  # noqa: PLC0415

        profiler_callback = ProfilerConfig(
            enabled=True,
            start_step=profile_start_step,
            num_steps=profile_steps,
            perfetto_link=False,
            profile_options=ProfileOptionsConfig(
                host_tracer_level=1,
                python_tracer_level=0,
                enable_hlo_proto=True,
            ),
        ).build(
            "/tmp/grug-moe-pipeline-profiler",
            run_id=profile_run_id,
        )

    step_times = []
    loss = None
    for step_index in range(steps):
        started = time.monotonic()
        if schedule in {"automatic_zero_bubble", "automatic_dualpipe_v"}:
            state, metrics = step(state, batches, loss_denominator)
        else:
            state, metrics = step(state, batches)
        jax.block_until_ready((state, metrics))
        elapsed = time.monotonic() - started
        step_times.append(elapsed)
        metric_loss = metrics["train/loss"]
        if profiler_callback is not None:
            profiler_callback(SimpleNamespace(step=step_index))
        if not hasattr(metric_loss, "is_partially_addressable") or metric_loss.is_partially_addressable:
            if hasattr(metric_loss, "to_mpmd_local_array"):
                metric_loss = metric_loss.to_mpmd_local_array
            loss = float(metric_loss)
            _log(
                "PIPELINE_STEP",
                step=step_index,
                elapsed_seconds=elapsed,
                tokens_per_second=batch_size * seq_len / elapsed,
                loss=loss,
            )

    measured = step_times[warmup_steps:]
    if not measured:
        measured = step_times
    mean_step_seconds = sum(measured) / len(measured)
    median_step_seconds = statistics.median(measured)
    tokens_per_second = batch_size * seq_len / mean_step_seconds
    median_tokens_per_second = batch_size * seq_len / median_step_seconds
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
    analytic_mfu = 3 * forward_flops_per_token * tokens_per_second / (jax.device_count() * H100_BF16_PEAK_FLOPS)
    median_analytic_mfu = (
        3 * forward_flops_per_token * median_tokens_per_second / (jax.device_count() * H100_BF16_PEAK_FLOPS)
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


if __name__ == "__main__":
    main()
