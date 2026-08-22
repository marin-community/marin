# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded multi-host H100 smoke test for the canonical Grug MoE pipeline."""

import dataclasses
import json
import os
import time
from typing import cast

H100_BF16_PEAK_FLOPS = 989e12


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if not value else int(value)


def _log(event: str, **values) -> None:
    print(f"{event} {json.dumps(values, sort_keys=True, default=str)}", flush=True)


def main() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from iris.runtime.jax_init import initialize_jax  # noqa: PLC0415

    initialize_jax()

    import jax  # noqa: PLC0415
    import jmp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import optax  # noqa: PLC0415
    from jax.sharding import NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415
    from levanter.data.text.examples import GrugLmExample  # noqa: PLC0415
    from levanter.utils.flop_utils import lm_flops_per_token  # noqa: PLC0415

    from experiments.grug.moe.grug_moe_pipeline import (  # noqa: PLC0415
        GrugMoePipelineConfig,
        batches_for_pipeline,
        make_explicit_1f1b_step,
        make_mpmd_pipeline_state,
        make_pipeline_mesh,
        place_pipeline_batches,
        prepare_explicit_step,
    )
    from experiments.grug.moe.heuristic import MoeHeuristic  # noqa: PLC0415
    from experiments.grug.moe.model import Transformer  # noqa: PLC0415

    stages = _env_int("PIPELINE_STAGES", 4)
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
    memory_threshold_value = os.environ.get("PIPELINE_RESHARD_THRESHOLD_BYTES")
    memory_threshold = None if not memory_threshold_value else int(memory_threshold_value)
    mp_policy = jmp.get_policy(os.environ.get("PIPELINE_MP", "params=float32,compute=bfloat16,output=bfloat16"))
    pipeline_config = GrugMoePipelineConfig(stages=stages, microbatches=microbatches)

    if jax.process_count() != stages:
        raise ValueError(f"expected one process per stage ({stages}), got {jax.process_count()}")
    if jax.local_device_count() != expert_axis_size:
        raise ValueError(
            f"expected {expert_axis_size} local devices for expert parallelism, got {jax.local_device_count()}"
        )
    if batch_size % microbatches != 0:
        raise ValueError(f"batch size {batch_size} must be divisible by {microbatches} microbatches")
    if (batch_size // microbatches) % expert_axis_size != 0:
        raise ValueError(
            f"microbatch size {batch_size // microbatches} must be divisible by expert axis {expert_axis_size}"
        )

    model_config = dataclasses.replace(
        MoeHeuristic().build_model_config(hidden_dim, seq_len=seq_len),
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        router_z_loss_coef=0.0,
        attention_implementation=cast(str, os.environ.get("PIPELINE_ATTENTION", "gpu_fa4_cute")),
        moe_implementation=cast(str, os.environ.get("PIPELINE_MOE", "ring")),
        remat_mode=cast(str, os.environ.get("PIPELINE_REMAT", "save_moe")),
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
        microbatches=microbatches,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        steps=steps,
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
        batches = batches_for_pipeline(batch, pipeline_config)
    state = make_mpmd_pipeline_state(
        model,
        optimizer,
        mpmd_mesh,
        num_stages=stages,
        memory_threshold=memory_threshold,
    )
    batches = place_pipeline_batches(
        mpmd_mesh,
        batches,
        memory_threshold=memory_threshold,
    )
    jax.block_until_ready((state, batches))
    _log(
        "PIPELINE_INIT",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - init_started,
    )

    lower_started = time.monotonic()
    step = make_explicit_1f1b_step(
        optimizer,
        mp_policy,
        config=pipeline_config,
        mpmd_mesh=mpmd_mesh,
        sample_state=state,
        sample_batches=batches,
    )
    step = prepare_explicit_step(step, state, batches, mpmd_mesh)
    _log(
        "PIPELINE_LOWER",
        process_index=jax.process_index(),
        elapsed_seconds=time.monotonic() - lower_started,
    )

    step_times = []
    loss = None
    for step_index in range(steps):
        started = time.monotonic()
        state, metrics = step(state, batches)
        jax.block_until_ready((state, metrics))
        elapsed = time.monotonic() - started
        step_times.append(elapsed)
        if jax.process_index() == 0:
            loss = float(metrics["train/loss"])
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
    tokens_per_second = batch_size * seq_len / mean_step_seconds
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
    if jax.process_index() == 0:
        _log(
            "PIPELINE_SUMMARY",
            mean_step_seconds=mean_step_seconds,
            tokens_per_second=tokens_per_second,
            analytic_mfu=analytic_mfu,
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
