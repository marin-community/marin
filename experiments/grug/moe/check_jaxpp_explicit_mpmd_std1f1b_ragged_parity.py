# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare direct and explicit-MPMD std-1F1B Grug MoE loss and gradients.

This is an authoritative production-mixed-precision check for the
device-initiated ``ragged_all_to_all`` path. It self-spawns four JAX processes
on one H100x8 host. Each pipeline rank owns two local GPUs as its ``expert=2``
mesh, and each of the four logical stages owns one transformer layer.

After installing the pinned JAX/JaxPP runtime with ``jaxpp_setup_scripts()``
from ``train.py``, run exactly:

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    JAXPP_SOURCE=/tmp/jaxpp \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.35 \
    XLA_FLAGS="--xla_gpu_autotune_level=0 \
      --xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true \
      --xla_gpu_ragged_all_to_all_mode=symmetric \
      --xla_enable_nccl_symmetric_buffers_for_collectives=RaggedAllToAll \
      --xla_gpu_nccl_termination_timeout_seconds=120" \
    .venv/bin/python -u \
      experiments/grug/moe/check_jaxpp_explicit_mpmd_std1f1b_ragged_parity.py

The direct reference is the mean of four independently differentiated
microbatch losses. The explicit arm executes ``train.py``'s production
``std_1f1b`` step with an observation-only Optax transform that leaves
parameters unchanged and stores each averaged stage-local gradient directly in
optimizer state. Loss and every gradient leaf must have relative-L2 at most
0.002.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import multiprocessing as mp
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe import check_jaxpp_eager_1f1b_parity as eager_parity
from experiments.grug.moe import train as grug_train
from experiments.grug.moe.model import GrugModelConfig

PIPELINE_STAGES = 4
MICROBATCHES = 4
DEVICES_PER_STAGE = 2
EXPERT_AXIS_SIZE = 2
MICROBATCH_SIZE = 2
SEQUENCE_LENGTH = 8
COORDINATOR_PORT = 5847
_BATCH_AXES = ("replica_dcn", "data", "expert")
_DEVICE_RAGGED_FLAG = "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true"


def validate_authoritative_topology(
    *,
    process_count: int,
    local_device_count: int,
    device_count: int,
) -> None:
    """Require one four-rank pipeline with an EP2 mesh on every rank."""
    expected_devices = PIPELINE_STAGES * DEVICES_PER_STAGE
    if process_count != PIPELINE_STAGES or local_device_count != DEVICES_PER_STAGE or device_count != expected_devices:
        raise ValueError(
            "explicit-MPMD ragged parity requires four JAX processes with two local devices each; "
            f"found process_count={process_count}, local_device_count={local_device_count}, "
            f"device_count={device_count}"
        )


def validate_device_ragged_flags(xla_flags: str) -> None:
    """Reject host-initiated ragged-all-to-all execution."""
    if _DEVICE_RAGGED_FLAG not in xla_flags.split():
        raise ValueError(f"device-ragged parity requires XLA_FLAGS to contain {_DEVICE_RAGGED_FLAG}")


def gradient_capture_optimizer() -> optax.GradientTransformation:
    """Return an observer that stores gradients without changing parameters."""

    def init_fn(params):
        return (jax.tree.map(jnp.zeros_like, params),)

    def update_fn(gradients, _state, _params=None):
        zero_updates = jax.tree.map(jnp.zeros_like, gradients)
        return zero_updates, (gradients,)

    return optax.GradientTransformation(init_fn, update_fn)


def captured_gradients(opt_state):
    """Return the gradient pytree stored by :func:`gradient_capture_optimizer`."""
    (gradients,) = opt_state
    return gradients


def build_stage_parity_report(
    *,
    stage_index: int,
    explicit_loss,
    direct_loss,
    explicit_gradients,
    direct_gradients,
) -> eager_parity.ParityReport:
    """Apply the fixed policy to one corresponding pair of stage pytrees."""
    stage_key = f"stage_{stage_index}"
    return eager_parity.build_parity_report(
        automatic_loss=explicit_loss,
        direct_loss=direct_loss,
        automatic_gradients={stage_key: explicit_gradients},
        direct_gradients={stage_key: direct_gradients},
        tolerance=eager_parity.DEFAULT_TOLERANCE,
    )


def _model_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=PIPELINE_STAGES,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=SEQUENCE_LENGTH,
        sliding_window=SEQUENCE_LENGTH,
        attention_implementation="reference",
        moe_implementation="ragged_all_to_all",
        remat_mode="save_moe",
    )


def _pipeline_config() -> grug_train.GrugJaxPPConfig:
    return grug_train.GrugJaxPPConfig(
        stages=PIPELINE_STAGES,
        microbatches=MICROBATCHES,
        schedule="std_1f1b",
        implementation="explicit_mpmd",
        mpmd_dim=PIPELINE_STAGES,
        explicit_mpmd_schedule_mode="default",
        explicit_mpmd_pipeline_wire_format="bf16",
    )


def _global_batch(mesh: jax.sharding.Mesh) -> GrugLmExample:
    global_batch_size = MICROBATCHES * MICROBATCH_SIZE
    tokens = jnp.arange(global_batch_size * SEQUENCE_LENGTH, dtype=jnp.int32)
    tokens = (tokens.reshape(global_batch_size, SEQUENCE_LENGTH) * 7 + 3) % _model_config().vocab_size
    loss_weight = jnp.ones_like(tokens, dtype=jnp.float32).at[:, -1].set(0)
    batch_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    batch = GrugLmExample(
        tokens=jax.device_put(tokens, batch_sharding),
        loss_weight=jax.device_put(loss_weight, batch_sharding),
    )
    with jax.set_mesh(mesh):
        return grug_train._reshape_batch_for_pipeline(batch, MICROBATCHES)


def _stage_batches(mpmd_mesh, batch: GrugLmExample):
    stage_mpmd_indices = grug_train._pipeline_stage_mpmd_indices(_pipeline_config())
    host_stage_batches = tuple(
        tuple(
            (
                grug_train._select_pipeline_microbatch(batch, microbatch_index, MICROBATCHES)
                if stage_index == 0
                else grug_train._copy_shardable_tree(
                    grug_train._select_pipeline_microbatch(batch, microbatch_index, MICROBATCHES)
                )
            )
            for stage_index in range(PIPELINE_STAGES)
        )
        for microbatch_index in range(MICROBATCHES)
    )
    return tuple(
        tuple(
            grug_train._put_batch_on_stage(mpmd_mesh, mpmd_index, stage_batch)
            for mpmd_index, stage_batch in zip(stage_mpmd_indices, microbatch_batches, strict=True)
        )
        for microbatch_batches in host_stage_batches
    )


def _direct_stage_gradients(direct_gradients, *, pipeline, mpmd_mesh):
    stage_gradients = direct_gradients.split_for_pipeline(
        pipeline.stages,
        pipeline.stage_layer_counts,
    )
    stage_mpmd_indices = grug_train._pipeline_stage_mpmd_indices(pipeline)
    target_shardings = tuple(
        grug_train._tree_mpmd_shardings_on_stage(mpmd_mesh, mpmd_index, gradients)
        for mpmd_index, gradients in zip(stage_mpmd_indices, stage_gradients, strict=True)
    )
    return grug_train._reshard_to_mpmd(mpmd_mesh, stage_gradients, target_shardings)


def _authoritative_environment() -> dict[str, Any]:
    reproducer = importlib.import_module("experiments.grug.moe.repro_jaxpp_jax011_ragged_all_to_all")
    environment = reproducer.check_environment("jaxpp-four-stage-ragged")
    validate_device_ragged_flags(environment["xla_flags"])
    return environment


def _run_worker(process_id: int, coordinator_port: int, local_device_ids: list[int]) -> None:
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{coordinator_port}",
        num_processes=PIPELINE_STAGES,
        process_id=process_id,
        local_device_ids=local_device_ids,
        cluster_detection_method="deactivate",
    )
    try:
        validate_authoritative_topology(
            process_count=jax.process_count(),
            local_device_count=jax.local_device_count(),
            device_count=jax.device_count(),
        )
        environment = _authoritative_environment()
        pipeline = _pipeline_config()
        mesh = grug_train._compact_or_pipeline_grug_mesh(
            expert_axis_size=EXPERT_AXIS_SIZE,
            replica_axis_size=1,
            pipeline=pipeline,
        )
        explicit_mpmd = grug_train._require_jaxpp_explicit_mpmd()
        mpmd_mesh = explicit_mpmd.MpmdMesh(mesh, pipeline.stage_axis_name)
        local_stage_index = mpmd_mesh.my_mpmd_axis_index
        optimizer = gradient_capture_optimizer()

        with jax.set_mesh(mesh):
            initial_state = grug_train.initial_state(
                _model_config(),
                optimizer=optimizer,
                mp=eager_parity._MIXED_PRECISION,
                key=jax.random.PRNGKey(0),
                ema_beta=None,
            )
            batch = _global_batch(mesh)
            direct_step = jax.jit(
                functools.partial(
                    eager_parity._direct_microbatch_mean,
                    precision=eager_parity.PrecisionMode.PRODUCTION_MIXED,
                )
            )
            direct_loss, direct_gradients = direct_step(initial_state.params, batch)
            jax.block_until_ready((direct_loss, direct_gradients))

        direct_stage_gradients = _direct_stage_gradients(
            direct_gradients,
            pipeline=pipeline,
            mpmd_mesh=mpmd_mesh,
        )
        pipeline_state = grug_train._split_state_for_explicit_mpmd(
            initial_state,
            pipeline=pipeline,
            optimizer=optimizer,
            mpmd_mesh=mpmd_mesh,
        )
        stage_batches = _stage_batches(mpmd_mesh, batch)

        explicit_step = grug_train._make_explicit_mpmd_train_step(
            optimizer,
            eager_parity._MIXED_PRECISION,
            z_loss_weight=0.0,
            pipeline=pipeline,
            mpmd_mesh=mpmd_mesh,
            sample_state=pipeline_state,
            sample_batches=stage_batches,
        )
        explicit_step = grug_train._LocalLoweredExplicitMpmdStep(explicit_step.lower(pipeline_state, stage_batches))
        next_state, metrics, _ = explicit_step(pipeline_state, stage_batches)
        jax.block_until_ready((next_state, metrics))

        explicit_gradients = captured_gradients(next_state.opt_state[local_stage_index])
        explicit_loss_local = np.zeros((), dtype=np.float32)
        if local_stage_index == 0:
            explicit_loss_local = np.asarray(jax.device_get(metrics["train/loss"]), dtype=np.float32)
        explicit_loss = multihost_utils.broadcast_one_to_all(
            explicit_loss_local,
            is_source=local_stage_index == 0,
        )
        direct_loss_host = np.asarray(jax.device_get(direct_loss), dtype=np.float32)
        report = build_stage_parity_report(
            stage_index=local_stage_index,
            explicit_loss=explicit_loss,
            direct_loss=direct_loss_host,
            explicit_gradients=explicit_gradients,
            direct_gradients=direct_stage_gradients[local_stage_index],
        )
        print(
            json.dumps(
                {
                    "stage_index": local_stage_index,
                    "precision": eager_parity.PrecisionMode.PRODUCTION_MIXED,
                    "moe_implementation": _model_config().moe_implementation,
                    "topology": {
                        "pipeline_stages": PIPELINE_STAGES,
                        "devices_per_stage": DEVICES_PER_STAGE,
                        "expert_axis_size": EXPERT_AXIS_SIZE,
                        "microbatches": MICROBATCHES,
                        "microbatch_size": MICROBATCH_SIZE,
                    },
                    "runtime": {
                        name: environment[name]
                        for name in (
                            "jax",
                            "jaxlib",
                            "jaxpp",
                            "jaxpp_revision",
                            "nvidia_nccl_cu13",
                            "xla_flags",
                        )
                    },
                    "report": report.as_dict(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        stage_results = multihost_utils.process_allgather(np.asarray(report.passed, dtype=np.int32))
        all_passed = bool(np.all(stage_results))
        multihost_utils.sync_global_devices("explicit_mpmd_std1f1b_ragged_parity_complete")
        raise SystemExit(0 if all_passed else 1)
    finally:
        jax.distributed.shutdown()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-port", type=int, default=COORDINATOR_PORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_run_worker,
            args=(
                process_id,
                args.coordinator_port,
                list(
                    range(
                        process_id * DEVICES_PER_STAGE,
                        (process_id + 1) * DEVICES_PER_STAGE,
                    )
                ),
            ),
            name=f"jaxpp-explicit-ragged-parity-rank-{process_id}",
        )
        for process_id in range(PIPELINE_STAGES)
    ]
    for process in processes:
        process.start()
    return eager_parity._monitor_workers(processes)


if __name__ == "__main__":
    raise SystemExit(main())
