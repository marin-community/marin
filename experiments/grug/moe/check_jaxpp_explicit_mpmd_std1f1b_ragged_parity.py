# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare direct and explicit-MPMD std-1F1B Grug MoE loss and gradients.

This is an authoritative production-mixed-precision check for the
device-initiated ``ragged_all_to_all`` path. The production gate reserves four
H100x8 Iris tasks so every rank has an isolated physical node, then selects one
working two-GPU pair per rank for its ``expert=2`` mesh. Each task attempts the
complete direct reference in a bounded single-process child before initializing
the distributed JAX runtime in a second fresh process with that pair visible.
Once all ranks have initialized, one successful child result is shared through
JaxPP's host-side distributed client. The single-host fallback self-spawns four
JAX processes on one H100x8 host and serializes only their direct-reference
executions.

After installing the pinned JAX/JaxPP runtime with ``jaxpp_setup_scripts()``
from ``train.py``, run the command below in every task of a four-replica
H100x8 Iris job:

    JAXPP_SOURCE=/tmp/jaxpp \
    GRUG_JAXPP_PRECOMPILE_LOCAL=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.35 \
    XLA_FLAGS="--xla_gpu_autotune_level=0 \
      --xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true \
      --xla_gpu_ragged_all_to_all_mode=symmetric \
      --xla_enable_nccl_symmetric_buffers_for_collectives=RaggedAllToAll \
      --xla_gpu_nccl_termination_timeout_seconds=120" \
    .venv/bin/python -u \
      experiments/grug/moe/check_jaxpp_explicit_mpmd_std1f1b_ragged_parity.py

The direct reference is the mean of the independently differentiated
microbatch losses. The explicit arm executes ``train.py``'s production
``std_1f1b`` step with an observation-only Optax transform that leaves
parameters unchanged and stores each averaged stage-local gradient directly in
optimizer state. Loss and every gradient leaf must have relative-L2 at most
0.002. ``GRUG_JAXPP_PRECOMPILE_LOCAL=1`` activates a research-only control
from ``jaxpp_jax_0_11_inline.patch``: every rank precompiles its exact local
task call-jaxprs, allocates its bufferized receive prologue, and enters a host
barrier before any transfer equation is evaluated.
"""

from __future__ import annotations

import argparse
import faulthandler
import functools
import importlib
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from iris.cluster.client.job_info import get_job_info
from iris.runtime.jax_init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe import check_jaxpp_eager_1f1b_parity as eager_parity
from experiments.grug.moe import train as grug_train
from experiments.grug.moe.model import GrugModelConfig

PIPELINE_STAGES = 4
MICROBATCHES = 1
DEVICES_PER_STAGE = 2
EXPERT_AXIS_SIZE = 2
MICROBATCH_SIZE = 2
SEQUENCE_LENGTH = 8
COORDINATOR_PORT = 5847
_BATCH_AXES = ("replica_dcn", "data", "expert")
_DEVICE_RAGGED_FLAG = "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true"
_DIRECT_REFERENCE_DEVICE_PAIRS = ("0,1", "2,3", "4,5", "6,7")
_PRECOMPILE_LOCAL_ENV = "GRUG_JAXPP_PRECOMPILE_LOCAL"


@dataclass(frozen=True)
class _AttentionGateFp32Policy:
    base: Any

    def cast_to_param(self, tree):
        return self.base.cast_to_param(tree)

    def cast_to_compute(self, tree):
        casted = self.base.cast_to_compute(tree)
        return grug_train._restore_attention_gate_parameters(tree, casted)

    def cast_to_output(self, tree):
        return self.base.cast_to_output(tree)


_PARITY_MIXED_PRECISION = _AttentionGateFp32Policy(eager_parity._MIXED_PRECISION)


def _event(process_id: int, event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "process_id": process_id, **fields}, sort_keys=True), flush=True)


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


def local_precompile_enabled(environment: dict[str, str]) -> bool:
    """Return whether the research-only JaxPP local precompile control is enabled."""
    value = environment.get(_PRECOMPILE_LOCAL_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_PRECOMPILE_LOCAL_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


@dataclass(frozen=True)
class _PrecompiledLoweredMpmdFun:
    lowered: Any
    execution: Any

    @property
    def in_shardings(self):
        return self.lowered.in_shardings

    @property
    def _local_jaxpr(self):
        return self.lowered._local_jaxpr

    @property
    def mpmd_mesh(self):
        return self.lowered.mpmd_mesh

    @property
    def out_shape(self):
        return self.lowered.out_shape

    def eval_local(self, *local_args):
        return self.lowered.eval_local_precompiled(self.execution, *local_args)


def _precompile_local_explicit_step(
    explicit_step: grug_train._LocalLoweredExplicitMpmdStep,
    state: grug_train.GrugPipelineTrainState,
    batches: tuple[GrugLmExample, ...],
) -> tuple[grug_train._LocalLoweredExplicitMpmdStep, Any]:
    flat_args, args_tree = jax.tree_util.tree_flatten((state, batches))
    lowered = explicit_step.lowered
    in_tree = jax.tree_util.tree_structure(lowered.in_shardings)
    if args_tree != in_tree:
        raise ValueError("local precompile received an unexpected input tree")
    local_jaxpr = lowered._local_jaxpr
    execution = lowered.precompile_local(*(flat_args[idx] for idx in local_jaxpr.global_invar_indices))
    return (
        grug_train._LocalLoweredExplicitMpmdStep(_PrecompiledLoweredMpmdFun(lowered, execution)),
        execution,
    )


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


def block_local_parity_outputs(
    process_id: int,
    local_stage_index: int,
    opt_state,
    metrics: dict[str, Any],
) -> None:
    """Materialize only outputs owned by this MPMD rank."""
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(opt_state)
    for leaf_index, (path, leaf) in enumerate(leaves_with_paths):
        fields = {
            "stage_index": local_stage_index,
            "leaf_index": leaf_index,
            "path": jax.tree_util.keystr(path),
        }
        _event(process_id, "explicit_gradient_ready_start", **fields)
        jax.block_until_ready(leaf)
        _event(process_id, "explicit_gradient_ready_complete", **fields)

    if local_stage_index == 0:
        _event(process_id, "explicit_loss_ready_start", stage_index=local_stage_index)
        jax.block_until_ready(metrics["train/loss"])
        _event(process_id, "explicit_loss_ready_complete", stage_index=local_stage_index)


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


def _batch_for_mesh(mesh: jax.sharding.Mesh) -> GrugLmExample:
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
    stage_batch_groups = tuple(
        tuple(
            grug_train._put_batch_on_stage(mpmd_mesh, mpmd_index, stage_batch)
            for mpmd_index, stage_batch in zip(stage_mpmd_indices, microbatch_batches, strict=True)
        )
        for microbatch_batches in host_stage_batches
    )
    if MICROBATCHES == 1:
        return stage_batch_groups[0]
    return stage_batch_groups


def _authoritative_environment() -> dict[str, Any]:
    reproducer = importlib.import_module("experiments.grug.moe.repro_jaxpp_jax011_ragged_all_to_all")
    environment = reproducer.check_environment("jaxpp-four-stage-ragged")
    validate_device_ragged_flags(environment["xla_flags"])
    return environment


def _wait_at_jaxpp_barrier(name: str) -> None:
    dime2 = grug_train.jaxpp_dime2
    if dime2 is None:
        raise ModuleNotFoundError("jaxpp.dime2 is required for the ragged parity barrier")
    client = dime2.get_distributed_client()
    client.wait_at_barrier(name, dime2.env_vars.jaxpp_client_timeout.value)


def _share_standalone_direct_result(
    process_id: int,
    local_result: tuple[Any, Any] | None,
) -> tuple[Any, Any]:
    dime2 = grug_train.jaxpp_dime2
    if dime2 is None:
        raise ModuleNotFoundError("jaxpp.dime2 is required to share the direct parity reference")

    client = dime2.get_distributed_client()
    timeout = dime2.env_vars.jaxpp_client_timeout.value
    status_key = f"grug-ragged-parity-direct-status-{process_id}"
    result_key = f"grug-ragged-parity-direct-result-{process_id}"
    client.key_value_set_bytes(status_key, b"1" if local_result is not None else b"0")
    if local_result is not None:
        client.key_value_set_bytes(result_key, pickle.dumps(local_result))
    client.wait_at_barrier("grug_ragged_parity_direct_results_published", timeout)

    source_process_id = next(
        (
            candidate_process_id
            for candidate_process_id in range(PIPELINE_STAGES)
            if client.blocking_key_value_get_bytes(
                f"grug-ragged-parity-direct-status-{candidate_process_id}",
                timeout,
            )
            == b"1"
        ),
        None,
    )
    if source_process_id is None:
        raise RuntimeError("no pipeline rank produced a standalone direct reference")

    shared_result = pickle.loads(
        client.blocking_key_value_get_bytes(
            f"grug-ragged-parity-direct-result-{source_process_id}",
            timeout,
        )
    )
    _event(
        process_id,
        "standalone_direct_reference_shared",
        source_process_id=source_process_id,
    )
    return shared_result


def _standalone_direct_reference(process_id: int, device_pair: str, output_path: Path) -> None:
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        _event(process_id, "standalone_direct_reference_start", device_pair=device_pair)
        pipeline = _pipeline_config()
        optimizer = gradient_capture_optimizer()
        mesh = grug_train._compact_or_pipeline_grug_mesh(
            expert_axis_size=EXPERT_AXIS_SIZE,
            replica_axis_size=1,
            pipeline=None,
        )
        with jax.set_mesh(mesh):
            direct_state = grug_train.initial_state(
                _model_config(),
                optimizer=optimizer,
                mp=_PARITY_MIXED_PRECISION,
                key=jax.random.PRNGKey(0),
                ema_beta=None,
            )
            direct_batch = _batch_for_mesh(mesh)
            direct_step = jax.jit(
                functools.partial(
                    eager_parity._direct_microbatch_mean,
                    precision=eager_parity.PrecisionMode.PRODUCTION_MIXED,
                    mixed_precision_policy=_PARITY_MIXED_PRECISION,
                )
            )
            direct_loss, direct_gradients = direct_step(direct_state.params, direct_batch)
            direct_stage_gradients = direct_gradients.split_for_pipeline(
                pipeline.stages,
                pipeline.stage_layer_counts,
            )
            host_result = jax.device_get((direct_loss, direct_stage_gradients))
        with output_path.open("wb") as output:
            pickle.dump(host_result, output)
        _event(process_id, "standalone_direct_reference_complete", device_pair=device_pair)
    finally:
        faulthandler.cancel_dump_traceback_later()


def _standalone_direct_reference_attempt(process_id: int, device_pair: str):
    descriptor, raw_output_path = tempfile.mkstemp(prefix=f"jaxpp-ragged-parity-direct-{process_id}-", suffix=".pkl")
    os.close(descriptor)
    output_path = Path(raw_output_path)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device_pair
    command = (
        sys.executable,
        "-m",
        "experiments.grug.moe.check_jaxpp_explicit_mpmd_std1f1b_ragged_parity",
        "--standalone-direct-output",
        str(output_path),
        "--standalone-direct-stage",
        str(process_id),
        "--standalone-direct-device-pair",
        device_pair,
    )
    try:
        try:
            completed = subprocess.run(command, env=environment, timeout=90, check=False)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"standalone direct reference on GPUs {device_pair} exceeded 90 seconds") from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"standalone direct reference on GPUs {device_pair} exited with status {completed.returncode}"
            )
        with output_path.open("rb") as output:
            return pickle.load(output)
    finally:
        output_path.unlink(missing_ok=True)


def _standalone_direct_reference_result(process_id: int):
    failures = []
    for device_pair in _DIRECT_REFERENCE_DEVICE_PAIRS:
        try:
            result = _standalone_direct_reference_attempt(process_id, device_pair)
            return result, device_pair
        except (RuntimeError, TimeoutError) as error:
            failures.append(f"{device_pair}: {error}")
            _event(
                process_id,
                "standalone_direct_reference_retry",
                device_pair=device_pair,
                error=str(error),
            )
    _event(
        process_id,
        "standalone_direct_reference_unavailable",
        error="; ".join(failures),
    )
    return None, _DIRECT_REFERENCE_DEVICE_PAIRS[0]


def _run_distributed_worker_subprocess(
    standalone_direct_result: tuple[Any, Any] | None,
    device_pair: str,
) -> int:
    descriptor, raw_input_path = tempfile.mkstemp(prefix="jaxpp-ragged-parity-direct-result-", suffix=".pkl")
    os.close(descriptor)
    input_path = Path(raw_input_path)
    with input_path.open("wb") as output:
        pickle.dump(standalone_direct_result, output)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device_pair
    command = (
        sys.executable,
        "-m",
        "experiments.grug.moe.check_jaxpp_explicit_mpmd_std1f1b_ragged_parity",
        "--distributed-worker-direct-input",
        str(input_path),
    )
    try:
        try:
            completed = subprocess.run(command, env=environment, timeout=1200, check=False)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("distributed parity worker exceeded 1200 seconds") from error
        return completed.returncode
    finally:
        input_path.unlink(missing_ok=True)


def _run_initialized_worker(
    process_id: int,
    direct_reference_barrier: Any | None,
    standalone_direct_result: tuple[Any, Any] | None = None,
    share_standalone_result: bool = False,
) -> None:
    faulthandler.dump_traceback_later(120, repeat=True)
    completed = False
    exit_code = 0
    try:
        validate_authoritative_topology(
            process_count=jax.process_count(),
            local_device_count=jax.local_device_count(),
            device_count=jax.device_count(),
        )
        environment = _authoritative_environment()
        if share_standalone_result:
            standalone_direct_result = _share_standalone_direct_result(
                process_id,
                standalone_direct_result,
            )
        pipeline = _pipeline_config()
        mesh = grug_train._compact_or_pipeline_grug_mesh(
            expert_axis_size=EXPERT_AXIS_SIZE,
            replica_axis_size=1,
            pipeline=pipeline,
        )
        explicit_mpmd = grug_train._require_jaxpp_explicit_mpmd()
        mpmd_mesh = explicit_mpmd.MpmdMesh(mesh, pipeline.stage_axis_name)
        local_stage_index = mpmd_mesh.my_mpmd_axis_index
        stage_mesh = mpmd_mesh.my_mpmd_group_mesh
        optimizer = gradient_capture_optimizer()

        with jax.set_mesh(mesh):
            initial_state = grug_train.initial_state(
                _model_config(),
                optimizer=optimizer,
                mp=_PARITY_MIXED_PRECISION,
                key=jax.random.PRNGKey(0),
                ema_beta=None,
            )
            batch = _batch_for_mesh(mesh)
        pipeline_state = grug_train._split_state_for_explicit_mpmd(
            initial_state,
            pipeline=pipeline,
            optimizer=optimizer,
            mpmd_mesh=mpmd_mesh,
        )
        stage_batches = _stage_batches(mpmd_mesh, batch)

        if standalone_direct_result is not None:
            direct_loss, direct_stage_gradients = standalone_direct_result
            direct_stage_gradient = direct_stage_gradients[local_stage_index]
            _event(process_id, "standalone_direct_reference_loaded", stage_index=local_stage_index)
            grug_train._warm_jaxpp_device_ragged(
                mpmd_mesh,
                global_microbatch_tokens=MICROBATCH_SIZE * SEQUENCE_LENGTH,
                hidden_dim=_model_config().hidden_dim,
                top_k=_model_config().num_experts_per_token,
            )
            _event(process_id, "device_ragged_warmup_complete", stage_index=local_stage_index)
        else:
            if direct_reference_barrier is None:
                raise ValueError("a local direct-reference barrier is required without a standalone result")
            direct_result = None
            for active_process_id in range(PIPELINE_STAGES):
                direct_reference_barrier.wait()
                if process_id == active_process_id:
                    _event(process_id, "direct_reference_start", stage_index=local_stage_index)
                    with jax.set_mesh(stage_mesh):
                        direct_state = grug_train.initial_state(
                            _model_config(),
                            optimizer=optimizer,
                            mp=_PARITY_MIXED_PRECISION,
                            key=jax.random.PRNGKey(0),
                            ema_beta=None,
                        )
                        direct_batch = _batch_for_mesh(stage_mesh)
                        direct_step = jax.jit(
                            functools.partial(
                                eager_parity._direct_microbatch_mean,
                                precision=eager_parity.PrecisionMode.PRODUCTION_MIXED,
                                mixed_precision_policy=_PARITY_MIXED_PRECISION,
                            )
                        )
                        direct_result = direct_step(direct_state.params, direct_batch)
                        jax.block_until_ready(direct_result)
                    _event(process_id, "direct_reference_complete", stage_index=local_stage_index)
                direct_reference_barrier.wait()

            if direct_result is None:
                raise RuntimeError(f"process {process_id} did not execute its direct reference")
            direct_loss, direct_gradients = direct_result
            direct_stage_gradient = direct_gradients.split_for_pipeline(
                pipeline.stages,
                pipeline.stage_layer_counts,
            )[local_stage_index]

        _wait_at_jaxpp_barrier("grug_ragged_parity_ragged_ready")
        _event(process_id, "device_ragged_barrier_complete", stage_index=local_stage_index)
        _event(process_id, "dime_prewarm_start", stage_index=local_stage_index)
        grug_train._prewarm_jaxpp_dime(mpmd_mesh, "all")
        _event(process_id, "dime_prewarm_complete", stage_index=local_stage_index)

        _event(process_id, "explicit_lower_start", stage_index=local_stage_index)
        explicit_step = grug_train._make_explicit_mpmd_train_step(
            optimizer,
            _PARITY_MIXED_PRECISION,
            z_loss_weight=0.0,
            pipeline=pipeline,
            mpmd_mesh=mpmd_mesh,
            sample_state=pipeline_state,
            sample_batches=stage_batches,
        )
        explicit_step = grug_train._LocalLoweredExplicitMpmdStep(explicit_step.lower(pipeline_state, stage_batches))
        _event(process_id, "explicit_lower_complete", stage_index=local_stage_index)
        if local_precompile_enabled(os.environ):
            _event(process_id, "explicit_precompile_start", stage_index=local_stage_index)
            explicit_step, precompiled_execution = _precompile_local_explicit_step(
                explicit_step,
                pipeline_state,
                stage_batches,
            )
            _event(
                process_id,
                "explicit_precompile_complete",
                stage_index=local_stage_index,
                task_count=precompiled_execution.task_count,
                recv_buffer_count=len(precompiled_execution.recv_buffers),
            )
            _wait_at_jaxpp_barrier("grug_ragged_parity_explicit_precompiled")
            _event(process_id, "explicit_precompile_barrier_complete", stage_index=local_stage_index)
        _event(process_id, "explicit_execute_start", stage_index=local_stage_index)
        next_state, metrics, _ = explicit_step(pipeline_state, stage_batches)
        block_local_parity_outputs(
            process_id,
            local_stage_index,
            next_state.opt_state[local_stage_index],
            metrics,
        )
        _event(process_id, "explicit_execute_complete", stage_index=local_stage_index)

        explicit_gradients = captured_gradients(next_state.opt_state[local_stage_index])
        explicit_loss_local = np.zeros((), dtype=np.float32)
        if local_stage_index == 0:
            explicit_loss_local = np.asarray(jax.device_get(metrics["train/loss"]), dtype=np.float32)
        explicit_loss = multihost_utils.broadcast_one_to_all(
            explicit_loss_local,
            is_source=local_stage_index == 0,
        )
        _event(process_id, "loss_broadcast_complete", stage_index=local_stage_index)
        direct_loss_host = np.asarray(jax.device_get(direct_loss), dtype=np.float32)
        report = build_stage_parity_report(
            stage_index=local_stage_index,
            explicit_loss=explicit_loss,
            direct_loss=direct_loss_host,
            explicit_gradients=explicit_gradients,
            direct_gradients=direct_stage_gradient,
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
                            "nccl_runtime_version",
                            "nccl_mapped_libraries",
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
        completed = True
        exit_code = 0 if all_passed else 1
    except BaseException as error:
        _event(
            process_id,
            "worker_error",
            error_type=type(error).__name__,
            error=str(error),
        )
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if completed:
            _event(process_id, "distributed_shutdown_start")
            jax.distributed.shutdown()
            _event(process_id, "distributed_shutdown_complete")
        else:
            _event(process_id, "distributed_shutdown_skipped_after_error")
    raise SystemExit(exit_code)


def _run_self_spawned_worker(
    process_id: int,
    coordinator_port: int,
    local_device_ids: list[int],
    direct_reference_barrier: Any,
) -> None:
    _event(process_id, "distributed_initialize_start", local_device_ids=local_device_ids)
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{coordinator_port}",
        num_processes=PIPELINE_STAGES,
        process_id=process_id,
        local_device_ids=local_device_ids,
        cluster_detection_method="deactivate",
    )
    _event(process_id, "distributed_initialize_complete")
    _run_initialized_worker(process_id, direct_reference_barrier)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-port", type=int, default=COORDINATOR_PORT)
    parser.add_argument("--standalone-direct-output", type=Path)
    parser.add_argument("--standalone-direct-stage", type=int)
    parser.add_argument("--standalone-direct-device-pair")
    parser.add_argument("--distributed-worker-direct-input", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.standalone_direct_output is not None:
        if args.standalone_direct_stage is None or args.standalone_direct_device_pair is None:
            raise ValueError(
                "--standalone-direct-stage and --standalone-direct-device-pair are required with "
                "--standalone-direct-output"
            )
        _standalone_direct_reference(
            args.standalone_direct_stage,
            args.standalone_direct_device_pair,
            args.standalone_direct_output,
        )
        return 0

    if args.distributed_worker_direct_input is not None:
        with args.distributed_worker_direct_input.open("rb") as direct_input:
            standalone_direct_result = pickle.load(direct_input)
        initialize_jax()
        _event(jax.process_index(), "distributed_initialize_complete")
        _run_initialized_worker(
            jax.process_index(),
            None,
            standalone_direct_result,
            share_standalone_result=True,
        )
        return 0

    job_info = get_job_info()
    if job_info is not None and job_info.num_tasks > 1:
        standalone_direct_result, device_pair = _standalone_direct_reference_result(job_info.task_index)
        return _run_distributed_worker_subprocess(standalone_direct_result, device_pair)

    context = mp.get_context("spawn")
    direct_reference_barrier = context.Barrier(PIPELINE_STAGES)
    processes = [
        context.Process(
            target=_run_self_spawned_worker,
            args=(
                process_id,
                args.coordinator_port,
                list(
                    range(
                        process_id * DEVICES_PER_STAGE,
                        (process_id + 1) * DEVICES_PER_STAGE,
                    )
                ),
                direct_reference_barrier,
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
