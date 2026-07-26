# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-block JaxPP gate for the exact paired MoE component boundary.

The authoritative GPU run self-spawns two JAX processes on one H100x8 node.
Each process owns four devices, while one JaxPP MPMD task spans the complete
EP8 mesh. The task keeps attention, shared/dense work, and their VJPs separate
for the two original microbatches; only the two exact-ring ``MoEMLP`` calls
share one task-local reverse pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import hashlib
import json
import multiprocessing
import os
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.experimental import multihost_utils
from jax.extend import core as jax_core
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask

from experiments.grug.moe import train as grug_train
from experiments.grug.moe.check_jaxpp_eager_1f1b_parity import (
    DEFAULT_TOLERANCE,
    build_parity_report,
    build_value_parity,
)
from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.model import (
    Block,
    GrugModelConfig,
    _run_block_with_remat,
    paired_moe_component_forward,
)

SOURCE_LINEAGE = "0adaf6156d"
JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
HIDDEN_DIM = 2560
NUM_EXPERTS = 64
TOP_K = 4
SEQUENCE_LENGTH = 4096
GLOBAL_MICROBATCH_SIZE = 32
LOCAL_MICROBATCH_SIZE = 4
EXPERT_AXIS_SIZE = 8
PROCESS_COUNT = 2
DEVICES_PER_PROCESS = 4
ROUTER_Z_LOSS_SCALE = 0.1
PARAMETER_LEAF_COUNT = 19
WORKER_TIMEOUT = 3300
WORKER_SHUTDOWN_TIMEOUT = 10
STACK_INTERVAL = 300
COORDINATOR_PORT = 5897
_BATCH_AXES = ("replica_dcn", "data", "expert")
_MESH_AXES = ("pipeline", "replica_dcn", "data", "expert", "model")
_MIXED_PRECISION = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
_REQUIRED_KERNEL_ENVIRONMENT = {
    "RAGGED_DOT_IMPL": "triton",
    "HALIAX_RAGGED_DOT_TRITON_BLOCK_K": "32",
    "HALIAX_RAGGED_DOT_TRITON_NUM_WARPS": "8",
}


def event(process_id: int | None, name: str, **fields: Any) -> None:
    """Emit one machine-readable gate event."""
    print(
        json.dumps(
            {"event": name, "process_id": process_id, "time": time.time(), **fields},
            default=str,
            sort_keys=True,
        ),
        flush=True,
    )


def target_model_config() -> GrugModelConfig:
    """Return the fixed one-block target model configuration."""
    return dataclasses.replace(
        MoeHeuristic().build_model_config(HIDDEN_DIM, seq_len=SEQUENCE_LENGTH),
        num_layers=1,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        router_z_loss_coef=ROUTER_Z_LOSS_SCALE,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
        remat_mode="save_moe",
    )


def validate_kernel_environment(environment: dict[str, str]) -> None:
    """Require the exact expert kernel used by the throughput treatment."""
    mismatches = {
        name: {"expected": expected, "actual": environment.get(name)}
        for name, expected in _REQUIRED_KERNEL_ENVIRONMENT.items()
        if environment.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"component parity requires target kernel environment: {mismatches}")


def validate_topology() -> None:
    """Require the one-node two-process H100x8 topology."""
    expected_devices = PROCESS_COUNT * DEVICES_PER_PROCESS
    if (
        jax.process_count() != PROCESS_COUNT
        or jax.local_device_count() != DEVICES_PER_PROCESS
        or jax.device_count() != expected_devices
    ):
        raise ValueError(
            "component parity requires two JAX processes with four devices each; "
            f"found process_count={jax.process_count()}, local_device_count={jax.local_device_count()}, "
            f"device_count={jax.device_count()}"
        )
    invalid = [
        {"platform": device.platform, "device_kind": device.device_kind}
        for device in jax.devices()
        if device.platform != "gpu" or "H100" not in device.device_kind
    ]
    if invalid:
        raise ValueError(f"component parity requires eight H100 GPUs: {invalid}")


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _initialize_array(
    function,
    *,
    sharding: NamedSharding,
):
    return jax.jit(function, out_shardings=sharding)()


def _problem(stage_mesh: Mesh):
    hidden_sharding = NamedSharding(stage_mesh, P(_BATCH_AXES, None, None))
    qb_sharding = NamedSharding(stage_mesh, P(None))
    with jax.set_mesh(stage_mesh):
        block = Block.init(target_model_config(), key=jax.random.PRNGKey(0))
        qb_beta = jax.device_put(np.linspace(-0.05, 0.05, NUM_EXPERTS, dtype=np.float32), qb_sharding)

        def random_hidden(seed: int):
            return _initialize_array(
                lambda: jax.random.normal(
                    jax.random.PRNGKey(seed),
                    (GLOBAL_MICROBATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_DIM),
                    dtype=jnp.bfloat16,
                )
                * jnp.asarray(0.02, dtype=jnp.bfloat16),
                sharding=hidden_sharding,
            )

        hiddens = (random_hidden(1), random_hidden(2))
        cotangents = (random_hidden(3), random_hidden(4))
    return block, qb_beta, hiddens, cotangents


def _project_output(output: jax.Array, cotangent: jax.Array) -> jax.Array:
    return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))


def ordered_block_value_and_grads(
    params: Block,
    qb_beta: jax.Array,
    hiddens: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
):
    """Differentiate the two complete block calls independently and in order."""
    losses = []
    outputs = []
    router_stats = []
    parameter_gradients = []
    input_gradients = []
    mask = AttentionMask.causal()
    for hidden, cotangent in zip(hiddens, cotangents, strict=True):

        def projected_block(master_params, current_hidden, cotangent=cotangent):
            compute_block = grug_train._compute_block(master_params, qb_beta, _MIXED_PRECISION)
            output, stats = _run_block_with_remat(
                compute_block,
                current_hidden,
                mask,
                use_pko=False,
                disable_rope=False,
                remat_mode="save_moe",
                effectful_moe=False,
            )
            loss = _project_output(output, cotangent) + ROUTER_Z_LOSS_SCALE * stats["router_z_loss"]
            return loss, (output, stats)

        (loss, (output, stats)), (parameter_gradient, input_gradient) = jax.value_and_grad(
            projected_block,
            argnums=(0, 1),
            has_aux=True,
        )(params, hidden)
        losses.append(loss)
        outputs.append(output)
        router_stats.append(stats)
        parameter_gradients.append(parameter_gradient)
        input_gradients.append(input_gradient)
    return (
        tuple(losses),
        tuple(outputs),
        tuple(router_stats),
        grug_train._sum_microbatch_group(tuple(parameter_gradients)),
        tuple(input_gradients),
    )


def paired_block_value_and_grads(
    params: Block,
    qb_beta: jax.Array,
    hiddens: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
):
    """Run the production component composition with only MoEMLP joined."""
    masks = (AttentionMask.causal(), AttentionMask.causal())
    return grug_train.paired_compute_block_value_and_grads(
        params,
        qb_beta,
        hiddens,
        masks,
        cotangents,
        _MIXED_PRECISION,
        use_pko=False,
        disable_rope=False,
        remat_mode="save_moe",
        router_z_loss_scale=ROUTER_Z_LOSS_SCALE,
    )


def _compile_and_run(process_id: int, name: str, function, arguments):
    lower_start = time.monotonic()
    event(process_id, f"{name}_lower_start")
    lowered = function.lower(*arguments)
    event(process_id, f"{name}_lower_done", elapsed_seconds=time.monotonic() - lower_start)
    compile_start = time.monotonic()
    event(process_id, f"{name}_compile_start")
    compiled = lowered.compile()
    event(process_id, f"{name}_compile_done", elapsed_seconds=time.monotonic() - compile_start)
    execute_start = time.monotonic()
    event(process_id, f"{name}_execute_start")
    result = compiled(*arguments)
    _block_until_ready(result)
    event(process_id, f"{name}_execute_done", elapsed_seconds=time.monotonic() - execute_start)
    return result


def _walk_jaxpr_equations(closed_jaxpr: jax_core.ClosedJaxpr):
    equations = []

    def walk_value(value) -> None:
        if isinstance(value, jax_core.ClosedJaxpr):
            walk_jaxpr(value.jaxpr)
        elif isinstance(value, jax_core.Jaxpr):
            walk_jaxpr(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk_value(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk_value(item)

    def walk_jaxpr(jaxpr: jax_core.Jaxpr) -> None:
        for equation in jaxpr.eqns:
            equations.append(equation)
            walk_value(equation.params)

    walk_jaxpr(closed_jaxpr.jaxpr)
    return tuple(equations)


def component_structure(closed_jaxpr: jax_core.ClosedJaxpr) -> dict[str, Any]:
    """Summarize the joined MoE boundary from a lowered task JAXPR."""
    equations = _walk_jaxpr_equations(closed_jaxpr)
    names = tuple(str(equation.source_info.name_stack) for equation in equations)
    ring_bodies = [
        equation
        for equation in equations
        if equation.primitive.name == "shard_map"
        and str(equation.source_info.name_stack).endswith("_paired_moe_calls/MoEMLP/MoEExpertMlp/moe_mlp")
    ]
    router_calls = [name for name in names if name.endswith("_paired_moe_calls/MoEMLP/td,de->te")]
    boundary_names = [name for name in names if "_paired_moe_calls" in name]
    attention_inside_boundary = [name for name in boundary_names if "Attention" in name or "_BlockAttentionView" in name]
    return {
        "ring_body_count": len(ring_bodies),
        "router_call_count": len(router_calls),
        "attention_inside_joined_moe_count": len(attention_inside_boundary),
        "joined_moe_equation_count": len(boundary_names),
    }


def validate_pure_moe_structure(structure: dict[str, Any]) -> None:
    """Require the untransposed paired MoE boundary to contain two calls."""
    expected = {
        "ring_body_count": 2,
        "router_call_count": 2,
        "attention_inside_joined_moe_count": 0,
    }
    actual = {name: structure[name] for name in expected}
    if actual != expected:
        raise AssertionError(f"unexpected pure paired MoE JAXPR structure: {structure}")


def validate_full_task_structure(structure: dict[str, Any]) -> None:
    """Reject attention captured inside the joined MoE boundary after VJP expansion."""
    if structure["attention_inside_joined_moe_count"] != 0:
        raise AssertionError(f"attention entered the joined MoE JAXPR boundary: {structure}")


def _target_pure_moe_structure(
    params: Block,
    qb_beta: jax.Array,
    hiddens: tuple[jax.Array, jax.Array],
) -> dict[str, Any]:
    compute_block = grug_train._compute_block(params, qb_beta, _MIXED_PRECISION)
    mlp_inputs = tuple(compute_block.mlp_gated_norm(compute_block.rms_mlp(hidden)) for hidden in hiddens)
    closed_jaxpr, _, _ = eqx.filter_make_jaxpr(
        lambda mlp, inputs: paired_moe_component_forward(mlp, inputs, remat_mode="save_moe")
    )(compute_block.mlp, mlp_inputs)
    return component_structure(closed_jaxpr)


def _parameter_tree_signature(params: Block) -> dict[str, Any]:
    leaves_with_paths, _ = jax.tree.flatten_with_path(params)
    records = [
        {
            "path": jax.tree_util.keystr(path),
            "shape": tuple(int(dimension) for dimension in leaf.shape),
            "dtype": str(leaf.dtype),
        }
        for path, leaf in leaves_with_paths
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "leaf_count": len(records),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _sum_losses(losses: tuple[jax.Array, jax.Array]) -> jax.Array:
    return losses[0] + losses[1]


def _exact_routing_mismatches(
    actual_stats: tuple[dict[str, jax.Array], dict[str, jax.Array]],
    reference_stats: tuple[dict[str, jax.Array], dict[str, jax.Array]],
) -> tuple[int, int]:
    mismatches = tuple(
        jnp.sum(actual["routing_counts"] != reference["routing_counts"], dtype=jnp.int32)
        for actual, reference in zip(actual_stats, reference_stats, strict=True)
    )
    return tuple(int(value) for value in jax.device_get(mismatches))


def _report_parity(process_id: int, actual, reference, parameter_signature: dict[str, Any]) -> bool:
    per_microbatch_loss = tuple(
        build_value_parity(actual_loss, reference_loss, tolerance=DEFAULT_TOLERANCE)
        for actual_loss, reference_loss in zip(actual[0], reference[0], strict=True)
    )
    value_report = build_parity_report(
        automatic_loss=_sum_losses(actual[0]),
        direct_loss=_sum_losses(reference[0]),
        automatic_gradients={"outputs": actual[1]},
        direct_gradients={"outputs": reference[1]},
        tolerance=DEFAULT_TOLERANCE,
        gradient_root="values",
    )
    gradient_report = build_parity_report(
        automatic_loss=_sum_losses(actual[0]),
        direct_loss=_sum_losses(reference[0]),
        automatic_gradients={"parameters": actual[3], "inputs": actual[4]},
        direct_gradients={"parameters": reference[3], "inputs": reference[4]},
        tolerance=DEFAULT_TOLERANCE,
        gradient_root="gradients",
    )
    router_report = build_parity_report(
        automatic_loss=_sum_losses(actual[0]),
        direct_loss=_sum_losses(reference[0]),
        automatic_gradients=actual[2],
        direct_gradients=reference[2],
        tolerance=DEFAULT_TOLERANCE,
        gradient_root="router_metrics",
    )
    routing_mismatches = _exact_routing_mismatches(actual[2], reference[2])
    gradient_leaf_count = len(gradient_report.gradients)
    parameter_gradient_leaf_count = len(jax.tree.leaves(actual[3]))
    input_gradient_leaf_count = len(jax.tree.leaves(actual[4]))
    structure_preserved = (
        jax.tree.structure(actual[3]) == jax.tree.structure(reference[3])
        and parameter_gradient_leaf_count == parameter_signature["leaf_count"] == PARAMETER_LEAF_COUNT
    )
    passed = (
        all(loss.passed for loss in per_microbatch_loss)
        and value_report.passed
        and gradient_report.passed
        and router_report.passed
        and not any(routing_mismatches)
        and structure_preserved
    )
    if process_id == 0:
        event(
            process_id,
            "component_parity_report",
            source_lineage=SOURCE_LINEAGE,
            tolerance=DEFAULT_TOLERANCE,
            per_microbatch_loss=[dataclasses.asdict(loss) for loss in per_microbatch_loss],
            value_report=value_report.as_dict(),
            gradient_report=gradient_report.as_dict(),
            router_report=router_report.as_dict(),
            routing_count_mismatches=routing_mismatches,
            gradient_leaf_count=gradient_leaf_count,
            parameter_gradient_leaf_count=parameter_gradient_leaf_count,
            input_gradient_leaf_count=input_gradient_leaf_count,
            parameter_tree=parameter_signature,
            checkpoint_parameter_structure_unchanged=structure_preserved,
            passed=passed,
        )
    return passed


def _reconstruct_local_outputs(lowered, local_outputs):
    flat_shapes, output_tree = jax.tree.flatten(lowered.out_shape)
    local_jaxpr = lowered._local_jaxpr
    outputs_by_index = dict(zip(local_jaxpr.global_outvar_indices, local_outputs, strict=True))
    if set(outputs_by_index) != set(range(len(flat_shapes))):
        raise ValueError(
            "single-stage gate expected every output on the local MPMD group; "
            f"found output indices={sorted(outputs_by_index)}, total={len(flat_shapes)}"
        )
    return jax.tree.unflatten(output_tree, [outputs_by_index[index] for index in range(len(flat_shapes))])


def _start_watchdog(process_id: int) -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(STACK_INTERVAL, repeat=True)

    def hard_stop() -> None:
        event(process_id, "worker_timeout", timeout_seconds=WORKER_TIMEOUT)
        os._exit(124)

    timer = threading.Timer(WORKER_TIMEOUT, hard_stop)
    timer.daemon = True
    timer.start()


def _run_worker(process_id: int, coordinator_address: str, local_device_ids: list[int]) -> None:
    event(process_id, "distributed_initialize_start", local_device_ids=local_device_ids)
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=PROCESS_COUNT,
        process_id=process_id,
        local_device_ids=local_device_ids,
        cluster_detection_method="deactivate",
    )
    _start_watchdog(process_id)
    completed = False
    try:
        validate_kernel_environment(dict(os.environ))
        validate_topology()
        devices = np.asarray(jax.devices(), dtype=object).reshape((1, 1, 1, EXPERT_AXIS_SIZE, 1))
        global_mesh = Mesh(
            devices,
            _MESH_AXES,
            axis_types=tuple(AxisType.Explicit for _ in _MESH_AXES),
        )
        mpmd = grug_train._require_jaxpp_explicit_mpmd()
        mpmd_mesh = mpmd.MpmdMesh(global_mesh, "pipeline")
        stage_mesh = mpmd_mesh.unstack[0]
        event(
            process_id,
            "configuration",
            source_lineage=SOURCE_LINEAGE,
            jaxpp_revision=JAXPP_REVISION,
            hidden_dim=HIDDEN_DIM,
            experts=NUM_EXPERTS,
            top_k=TOP_K,
            sequence_length=SEQUENCE_LENGTH,
            global_microbatch_size=GLOBAL_MICROBATCH_SIZE,
            local_microbatch_size=LOCAL_MICROBATCH_SIZE,
            expert_axis_size=EXPERT_AXIS_SIZE,
            attention_implementation="gpu_fa4_cute",
            moe_implementation="ring",
            remat_mode="save_moe",
            compute_dtype="bfloat16",
        )
        event(process_id, "problem_init_start")
        with jax.set_mesh(stage_mesh):
            params, qb_beta, hiddens, cotangents = _problem(stage_mesh)
        parameter_signature = _parameter_tree_signature(params)
        if parameter_signature["leaf_count"] != PARAMETER_LEAF_COUNT:
            raise ValueError(f"unexpected Block parameter structure: {parameter_signature}")
        event(process_id, "problem_init_done", parameter_tree=parameter_signature)
        with jax.set_mesh(stage_mesh):
            pure_moe_structure = _target_pure_moe_structure(params, qb_beta, hiddens)
        validate_pure_moe_structure(pure_moe_structure)
        event(process_id, "pure_paired_moe_structure", **pure_moe_structure)

        arguments = (params, qb_beta, hiddens, cotangents)
        with jax.set_mesh(stage_mesh):
            ordered = jax.jit(ordered_block_value_and_grads)
            reference = _compile_and_run(process_id, "ordered", ordered, arguments)
        out_shardings = grug_train._tree_named_shardings_on_stage(mpmd_mesh, 0, reference)
        in_shardings = grug_train._tree_named_shardings_on_stage(mpmd_mesh, 0, arguments)

        @mpmd.mpmd(mpmd_mesh, in_shardings=in_shardings, infer_donation=False)
        def program(current_params, current_qb_beta, current_hiddens, current_cotangents):
            return mpmd.task(
                paired_block_value_and_grads,
                name="grug_group2_component_block_vjp",
                out_shardings=out_shardings,
            )(current_params, current_qb_beta, current_hiddens, current_cotangents)

        event(process_id, "jaxpp_lower_start")
        lower_start = time.monotonic()
        lowered = program.lower(*arguments)
        event(process_id, "jaxpp_lower_done", elapsed_seconds=time.monotonic() - lower_start)
        structure = component_structure(lowered._local_jaxpr.closed_jaxpr)
        validate_full_task_structure(structure)
        event(process_id, "jaxpp_full_task_component_structure", **structure)

        flat_args, argument_tree = jax.tree.flatten(arguments)
        if argument_tree != jax.tree.structure(lowered.in_shardings):
            raise ValueError("JaxPP lowered input tree differs from gate argument tree")
        local_args = [flat_args[index] for index in lowered._local_jaxpr.global_invar_indices]
        event(process_id, "jaxpp_precompile_start", local_input_count=len(local_args))
        compile_start = time.monotonic()
        execution = lowered.precompile_local(*local_args)
        event(
            process_id,
            "jaxpp_precompile_done",
            elapsed_seconds=time.monotonic() - compile_start,
            task_count=execution.task_count,
            recv_buffer_count=len(execution.recv_buffers),
        )
        if execution.task_count != 1:
            raise AssertionError(f"component gate expected one JaxPP task, got {execution.task_count}")
        multihost_utils.sync_global_devices("group2_component_precompiled")

        event(process_id, "jaxpp_execute_start")
        execute_start = time.monotonic()
        local_outputs = lowered.eval_local_precompiled(execution, *local_args)
        actual = _reconstruct_local_outputs(lowered, local_outputs)
        _block_until_ready(actual)
        event(process_id, "jaxpp_execute_done", elapsed_seconds=time.monotonic() - execute_start)
        passed = _report_parity(process_id, actual, reference, parameter_signature)
        multihost_utils.sync_global_devices("group2_component_parity_complete")
        if not passed:
            raise AssertionError("component parity exceeded the fixed per-leaf tolerance")
        completed = True
    except BaseException as error:
        event(
            process_id,
            "worker_error",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if completed:
            event(process_id, "distributed_shutdown_start")
            jax.distributed.shutdown()
            event(process_id, "distributed_shutdown_done")
        else:
            event(process_id, "distributed_shutdown_skipped_after_error")


def _monitor_workers(processes: list[multiprocessing.Process]) -> int:
    deadline = time.monotonic() + WORKER_TIMEOUT + 60
    try:
        while any(process.is_alive() for process in processes):
            failed = next(
                (process.exitcode for process in processes if process.exitcode not in (None, 0)),
                None,
            )
            if failed is not None:
                return failed
            if time.monotonic() >= deadline:
                raise TimeoutError(f"component parity exceeded {WORKER_TIMEOUT + 60} seconds")
            time.sleep(0.5)
        return next((process.exitcode for process in processes if process.exitcode), 0)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=WORKER_SHUTDOWN_TIMEOUT)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-port", type=int, default=COORDINATOR_PORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    context = multiprocessing.get_context("spawn")
    coordinator_address = f"127.0.0.1:{args.coordinator_port}"
    processes = [
        context.Process(
            target=_run_worker,
            args=(
                process_id,
                coordinator_address,
                list(
                    range(
                        process_id * DEVICES_PER_PROCESS,
                        (process_id + 1) * DEVICES_PER_PROCESS,
                    )
                ),
            ),
            name=f"jaxpp-group2-component-rank-{process_id}",
        )
        for process_id in range(PROCESS_COUNT)
    ]
    for process in processes:
        process.start()
    return _monitor_workers(processes)


if __name__ == "__main__":
    sys.exit(main())
