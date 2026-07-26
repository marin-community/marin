# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H100x8 parity gate for exact-ring grouped stage tasks.

The gate compares the interleaved MoE-boundary group-size-2 final-stage path
against two ordered single-microbatch calls. It uses the production d2560,
CuTe FA4, EP8 bulk-ring, BF16-compute, and Pallas-Triton expert kernels. The
global microbatch is 32 examples, matching the target L24 run, so each H100
owns four examples.
"""

import argparse
import dataclasses
import faulthandler
import json
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
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
    RematMode,
    Transformer,
    TransformerPipelineStage,
    _run_grouped_block_with_remat,
)

SOURCE_LINEAGE = "cb39dc4c7c"
HIDDEN_DIM = 2560
NUM_EXPERTS = 64
TOP_K = 4
SEQUENCE_LENGTH = 4096
VOCAB_SIZE = 8192
EXPERT_AXIS_SIZE = 8
GLOBAL_MICROBATCH_SIZE = 32
LOCAL_MICROBATCH_SIZE = 4
PIPELINE_STAGES = 2
TARGET_STAGE = 1
LOGSUMEXP_WEIGHT = None
STACK_INTERVAL = 600
_BATCH_AXES = ("replica_dcn", "data", "expert")
_PRODUCTION_MIXED_PRECISION = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
_REQUIRED_KERNEL_ENVIRONMENT = {
    "RAGGED_DOT_IMPL": "triton",
    "HALIAX_RAGGED_DOT_TRITON_BLOCK_K": "32",
    "HALIAX_RAGGED_DOT_TRITON_NUM_WARPS": "8",
}
_DIAGNOSTIC_MODES = ("full", "moe-pair")


def _event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def target_model_config() -> GrugModelConfig:
    """Return the fixed final-stage model shape used by the throughput runs."""
    return dataclasses.replace(
        MoeHeuristic().build_model_config(HIDDEN_DIM, seq_len=SEQUENCE_LENGTH),
        vocab_size=VOCAB_SIZE,
        num_layers=PIPELINE_STAGES,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        router_z_loss_coef=0.0,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
        loss_implementation="xla",
        remat_mode="save_moe",
    )


def validate_kernel_environment(environment: Mapping[str, str]) -> None:
    """Require the exact expert-kernel geometry used by the matched runs."""
    mismatches = {
        name: (expected, environment.get(name))
        for name, expected in _REQUIRED_KERNEL_ENVIRONMENT.items()
        if environment.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"group2 parity requires target kernel environment; mismatches={mismatches}")


def validate_h100x8_topology() -> None:
    """Require one JAX process controlling one complete H100x8 node."""
    devices = jax.devices()
    if jax.process_count() != 1 or jax.local_device_count() != EXPERT_AXIS_SIZE or len(devices) != EXPERT_AXIS_SIZE:
        raise ValueError(
            "group2 parity requires one JAX process with eight local/global devices; "
            f"found process_count={jax.process_count()}, local_device_count={jax.local_device_count()}, "
            f"device_count={len(devices)}"
        )
    invalid_devices = [
        {"platform": device.platform, "device_kind": device.device_kind}
        for device in devices
        if device.platform != "gpu" or "H100" not in device.device_kind
    ]
    if invalid_devices:
        raise ValueError(f"group2 parity requires eight H100 GPUs; invalid_devices={invalid_devices}")


def host_microbatches() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Create deterministic tokens and unequal weighted-CE denominators."""
    rows = np.arange(GLOBAL_MICROBATCH_SIZE, dtype=np.int32)[:, None]
    positions = np.arange(SEQUENCE_LENGTH, dtype=np.int32)[None, :]
    first_tokens = (rows * 97 + positions * 17 + 3) % VOCAB_SIZE
    second_tokens = (rows * 193 + positions * 29 + 11) % VOCAB_SIZE

    first_weight = np.ones((GLOBAL_MICROBATCH_SIZE, SEQUENCE_LENGTH), dtype=np.float32)
    first_weight[:, -1] = 0.0
    second_weight = np.broadcast_to((positions % 3 != 0), first_weight.shape).astype(np.float32).copy()
    second_weight[:, -1] = 0.0
    if float(first_weight.sum()) == float(second_weight.sum()):
        raise AssertionError("parity gate requires unequal loss_weight denominators")
    return (first_tokens, first_weight), (second_tokens, second_weight)


def ordered_last_stage_loss_and_grads(
    params: TransformerPipelineStage,
    qb_betas: jax.Array,
    hiddens: tuple[jax.Array, jax.Array],
    batches: tuple[GrugLmExample, GrugLmExample],
    mp: jmp.Policy,
    *,
    logsumexp_weight: float | None,
):
    """Run and differentiate the two microbatches independently, in order."""
    losses = []
    qb_betas_next = []
    gradients = []
    hidden_gradients = []
    for hidden, batch in zip(hiddens, batches, strict=True):

        def loss_fn(stage_params, stage_hidden, batch=batch):
            compute_params = grug_train._compute_stage(stage_params, qb_betas, mp)
            output_hidden, router_metrics = compute_params.block_range(stage_hidden, mask=batch.attn_mask)
            final_hidden = compute_params.finalize_hidden(output_hidden)
            loss, metrics = compute_params.hidden_next_token_loss(
                final_hidden,
                batch.tokens,
                batch.loss_weight,
                router_metrics,
                reduction="mean",
                logsumexp_weight=logsumexp_weight,
                return_router_metrics=True,
            )
            return loss, metrics["qb_beta_per_layer"]

        (loss, qb_next), (gradient, hidden_gradient) = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(params, hidden)
        losses.append(loss)
        qb_betas_next.append(qb_next)
        gradients.append(gradient)
        hidden_gradients.append(hidden_gradient)

    return (
        grug_train._sum_microbatch_group(tuple(losses)),
        grug_train._sum_microbatch_group(tuple(qb_betas_next)),
        grug_train._sum_microbatch_group(tuple(gradients)),
        tuple(hidden_gradients),
    )


def _project_output(output: jax.Array, cotangent: jax.Array) -> jax.Array:
    return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))


def _selected_experts_and_margin(block: Block, mlp_input: jax.Array) -> tuple[jax.Array, jax.Array]:
    flat_input = jnp.reshape(mlp_input, (-1, mlp_input.shape[-1]))
    router_logits = jnp.einsum(
        "td,de->te",
        flat_input,
        reshard(block.mlp.router, P(None, None)),
    ).astype(jnp.float32)
    router_logits = reshard(router_logits, P(_BATCH_AXES, None))
    biased_logits = router_logits + jax.lax.stop_gradient(reshard(block.mlp.router_bias, P(None)))
    topk_logits, selected_experts = jax.lax.top_k(biased_logits, block.mlp.cfg.num_experts_per_token + 1)
    top_k = block.mlp.cfg.num_experts_per_token
    boundary_margin = topk_logits[:, top_k - 1] - topk_logits[:, top_k]
    return selected_experts[:, :top_k], boundary_margin


def ordered_moe_preparation_and_routes(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
):
    """Prepare and route each microbatch independently."""
    mlp_inputs = tuple(block.mlp_gated_norm(block.rms_mlp(hidden)) for hidden in hiddens)
    routes = tuple(_selected_experts_and_margin(block, mlp_input) for mlp_input in mlp_inputs)
    return mlp_inputs, tuple(route[0] for route in routes), tuple(route[1] for route in routes)


def grouped_moe_preparation_and_routes(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
):
    """Prepare the interleaved pair together and inspect its routes."""
    packed_hidden = grug_train._pack_microbatch_pair(hiddens, name="MoE hidden")
    packed_mlp_input = block.mlp_gated_norm(block.rms_mlp(packed_hidden))
    mlp_inputs = grug_train._unpack_microbatch_pair(packed_mlp_input)
    routes = tuple(_selected_experts_and_margin(block, mlp_input) for mlp_input in mlp_inputs)
    return mlp_inputs, tuple(route[0] for route in routes), tuple(route[1] for route in routes)


def joined_moe_pair_value_and_grads(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
):
    """Differentiate two learned-router MoE calls in one reverse pass."""

    def projected_pair(current_block, current_hiddens):
        outputs = []
        router_stats = []
        loss = jnp.asarray(0.0, dtype=jnp.float32)
        for hidden, cotangent in zip(current_hiddens, cotangents, strict=True):
            output, stats = current_block.moe_residual(hidden)
            outputs.append(output)
            router_stats.append(stats)
            loss = loss + _project_output(output, cotangent)
        return loss, (tuple(outputs), tuple(router_stats))

    (loss, auxiliary), gradients = jax.value_and_grad(
        projected_pair,
        argnums=(0, 1),
        has_aux=True,
    )(block, hiddens)
    return loss, auxiliary, gradients[0], gradients[1]


def grouped_moe_pair_value_and_grads(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
):
    """Differentiate the production packed-preparation MoE boundary."""
    packed_hidden = grug_train._pack_microbatch_pair(hiddens, name="MoE hidden")
    packed_cotangent = grug_train._pack_microbatch_pair(cotangents, name="MoE cotangent")

    def projected_grouped_moe(current_block, current_hidden):
        output, router_stats = current_block.grouped_moe_residual(current_hidden)
        return _project_output(output, packed_cotangent), (
            grug_train._unpack_microbatch_pair(output),
            router_stats,
        )

    (loss, auxiliary), (block_gradient, hidden_gradient) = jax.value_and_grad(
        projected_grouped_moe,
        argnums=(0, 1),
        has_aux=True,
    )(block, packed_hidden)
    return (
        loss,
        auxiliary,
        block_gradient,
        grug_train._unpack_microbatch_pair(hidden_gradient),
    )


def ordered_moe_pair_value_and_grads(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
):
    """Differentiate two learned-router MoE calls separately and sum parameters."""
    losses = []
    outputs = []
    router_stats = []
    block_gradients = []
    hidden_gradients = []
    for hidden, cotangent in zip(hiddens, cotangents, strict=True):

        def projected_single(current_block, current_hidden, cotangent=cotangent):
            output, stats = current_block.moe_residual(current_hidden)
            return _project_output(output, cotangent), (output, stats)

        (loss, (output, stats)), (block_gradient, hidden_gradient) = jax.value_and_grad(
            projected_single,
            argnums=(0, 1),
            has_aux=True,
        )(block, hidden)
        losses.append(loss)
        outputs.append(output)
        router_stats.append(stats)
        block_gradients.append(block_gradient)
        hidden_gradients.append(hidden_gradient)
    return (
        grug_train._sum_microbatch_group(tuple(losses)),
        (tuple(outputs), tuple(router_stats)),
        grug_train._sum_microbatch_group(tuple(block_gradients)),
        tuple(hidden_gradients),
    )


def packed_attention_value_and_grads(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
    mask: AttentionMask | jax.Array,
):
    """Differentiate one interleaved packed attention call."""
    packed_hidden = grug_train._pack_microbatch_pair(hiddens, name="attention hidden")
    packed_cotangent = grug_train._pack_microbatch_pair(cotangents, name="attention cotangent")

    def projected_attention(current_block, current_hidden):
        output = current_block.attention_residual(current_hidden, mask)
        return _project_output(output, packed_cotangent), output

    (loss, output), (block_gradient, hidden_gradient) = jax.value_and_grad(
        projected_attention,
        argnums=(0, 1),
        has_aux=True,
    )(block, packed_hidden)
    return (
        loss,
        grug_train._unpack_microbatch_pair(output),
        block_gradient,
        grug_train._unpack_microbatch_pair(hidden_gradient),
    )


def ordered_attention_value_and_grads(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
    cotangents: tuple[jax.Array, jax.Array],
    masks: tuple[AttentionMask | jax.Array, AttentionMask | jax.Array],
):
    """Differentiate two attention calls separately and sum parameters."""
    losses = []
    outputs = []
    block_gradients = []
    hidden_gradients = []
    for hidden, cotangent, mask in zip(hiddens, cotangents, masks, strict=True):

        def projected_attention(current_block, current_hidden, mask=mask, cotangent=cotangent):
            output = current_block.attention_residual(current_hidden, mask)
            return _project_output(output, cotangent), output

        (loss, output), (block_gradient, hidden_gradient) = jax.value_and_grad(
            projected_attention,
            argnums=(0, 1),
            has_aux=True,
        )(block, hidden)
        losses.append(loss)
        outputs.append(output)
        block_gradients.append(block_gradient)
        hidden_gradients.append(hidden_gradient)
    return (
        grug_train._sum_microbatch_group(tuple(losses)),
        tuple(outputs),
        grug_train._sum_microbatch_group(tuple(block_gradients)),
        tuple(hidden_gradients),
    )


def grouped_block_value_and_grads(
    block: Block,
    packed_hidden: jax.Array,
    packed_cotangent: jax.Array,
    mask: AttentionMask | jax.Array,
    *,
    remat_mode: RematMode | None,
):
    """Differentiate a grouped block with no checkpoint or one production mode."""

    def projected_block(current_block, current_hidden):
        if remat_mode is None:
            output, router_stats = current_block.grouped_call(current_hidden, mask)
        else:
            output, router_stats = _run_grouped_block_with_remat(
                current_block,
                current_hidden,
                mask,
                use_pko=False,
                disable_rope=False,
                remat_mode=remat_mode,
            )
        return _project_output(output, packed_cotangent), (output, router_stats)

    (loss, auxiliary), (block_gradient, hidden_gradient) = jax.value_and_grad(
        projected_block,
        argnums=(0, 1),
        has_aux=True,
    )(block, packed_hidden)
    return loss, auxiliary, block_gradient, hidden_gradient


def joined_final_head_loss_and_grads(
    stage: TransformerPipelineStage,
    hiddens: tuple[jax.Array, jax.Array],
    batches: tuple[GrugLmExample, GrugLmExample],
    router_metrics: tuple[dict[str, jax.Array], dict[str, jax.Array]],
    *,
    logsumexp_weight: float | None,
):
    """Differentiate both weighted final-head losses in one reverse pass."""

    def pair_loss(current_stage, current_hiddens):
        losses = []
        for hidden, batch, metrics in zip(current_hiddens, batches, router_metrics, strict=True):
            final_hidden = current_stage.finalize_hidden(hidden)
            losses.append(
                current_stage.hidden_next_token_loss(
                    final_hidden,
                    batch.tokens,
                    batch.loss_weight,
                    metrics,
                    reduction="mean",
                    logsumexp_weight=logsumexp_weight,
                )
            )
        return grug_train._sum_microbatch_group(tuple(losses))

    loss, (stage_gradient, hidden_gradients) = jax.value_and_grad(pair_loss, argnums=(0, 1))(stage, hiddens)
    return loss, stage_gradient, hidden_gradients


def ordered_final_head_loss_and_grads(
    stage: TransformerPipelineStage,
    hiddens: tuple[jax.Array, jax.Array],
    batches: tuple[GrugLmExample, GrugLmExample],
    router_metrics: tuple[dict[str, jax.Array], dict[str, jax.Array]],
    *,
    logsumexp_weight: float | None,
):
    """Differentiate the weighted final-head losses separately and sum parameters."""
    losses = []
    stage_gradients = []
    hidden_gradients = []
    for hidden, batch, metrics in zip(hiddens, batches, router_metrics, strict=True):

        def single_loss(current_stage, current_hidden, batch=batch, metrics=metrics):
            final_hidden = current_stage.finalize_hidden(current_hidden)
            return current_stage.hidden_next_token_loss(
                final_hidden,
                batch.tokens,
                batch.loss_weight,
                metrics,
                reduction="mean",
                logsumexp_weight=logsumexp_weight,
            )

        loss, (stage_gradient, hidden_gradient) = jax.value_and_grad(single_loss, argnums=(0, 1))(stage, hidden)
        losses.append(loss)
        stage_gradients.append(stage_gradient)
        hidden_gradients.append(hidden_gradient)
    return (
        grug_train._sum_microbatch_group(tuple(losses)),
        grug_train._sum_microbatch_group(tuple(stage_gradients)),
        tuple(hidden_gradients),
    )


def _device_problem():
    mesh = grug_train._compact_or_pipeline_grug_mesh(
        expert_axis_size=EXPERT_AXIS_SIZE,
        replica_axis_size=1,
        pipeline=None,
    )
    token_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    hidden_sharding = NamedSharding(mesh, P(_BATCH_AXES, None, None))
    replicated_sharding = NamedSharding(mesh, P(None, None))

    with jax.set_mesh(mesh):
        model = Transformer.init(target_model_config(), key=jax.random.PRNGKey(0))
        stage = model.split_for_pipeline(PIPELINE_STAGES)[TARGET_STAGE]
        del model

        batches = tuple(
            GrugLmExample(
                tokens=jax.device_put(tokens, token_sharding),
                loss_weight=jax.device_put(loss_weight, token_sharding),
            )
            for tokens, loss_weight in host_microbatches()
        )

        def random_hidden(key):
            hidden = jax.random.normal(
                key,
                (GLOBAL_MICROBATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_DIM),
                dtype=jnp.bfloat16,
            )
            return hidden * jnp.asarray(0.02, dtype=jnp.bfloat16)

        make_hidden = jax.jit(random_hidden, out_shardings=hidden_sharding)
        hiddens = tuple(make_hidden(jax.random.PRNGKey(seed)) for seed in (1, 2))
        qb_betas = jax.device_put(
            np.linspace(-0.05, 0.05, NUM_EXPERTS, dtype=np.float32)[None, :],
            replicated_sharding,
        )
    return mesh, stage, qb_betas, hiddens, batches


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _compile_and_run(name: str, function, arguments):
    start = time.monotonic()
    _event("lower_start", arm=name)
    lowered = function.lower(*arguments)
    _event("lower_done", arm=name, elapsed_seconds=time.monotonic() - start)
    compile_start = time.monotonic()
    _event("compile_start", arm=name)
    compiled = lowered.compile()
    _event("compile_done", arm=name, elapsed_seconds=time.monotonic() - compile_start)
    execute_start = time.monotonic()
    _event("execute_start", arm=name)
    result = compiled(*arguments)
    _block_until_ready(result)
    _event("execute_done", arm=name, elapsed_seconds=time.monotonic() - execute_start)
    return result


def _moe_pair_diagnostic(
    mesh,
    stage: TransformerPipelineStage,
    qb_betas: jax.Array,
    hiddens: tuple[jax.Array, jax.Array],
) -> bool:
    compute_block = grug_train._compute_stage(stage, qb_betas, _PRODUCTION_MIXED_PRECISION).blocks[0]
    cotangents = (hiddens[1], hiddens[0])
    joined = jax.jit(joined_moe_pair_value_and_grads)
    grouped = jax.jit(grouped_moe_pair_value_and_grads)
    ordered = jax.jit(ordered_moe_pair_value_and_grads)
    grouped_preparation = jax.jit(grouped_moe_preparation_and_routes)
    ordered_preparation = jax.jit(ordered_moe_preparation_and_routes)
    arguments = (compute_block, hiddens, cotangents)
    with jax.set_mesh(mesh):
        ordered_preparation_result = _compile_and_run(
            "moe_preparation_ordered",
            ordered_preparation,
            (compute_block, hiddens),
        )
        grouped_preparation_result = _compile_and_run(
            "moe_preparation_grouped",
            grouped_preparation,
            (compute_block, hiddens),
        )
        reference = _compile_and_run("moe_pair_ordered", ordered, arguments)
        joined_result = _compile_and_run("moe_pair_joined", joined, arguments)
        grouped_result = _compile_and_run("moe_pair_grouped_boundary", grouped, arguments)
        pair_value_report = build_parity_report(
            automatic_loss=joined_result[0],
            direct_loss=reference[0],
            automatic_gradients={"outputs": joined_result[1][0], "router_stats": joined_result[1][1]},
            direct_gradients={"outputs": reference[1][0], "router_stats": reference[1][1]},
            tolerance=DEFAULT_TOLERANCE,
            gradient_root="values",
        )
        pair_gradient_report = build_parity_report(
            automatic_loss=joined_result[0],
            direct_loss=reference[0],
            automatic_gradients={"parameters": joined_result[2], "inputs": joined_result[3]},
            direct_gradients={"parameters": reference[2], "inputs": reference[3]},
            tolerance=DEFAULT_TOLERANCE,
            gradient_root="gradients",
        )
        grouped_value_report = build_parity_report(
            automatic_loss=grouped_result[0],
            direct_loss=joined_result[0],
            automatic_gradients={"outputs": grouped_result[1][0], "router_stats": grouped_result[1][1]},
            direct_gradients={"outputs": joined_result[1][0], "router_stats": joined_result[1][1]},
            tolerance=DEFAULT_TOLERANCE,
            gradient_root="values",
        )
        grouped_gradient_report = build_parity_report(
            automatic_loss=grouped_result[0],
            direct_loss=joined_result[0],
            automatic_gradients={"parameters": grouped_result[2], "inputs": grouped_result[3]},
            direct_gradients={"parameters": joined_result[2], "inputs": joined_result[3]},
            tolerance=DEFAULT_TOLERANCE,
            gradient_root="gradients",
        )
        preparation_report = build_parity_report(
            automatic_loss=jnp.sum(grouped_preparation_result[0][0].astype(jnp.float32)),
            direct_loss=jnp.sum(ordered_preparation_result[0][0].astype(jnp.float32)),
            automatic_gradients=grouped_preparation_result[0],
            direct_gradients=ordered_preparation_result[0],
            tolerance=DEFAULT_TOLERANCE,
            gradient_root="mlp_inputs",
        )
        route_mismatch_assignments = int(
            sum(
                jnp.sum(grouped_selected != ordered_selected)
                for grouped_selected, ordered_selected in zip(
                    grouped_preparation_result[1],
                    ordered_preparation_result[1],
                    strict=True,
                )
            )
        )
        route_mismatch_tokens = int(
            sum(
                jnp.sum(jnp.any(grouped_selected != ordered_selected, axis=-1))
                for grouped_selected, ordered_selected in zip(
                    grouped_preparation_result[1],
                    ordered_preparation_result[1],
                    strict=True,
                )
            )
        )
        ordered_min_boundary_margin = float(
            jnp.min(jnp.stack(tuple(jnp.min(margin) for margin in ordered_preparation_result[2])))
        )
        grouped_min_boundary_margin = float(
            jnp.min(jnp.stack(tuple(jnp.min(margin) for margin in grouped_preparation_result[2])))
        )

    passed = (
        preparation_report.passed
        and route_mismatch_assignments == 0
        and pair_value_report.passed
        and pair_gradient_report.passed
        and grouped_value_report.passed
        and grouped_gradient_report.passed
    )
    _event(
        "moe_pair_parity_report",
        source_lineage=SOURCE_LINEAGE,
        preparation_report=preparation_report.as_dict(),
        route_mismatch_assignments=route_mismatch_assignments,
        route_mismatch_tokens=route_mismatch_tokens,
        ordered_min_boundary_margin=ordered_min_boundary_margin,
        grouped_min_boundary_margin=grouped_min_boundary_margin,
        pair_value_report=pair_value_report.as_dict(),
        pair_gradient_report=pair_gradient_report.as_dict(),
        grouped_value_report=grouped_value_report.as_dict(),
        grouped_gradient_report=grouped_gradient_report.as_dict(),
        passed=passed,
    )
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", choices=_DIAGNOSTIC_MODES, default="full")
    args = parser.parse_args(argv)
    faulthandler.enable()
    faulthandler.dump_traceback_later(STACK_INTERVAL, repeat=True)
    validate_kernel_environment(os.environ)
    validate_h100x8_topology()
    _event(
        "configuration",
        source_lineage=SOURCE_LINEAGE,
        global_microbatch_size=GLOBAL_MICROBATCH_SIZE,
        local_microbatch_size=LOCAL_MICROBATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=HIDDEN_DIM,
        experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_axis_size=EXPERT_AXIS_SIZE,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
        compute_dtype="bfloat16",
        remat_mode="save_moe",
        diagnostic=args.diagnostic,
        tolerance=DEFAULT_TOLERANCE,
    )
    _event("problem_init_start")
    mesh, stage, qb_betas, hiddens, batches = _device_problem()
    _event("problem_init_done")
    if args.diagnostic == "moe-pair":
        passed = _moe_pair_diagnostic(mesh, stage, qb_betas, hiddens)
        faulthandler.cancel_dump_traceback_later()
        return 0 if passed else 1

    ordered = jax.jit(
        lambda params, biases, hidden_pair, batch_pair: ordered_last_stage_loss_and_grads(
            params,
            biases,
            hidden_pair,
            batch_pair,
            _PRODUCTION_MIXED_PRECISION,
            logsumexp_weight=LOGSUMEXP_WEIGHT,
        )
    )
    grouped = jax.jit(
        lambda params, biases, hidden_pair, batch_pair: grug_train._grouped_last_stage_loss_and_grads(
            params,
            biases,
            hidden_pair,
            batch_pair,
            _PRODUCTION_MIXED_PRECISION,
            logsumexp_weight=LOGSUMEXP_WEIGHT,
        )
    )
    arguments = (stage, qb_betas, hiddens, batches)
    with jax.set_mesh(mesh):
        reference = _compile_and_run("ordered", ordered, arguments)
        actual = _compile_and_run("grouped", grouped, arguments)
        report = build_parity_report(
            automatic_loss=actual[0],
            direct_loss=reference[0],
            automatic_gradients={"parameters": actual[2], "inputs": actual[3]},
            direct_gradients={"parameters": reference[2], "inputs": reference[3]},
            tolerance=DEFAULT_TOLERANCE,
            gradient_root="gradients",
        )
        qb_beta = build_value_parity(actual[1], reference[1], tolerance=DEFAULT_TOLERANCE)

    loss_weight_sums = [float(np.asarray(batch.loss_weight).sum()) for batch in batches]
    passed = report.passed and qb_beta.passed
    _event(
        "parity_report",
        source_lineage=SOURCE_LINEAGE,
        loss_weight_sums=loss_weight_sums,
        gradient_leaf_count=len(report.gradients),
        report=report.as_dict(),
        qb_beta=dataclasses.asdict(qb_beta),
        passed=passed,
    )
    faulthandler.cancel_dump_traceback_later()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
