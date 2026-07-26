# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H100x8 parity gate for exact-ring grouped stage tasks.

The gate compares the interleaved MoE-boundary group-size-2 final-stage path
against two ordered single-microbatch calls. It uses the production d2560,
CuTe FA4, EP8 bulk-ring, BF16-compute, and Pallas-Triton expert kernels. The
global microbatch is 32 examples, matching the target L24 run, so each H100
owns four examples.
"""

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
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe import train as grug_train
from experiments.grug.moe.check_jaxpp_eager_1f1b_parity import (
    DEFAULT_TOLERANCE,
    build_parity_report,
    build_value_parity,
)
from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.model import GrugModelConfig, Transformer, TransformerPipelineStage

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


def main() -> int:
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
        tolerance=DEFAULT_TOLERANCE,
    )
    _event("problem_init_start")
    mesh, stage, qb_betas, hiddens, batches = _device_problem()
    _event("problem_init_done")

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
