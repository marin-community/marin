# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.quantization import Fp8RaggedDotOp
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.model import MoEMLP, TransformerPipelineStage
from experiments.grug.moe.repro_jaxpp_fp8_expert_compile import (
    Config,
    Fp8ExpertLayer,
    LastStageParameters,
    _stage_mesh,
    accumulate_gradients,
    average_gradients,
    external_distributed_context,
    last_stage_loss_and_gradients,
    materialize_parameters,
    parse_config,
    routed_last_stage_loss_and_gradients,
    stage_partition_specs,
    stage_shardings,
)


@dataclass(frozen=True)
class FakeIrisJobInfo:
    task_index: int
    num_tasks: int


def _host_array(shape, dtype, fill):
    return jnp.full(shape, fill, dtype)


def _gradient_layer(weight_value: float, state_value: float) -> Fp8ExpertLayer:
    op = Fp8RaggedDotOp(
        input_scale=_host_array((1,), jnp.float32, state_value),
        output_grad_scale=_host_array((1,), jnp.float32, state_value),
        kernel_scale=_host_array((1,), jnp.float32, state_value),
        input_amax_history=_host_array((4,), jnp.float32, state_value),
        output_grad_amax_history=_host_array((4,), jnp.float32, state_value),
        kernel_amax_history=_host_array((4,), jnp.float32, state_value),
        compute_dtype=None,
        fwd_dtype=jnp.float8_e4m3fn,
        rev_dtype=jnp.float8_e4m3fn,
    )
    return Fp8ExpertLayer(
        w13=jnp.full((1, 2, 4), weight_value, jnp.float32),
        w2=jnp.full((1, 2, 2), weight_value, jnp.float32),
        w13_op=op,
        w2_op=op,
    )


def test_gradient_reduction_adds_weights_maxes_state_and_averages_only_weights() -> None:
    first = (_gradient_layer(2.0, 3.0),)
    second = (_gradient_layer(4.0, 5.0),)

    accumulated = accumulate_gradients(first, second)
    averaged = average_gradients(accumulated, microbatches=2)

    assert isinstance(averaged, tuple)
    np.testing.assert_array_equal(averaged[0].w13, jnp.full((1, 2, 4), 3.0))
    np.testing.assert_array_equal(averaged[0].w2, jnp.full((1, 2, 2), 3.0))
    np.testing.assert_array_equal(averaged[0].w13_op.input_scale, jnp.full((1,), 5.0))
    np.testing.assert_array_equal(averaged[0].w2_op.output_grad_amax_history, jnp.full((4,), 5.0))


@pytest.mark.parametrize(
    ("argument", "value"),
    (("--tokens", "127"), ("--hidden", "96"), ("--intermediate", "192")),
)
def test_fp8_config_rejects_shapes_that_cannot_lower(argument: str, value: str) -> None:
    with pytest.raises(ValueError, match="must be divisible by 128"):
        parse_config(["--runtime", "direct", "--kernel", "fp8", argument, value])


def test_bf16_control_accepts_non_fp8_aligned_shapes() -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "bf16",
            "--experts",
            "3",
            "--tokens",
            "12",
            "--hidden",
            "6",
            "--intermediate",
            "5",
        ]
    )

    assert config.kernel == "bf16"
    assert config.tokens_per_expert == 4


def test_config_rejects_unequal_expert_groups() -> None:
    config = Config(
        runtime="direct",
        worker_mode="local",
        kernel="bf16",
        layers=1,
        experts=3,
        tokens=10,
        hidden=8,
        intermediate=8,
        loss_boundary="mse",
        remat_mode="none",
        routing_mode="fixed",
        block_boundary="moe_only",
        attention_implementation="reference",
        num_heads=1,
        num_kv_heads=1,
        total_layers=8,
        sliding_window=2048,
        sequence_length=8,
        vocab_size=8,
        top_k=1,
        devices_per_stage=1,
        microbatches=1,
        amax_history=4,
        timeout=30,
        stack_after=10,
        coordinator_port=5793,
        stop_after="execute",
        dump_dir=None,
    )

    with pytest.raises(ValueError, match="tokens=10 must be divisible by experts=3"):
        config.validate()


def test_ring_partition_specs_preserve_production_sharding_contract() -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "fp8_ring",
            "--devices-per-stage",
            "2",
            "--experts",
            "4",
            "--top-k",
            "4",
        ]
    )

    specs = stage_partition_specs(config)

    assert specs.activation == P(("replica_dcn", "data", "expert"), None)
    assert specs.sequence_activation == P(("replica_dcn", "data", "expert"), None, None)
    assert specs.token == P(("replica_dcn", "data", "expert"), None)
    assert specs.weight == P("expert", None, None)
    assert specs.attention_input_weight == P("data", "model")
    assert specs.attention_output_weight == P("model", "data")
    assert specs.lm_head == P(("replica_dcn", "data"), "model")
    assert specs.qb_beta == P(None, None)
    assert specs.state == P()


def test_learned_qb_boundary_accepts_production_last_stage_shape() -> None:
    config = parse_config(
        [
            "--runtime",
            "jaxpp",
            "--worker-mode",
            "external",
            "--kernel",
            "fp8_ring",
            "--loss-boundary",
            "next_token",
            "--remat-mode",
            "save_moe",
            "--routing-mode",
            "learned_qb",
            "--devices-per-stage",
            "8",
            "--layers",
            "2",
            "--microbatches",
            "4",
            "--experts",
            "64",
            "--top-k",
            "4",
            "--tokens",
            "32768",
            "--sequence-length",
            "4096",
            "--hidden",
            "2560",
            "--intermediate",
            "1280",
            "--vocab-size",
            "8192",
            "--amax-history",
            "1024",
        ]
    )

    assert config.batch_size == 8
    assert config.loss_boundary == "next_token"
    assert config.routing_mode == "learned_qb"


def test_full_block_boundary_preserves_production_last_stage_geometry() -> None:
    config = parse_config(
        [
            "--runtime",
            "jaxpp",
            "--worker-mode",
            "external",
            "--kernel",
            "fp8_ring",
            "--loss-boundary",
            "next_token",
            "--remat-mode",
            "save_moe",
            "--routing-mode",
            "learned_qb",
            "--block-boundary",
            "full",
            "--attention-implementation",
            "gpu_fa4_cute",
            "--num-heads",
            "20",
            "--num-kv-heads",
            "5",
            "--total-layers",
            "8",
            "--devices-per-stage",
            "8",
            "--layers",
            "2",
            "--microbatches",
            "4",
            "--experts",
            "64",
            "--top-k",
            "4",
            "--tokens",
            "32768",
            "--sequence-length",
            "4096",
            "--hidden",
            "2560",
            "--intermediate",
            "1280",
            "--vocab-size",
            "8192",
            "--amax-history",
            "1024",
        ]
    )

    assert config.hidden // config.num_heads == 128
    assert config.num_heads // config.num_kv_heads == 4
    assert config.total_layers - config.layers == 6
    assert config.sliding_window == 2048


def test_full_block_boundary_runs_production_attention_norm_moe_and_loss_tree() -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "bf16",
            "--loss-boundary",
            "next_token",
            "--remat-mode",
            "save_moe",
            "--routing-mode",
            "learned_qb",
            "--block-boundary",
            "full",
            "--attention-implementation",
            "reference",
            "--num-heads",
            "2",
            "--num-kv-heads",
            "1",
            "--total-layers",
            "8",
            "--layers",
            "2",
            "--experts",
            "2",
            "--top-k",
            "1",
            "--tokens",
            "8",
            "--sequence-length",
            "4",
            "--hidden",
            "8",
            "--intermediate",
            "4",
            "--vocab-size",
            "8",
            "--sliding-window",
            "2",
        ]
    )
    mesh = _stage_mesh(config, [jax.devices()[0]])
    shardings = stage_shardings(config, mesh)
    params = materialize_parameters(config, shardings)
    assert isinstance(params, TransformerPipelineStage)
    assert params.start_layer == 6
    assert params.end_layer == 8
    assert params.blocks[0].shared is None

    hidden_values = jnp.arange(64, dtype=jnp.float32).reshape(2, 4, 8) / 64
    hidden = jax.device_put(hidden_values.astype(jnp.bfloat16), shardings.sequence_activation)
    qb_betas = jax.device_put(jnp.zeros((2, 2), jnp.float32), shardings.qb_beta)
    token_ids = jax.device_put(jnp.arange(8, dtype=jnp.int32).reshape(2, 4), shardings.token)
    loss_weight = jax.device_put(jnp.ones((2, 4), jnp.float32), shardings.token)
    dependency = jax.device_put(jnp.asarray(0.0, jnp.float32), shardings.state)

    with jax.set_mesh(mesh):
        backward_jaxpr = jax.make_jaxpr(
            lambda p, x: routed_last_stage_loss_and_gradients(
                p,
                x,
                qb_betas,
                token_ids,
                loss_weight,
                dependency,
                config,
                mesh,
            )
        )(params, hidden)
        loss, next_qb_betas, grads, d_hidden = routed_last_stage_loss_and_gradients(
            params,
            hidden,
            qb_betas,
            token_ids,
            loss_weight,
            dependency,
            config,
            mesh,
        )

    graph = str(backward_jaxpr)
    assert "top_k" in graph
    assert "remat2" in graph
    assert graph.count("dot_general") >= 8
    assert np.isfinite(loss)
    assert next_qb_betas.shape == (2, 2)
    assert isinstance(grads, TransformerPipelineStage)
    assert grads.blocks[0].attn.w_q.shape == (8, 8)
    assert np.any(np.asarray(grads.blocks[0].attn.w_q) != 0)
    assert grads.blocks[0].rms_attn.weight.shape == (8,)
    assert grads.blocks[0].mlp.router.shape == (8, 2)
    assert grads.output_proj is not None
    assert grads.output_proj.shape == (8, 8)
    assert d_hidden.shape == hidden.shape
    assert np.any(np.asarray(d_hidden) != 0)


def test_learned_qb_boundary_uses_production_router_and_dispatch_graph() -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "bf16",
            "--loss-boundary",
            "next_token",
            "--remat-mode",
            "save_moe",
            "--routing-mode",
            "learned_qb",
            "--experts",
            "2",
            "--top-k",
            "1",
            "--tokens",
            "4",
            "--sequence-length",
            "4",
            "--hidden",
            "4",
            "--intermediate",
            "4",
            "--vocab-size",
            "8",
        ]
    )
    mesh = _stage_mesh(config, [jax.devices()[0]])
    shardings = stage_shardings(config, mesh)
    params = materialize_parameters(config, shardings)
    assert isinstance(params, LastStageParameters)
    assert isinstance(params.expert_layers[0], MoEMLP)
    hidden = jax.device_put(jnp.full((1, 4, 4), 0.02, jnp.bfloat16), shardings.sequence_activation)
    qb_betas = jax.device_put(jnp.zeros((1, 2), jnp.float32), shardings.qb_beta)
    token_ids = jax.device_put(jnp.ones((1, 4), jnp.int32), shardings.token)
    loss_weight = jax.device_put(jnp.ones((1, 4), jnp.float32), shardings.token)
    dependency = jax.device_put(jnp.asarray(0.0, jnp.float32), shardings.state)

    with jax.set_mesh(mesh):
        routed_jaxpr = jax.make_jaxpr(
            lambda p, x, qb: routed_last_stage_loss_and_gradients(
                p,
                x,
                qb,
                token_ids,
                loss_weight,
                dependency,
                config,
                mesh,
            )
        )(params, hidden, qb_betas)
        loss, next_qb_betas, grads, d_hidden = routed_last_stage_loss_and_gradients(
            params,
            hidden,
            qb_betas,
            token_ids,
            loss_weight,
            dependency,
            config,
            mesh,
        )
        _, biased_next_qb_betas, _, _ = routed_last_stage_loss_and_gradients(
            params,
            hidden,
            jax.device_put(jnp.asarray([[10.0, -10.0]], jnp.float32), shardings.qb_beta),
            token_ids,
            loss_weight,
            dependency,
            config,
            mesh,
        )

    lowered_text = str(routed_jaxpr)
    assert "top_k" in lowered_text
    assert "sort" in lowered_text
    assert "remat2" in lowered_text
    assert np.isfinite(loss)
    assert next_qb_betas.shape == (1, 2)
    assert not np.array_equal(next_qb_betas, biased_next_qb_betas)
    assert isinstance(grads, LastStageParameters)
    assert isinstance(grads.expert_layers[0], MoEMLP)
    assert grads.expert_layers[0].router.shape == (4, 2)
    assert d_hidden.shape == hidden.shape


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--routing-mode", "learned_qb"), "requires --loss-boundary next_token"),
        (
            (
                "--routing-mode",
                "learned_qb",
                "--loss-boundary",
                "next_token",
                "--experts",
                "2",
                "--top-k",
                "2",
            ),
            "requires top_k < experts",
        ),
        (
            (
                "--routing-mode",
                "learned_qb",
                "--loss-boundary",
                "next_token",
                "--kernel",
                "fp8",
                "--experts",
                "2",
            ),
            "requires the expert-parallel fp8_ring kernel",
        ),
    ),
)
def test_learned_qb_config_rejects_incomplete_production_boundaries(
    arguments: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_config(["--runtime", "direct", *arguments])


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ("--block-boundary", "full"),
            "requires --loss-boundary next_token and --routing-mode learned_qb",
        ),
        (
            (
                "--block-boundary",
                "full",
                "--loss-boundary",
                "next_token",
                "--routing-mode",
                "learned_qb",
                "--experts",
                "2",
            ),
            "requires --remat-mode save_moe",
        ),
        (
            (
                "--block-boundary",
                "full",
                "--loss-boundary",
                "next_token",
                "--routing-mode",
                "learned_qb",
                "--remat-mode",
                "save_moe",
                "--attention-implementation",
                "gpu_fa4_cute",
                "--num-heads",
                "2",
                "--hidden",
                "16",
                "--experts",
                "2",
            ),
            "requires head dimension 128",
        ),
    ),
)
def test_full_block_config_rejects_partial_production_boundary(
    arguments: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_config(["--runtime", "direct", "--kernel", "bf16", *arguments])


@pytest.mark.parametrize("remat_mode", ("none", "recompute_all", "save_moe"))
def test_block_remat_wraps_the_differentiated_expert_residual(remat_mode: str) -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "bf16",
            "--loss-boundary",
            "next_token",
            "--remat-mode",
            remat_mode,
            "--experts",
            "1",
            "--tokens",
            "4",
            "--sequence-length",
            "4",
            "--hidden",
            "4",
            "--intermediate",
            "4",
            "--vocab-size",
            "8",
        ]
    )
    mesh = _stage_mesh(config, [jax.devices()[0]])
    shardings = stage_shardings(config, mesh)
    params = materialize_parameters(config, shardings)
    hidden = jax.device_put(jnp.full((1, 4, 4), 0.02, jnp.bfloat16), shardings.sequence_activation)
    token_ids = jax.device_put(jnp.ones((1, 4), jnp.int32), shardings.token)
    loss_weight = jax.device_put(jnp.ones((1, 4), jnp.float32), shardings.token)
    dependency = jax.device_put(jnp.asarray(0.0, jnp.float32), shardings.state)

    with jax.set_mesh(mesh):
        backward_jaxpr = jax.make_jaxpr(
            lambda p, x: last_stage_loss_and_gradients(
                p,
                x,
                token_ids,
                loss_weight,
                dependency,
                config,
                mesh,
            )
        )(params, hidden)

    remat_equations = [equation for equation in backward_jaxpr.jaxpr.eqns if equation.primitive.name == "remat2"]
    if remat_mode == "none":
        assert remat_equations == []
        return
    assert len(remat_equations) == config.layers
    assert remat_equations[0].params["differentiated"] is True
    assert remat_equations[0].params["prevent_cse"] is True
    if remat_mode == "save_moe":
        assert remat_equations[0].params["policy"] is not None
    else:
        assert remat_equations[0].params["policy"] is None


def test_default_mode_does_not_add_a_remat_boundary() -> None:
    config = parse_config(["--runtime", "direct"])

    assert config.remat_mode == "none"
    assert config.routing_mode == "fixed"
    assert config.block_boundary == "moe_only"
    assert config.attention_implementation == "reference"


def test_next_token_boundary_rejects_batch_smaller_than_expert_mesh() -> None:
    with pytest.raises(ValueError, match="next-token batch size=1 must be divisible by devices_per_stage=2"):
        parse_config(
            [
                "--runtime",
                "jaxpp",
                "--kernel",
                "bf16_ring",
                "--loss-boundary",
                "next_token",
                "--devices-per-stage",
                "2",
                "--experts",
                "2",
                "--top-k",
                "2",
                "--tokens",
                "4",
                "--sequence-length",
                "4",
            ]
        )


def test_next_token_backward_returns_complete_last_stage_tree() -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "bf16",
            "--loss-boundary",
            "next_token",
            "--experts",
            "1",
            "--tokens",
            "4",
            "--sequence-length",
            "4",
            "--hidden",
            "4",
            "--intermediate",
            "4",
            "--vocab-size",
            "8",
        ]
    )
    mesh = _stage_mesh(config, [jax.devices()[0]])
    shardings = stage_shardings(config, mesh)
    params = materialize_parameters(config, shardings)
    hidden = jax.device_put(jnp.full((1, 4, 4), 0.02, jnp.bfloat16), shardings.sequence_activation)
    token_ids = jax.device_put(jnp.ones((1, 4), jnp.int32), shardings.token)
    loss_weight = jax.device_put(jnp.ones((1, 4), jnp.float32), shardings.token)
    dependency = jax.device_put(jnp.asarray(0.0, jnp.float32), shardings.state)

    with jax.set_mesh(mesh):
        loss, qb_beta_per_layer, grads, d_hidden = last_stage_loss_and_gradients(
            params,
            hidden,
            token_ids,
            loss_weight,
            dependency,
            config,
            mesh,
        )

    assert isinstance(grads, LastStageParameters)
    assert np.isfinite(loss)
    assert qb_beta_per_layer.shape == (1, 1)
    assert grads.final_norm_weight.shape == (4,)
    assert grads.final_gate_down.shape == (4, 128)
    assert grads.lm_head.shape == (4, 8)
    assert d_hidden.shape == hidden.shape


def test_ring_config_balances_assignments_instead_of_flat_token_groups() -> None:
    config = parse_config(
        [
            "--runtime",
            "direct",
            "--kernel",
            "bf16_ring",
            "--devices-per-stage",
            "2",
            "--experts",
            "4",
            "--top-k",
            "2",
            "--tokens",
            "6",
        ]
    )

    assert config.tokens % config.experts != 0
    assert config.tokens * config.top_k % config.experts == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--devices-per-stage", "1"), "requires devices_per_stage greater than 1"),
        (
            ("--devices-per-stage", "2", "--experts", "3", "--tokens", "126"),
            "experts=3 must be divisible",
        ),
        (
            ("--devices-per-stage", "2", "--experts", "4", "--top-k", "1", "--tokens", "6"),
            "balanced ring routing requires tokens \\* top_k to be divisible by experts",
        ),
        (
            ("--devices-per-stage", "2", "--experts", "4", "--top-k", "2", "--tokens", "64"),
            "assignments_per_device=64 must be divisible by 128",
        ),
    ),
)
def test_fp8_ring_config_rejects_non_production_mesh_contracts(
    arguments: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_config(["--runtime", "direct", "--kernel", "fp8_ring", *arguments])


def test_external_context_prefers_complete_jax_environment() -> None:
    context = external_distributed_context(
        {
            "JAX_COORDINATOR_ADDRESS": "coordinator.internal:8476",
            "JAX_NUM_PROCESSES": "2",
            "JAX_PROCESS_ID": "1",
        },
        FakeIrisJobInfo(task_index=0, num_tasks=4),
    )

    assert context.coordinator_address == "coordinator.internal:8476"
    assert context.num_processes == 2
    assert context.process_id == 1
    assert context.bootstrap == "jax_environment"


def test_external_context_uses_iris_job_info_without_jax_environment() -> None:
    context = external_distributed_context({}, FakeIrisJobInfo(task_index=1, num_tasks=2))

    assert context.coordinator_address is None
    assert context.num_processes == 2
    assert context.process_id == 1
    assert context.bootstrap == "iris_job_info"


def test_external_context_rejects_partial_jax_environment() -> None:
    with pytest.raises(ValueError, match="requires all JAX distributed environment variables"):
        external_distributed_context(
            {
                "JAX_COORDINATOR_ADDRESS": "coordinator.internal:8476",
                "JAX_NUM_PROCESSES": "2",
            },
            FakeIrisJobInfo(task_index=0, num_tasks=2),
        )


def test_external_worker_mode_rejects_single_process_runtime() -> None:
    with pytest.raises(ValueError, match="supports only distributed_direct or jaxpp"):
        parse_config(["--runtime", "direct", "--worker-mode", "external"])
