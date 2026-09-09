# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample

from experiments.grug.moe_pipeline.benchmark import _resolve_benchmark_config
from experiments.grug.moe_pipeline.checkpoint import restore_checkpoint, save_checkpoint
from experiments.grug.moe_pipeline.model import GrugModelConfig, Transformer
from experiments.grug.moe_pipeline.pipeline import (
    AutomaticPipelineSchedule,
    GrugMoeAutomaticPipelineState,
    GrugMoePipelineConfig,
    _apply_qb_betas,
    automatic_stage_to_mpmd_indices,
    split_automatic_stages,
    split_transformer,
)
from experiments.grug.moe_pipeline.train import PipelineSchedule, _validate_local_mesh


def _tiny_model(*, num_layers: int = 2) -> tuple[Mesh, Transformer]:
    config = GrugModelConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=num_layers,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=4,
        sliding_window=4,
        router_z_loss_coef=0.1,
        attention_implementation="reference",
        moe_implementation="scatter",
    )
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.PRNGKey(0))
    return mesh, model


def _batch() -> GrugLmExample:
    return GrugLmExample(
        tokens=jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32),
        loss_weight=jnp.ones((2, 4), dtype=jnp.float32),
    )


def _assert_trees_close(actual, expected) -> None:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-5, atol=1e-5)


def _staged_loss(stages, batch: GrugLmExample, *, logsumexp_weight: float | None = None):
    hidden = stages[0].embed(batch.tokens)
    router_loss = jnp.array(0.0, dtype=jnp.float32)
    for stage in stages:
        hidden, metrics = stage.run_blocks(hidden, batch.attn_mask)
        router_loss = router_loss + stage.local_router_loss(metrics)
    hidden = stages[-1].finish(hidden)
    return (
        stages[-1].cross_entropy_loss(
            hidden,
            batch.tokens,
            batch.loss_weight,
            logsumexp_weight=logsumexp_weight,
        )
        + router_loss
    )


def test_split_transformer_assigns_uneven_contiguous_stages():
    _, model = _tiny_model(num_layers=4)

    stages = split_transformer(model, 2, layer_counts=(3, 1))

    assert [(stage.start_layer, stage.end_layer) for stage in stages] == [(0, 3), (3, 4)]
    assert all(actual is expected for actual, expected in zip(stages[0].blocks, model.blocks[:3], strict=True))
    assert stages[1].blocks[0] is model.blocks[3]
    assert stages[0].token_embed is model.token_embed
    assert stages[0].output_proj is None
    assert stages[1].token_embed is None
    assert stages[1].output_proj is model.output_proj


def test_staged_loss_and_gradients_match_the_unsplit_model():
    mesh, model = _tiny_model()
    batch = _batch()

    def ordinary_loss(params):
        return params.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="mean",
            logsumexp_weight=0.01,
        )

    def pipeline_loss(params):
        return _staged_loss(split_transformer(params, 2), batch, logsumexp_weight=0.01)

    with jax.set_mesh(mesh):
        ordinary_value, ordinary_grads = jax.value_and_grad(ordinary_loss)(model)
        pipeline_value, pipeline_grads = jax.value_and_grad(pipeline_loss)(model)

    np.testing.assert_allclose(pipeline_value, ordinary_value, rtol=1e-5, atol=1e-5)
    _assert_trees_close(pipeline_grads, ordinary_grads)


def test_dualpipe_v_maps_two_logical_stages_to_each_physical_rank():
    pytest.importorskip("jaxpp")
    config = GrugMoePipelineConfig(stages=4, physical_stages=2, microbatches=4)

    assert automatic_stage_to_mpmd_indices(config, AutomaticPipelineSchedule.DUALPIPE_V) == (0, 1, 1, 0)


def test_automatic_pipeline_excludes_qb_bias_from_differentiated_parameters():
    _, model = _tiny_model(num_layers=4)

    trainable_stages, _ = split_automatic_stages(model, num_stages=2)

    for trainable_stage in trainable_stages:
        for trainable_block in trainable_stage.blocks:
            assert trainable_block.mlp.router_bias is None


def test_automatic_pipeline_installs_pending_qb_bias():
    _, model = _tiny_model()
    trainable_stages, _ = split_automatic_stages(model, num_stages=2)
    pending_bias = jnp.array([[2.0, -1.0]], dtype=jnp.float32)

    stage = _apply_qb_betas(trainable_stages[0], pending_bias)

    np.testing.assert_allclose(stage.blocks[0].mlp.router_bias, jnp.array([-1.5, 1.5]))


def test_pipeline_mesh_validation_uses_full_stage_shard_count():
    # Regression: a stage may contain both FSDP and expert axes, so its device
    # count can exceed the expert-axis size.
    _validate_local_mesh(
        local_device_count=8,
        devices_per_stage=16,
        expert_axis_size=4,
        batch_size=64,
        microbatches=4,
    )

    with pytest.raises(ValueError):
        _validate_local_mesh(
            local_device_count=8,
            devices_per_stage=16,
            expert_axis_size=4,
            batch_size=32,
            microbatches=4,
        )


def test_dualpipe_v_train_config_rejects_too_few_microbatches():
    zero_bubble = _resolve_benchmark_config({})

    with pytest.raises(ValueError):
        replace(
            zero_bubble,
            physical_stages=2,
            microbatches=3,
            schedule=PipelineSchedule.DUALPIPE_V,
        )


def test_checkpoint_restores_optimizer_and_pending_router_updates(tmp_path):
    mesh, model = _tiny_model()
    params, _ = split_automatic_stages(model, num_stages=2)
    optimizer = optax.adamw(1e-4)
    with jax.set_mesh(mesh):
        state = GrugMoeAutomaticPipelineState(
            params,
            tuple(optimizer.init(stage) for stage in params),
            (jnp.array([[2.0, -1.0]]), jnp.array([[-3.0, 1.0]])),
        )
        # Populate Adam moments and counts; a fresh optimizer must differ.
        gradients = jax.tree.map(jnp.ones_like, params)
        updated = [
            optimizer.update(grad, opt, param)
            for grad, opt, param in zip(gradients, state.opt_state, params, strict=True)
        ]
        state = replace(
            state,
            trainable_params=tuple(
                optax.apply_updates(param, item[0]) for param, item in zip(params, updated, strict=True)
            ),
            opt_state=tuple(item[1] for item in updated),
        )
        root = str(tmp_path)
        checkpoint = save_checkpoint(root, state, step=1, contract={"schedule": "zero_bubble"})
        assert json.loads((tmp_path / "latest.json").read_text()) == {
            "checkpoint": checkpoint.rsplit("/", 1)[-1],
            "step": 1,
        }
        # An interrupted newer save must not hide the committed checkpoint.
        (tmp_path / "step-000000000002-incomplete").mkdir()
        empty = jax.tree.map(jnp.zeros_like, state)
        shardings = jax.tree.map(lambda value: value.sharding, empty)
        restored, step = restore_checkpoint(root, empty, shardings, contract={"schedule": "zero_bubble"})
        assert step == 1
        for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
            np.testing.assert_array_equal(actual, expected)
        for stage_optimizer in restored.opt_state:
            np.testing.assert_array_equal(stage_optimizer[0].count, step)
        latest = json.loads((tmp_path / "latest.json").read_text())
        (tmp_path / "latest.json").write_text(json.dumps({**latest, "step": 2}))
        with pytest.raises(ValueError, match="step disagrees"):
            restore_checkpoint(root, empty, shardings, contract={"schedule": "zero_bubble"})
        # Recover committed data even if the first latest-pointer write failed.
        (tmp_path / "latest.json").unlink()
        recovered, recovered_step = restore_checkpoint(root, empty, shardings, contract={"schedule": "zero_bubble"})
        assert recovered_step == step
        _assert_trees_close(recovered, state)
        with pytest.raises(ValueError, match="training configuration"):
            restore_checkpoint(root, empty, shardings, contract={"schedule": "dualpipe_v"})
