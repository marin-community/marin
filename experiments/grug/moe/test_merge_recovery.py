# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.expert_merge import convert_one_expert_pair, forward_with_moe_traces
from experiments.grug.moe.merge_recovery import (
    MergeRecoveryConfig,
    RecoveryStage,
    chunked_output_kl,
    initial_recovery_state,
    make_recovery_train_step,
    recovery_forward,
    recovery_trainable_filter,
)
from experiments.grug.moe.model import GrugModelConfig, Transformer

_AFFECTED_LAYERS = (1, 2)


def test_chunked_output_kl_matches_dense_teacher_to_student_kl():
    student_hidden = jax.random.normal(jax.random.key(10), (2, 3, 5))
    teacher_hidden = jax.random.normal(jax.random.key(11), (2, 3, 5))
    student_output = jax.random.normal(jax.random.key(12), (5, 13))
    teacher_output = jax.random.normal(jax.random.key(13), (5, 13))
    weights = jnp.asarray([[1.0, 1.0, 0.0], [0.5, 1.0, 1.0]], dtype=jnp.float32)

    actual = chunked_output_kl(
        student_hidden,
        student_output,
        teacher_hidden,
        teacher_output,
        weights,
        vocab_chunk_size=4,
    )
    student_logits = jnp.einsum("bsh,hv->bsv", student_hidden, student_output)
    teacher_logits = jnp.einsum("bsh,hv->bsv", teacher_hidden, teacher_output)
    teacher_probs = jax.nn.softmax(teacher_logits.astype(jnp.float32), axis=-1)
    dense_by_token = jnp.sum(
        teacher_probs
        * (
            jax.nn.log_softmax(teacher_logits.astype(jnp.float32), axis=-1)
            - jax.nn.log_softmax(student_logits.astype(jnp.float32), axis=-1)
        ),
        axis=-1,
    )
    expected = jnp.sum(dense_by_token * weights) / jnp.sum(weights)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def _tiny_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="scatter",
    )


def _teacher_and_student() -> tuple[Transformer, Transformer]:
    teacher = Transformer.init(_tiny_config(), key=jax.random.key(0))
    student = convert_one_expert_pair(
        teacher,
        representative_layer=_AFFECTED_LAYERS[0],
        source_layer=_AFFECTED_LAYERS[1],
        source_to_shared=np.arange(teacher.config.num_experts),
    )
    return teacher, student


def _array_leaves(tree) -> list[jax.Array]:
    return jax.tree.leaves(eqx.filter(tree, eqx.is_array))


def _tree_changed(before, after) -> bool:
    return any(
        not np.array_equal(before_leaf, after_leaf)
        for before_leaf, after_leaf in zip(_array_leaves(before), _array_leaves(after), strict=True)
    )


def _assert_tree_equal(actual, expected) -> None:
    for actual_leaf, expected_leaf in zip(_array_leaves(actual), _array_leaves(expected), strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_recovery_forward_evaluates_teacher_moe_on_current_student_state():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher, student = _teacher_and_student()
        student = eqx.tree_at(
            lambda model: model.expert_banks[0].w_down,
            student,
            student.expert_banks[0].w_down * 2.0,
        )
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)

        actual = recovery_forward(student, teacher, tokens, affected_layers=_AFFECTED_LAYERS)
        _, student_traces = forward_with_moe_traces(student, tokens, target_layers=_AFFECTED_LAYERS)
        _, teacher_traces = forward_with_moe_traces(teacher, tokens, target_layers=_AFFECTED_LAYERS)

        expected_errors = []
        teacher_rollout_errors = []
        for layer in _AFFECTED_LAYERS:
            teacher_block = teacher.blocks[layer]
            teacher_bank = teacher.expert_banks[teacher_block.expert_bank_index]
            teacher_on_student, _ = teacher_block.mlp(student_traces[layer].mlp_input, teacher_bank)
            teacher_on_teacher, _ = teacher_block.mlp(teacher_traces[layer].mlp_input, teacher_bank)
            student_output = student_traces[layer].routed_output.astype(jnp.float32)
            expected_errors.append(
                jnp.mean(jnp.square(student_output - teacher_on_student))
                / (jnp.mean(jnp.square(teacher_on_student)) + 1e-8)
            )
            teacher_rollout_errors.append(
                jnp.mean(jnp.square(student_output - teacher_on_teacher))
                / (jnp.mean(jnp.square(teacher_on_teacher)) + 1e-8)
            )

    expected_nrmse = jnp.sqrt(jnp.stack(expected_errors))
    np.testing.assert_allclose(actual.moe_output_nrmse, expected_nrmse, rtol=1e-6, atol=1e-6)
    assert not np.allclose(actual.moe_output_nrmse, jnp.sqrt(jnp.stack(teacher_rollout_errors)))
    np.testing.assert_allclose(actual.router_top1_agreement_with_teacher, 1.0)
    np.testing.assert_allclose(actual.router_topk_agreement_with_teacher, 1.0)


def test_recovery_router_agreement_accounts_for_source_expert_permutation():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher = Transformer.init(_tiny_config(), key=jax.random.key(21))
        permutation = (1, 0, 3, 2)
        student = convert_one_expert_pair(
            teacher,
            representative_layer=_AFFECTED_LAYERS[0],
            source_layer=_AFFECTED_LAYERS[1],
            source_to_shared=np.asarray(permutation),
        )
        actual = recovery_forward(
            student,
            teacher,
            jnp.arange(8, dtype=jnp.int32).reshape(1, 8),
            affected_layers=_AFFECTED_LAYERS,
            source_to_shared=permutation,
        )

    np.testing.assert_allclose(actual.router_top1_agreement_with_teacher, 1.0)
    np.testing.assert_allclose(actual.router_topk_agreement_with_teacher, 1.0)


def test_local_recovery_updates_only_shared_bank_and_keeps_qb_frozen():
    optimizer = optax.sgd(1e-3)
    config = MergeRecoveryConfig(affected_layers=_AFFECTED_LAYERS, stage=RecoveryStage.LOCAL)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher, student = _teacher_and_student()
        pending_qb_betas = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10.0
        state = initial_recovery_state(
            student,
            optimizer=optimizer,
            pending_qb_betas=pending_qb_betas,
            config=config,
        )
        updated, losses = make_recovery_train_step(optimizer, config)(
            state,
            teacher,
            jnp.arange(8, dtype=jnp.int32).reshape(1, 8),
            None,
        )

    assert int(updated.step) == 1
    assert float(losses.moe) > 0
    assert _tree_changed(student.expert_banks[1], updated.params.expert_banks[1])
    _assert_tree_equal(updated.params.expert_banks[0], student.expert_banks[0])
    _assert_tree_equal(updated.params.expert_banks[2], student.expert_banks[2])
    for before_block, after_block in zip(student.blocks, updated.params.blocks, strict=True):
        np.testing.assert_array_equal(after_block.mlp.router, before_block.mlp.router)
        np.testing.assert_array_equal(after_block.mlp.router_bias, before_block.mlp.router_bias)
    np.testing.assert_array_equal(updated.pending_qb_betas, pending_qb_betas)


def test_preservation_recovery_trains_affected_routers_and_updates_only_their_qb_state():
    optimizer = optax.sgd(1e-3)
    config = MergeRecoveryConfig(affected_layers=_AFFECTED_LAYERS, stage=RecoveryStage.PRESERVATION)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        teacher, student = _teacher_and_student()
        trainable, _ = eqx.partition(student, recovery_trainable_filter(student, config))
        pending_qb_betas = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10.0
        state = initial_recovery_state(
            student,
            optimizer=optimizer,
            pending_qb_betas=pending_qb_betas,
            config=config,
        )
        updated, losses = make_recovery_train_step(optimizer, config)(
            state,
            teacher,
            jnp.arange(8, dtype=jnp.int32).reshape(1, 8),
            jnp.ones((1, 8), dtype=jnp.float32),
        )

    assert trainable.blocks[0].mlp.router is None
    assert trainable.blocks[3].mlp.router is None
    assert trainable.blocks[1].mlp.router is not None
    assert trainable.blocks[2].mlp.router is not None
    assert all(block.mlp.router_bias is None for block in trainable.blocks)
    assert _tree_changed(student.expert_banks[1], updated.params.expert_banks[1])
    _assert_tree_equal(updated.params.expert_banks[0], student.expert_banks[0])
    _assert_tree_equal(updated.params.expert_banks[2], student.expert_banks[2])

    affected_router_changed = False
    for layer, (before_block, after_block) in enumerate(zip(student.blocks, updated.params.blocks, strict=True)):
        if layer in _AFFECTED_LAYERS:
            affected_router_changed |= not np.array_equal(after_block.mlp.router, before_block.mlp.router)
            expected_bias = -pending_qb_betas[layer]
            expected_bias -= jnp.mean(expected_bias)
            np.testing.assert_allclose(after_block.mlp.router_bias, expected_bias, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(updated.pending_qb_betas[layer], losses.qb_beta_per_layer[layer])
        else:
            np.testing.assert_array_equal(after_block.mlp.router, before_block.mlp.router)
            np.testing.assert_array_equal(after_block.mlp.router_bias, before_block.mlp.router_bias)
            np.testing.assert_array_equal(updated.pending_qb_betas[layer], pending_qb_betas[layer])
    assert affected_router_changed
