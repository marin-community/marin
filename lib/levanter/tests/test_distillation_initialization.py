# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax

from levanter.distillation_initialization import (
    TeacherInitialization,
    default_qwen_axis_mapping,
    initialize_qwen_from_teacher,
    saliency_qwen_axis_mapping,
)
from levanter.models.qwen import Qwen3Config


def _qwen_pair():
    Vocab = hax.Axis("vocab", 32)
    teacher_config = Qwen3Config(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        tie_word_embeddings=False,
    )
    student_config = Qwen3Config(
        max_seq_len=8,
        hidden_dim=8,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        tie_word_embeddings=True,
    )
    teacher = teacher_config.build(Vocab, key=jrandom.PRNGKey(0))
    student = student_config.build(Vocab, key=jrandom.PRNGKey(1))
    return student, teacher


def test_structured_qwen_initialization_selects_named_axes_and_ties_embeddings():
    student, teacher = _qwen_pair()
    mapping = default_qwen_axis_mapping(student, teacher)
    initialized = initialize_qwen_from_teacher(
        student,
        teacher,
        method=TeacherInitialization.STRUCTURED,
        axis_mapping=mapping,
        key=jrandom.PRNGKey(2),
    )

    assert initialized.lm_head is None
    assert initialized.transformer.layers.stacked.mlp.gate_proj.weight.axes == (
        student.transformer.layers.stacked.mlp.gate_proj.weight.axes
    )
    layer_indices = jnp.asarray(mapping.indices["layer"])
    mlp_indices = jnp.asarray(mapping.indices["mlp"])
    embed_indices = jnp.asarray(mapping.indices["embed"])
    expected_gate = teacher.transformer.layers.stacked.mlp.gate_proj.weight.array[
        layer_indices[:, None, None],
        mlp_indices[None, :, None],
        embed_indices[None, None, :],
    ]
    np.testing.assert_allclose(
        initialized.transformer.layers.stacked.mlp.gate_proj.weight.array,
        expected_gate,
    )

    teacher_embedding = teacher.embeddings.token_embeddings.weight.array[:, embed_indices]
    teacher_head = teacher.lm_head.weight.array[:, embed_indices]
    np.testing.assert_allclose(
        initialized.embeddings.token_embeddings.weight.array,
        0.5 * (teacher_embedding + teacher_head),
    )


def test_factorized_qwen_initialization_has_requested_matrix_rank():
    student, teacher = _qwen_pair()
    initialized = initialize_qwen_from_teacher(
        student,
        teacher,
        method=TeacherInitialization.FACTORIZED,
        rank=2,
        key=jrandom.PRNGKey(3),
    )

    gate = initialized.transformer.layers.stacked.mlp.gate_proj.weight.array
    for layer in gate:
        assert int(jnp.linalg.matrix_rank(layer, tol=1e-4)) <= 2
    embedding = initialized.embeddings.token_embeddings.weight.array
    assert int(jnp.linalg.matrix_rank(embedding, tol=1e-4)) <= 4


def test_qwen_initialization_is_deterministic():
    student, teacher = _qwen_pair()
    first = initialize_qwen_from_teacher(
        student,
        teacher,
        method=TeacherInitialization.FACTORIZED,
        rank=2,
        key=jrandom.PRNGKey(4),
    )
    second = initialize_qwen_from_teacher(
        student,
        teacher,
        method=TeacherInitialization.FACTORIZED,
        rank=2,
        key=jrandom.PRNGKey(4),
    )
    first_leaves = jax.tree.leaves(first, is_leaf=lambda value: isinstance(value, hax.NamedArray))
    second_leaves = jax.tree.leaves(second, is_leaf=lambda value: isinstance(value, hax.NamedArray))
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        if isinstance(first_leaf, hax.NamedArray):
            np.testing.assert_array_equal(first_leaf.array, second_leaf.array)


def test_saliency_mapping_preserves_complete_attention_heads():
    student, teacher = _qwen_pair()
    mapping = saliency_qwen_axis_mapping(student, teacher)
    teacher_groups = teacher.config.attention_config().QHeadsPerGroup.size
    expected_heads = tuple(
        kv_head * teacher_groups + q_group
        for kv_head in mapping.indices["kv_head"]
        for q_group in mapping.indices["q_heads_per_group"]
    )
    assert mapping.indices["heads"] == expected_heads
    assert len(mapping.indices["embed"]) == student.config.Embed.size
    assert len(mapping.indices["mlp"]) == student.config.Mlp.size
