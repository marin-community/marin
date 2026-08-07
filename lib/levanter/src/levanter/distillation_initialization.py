# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray

from levanter.models.qwen import Qwen3LMHeadModel


class TeacherInitialization(StrEnum):
    STRUCTURED = "structured"
    FACTORIZED = "factorized"


@dataclass(frozen=True)
class QwenAxisMapping:
    indices: dict[str, tuple[int, ...]]


_FACTORIZED_WEIGHT_AXES = {
    "token_embeddings.weight": (("vocab",), ("embed",)),
    "lm_head.weight": (("vocab",), ("embed",)),
    "q_proj.weight": (("kv_head", "q_heads_per_group", "head_size"), ("embed",)),
    "k_proj.weight": (("kv_head", "head_size"), ("embed",)),
    "v_proj.weight": (("kv_head", "head_size"), ("embed",)),
    "o_proj.weight": (("embed",), ("heads", "head_size")),
    "gate_proj.weight": (("mlp",), ("embed",)),
    "up_proj.weight": (("mlp",), ("embed",)),
    "down_proj.weight": (("embed",), ("mlp",)),
}


def evenly_spaced_indices(source_size: int, target_size: int) -> tuple[int, ...]:
    if source_size < target_size:
        raise ValueError(f"Cannot select {target_size} coordinates from source size {source_size}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if target_size == source_size:
        return tuple(range(source_size))
    return tuple(int(index) for index in jnp.linspace(0, source_size - 1, target_size).round())


def default_qwen_axis_mapping(
    student: Qwen3LMHeadModel,
    teacher: Qwen3LMHeadModel,
) -> QwenAxisMapping:
    student_axes = {
        axis.name: axis
        for axis in (
            student.config.Layers,
            student.config.Embed,
            student.config.Mlp,
            student.config.attention_config().Heads,
            student.config.attention_config().KVHeads,
            student.config.attention_config().QHeadsPerGroup,
            student.config.attention_config().HeadSize,
            student.Vocab,
        )
    }
    teacher_axes = {
        axis.name: axis
        for axis in (
            teacher.config.Layers,
            teacher.config.Embed,
            teacher.config.Mlp,
            teacher.config.attention_config().Heads,
            teacher.config.attention_config().KVHeads,
            teacher.config.attention_config().QHeadsPerGroup,
            teacher.config.attention_config().HeadSize,
            teacher.Vocab,
        )
    }
    indices = {}
    for name, student_axis in student_axes.items():
        teacher_axis = teacher_axes[name]
        indices[name] = evenly_spaced_indices(teacher_axis.size, student_axis.size)
    return QwenAxisMapping(indices)


def _highest_scoring_indices(scores: jax.Array, target_size: int) -> tuple[int, ...]:
    if scores.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {scores.shape}")
    if target_size > scores.shape[0]:
        raise ValueError(f"Cannot select {target_size} coordinates from {scores.shape[0]} scores")
    selected = jnp.argsort(scores, descending=True)[:target_size]
    return tuple(int(index) for index in jnp.sort(selected))


def saliency_qwen_axis_mapping(
    student: Qwen3LMHeadModel,
    teacher: Qwen3LMHeadModel,
) -> QwenAxisMapping:
    """Select teacher coordinates by weight energy while preserving complete attention heads."""
    mapping = default_qwen_axis_mapping(student, teacher)
    indices = dict(mapping.indices)

    embedding = teacher.embeddings.token_embeddings.weight
    embed_scores = hax.mean(hax.square(embedding.astype(jnp.float32)), axis=teacher.Vocab).array
    indices["embed"] = _highest_scoring_indices(embed_scores, student.config.Embed.size)

    layers = teacher.transformer.layers.stacked
    gate_scores = hax.mean(
        hax.square(layers.mlp.gate_proj.weight.astype(jnp.float32)),
        axis=("layer", "embed"),
    )
    up_scores = hax.mean(
        hax.square(layers.mlp.up_proj.weight.astype(jnp.float32)),
        axis=("layer", "embed"),
    )
    down_scores = hax.mean(
        hax.square(layers.mlp.down_proj.weight.astype(jnp.float32)),
        axis=("layer", "embed"),
    )
    mlp_scores = gate_scores + up_scores + down_scores
    indices["mlp"] = _highest_scoring_indices(mlp_scores.array, student.config.Mlp.size)

    q_scores = hax.mean(
        hax.square(layers.self_attn.q_proj.weight.astype(jnp.float32)),
        axis=("layer", "head_size", "embed"),
    )
    q_group_scores = hax.mean(q_scores, axis="kv_head").array
    q_groups = _highest_scoring_indices(
        q_group_scores,
        student.config.attention_config().QHeadsPerGroup.size,
    )
    indices["q_heads_per_group"] = q_groups

    teacher_q_groups = teacher.config.attention_config().QHeadsPerGroup.size
    kv_heads = indices["kv_head"]
    indices["heads"] = tuple(kv_head * teacher_q_groups + q_group for kv_head in kv_heads for q_group in q_groups)
    return QwenAxisMapping(indices)


def _select_named_array(
    source: NamedArray,
    target: NamedArray,
    axis_mapping: QwenAxisMapping,
) -> NamedArray:
    selected = source
    for target_axis in target.axes:
        source_axis = selected.resolve_axis(target_axis.name)
        if source_axis.size == target_axis.size:
            continue
        indices = axis_mapping.indices[target_axis.name]
        index = hax.named(jnp.asarray(indices, dtype=jnp.int32), target_axis)
        selected = hax.take(selected, source_axis, index)
    return selected.rearrange(target.axes)


def _randomized_low_rank_approximation(
    matrix: jax.Array,
    rank: int,
    *,
    key: jax.Array,
    oversample: int = 16,
    power_iterations: int = 1,
) -> jax.Array:
    output_size, input_size = matrix.shape[-2:]
    rank = min(rank, output_size, input_size)
    projection_size = min(rank + oversample, output_size, input_size)
    omega_shape = (*matrix.shape[:-2], input_size, projection_size)
    omega = jax.random.normal(key, omega_shape, dtype=jnp.float32)
    matrix = matrix.astype(jnp.float32)
    projected = matrix @ omega
    for _ in range(power_iterations):
        projected = matrix @ (jnp.swapaxes(matrix, -1, -2) @ projected)
    basis, _ = jnp.linalg.qr(projected, mode="reduced")
    compressed = jnp.swapaxes(basis, -1, -2) @ matrix
    left, singular_values, right = jnp.linalg.svd(compressed, full_matrices=False)
    left = left[..., :rank]
    singular_values = singular_values[..., :rank]
    right = right[..., :rank, :]
    approximation = (basis @ left) * singular_values[..., None, :]
    approximation = approximation @ right

    source_rms = jnp.sqrt(jnp.mean(jnp.square(matrix), axis=-1, keepdims=True))
    approximation_rms = jnp.sqrt(jnp.mean(jnp.square(approximation), axis=-1, keepdims=True))
    approximation = approximation * source_rms / jnp.maximum(approximation_rms, 1e-8)
    return approximation


def _factorize_named_weight(
    weight: NamedArray,
    path: str,
    *,
    rank: int,
    key: jax.Array,
) -> NamedArray:
    matching_suffix = next((suffix for suffix in _FACTORIZED_WEIGHT_AXES if suffix in path), None)
    if matching_suffix is None:
        return weight
    output_names, input_names = _FACTORIZED_WEIGHT_AXES[matching_suffix]
    layer_axes = tuple(axis for axis in weight.axes if axis.name == "layer")
    output_axes = tuple(weight.resolve_axis(name) for name in output_names)
    input_axes = tuple(weight.resolve_axis(name) for name in input_names)
    arranged = weight.rearrange((*layer_axes, *output_axes, *input_axes))
    layer_shape = tuple(axis.size for axis in layer_axes)
    output_size = hax.axis_size(output_axes)
    input_size = hax.axis_size(input_axes)
    matrix = arranged.array.reshape((*layer_shape, output_size, input_size))
    approximation = _randomized_low_rank_approximation(matrix, rank, key=key)
    approximation = approximation.reshape(tuple(axis.size for axis in arranged.axes))
    return hax.named(approximation.astype(weight.dtype), arranged.axes).rearrange(weight.axes)


def initialize_qwen_from_teacher(
    student: Qwen3LMHeadModel,
    teacher: Qwen3LMHeadModel,
    *,
    method: TeacherInitialization,
    axis_mapping: QwenAxisMapping | None = None,
    rank: int = 512,
    key: jax.Array,
) -> Qwen3LMHeadModel:
    """Initialize a smaller Qwen model from deterministic teacher coordinates."""
    if student.Vocab.size != teacher.Vocab.size:
        raise ValueError(
            f"Teacher and student vocabularies must match, got {teacher.Vocab.size} and {student.Vocab.size}"
        )
    if axis_mapping is None:
        axis_mapping = default_qwen_axis_mapping(student, teacher)

    teacher_leaves = {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(
            teacher,
            is_leaf=lambda value: isinstance(value, NamedArray),
        )[0]
        if isinstance(leaf, NamedArray)
    }

    def initialize_leaf(path, target):
        if not isinstance(target, NamedArray):
            return target
        path_string = jax.tree_util.keystr(path)
        source = teacher_leaves.get(path_string)
        if source is None:
            return target
        initialized = _select_named_array(source, target, axis_mapping)
        if method == TeacherInitialization.FACTORIZED:
            path_key = jax.random.fold_in(key, sum(path_string.encode("utf-8")))
            initialized = _factorize_named_weight(initialized, path_string, rank=rank, key=path_key)
        return initialized

    initialized = jax.tree_util.tree_map_with_path(
        initialize_leaf,
        student,
        is_leaf=lambda value: isinstance(value, NamedArray),
    )

    selected_embedding = initialized.embeddings.token_embeddings.weight
    if teacher.lm_head is not None:
        selected_head = _select_named_array(
            teacher.lm_head.weight,
            selected_embedding,
            axis_mapping,
        )
        if method == TeacherInitialization.FACTORIZED:
            selected_head = _factorize_named_weight(
                selected_head,
                "lm_head.weight",
                rank=rank,
                key=jax.random.fold_in(key, sum(b"lm_head.weight")),
            )
        selected_embedding = (selected_embedding + selected_head) * 0.5
        initialized = eqx.tree_at(
            lambda model: model.embeddings.token_embeddings.weight,
            initialized,
            selected_embedding,
        )

    return dataclasses.replace(initialized, lm_head=None) if student.lm_head is None else initialized
