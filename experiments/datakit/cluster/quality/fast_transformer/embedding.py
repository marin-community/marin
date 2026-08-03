# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training loss and batched inference for fast-transformer embeddings."""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from experiments.datakit.cluster.quality.fast_transformer.inference import data_parallel_shardings
from experiments.datakit.cluster.quality.fast_transformer.model import FastEmbeddingTransformer

_PREDICT_TOKEN_BUDGET = 262_144


def source_balanced_token_remap(
    source_token_counts: Sequence[np.ndarray],
    compact_vocab_size: int,
) -> np.ndarray:
    """Select tokens by mean within-source frequency and return a dense remap."""
    if compact_vocab_size < 3:
        raise ValueError(f"Compact vocabulary must hold reserved and text tokens, got {compact_vocab_size}")
    if not source_token_counts:
        raise ValueError("At least one source token count is required")
    raw_vocab_size = len(source_token_counts[0])
    if any(counts.shape != (raw_vocab_size,) for counts in source_token_counts):
        raise ValueError("Source token count arrays have different shapes")
    scores = np.zeros(raw_vocab_size, dtype=np.float64)
    for counts in source_token_counts:
        total = int(counts.sum())
        if total == 0:
            raise ValueError("A source has no tokens")
        scores += counts / total
    raw_ids = np.arange(raw_vocab_size)
    present = raw_ids[scores > 0]
    ranked = present[np.lexsort((present, -scores[present]))]
    selected = ranked[: compact_vocab_size - 2]
    remap = np.ones(raw_vocab_size, dtype=np.int32)
    remap[selected] = np.arange(2, len(selected) + 2, dtype=np.int32)
    return remap


def pack_remapped_windows(
    raw_windows: Sequence[Sequence[Sequence[int]]],
    raw_to_compact: np.ndarray,
    max_tokens: int,
    tokens_per_window: int,
) -> np.ndarray:
    """Pack fixed document windows into remapped, padded token rows."""
    if raw_to_compact.ndim != 1:
        raise ValueError(f"Expected a one-dimensional token remap, got {raw_to_compact.shape}")
    if raw_windows and len(raw_windows[0]) * tokens_per_window > max_tokens:
        raise ValueError("Document windows do not fit in the fixed token row")
    output = np.zeros((len(raw_windows), max_tokens), dtype=np.int32)
    for document_index, windows in enumerate(raw_windows):
        if document_index and len(windows) != len(raw_windows[0]):
            raise ValueError("Documents have different window counts")
        for window_index, raw_ids in enumerate(windows):
            row = np.asarray(raw_ids[:tokens_per_window], dtype=np.int64)
            if len(row) == 0:
                continue
            if int(row.max()) >= len(raw_to_compact) or int(row.min()) < 0:
                raise ValueError("Tokenizer returned an ID outside the remap")
            start = window_index * tokens_per_window
            output[document_index, start : start + len(row)] = raw_to_compact[row]
    return output


def _off_diagonal(matrix: Array) -> Array:
    batch_size = matrix.shape[0]
    return matrix.reshape(-1)[1:].reshape(batch_size - 1, batch_size + 1)[:, :-1].reshape(batch_size, batch_size - 1)


def contrastive_embedding_loss(student: Array, teacher: Array, temperature: float) -> Array:
    """Match teacher pairwise geometry with the Luxical Gram-KL objective."""
    if student.ndim != 2 or teacher.ndim != 2 or student.shape[0] != teacher.shape[0]:
        raise ValueError(f"Student rows {student.shape} do not match teacher rows {teacher.shape}")
    if student.shape[0] < 2:
        raise ValueError(f"Expected at least two embedding rows, got {student.shape}")
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    student = student / jnp.maximum(jnp.linalg.norm(student, axis=1, keepdims=True), 1e-12)
    teacher = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    student_logits = _off_diagonal(student @ student.T) / temperature
    teacher_logits = _off_diagonal(teacher @ teacher.T) / temperature
    student_log_probabilities = jax.nn.log_softmax(student_logits, axis=1)
    teacher_log_probabilities = jax.nn.log_softmax(teacher_logits, axis=1)
    divergence = jnp.sum(
        jnp.exp(teacher_log_probabilities) * (teacher_log_probabilities - student_log_probabilities),
        axis=1,
    )
    return temperature**2 * divergence.mean()


def cross_source_teacher_neighbor_loss(
    student: Array,
    teacher: Array,
    source_ids: Array,
    positive_count: int,
    temperature: float,
) -> Array:
    """Move each student vector toward the teacher's nearest cross-source rows."""
    if student.ndim != 2 or teacher.ndim != 2 or student.shape[0] != teacher.shape[0]:
        raise ValueError(f"Student rows {student.shape} do not match teacher rows {teacher.shape}")
    if source_ids.shape != (student.shape[0],):
        raise ValueError(f"Source shape {source_ids.shape} does not match embedding rows {student.shape[0]}")
    if positive_count < 1 or positive_count >= student.shape[0]:
        raise ValueError(f"Positive count {positive_count} is invalid for {student.shape[0]} rows")
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    student = student / jnp.maximum(jnp.linalg.norm(student, axis=1, keepdims=True), 1e-12)
    teacher = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    cross_source = source_ids[:, None] != source_ids[None, :]
    teacher_similarity = jnp.where(cross_source, teacher @ teacher.T, -jnp.inf)
    _, positive_indices = jax.lax.top_k(teacher_similarity, positive_count)
    student_logits = jnp.where(cross_source, student @ student.T / temperature, -jnp.inf)
    student_log_probabilities = jax.nn.log_softmax(student_logits, axis=1)
    positive_log_probabilities = jnp.take_along_axis(student_log_probabilities, positive_indices, axis=1)
    return -positive_log_probabilities.mean()


def embedding_spread_loss(student: Array, standard_deviation_target: float, covariance_weight: float) -> Array:
    """Keep normalized student dimensions variable and weakly correlated."""
    if student.ndim != 2 or student.shape[0] < 2:
        raise ValueError(f"Expected at least two embedding rows, got {student.shape}")
    if standard_deviation_target <= 0:
        raise ValueError(f"Standard-deviation target must be positive, got {standard_deviation_target}")
    if covariance_weight < 0:
        raise ValueError(f"Covariance weight must be nonnegative, got {covariance_weight}")

    student = student / jnp.maximum(jnp.linalg.norm(student, axis=1, keepdims=True), 1e-12)
    centered = student - student.mean(axis=0, keepdims=True)
    dimension_variance = jnp.sum(jnp.square(centered), axis=0) / (student.shape[0] - 1)
    variance = jnp.mean(jax.nn.relu(standard_deviation_target - jnp.sqrt(dimension_variance + 1e-6)))
    covariance = centered.T @ centered / (student.shape[0] - 1)
    off_diagonal_covariance = _off_diagonal(covariance)
    decorrelation = jnp.sum(jnp.square(off_diagonal_covariance)) / student.shape[1]
    return variance + covariance_weight * decorrelation


def direct_cosine_embedding_loss(student: Array, teacher: Array) -> Array:
    """Align each student vector with its matching teacher vector."""
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError(f"Student shape {student.shape} does not match the teacher matrix {teacher.shape}")
    student = student / jnp.maximum(jnp.linalg.norm(student, axis=1, keepdims=True), 1e-12)
    teacher = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    return jnp.mean(1.0 - jnp.sum(student * teacher, axis=1))


def projected_embedding_distillation_loss(
    student: Array,
    teacher: Array,
    projection: Array,
    temperature: float,
    direct_cosine_weight: float,
) -> Array:
    """Match cross-dimension geometry and align projected student vectors."""
    if student.ndim != 2 or teacher.ndim != 2 or student.shape[0] != teacher.shape[0]:
        raise ValueError(f"Student rows {student.shape} do not match teacher rows {teacher.shape}")
    expected_projection_shape = (student.shape[1], teacher.shape[1])
    if projection.shape != expected_projection_shape:
        raise ValueError(f"Projection shape {projection.shape} does not match {expected_projection_shape}")
    if direct_cosine_weight < 0:
        raise ValueError(f"Direct cosine weight must be nonnegative, got {direct_cosine_weight}")
    projected_student = student @ projection
    return contrastive_embedding_loss(
        student, teacher, temperature
    ) + direct_cosine_weight * direct_cosine_embedding_loss(
        projected_student,
        teacher,
    )


def source_conditioned_geometry_loss(student: Array, teacher: Array, source_ids: Array) -> Array:
    """Match student and teacher cosine pairs from the same source."""
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError(f"Student shape {student.shape} does not match the teacher matrix {teacher.shape}")
    if source_ids.shape != (student.shape[0],):
        raise ValueError(f"Source shape {source_ids.shape} does not match embedding rows {student.shape[0]}")
    student = student / jnp.maximum(jnp.linalg.norm(student, axis=1, keepdims=True), 1e-12)
    teacher = teacher / jnp.maximum(jnp.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    squared_error = jnp.square(student @ student.T - teacher @ teacher.T)
    same_source = source_ids[:, None] == source_ids[None, :]
    off_diagonal = ~jnp.eye(student.shape[0], dtype=bool)
    selected = same_source & off_diagonal
    pair_count = jnp.sum(selected)
    return jnp.sum(jnp.where(selected, squared_error, 0.0)) / jnp.maximum(pair_count, 1)


def embedding_distillation_loss(
    student: Array,
    teacher: Array,
    temperature: float,
    direct_cosine_weight: float,
) -> Array:
    """Combine pairwise geometry and direct teacher alignment."""
    if direct_cosine_weight < 0:
        raise ValueError(f"Direct cosine weight must be nonnegative, got {direct_cosine_weight}")
    return contrastive_embedding_loss(
        student, teacher, temperature
    ) + direct_cosine_weight * direct_cosine_embedding_loss(student, teacher)


@eqx.filter_jit
def _predict_batch(model: FastEmbeddingTransformer, ids: Array) -> Array:
    return model(ids, key=None, inference=True)


def predict_embeddings(
    model: FastEmbeddingTransformer,
    ids: np.ndarray,
    batch_size: int | None = None,
) -> np.ndarray:
    """Return normalized vectors for each row in a fixed-width token array."""
    if ids.ndim != 2:
        raise ValueError(f"Expected a two-dimensional token array, got {ids.shape}")
    if batch_size is None:
        batch_size = max(8, _PREDICT_TOKEN_BUDGET // ids.shape[1])
    device_count, _, batch_sharding = data_parallel_shardings()
    batch_size = max(device_count, (batch_size // device_count) * device_count)
    output: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        padding = batch_size - len(chunk)
        if padding:
            chunk = np.concatenate(
                [chunk, np.zeros((padding, ids.shape[1]), dtype=ids.dtype)],
                axis=0,
            )
        vectors = np.asarray(_predict_batch(model, jax.device_put(jnp.asarray(chunk), batch_sharding)))
        output.append(vectors[: len(chunk) - padding] if padding else vectors)
    return np.concatenate(output)
