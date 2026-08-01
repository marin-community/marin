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
    mask = ~jnp.eye(batch_size, dtype=jnp.bool_)
    return matrix[mask].reshape(batch_size, batch_size - 1)


def contrastive_embedding_loss(student: Array, teacher: Array, temperature: float) -> Array:
    """Match teacher pairwise geometry with the Luxical Gram-KL objective."""
    if student.shape != teacher.shape:
        raise ValueError(f"Student shape {student.shape} does not match teacher shape {teacher.shape}")
    if student.ndim != 2 or student.shape[0] < 2:
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
